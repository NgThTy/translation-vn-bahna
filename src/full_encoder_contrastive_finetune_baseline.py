"""
Baseline 8: Full encoder contrastive fine-tuning.

This baseline fine-tunes the full source and target encoders with a contrastive loss.

Pipeline:
1. Load Bahnaric-Vietnamese training pairs from train.csv.
2. Encode Bahnaric sentences with a source encoder.
3. Encode Vietnamese sentences with a target encoder.
4. Pool token representations into sentence embeddings.
5. Optimize symmetric InfoNCE contrastive loss.
6. After training, encode all test Bahnaric queries and Vietnamese candidates.
7. Retrieve Vietnamese candidates by cosine similarity or CSLS.
8. Report Top1 accuracy, MRR, Hit@K, Recall@K, Precision@K.

Important:
- This baseline updates the full encoder parameters.
- It is much heavier than LoRA fine-tuning.
- Run this on the GPU compute node, not on the head node.

Recommended first GPU run:
python src/full_encoder_contrastive_finetune_baseline.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --src_model xlm-roberta-base \
  --tgt_model xlm-roberta-base \
  --output_dir results/baselines/full_encoder_xlmr_infonce_3ep_cosine \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --lr 2e-5 \
  --pooling sentence_mean \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10 \
  --fp16
"""

import argparse
import json
import math
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(
    text: str,
    lowercase: bool = False,
    strip_accents: bool = False,
    remove_punct: bool = False,
) -> str:
    text = str(text)
    text = unicodedata.normalize("NFC", text)

    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")

    if lowercase:
        text = text.lower()

    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = unicodedata.normalize("NFC", text)

    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    text = re.sub(r"\s+", " ", text).strip()
    return text


class BahVnDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        lowercase: bool = False,
        strip_accents: bool = False,
        remove_punct: bool = False,
    ):
        df = pd.read_csv(csv_path).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)
        if "Bahnaric" not in df.columns or "Vietnamese" not in df.columns:
            raise ValueError("CSV must contain columns: Bahnaric,Vietnamese")

        self.src_texts = [
            normalize_text(x, lowercase=lowercase, strip_accents=strip_accents, remove_punct=remove_punct)
            for x in df["Bahnaric"].astype(str).tolist()
        ]
        self.tgt_texts = [
            normalize_text(x, lowercase=lowercase, strip_accents=strip_accents, remove_punct=remove_punct)
            for x in df["Vietnamese"].astype(str).tolist()
        ]

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.src_texts[idx], self.tgt_texts[idx]


def collate_batch(
    batch,
    src_tok,
    tgt_tok,
    src_max_len: int,
    tgt_max_len: int,
):
    src_texts, tgt_texts = zip(*batch)

    src_inputs = src_tok(
        list(src_texts),
        padding=True,
        truncation=True,
        max_length=src_max_len,
        return_tensors="pt",
    )
    tgt_inputs = tgt_tok(
        list(tgt_texts),
        padding=True,
        truncation=True,
        max_length=tgt_max_len,
        return_tensors="pt",
    )

    return src_inputs, tgt_inputs


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def cls_pool(last_hidden: torch.Tensor) -> torch.Tensor:
    return last_hidden[:, 0, :]


def encode_batch(model, inputs: Dict[str, torch.Tensor], pooling: str) -> torch.Tensor:
    outputs = model(**inputs, return_dict=True)
    last = outputs.last_hidden_state

    if pooling == "sentence_mean":
        return mean_pool(last, inputs["attention_mask"])

    if pooling == "cls":
        return cls_pool(last)

    raise ValueError(f"Unknown pooling mode: {pooling}")


def symmetric_infonce(src_emb: torch.Tensor, tgt_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    src = F.normalize(src_emb, p=2, dim=-1)
    tgt = F.normalize(tgt_emb, p=2, dim=-1)

    logits = src @ tgt.T
    logits = logits / max(temperature, 1e-8)

    labels = torch.arange(src.size(0), device=src.device)

    loss_src = F.cross_entropy(logits, labels)
    loss_tgt = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_src + loss_tgt)


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def cosine_topk(query: np.ndarray, index: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    q = l2_normalize_rows(query)
    z = l2_normalize_rows(index)

    sims = q @ z.T
    k = int(max(1, min(topk, index.shape[0])))
    topk_idx = np.argsort(-sims, axis=1)[:, :k]

    return topk_idx, sims


def csls_topk(query: np.ndarray, index: np.ndarray, topk: int, csls_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    q = l2_normalize_rows(query)
    z = l2_normalize_rows(index)
    sims = q @ z.T

    if q.shape[0] <= 1 or z.shape[0] <= 1:
        return cosine_topk(query, index, topk)

    k_csls = int(max(1, min(csls_k, q.shape[0] - 1, z.shape[0] - 1)))

    rq = np.partition(sims, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
    rz = np.partition(sims, -k_csls, axis=0)[-k_csls:, :].mean(axis=0)

    csls = 2.0 * sims - rq[:, None] - rz[None, :]

    k = int(max(1, min(topk, index.shape[0])))
    topk_idx = np.argsort(-csls, axis=1)[:, :k]

    return topk_idx, csls


def ranking_metrics(
    topk_idx: np.ndarray,
    gold_idx: np.ndarray,
    eval_ks: List[int],
) -> Tuple[Dict[str, float], np.ndarray]:
    n_items = topk_idx.shape[0]
    ranks = np.full(n_items, np.inf, dtype=np.float64)

    for i in range(n_items):
        hits = np.where(topk_idx[i] == gold_idx[i])[0]
        if len(hits) > 0:
            ranks[i] = float(hits[0] + 1)

    metrics: Dict[str, float] = {}
    metrics["MRR"] = float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)))
    metrics["Top1_acc"] = float(np.mean(ranks == 1.0))

    for k in eval_ks:
        k_eff = int(max(1, min(k, topk_idx.shape[1])))
        hit = float(np.mean(ranks <= k_eff))
        metrics[f"Hit@{k_eff}"] = hit
        metrics[f"Recall@{k_eff}"] = hit
        metrics[f"Precision@{k_eff}"] = hit / float(k_eff)

    return metrics, ranks


def make_bucket_columns(df_out: pd.DataFrame) -> pd.DataFrame:
    df_out = df_out.copy()

    df_out["Bahnaric_len_chars"] = df_out["Bahnaric"].astype(str).str.len()
    df_out["Vietnamese_len_chars"] = df_out["Gold_VN"].astype(str).str.len()
    df_out["Vietnamese_len_words"] = df_out["Gold_VN"].astype(str).str.split().map(len)

    def vn_len_bin(n: int) -> str:
        if n <= 5:
            return "1-5"
        if n <= 15:
            return "6-15"
        if n <= 30:
            return "16-30"
        return ">30"

    df_out["VN_len_bin"] = df_out["Vietnamese_len_words"].map(vn_len_bin)
    return df_out


def save_bucket_metrics(df_out: pd.DataFrame, output_dir: Path, eval_ks: List[int]) -> None:
    metric_cols = ["P@1", "MRR"] + [f"Hit@{k}" for k in eval_ks if f"Hit@{k}" in df_out.columns]
    available_cols = [col for col in metric_cols if col in df_out.columns]

    grouped = df_out.groupby("VN_len_bin", dropna=False)
    bucket_df = grouped[available_cols].mean().reset_index()
    bucket_df["count"] = grouped.size().values

    for col in available_cols:
        bucket_df[col] = bucket_df[col].astype(float).round(4)

    bucket_df.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_len: int,
    batch_size: int,
    pooling: str,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> np.ndarray:
    model.eval()
    all_embs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch = texts[start : start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        inputs = move_to_device(inputs, device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp and device.type == "cuda"):
            emb = encode_batch(model, inputs, pooling=pooling)

        all_embs.append(emb.detach().float().cpu().numpy())

    return np.vstack(all_embs).astype(np.float32)


def save_models(src_model, tgt_model, src_tok, tgt_tok, output_dir: Path) -> None:
    src_dir = output_dir / "src_encoder"
    tgt_dir = output_dir / "tgt_encoder"

    src_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    src_model.save_pretrained(src_dir)
    tgt_model.save_pretrained(tgt_dir)
    src_tok.save_pretrained(src_dir)
    tgt_tok.save_pretrained(tgt_dir)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print(f"Using device: {device}")
    print("Baseline: full_encoder_contrastive_finetuning")
    print(f"Source model: {args.src_model}")
    print(f"Target model: {args.tgt_model}")
    print(f"Pooling: {args.pooling}")
    print(f"Loss: symmetric InfoNCE")
    print(f"Full encoder fine-tuning: true")

    if device.type == "cpu":
        print("WARNING: Running full encoder fine-tuning on CPU will be very slow. Use a GPU compute node.")

    src_tok = AutoTokenizer.from_pretrained(args.src_model)
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)

    src_model = AutoModel.from_pretrained(args.src_model).to(device)
    tgt_model = AutoModel.from_pretrained(args.tgt_model).to(device)

    src_model.train()
    tgt_model.train()

    train_ds = BahVnDataset(
        args.train_csv,
        lowercase=args.lowercase,
        strip_accents=args.strip_accents,
        remove_punct=args.remove_punct,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda b: collate_batch(
            b,
            src_tok,
            tgt_tok,
            args.src_max_len,
            args.tgt_max_len,
        ),
    )

    trainable_params = list(src_model.parameters()) + list(tgt_model.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_steps = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = bool(args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        src_model.train()
        tgt_model.train()

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for step, (src_inputs, tgt_inputs) in enumerate(pbar, start=1):
            src_inputs = move_to_device(src_inputs, device)
            tgt_inputs = move_to_device(tgt_inputs, device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                src_emb = encode_batch(src_model, src_inputs, pooling=args.pooling)
                tgt_emb = encode_batch(tgt_model, tgt_inputs, pooling=args.pooling)
                loss = symmetric_infonce(src_emb, tgt_emb, temperature=args.temperature)
                loss = loss / max(1, args.grad_accum_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.detach().cpu()) * max(1, args.grad_accum_steps)

            should_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))

            if should_step:
                if args.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            avg_loss = running_loss / max(1, step)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} train_loss={epoch_loss:.6f}")

        if args.save_each_epoch:
            epoch_dir = output_dir / f"checkpoint_epoch_{epoch}"
            save_models(src_model, tgt_model, src_tok, tgt_tok, epoch_dir)

    print("Saving final full fine-tuned encoders...")
    save_models(src_model, tgt_model, src_tok, tgt_tok, output_dir)

    print("Evaluating sentence retrieval on test set...")

    test_df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

    raw_bah = test_df["Bahnaric"].astype(str).tolist()
    raw_vn = test_df["Vietnamese"].astype(str).tolist()

    bah = [
        normalize_text(x, lowercase=args.lowercase, strip_accents=args.strip_accents, remove_punct=args.remove_punct)
        for x in raw_bah
    ]
    vn = [
        normalize_text(x, lowercase=args.lowercase, strip_accents=args.strip_accents, remove_punct=args.remove_punct)
        for x in raw_vn
    ]

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    print("[1/3] Encoding Bahnaric queries...")
    bah_emb = encode_texts(
        bah,
        src_tok,
        src_model,
        device,
        max_len=args.src_max_len,
        batch_size=eval_batch_size,
        pooling=args.pooling,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    print("[2/3] Encoding Vietnamese candidates...")
    vn_emb = encode_texts(
        vn,
        tgt_tok,
        tgt_model,
        device,
        max_len=args.tgt_max_len,
        batch_size=eval_batch_size,
        pooling=args.pooling,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    print(f"Bahnaric embedding shape: {bah_emb.shape}")
    print(f"Vietnamese embedding shape: {vn_emb.shape}")

    print("[3/3] Retrieving candidates...")
    if args.use_csls:
        topk_idx, scores = csls_topk(
            query=bah_emb,
            index=vn_emb,
            topk=args.topk_eval,
            csls_k=args.csls_k,
        )
        retrieval = "csls"
    else:
        topk_idx, scores = cosine_topk(
            query=bah_emb,
            index=vn_emb,
            topk=args.topk_eval,
        )
        retrieval = "cosine"

    gold_idx = np.arange(len(test_df), dtype=np.int64)
    eval_ks = [int(k) for k in args.eval_ks]

    metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)

    pred_top1_idx = topk_idx[:, 0]
    preds = [raw_vn[i] for i in pred_top1_idx]
    topk_preds = ["|".join([raw_vn[j] for j in row]) for row in topk_idx]

    is_top1 = (pred_top1_idx == gold_idx).astype(np.float32)
    rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).astype(np.float32)
    gold_rank = [None if not np.isfinite(r) else int(r) for r in ranks.tolist()]

    out = {
        "Bahnaric": raw_bah,
        "Bahnaric_normalized": bah,
        "Predicted_VN": preds,
        "Gold_VN": raw_vn,
        "Gold_VN_normalized": vn,
        "Gold_rank": gold_rank,
        "TopK_Preds": topk_preds,
        "P@1": is_top1.tolist(),
        "MRR": rr.tolist(),
        "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(test_df))],
        "Gold_score": [float(scores[i, i]) for i in range(len(test_df))],
    }

    for k in eval_ks:
        k_eff = int(max(1, min(k, topk_idx.shape[1])))
        out[f"Hit@{k_eff}"] = (ranks <= float(k_eff)).astype(np.float32).tolist()

    df_out = pd.DataFrame(out)
    df_out = make_bucket_columns(df_out)
    df_out.to_csv(output_dir / "sentence_predictions.csv", index=False)

    save_bucket_metrics(df_out, output_dir, eval_ks)

    rounded_metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    rounded_metrics.update(
        {
            "method": "full_encoder_contrastive_finetuning",
            "pipeline": "full_encoder_finetune_pooling_infonce_retrieval",
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "src_model": args.src_model,
            "tgt_model": args.tgt_model,
            "pooling": args.pooling,
            "retrieval": retrieval,
            "num_train_pairs": int(len(train_ds)),
            "num_queries": int(len(test_df)),
            "candidate_pool_size": int(len(test_df)),
            "embedding_dim": int(bah_emb.shape[1]),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "grad_accum_steps": int(args.grad_accum_steps),
            "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "warmup_ratio": float(args.warmup_ratio),
            "src_max_len": int(args.src_max_len),
            "tgt_max_len": int(args.tgt_max_len),
            "topk_eval": int(args.topk_eval),
            "eval_ks": eval_ks,
            "use_csls": bool(args.use_csls),
            "csls_k": int(args.csls_k),
            "fp16": bool(args.fp16),
            "bf16": bool(args.bf16),
            "strip_accents": bool(args.strip_accents),
            "lowercase": bool(args.lowercase),
            "remove_punct": bool(args.remove_punct),
            "seed": int(args.seed),
        }
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rounded_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Saved bucket metrics to {output_dir / 'bucket_metrics_vn_len.csv'}")
    print(f"Saved source encoder to {output_dir / 'src_encoder'}")
    print(f"Saved target encoder to {output_dir / 'tgt_encoder'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline 8: full encoder contrastive fine-tuning for Bahnaric-Vietnamese retrieval"
    )

    parser.add_argument("--train_csv", required=True, help="CSV with columns Bahnaric,Vietnamese for training")
    parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese for retrieval evaluation")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--src_model", default="xlm-roberta-base")
    parser.add_argument("--tgt_model", default="xlm-roberta-base")

    parser.add_argument(
        "--pooling",
        choices=["sentence_mean", "cls"],
        default="sentence_mean",
        help="How to pool encoder token representations into a sentence embedding.",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=8)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--src_max_len", type=int, default=256)
    parser.add_argument("--tgt_max_len", type=int, default=256)

    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--use_csls", action="store_true")
    parser.add_argument("--csls_k", type=int, default=10)

    parser.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 autocast on CUDA. Prefer this on A100/H100 if supported.")

    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_each_epoch", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Use only one of --fp16 or --bf16, not both.")

    main(args)
