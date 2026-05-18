"""
Baseline-6-style LoRA + projection generalization experiment.

Purpose:
  Train a new LoRA + projection checkpoint for each FLORES language pair,
  then evaluate no-Kabsch sentence retrieval with cosine or CSLS.

This reproduces the strongest part of Baseline 6 for new language pairs:
  frozen XLM-R + LoRA adapters + projection heads + InfoNCE
  no Kabsch
  CSLS retrieval

Expected CSV columns:
  Bahnaric,Vietnamese
"""

import argparse
import json
import math
import random
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str, lowercase: bool = False) -> str:
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    if lowercase:
        text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ParallelDataset(Dataset):
    def __init__(self, csv_path: str, lowercase: bool = False):
        df = pd.read_csv(csv_path).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)
        self.src = [normalize_text(x, lowercase=lowercase) for x in df["Bahnaric"].astype(str).tolist()]
        self.tgt = [normalize_text(x, lowercase=lowercase) for x in df["Vietnamese"].astype(str).tolist()]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def collate_batch(batch, src_tok, tgt_tok, src_max_len, tgt_max_len):
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


def move_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def get_hidden_size(model) -> int:
    hidden = getattr(model.config, "hidden_size", None)
    if hidden is None:
        hidden = getattr(model.config, "dim", None)
    if hidden is None:
        raise ValueError("Could not infer hidden size from model config.")
    return int(hidden)


def infer_lora_target_modules(base_model) -> List[str]:
    module_names = [name for name, _ in base_model.named_modules()]

    bert_style = ["query", "key", "value", "dense"]
    roberta_style = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    bert_matches = sum(
        1 for name in module_names
        if any(part in name.split(".") for part in bert_style)
    )
    roberta_matches = sum(
        1 for name in module_names
        if any(part in name.split(".") for part in roberta_style)
    )

    if bert_matches > 0:
        return bert_style

    if roberta_matches > 0:
        return roberta_style

    raise RuntimeError(
        "Could not infer LoRA target modules. "
        "Try passing --lora_target_modules query key value dense"
    )


def apply_lora(base_model, args):
    target_modules = args.lora_target_modules
    if target_modules is None or len(target_modules) == 0:
        target_modules = infer_lora_target_modules(base_model)

    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    model = get_peft_model(base_model, cfg)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    print(f"LoRA target modules: {target_modules}")
    return model


def encode_batch(base_model, proj_head, inputs, pooling: str):
    outputs = base_model(**inputs, return_dict=True)
    last = outputs.last_hidden_state
    attn = inputs["attention_mask"].float()

    if pooling == "sentence_mean":
        mask = attn.unsqueeze(-1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return proj_head(pooled)

    if pooling == "token_mean":
        bsz, seq_len, hidden = last.shape
        flat = last.reshape(bsz * seq_len, hidden)
        projected = proj_head(flat).reshape(bsz, seq_len, -1)
        mask = attn.unsqueeze(-1)
        return (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    raise ValueError(f"Unknown pooling: {pooling}")


def symmetric_infonce(src_emb, tgt_emb, temperature: float):
    src = F.normalize(src_emb, p=2, dim=-1)
    tgt = F.normalize(tgt_emb, p=2, dim=-1)

    logits = src @ tgt.T
    logits = logits / max(temperature, 1e-8)
    labels = torch.arange(src.size(0), device=src.device)

    loss_src = F.cross_entropy(logits, labels)
    loss_tgt = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_src + loss_tgt)


def l2_normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def cosine_topk(query, index, topk):
    q = l2_normalize_rows(query)
    z = l2_normalize_rows(index)
    sims = q @ z.T
    k = int(max(1, min(topk, index.shape[0])))
    topk_idx = np.argsort(-sims, axis=1)[:, :k]
    return topk_idx, sims


def csls_topk(query, index, topk, csls_k=10):
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


def ranking_metrics(topk_idx, gold_idx, eval_ks):
    n_items = topk_idx.shape[0]
    ranks = np.full(n_items, np.inf, dtype=np.float64)

    for i in range(n_items):
        hits = np.where(topk_idx[i] == gold_idx[i])[0]
        if len(hits) > 0:
            ranks[i] = float(hits[0] + 1)

    metrics = {}
    metrics["MRR"] = float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)))
    metrics["Top1_acc"] = float(np.mean(ranks == 1.0))

    for k in eval_ks:
        k_eff = int(max(1, min(k, topk_idx.shape[1])))
        hit = float(np.mean(ranks <= k_eff))
        metrics[f"Hit@{k_eff}"] = hit
        metrics[f"Recall@{k_eff}"] = hit
        metrics[f"Precision@{k_eff}"] = hit / float(k_eff)

    return metrics, ranks


@torch.no_grad()
def encode_texts(texts, tokenizer, base_model, proj_head, device, max_len, batch_size, pooling):
    base_model.eval()
    proj_head.eval()

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
        emb = encode_batch(base_model, proj_head, inputs, pooling=pooling)
        all_embs.append(emb.detach().cpu().numpy())

    return np.vstack(all_embs).astype(np.float32)


def save_checkpoint(src_base, tgt_base, src_proj, tgt_proj, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(src_proj.state_dict(), output_dir / "src_proj.pt")
    torch.save(tgt_proj.state_dict(), output_dir / "tgt_proj.pt")

    src_adapter_dir = output_dir / "src_adapters"
    tgt_adapter_dir = output_dir / "tgt_adapters"
    src_adapter_dir.mkdir(parents=True, exist_ok=True)
    tgt_adapter_dir.mkdir(parents=True, exist_ok=True)

    src_base.save_pretrained(src_adapter_dir)
    tgt_base.save_pretrained(tgt_adapter_dir)


def evaluate(csv_path, src_tok, tgt_tok, src_base, tgt_base, src_proj, tgt_proj, device, args, output_dir, retrieval):
    df = pd.read_csv(csv_path).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)
    src_texts = [normalize_text(x, lowercase=args.lowercase) for x in df["Bahnaric"].astype(str).tolist()]
    tgt_texts = [normalize_text(x, lowercase=args.lowercase) for x in df["Vietnamese"].astype(str).tolist()]

    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size

    src_emb = encode_texts(
        src_texts,
        src_tok,
        src_base,
        src_proj,
        device,
        args.src_max_len,
        eval_batch_size,
        args.pooling,
    )
    tgt_emb = encode_texts(
        tgt_texts,
        tgt_tok,
        tgt_base,
        tgt_proj,
        device,
        args.tgt_max_len,
        eval_batch_size,
        args.pooling,
    )

    if retrieval == "csls":
        topk_idx, scores = csls_topk(src_emb, tgt_emb, topk=args.topk_eval, csls_k=args.csls_k)
    elif retrieval == "cosine":
        topk_idx, scores = cosine_topk(src_emb, tgt_emb, topk=args.topk_eval)
    else:
        raise ValueError(f"Unknown retrieval: {retrieval}")

    gold_idx = np.arange(len(df), dtype=np.int64)
    metrics, ranks = ranking_metrics(topk_idx, gold_idx, args.eval_ks)

    result_dir = output_dir / f"test_{retrieval}"
    result_dir.mkdir(parents=True, exist_ok=True)

    rounded = {k: round(float(v), 4) for k, v in metrics.items()}
    rounded.update(
        {
            "method": "baseline6_style_lora_projection_generalization",
            "pipeline": "frozen_xlmr_lora_projection_infonce_no_kabsch_retrieval",
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "src_model": args.src_model,
            "tgt_model": args.tgt_model,
            "pooling": args.pooling,
            "projection_dim": int(args.projection_dim),
            "retrieval": retrieval,
            "num_queries": int(len(df)),
            "candidate_pool_size": int(len(df)),
            "embedding_dim": int(src_emb.shape[1]),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "grad_accum_steps": int(args.grad_accum_steps),
            "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
            "lr": float(args.lr),
            "temperature": float(args.temperature),
            "kabsch_used": False,
            "use_lora": True,
            "use_csls": bool(retrieval == "csls"),
            "csls_k": int(args.csls_k),
            "topk_eval": int(args.topk_eval),
            "eval_ks": [int(k) for k in args.eval_ks],
        }
    )

    with open(result_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rounded, f, ensure_ascii=False, indent=2)

    pred_top1_idx = topk_idx[:, 0]
    out = pd.DataFrame(
        {
            "Source": df["Bahnaric"].astype(str).tolist(),
            "Gold_VN": df["Vietnamese"].astype(str).tolist(),
            "Predicted_VN": [df["Vietnamese"].iloc[i] for i in pred_top1_idx],
            "Gold_rank": [None if not np.isfinite(r) else int(r) for r in ranks.tolist()],
            "TopK_indices": ["|".join(map(str, row)) for row in topk_idx],
        }
    )
    out.to_csv(result_dir / "sentence_predictions.csv", index=False)

    print(json.dumps(rounded, ensure_ascii=False, indent=2))
    return rounded


def main(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    src_tok = AutoTokenizer.from_pretrained(args.src_model)
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)

    src_base_raw = AutoModel.from_pretrained(args.src_model)
    tgt_base_raw = AutoModel.from_pretrained(args.tgt_model)

    for p in src_base_raw.parameters():
        p.requires_grad = False
    for p in tgt_base_raw.parameters():
        p.requires_grad = False

    src_base = apply_lora(src_base_raw, args).to(device)
    tgt_base = apply_lora(tgt_base_raw, args).to(device)

    src_hidden = get_hidden_size(src_base)
    tgt_hidden = get_hidden_size(tgt_base)

    src_proj = ProjectionHead(src_hidden, out_dim=args.projection_dim, dropout=args.proj_dropout).to(device)
    tgt_proj = ProjectionHead(tgt_hidden, out_dim=args.projection_dim, dropout=args.proj_dropout).to(device)

    train_ds = ParallelDataset(args.train_csv, lowercase=args.lowercase)
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

    trainable_params = []
    trainable_params += [p for p in src_base.parameters() if p.requires_grad]
    trainable_params += [p for p in tgt_base.parameters() if p.requires_grad]
    trainable_params += list(src_proj.parameters())
    trainable_params += list(tgt_proj.parameters())

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

    best_ckpt = output_dir / "checkpoint_final"

    for epoch in range(1, args.epochs + 1):
        src_base.train()
        tgt_base.train()
        src_proj.train()
        tgt_proj.train()

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for step, (src_inputs, tgt_inputs) in enumerate(pbar, start=1):
            src_inputs = move_to_device(src_inputs, device)
            tgt_inputs = move_to_device(tgt_inputs, device)

            src_emb = encode_batch(src_base, src_proj, src_inputs, pooling=args.pooling)
            tgt_emb = encode_batch(tgt_base, tgt_proj, tgt_inputs, pooling=args.pooling)

            loss = symmetric_infonce(src_emb, tgt_emb, temperature=args.temperature)
            loss = loss / max(1, args.grad_accum_steps)
            loss.backward()

            running_loss += float(loss.detach().cpu()) * max(1, args.grad_accum_steps)

            should_step = (step % args.grad_accum_steps == 0) or (step == len(train_loader))
            if should_step:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = running_loss / max(1, step)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        epoch_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} train_loss={epoch_loss:.6f}")

    save_checkpoint(src_base, tgt_base, src_proj, tgt_proj, best_ckpt)
    print(f"Saved final checkpoint to {best_ckpt}")

    if args.eval_retrievals in {"cosine", "both"}:
        evaluate(
            args.test_csv,
            src_tok,
            tgt_tok,
            src_base,
            tgt_base,
            src_proj,
            tgt_proj,
            device,
            args,
            output_dir,
            retrieval="cosine",
        )

    if args.eval_retrievals in {"csls", "both"}:
        evaluate(
            args.test_csv,
            src_tok,
            tgt_tok,
            src_base,
            tgt_base,
            src_proj,
            tgt_proj,
            device,
            args,
            output_dir,
            retrieval="csls",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--src_model", default="xlm-roberta-base")
    parser.add_argument("--tgt_model", default="xlm-roberta-base")

    parser.add_argument("--pooling", choices=["sentence_mean", "token_mean"], default="token_mean")
    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument("--proj_dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=2)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--src_max_len", type=int, default=256)
    parser.add_argument("--tgt_max_len", type=int, default=256)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=None)

    parser.add_argument("--eval_retrievals", choices=["cosine", "csls", "both"], default="both")
    parser.add_argument("--csls_k", type=int, default=10)
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    main(args)
