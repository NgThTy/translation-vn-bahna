"""
Baseline 7: LoRA contrastive fine-tuning for Bahnaric -> Vietnamese sentence retrieval.

Pipeline:
1. Load Bahnaric-Vietnamese training sentence pairs.
2. Encode Bahnaric and Vietnamese sentences with two XLM-R encoders.
3. Attach LoRA adapters to both encoders.
4. Add projection heads on top of encoder outputs.
5. Train with symmetric InfoNCE contrastive loss.
6. Encode all Bahnaric test queries and Vietnamese test candidates.
7. Retrieve Vietnamese candidates using cosine similarity or CSLS.
8. Report Top1 accuracy, MRR, Hit@K, Recall@K, Precision@K.

This baseline is different from Baseline 6:
- Baseline 6 loads an old trained pipeline and optional Kabsch alignment.
- Baseline 7 actively fine-tunes LoRA adapters with contrastive learning.
- No Kabsch/Procrustes alignment is used here by default.

Expected CSV columns:
- Bahnaric
- Vietnamese
"""

import argparse
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from peft import LoraConfig, get_peft_model


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Text normalization
# -----------------------
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


# -----------------------
# Dataset
# -----------------------
class ParallelSentenceDataset(Dataset):
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
            normalize_text(
                x,
                lowercase=lowercase,
                strip_accents=strip_accents,
                remove_punct=remove_punct,
            )
            for x in df["Bahnaric"].astype(str).tolist()
        ]

        self.tgt_texts = [
            normalize_text(
                x,
                lowercase=lowercase,
                strip_accents=strip_accents,
                remove_punct=remove_punct,
            )
            for x in df["Vietnamese"].astype(str).tolist()
        ]

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.src_texts[idx], self.tgt_texts[idx]


def collate_parallel(
    batch,
    src_tokenizer,
    tgt_tokenizer,
    src_max_len: int,
    tgt_max_len: int,
    device: torch.device,
):
    src_texts, tgt_texts = zip(*batch)

    src_inputs = src_tokenizer(
        list(src_texts),
        padding=True,
        truncation=True,
        max_length=src_max_len,
        return_tensors="pt",
    )

    tgt_inputs = tgt_tokenizer(
        list(tgt_texts),
        padding=True,
        truncation=True,
        max_length=tgt_max_len,
        return_tensors="pt",
    )

    src_inputs = {k: v.to(device) for k, v in src_inputs.items()}
    tgt_inputs = {k: v.to(device) for k, v in tgt_inputs.items()}

    return src_inputs, tgt_inputs


# -----------------------
# Model
# -----------------------
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


def get_hidden_size(config) -> int:
    hidden = getattr(config, "hidden_size", None)
    if hidden is None:
        hidden = getattr(config, "dim", None)
    if hidden is None:
        raise ValueError("Cannot infer hidden size from model config.")
    return int(hidden)


def infer_lora_target_modules(model_name: str, base_model) -> List[str]:
    """
    Infer LoRA target modules from the actual module names.

    HuggingFace xlm-roberta-base usually uses BERT-style names:
      query, key, value, dense

    Some RoBERTa-like implementations use:
      q_proj, k_proj, v_proj, out_proj, fc1, fc2

    We inspect the loaded model and choose the names that actually exist.
    """
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
        "Could not infer LoRA target modules from the loaded model. "
        "Try passing --lora_target_modules manually, for example: "
        "query key value dense"
    )


def count_matching_modules(base_model, target_modules: List[str]) -> int:
    count = 0
    for name, _ in base_model.named_modules():
        if any(t in name for t in target_modules):
            count += 1
    return count


def apply_lora(
    base_model,
    model_name: str,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_target_modules: Optional[List[str]],
):
    target_modules = lora_target_modules
    if target_modules is None:
        target_modules = infer_lora_target_modules(model_name, base_model)

    match_count = count_matching_modules(base_model, target_modules)
    if match_count == 0:
        raise RuntimeError(
            f"No modules matched LoRA target_modules={target_modules}. "
            "For XLM-R, try: --lora_target_modules q_proj k_proj v_proj out_proj fc1 fc2"
        )

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    model = get_peft_model(base_model, lora_cfg)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    print(f"LoRA enabled for {model_name}")
    print(f"LoRA target modules: {target_modules}")
    print(f"Matched modules: {match_count}")

    return model


class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        projection_dim: int,
        pooling: str,
        use_lora: bool,
        lora_r: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_target_modules: Optional[List[str]],
        freeze_base: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.pooling = pooling

        config = AutoConfig.from_pretrained(model_name)
        self.base = AutoModel.from_pretrained(model_name, config=config)

        hidden_size = get_hidden_size(config)
        self.proj = ProjectionHead(hidden_size, out_dim=projection_dim)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        if use_lora:
            self.base = apply_lora(
                self.base,
                model_name=model_name,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
            )

    def forward(self, inputs):
        outputs = self.base(**inputs, return_dict=True)
        last = outputs.last_hidden_state
        attn = inputs["attention_mask"].float()

        if self.pooling == "sentence_mean":
            mask = attn.unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = self.proj(pooled)

        elif self.pooling == "token_mean":
            bsz, seq_len, hidden = last.shape
            flat = last.reshape(bsz * seq_len, hidden)
            projected = self.proj(flat).reshape(bsz, seq_len, -1)

            mask = attn.unsqueeze(-1)
            emb = (projected * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return emb


def count_trainable(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# -----------------------
# Loss
# -----------------------
def infonce_loss(src_emb, tgt_emb, temperature: float = 0.07):
    src = F.normalize(src_emb, p=2, dim=-1)
    tgt = F.normalize(tgt_emb, p=2, dim=-1)

    logits = src @ tgt.T
    logits = logits / max(temperature, 1e-6)

    labels = torch.arange(src.size(0), device=src.device)

    loss_src = F.cross_entropy(logits, labels)
    loss_tgt = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_src + loss_tgt)


def mse_loss(src_emb, tgt_emb):
    return F.mse_loss(src_emb, tgt_emb)


def hybrid_loss(src_emb, tgt_emb, temperature: float, alpha: float):
    return alpha * infonce_loss(src_emb, tgt_emb, temperature) + (1.0 - alpha) * mse_loss(src_emb, tgt_emb)


def compute_loss(src_emb, tgt_emb, loss_type: str, temperature: float, alpha: float):
    if loss_type == "infonce":
        return infonce_loss(src_emb, tgt_emb, temperature)
    if loss_type == "mse":
        return mse_loss(src_emb, tgt_emb)
    if loss_type == "hybrid":
        return hybrid_loss(src_emb, tgt_emb, temperature, alpha)
    raise ValueError(f"Unknown loss_type: {loss_type}")


# -----------------------
# Training
# -----------------------
def train_one_epoch(
    src_model,
    tgt_model,
    loader,
    optimizer,
    device,
    loss_type: str,
    temperature: float,
    alpha: float,
    max_grad_norm: float,
    epoch: int,
):
    src_model.train()
    tgt_model.train()

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Training epoch {epoch}", unit="batch", leave=False)

    for src_inputs, tgt_inputs in pbar:
        optimizer.zero_grad(set_to_none=True)

        src_emb = src_model(src_inputs)
        tgt_emb = tgt_model(tgt_inputs)

        loss = compute_loss(src_emb, tgt_emb, loss_type, temperature, alpha)
        loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            params = list(src_model.parameters()) + list(tgt_model.parameters())
            torch.nn.utils.clip_grad_norm_(
                [p for p in params if p.requires_grad and p.grad is not None],
                max_grad_norm,
            )

        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / max(1, n_batches):.4f}")

    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_loss(
    src_model,
    tgt_model,
    loader,
    loss_type: str,
    temperature: float,
    alpha: float,
):
    src_model.eval()
    tgt_model.eval()

    total_loss = 0.0
    n_batches = 0

    for src_inputs, tgt_inputs in loader:
        src_emb = src_model(src_inputs)
        tgt_emb = tgt_model(tgt_inputs)

        loss = compute_loss(src_emb, tgt_emb, loss_type, temperature, alpha)

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def save_checkpoint(src_model, tgt_model, output_dir: Path, use_lora: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(src_model.proj.state_dict(), output_dir / "src_proj.pt")
    torch.save(tgt_model.proj.state_dict(), output_dir / "tgt_proj.pt")

    if use_lora:
        (output_dir / "src_adapters").mkdir(parents=True, exist_ok=True)
        (output_dir / "tgt_adapters").mkdir(parents=True, exist_ok=True)
        src_model.base.save_pretrained(output_dir / "src_adapters")
        tgt_model.base.save_pretrained(output_dir / "tgt_adapters")


# -----------------------
# Retrieval
# -----------------------
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


@torch.no_grad()
def encode_texts(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_len: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    all_embs = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        emb = model(inputs)
        all_embs.append(emb.detach().cpu().numpy())

        print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embs).astype(np.float32)


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
    available_cols = [c for c in metric_cols if c in df_out.columns]

    grouped = df_out.groupby("VN_len_bin", dropna=False)
    bucket_df = grouped[available_cols].mean().reset_index()
    bucket_df["count"] = grouped.size().values

    for col in available_cols:
        bucket_df[col] = bucket_df[col].astype(float).round(4)

    bucket_df.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


def evaluate_retrieval(args, src_model, tgt_model, src_tokenizer, tgt_tokenizer, device, output_dir: Path):
    df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

    raw_bah = df["Bahnaric"].astype(str).tolist()
    raw_vn = df["Vietnamese"].astype(str).tolist()

    bah = [
        normalize_text(
            x,
            lowercase=args.lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for x in raw_bah
    ]

    vn = [
        normalize_text(
            x,
            lowercase=args.lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for x in raw_vn
    ]

    print("[Eval 1/3] Encoding Bahnaric queries...")
    bah_emb = encode_texts(
        src_model,
        src_tokenizer,
        bah,
        device=device,
        max_len=args.src_max_len,
        batch_size=args.eval_batch_size,
    )

    print("[Eval 2/3] Encoding Vietnamese candidates...")
    vn_emb = encode_texts(
        tgt_model,
        tgt_tokenizer,
        vn,
        device=device,
        max_len=args.tgt_max_len,
        batch_size=args.eval_batch_size,
    )

    print(f"Bahnaric embedding shape: {bah_emb.shape}")
    print(f"Vietnamese embedding shape: {vn_emb.shape}")

    print("[Eval 3/3] Retrieving candidates...")
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

    gold_idx = np.arange(len(df), dtype=np.int64)
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
        "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(df))],
        "Gold_score": [float(scores[i, i]) for i in range(len(df))],
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
            "method": "lora_contrastive_finetuning",
            "pipeline": "encoder_lora_pool_projection_infonce_retrieval",
            "train_csv": args.train_csv,
            "valid_csv": args.valid_csv,
            "test_csv": args.test_csv,
            "src_model": args.src_model,
            "tgt_model": args.tgt_model,
            "projection_dim": int(args.projection_dim),
            "pooling": args.pooling,
            "loss_type": args.loss_type,
            "temperature": float(args.temperature),
            "alpha": float(args.alpha),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "use_lora": bool(args.use_lora),
            "lora_r": int(args.lora_r),
            "lora_alpha": float(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "retrieval": retrieval,
            "use_csls": bool(args.use_csls),
            "csls_k": int(args.csls_k),
            "num_queries": int(len(df)),
            "candidate_pool_size": int(len(df)),
            "topk_eval": int(args.topk_eval),
            "eval_ks": eval_ks,
            "src_max_len": int(args.src_max_len),
            "tgt_max_len": int(args.tgt_max_len),
            "strip_accents": bool(args.strip_accents),
            "lowercase": bool(args.lowercase),
            "remove_punct": bool(args.remove_punct),
        }
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rounded_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Saved bucket metrics to {output_dir / 'bucket_metrics_vn_len.csv'}")


# -----------------------
# Main
# -----------------------
def main(args):
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print(f"Using device: {device}")
    print("Baseline: lora_contrastive_finetuning")
    print(f"Source model: {args.src_model}")
    print(f"Target model: {args.tgt_model}")
    print(f"Pooling: {args.pooling}")
    print(f"Loss: {args.loss_type}")
    print(f"Use LoRA: {args.use_lora}")

    src_tokenizer = AutoTokenizer.from_pretrained(args.src_model)
    tgt_tokenizer = AutoTokenizer.from_pretrained(args.tgt_model)

    train_ds = ParallelSentenceDataset(
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
        collate_fn=lambda b: collate_parallel(
            b,
            src_tokenizer,
            tgt_tokenizer,
            args.src_max_len,
            args.tgt_max_len,
            device,
        ),
    )

    valid_loader = None
    if args.valid_csv:
        valid_ds = ParallelSentenceDataset(
            args.valid_csv,
            lowercase=args.lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )

        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: collate_parallel(
                b,
                src_tokenizer,
                tgt_tokenizer,
                args.src_max_len,
                args.tgt_max_len,
                device,
            ),
        )

    lora_targets = args.lora_target_modules
    if lora_targets is not None and len(lora_targets) == 0:
        lora_targets = None

    src_model = ContrastiveEncoder(
        model_name=args.src_model,
        projection_dim=args.projection_dim,
        pooling=args.pooling,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        freeze_base=True,
    ).to(device)

    tgt_model = ContrastiveEncoder(
        model_name=args.tgt_model,
        projection_dim=args.projection_dim,
        pooling=args.pooling,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets,
        freeze_base=True,
    ).to(device)

    print(f"Trainable src params: {count_trainable(src_model)}")
    print(f"Trainable tgt params: {count_trainable(tgt_model)}")

    trainable_params = [p for p in src_model.parameters() if p.requires_grad]
    trainable_params += [p for p in tgt_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_valid = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            src_model=src_model,
            tgt_model=tgt_model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_type=args.loss_type,
            temperature=args.temperature,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm,
            epoch=epoch,
        )

        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.6f}")

        if valid_loader is not None:
            valid_loss = evaluate_loss(
                src_model=src_model,
                tgt_model=tgt_model,
                loader=valid_loader,
                loss_type=args.loss_type,
                temperature=args.temperature,
                alpha=args.alpha,
            )
            print(f"Epoch {epoch}/{args.epochs} valid_loss={valid_loss:.6f}")

            if valid_loss < best_valid:
                best_valid = valid_loss
                best_epoch = epoch
                save_checkpoint(src_model, tgt_model, output_dir, use_lora=args.use_lora)
                print(f"Saved best checkpoint to {output_dir}")
        else:
            save_checkpoint(src_model, tgt_model, output_dir, use_lora=args.use_lora)

    if valid_loader is not None:
        print(f"Best validation loss: {best_valid:.6f} at epoch {best_epoch}")

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    evaluate_retrieval(
        args=args,
        src_model=src_model,
        tgt_model=tgt_model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline 7: LoRA contrastive fine-tuning for Bahnaric-Vietnamese retrieval"
    )

    parser.add_argument("--train_csv", required=True, help="Training CSV with Bahnaric,Vietnamese columns")
    parser.add_argument("--valid_csv", default=None, help="Optional validation CSV")
    parser.add_argument("--test_csv", required=True, help="Test CSV with Bahnaric,Vietnamese columns")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--src_model", default="xlm-roberta-base")
    parser.add_argument("--tgt_model", default="xlm-roberta-base")

    parser.add_argument("--projection_dim", type=int, default=256)
    parser.add_argument(
        "--pooling",
        choices=["sentence_mean", "token_mean"],
        default="sentence_mean",
        help="sentence_mean = mean pool then project; token_mean = project tokens then mean pool",
    )

    parser.add_argument("--loss_type", choices=["infonce", "mse", "hybrid"], default="infonce")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--src_max_len", type=int, default=256)
    parser.add_argument("--tgt_max_len", type=int, default=256)

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=None,
        help="Optional explicit LoRA target modules. For XLM-R: q_proj k_proj v_proj out_proj fc1 fc2",
    )

    parser.add_argument("--use_csls", action="store_true")
    parser.add_argument("--csls_k", type=int, default=10)
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    main(args)
