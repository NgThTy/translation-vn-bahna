"""
Baseline 6: Previous pipeline / Projection + Procrustes baseline.

This baseline evaluates the old/current system as a retrieval baseline.

Recommended main setting:
- Backbone: XLM-R
- Epochs: 50
- Kabsch alignment: 10K lexicon
- Projection dir: results/models/b2_xlmr_50ep
- Alignment dir: results/alignment/alignment_B2_xlmr_10K_50ep

Pipeline:
1. Encode Bahnaric and Vietnamese sentences with pretrained encoders.
2. Pool encoder token representations.
3. Apply trained projection heads: src_proj.pt and tgt_proj.pt.
4. Optionally apply Kabsch/Procrustes alignment R.npy, t.npy to Bahnaric embeddings.
5. Retrieve nearest Vietnamese sentence using cosine similarity or CSLS.
6. Evaluate Top1 accuracy, MRR, Hit@K, Recall@K, Precision@K.

This script does NOT train anything.
It only loads existing projection heads and optional Kabsch alignment.

Expected files:
- --proj_dir/src_proj.pt
- --proj_dir/tgt_proj.pt
- optionally --alignment_dir/R.npy
- optionally --alignment_dir/t.npy

Main XLM-R 50ep + 10K Kabsch example:
python src/previous_pipeline_baseline.py \
  --test_csv data/test.csv \
  --proj_dir results/models/b2_xlmr_50ep \
  --alignment_dir results/alignment/alignment_B2_xlmr_10K_50ep \
  --src_model xlm-roberta-base \
  --tgt_model xlm-roberta-base \
  --pooling token_mean \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_cosine \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10

CSLS variant:
python src/previous_pipeline_baseline.py \
  --test_csv data/test.csv \
  --proj_dir results/models/b2_xlmr_50ep \
  --alignment_dir results/alignment/alignment_B2_xlmr_10K_50ep \
  --src_model xlm-roberta-base \
  --tgt_model xlm-roberta-base \
  --pooling token_mean \
  --use_csls \
  --csls_k 10 \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_10K_kabsch_csls \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10

No-Kabsch ablation:
python src/previous_pipeline_baseline.py \
  --test_csv data/test.csv \
  --proj_dir results/models/b2_xlmr_50ep \
  --src_model xlm-roberta-base \
  --tgt_model xlm-roberta-base \
  --pooling token_mean \
  --no_kabsch \
  --output_dir results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_cosine \
  --batch_size 8 \
  --src_max_len 256 \
  --tgt_max_len 256 \
  --topk_eval 10 \
  --eval_ks 1 5 10
"""

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


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


def get_hidden_size(model) -> int:
    hidden = getattr(model.config, "hidden_size", None)
    if hidden is None:
        hidden = getattr(model.config, "dim", None)
    if hidden is None:
        raise ValueError("Could not infer hidden size from model config.")
    return int(hidden)


def load_projection_head(path: Path, hidden_size: int, device: torch.device) -> Tuple[nn.Module, int]:
    if not path.exists():
        raise FileNotFoundError(
            f"Projection head not found: {path}\n"
            "Fix: pass a checkpoint subfolder, for example:\n"
            "  --proj_dir results/models/b2_xlmr_50ep\n"
            "Do not pass the parent directory results/models unless src_proj.pt and tgt_proj.pt are directly inside it."
        )

    sd = torch.load(path, map_location=device)

    if "net.0.weight" not in sd:
        raise ValueError(
            f"{path} does not look like a ProjectionHead state_dict. "
            "Expected key: net.0.weight"
        )

    out_dim = int(sd["net.0.weight"].shape[0])
    in_dim = int(sd["net.0.weight"].shape[1])

    if in_dim != hidden_size:
        raise ValueError(
            f"Projection input dimension mismatch for {path}. "
            f"Projection expects in_dim={in_dim}, but encoder hidden_size={hidden_size}. "
            "Check that --src_model/--tgt_model match the model used to train the projection heads."
        )

    head = ProjectionHead(in_dim=hidden_size, out_dim=out_dim).to(device)
    head.load_state_dict(sd, strict=True)
    head.eval()
    return head, out_dim


def maybe_load_lora(base_model, adapters_dir: Path):
    if not adapters_dir.is_dir():
        return base_model

    if PeftModel is None:
        raise ImportError("peft is not available, but --use_lora was requested.")

    return PeftModel.from_pretrained(base_model, str(adapters_dir))


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


def build_idf(tokenizer, texts: List[str], max_len: int) -> Dict[int, float]:
    from collections import Counter

    df = Counter()
    n_docs = 0

    for i in range(0, len(texts), 256):
        batch = texts[i : i + 256]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        for ids in enc["input_ids"]:
            n_docs += 1
            df.update(set(int(x) for x in ids.tolist()))

    return {tid: math.log((n_docs + 1) / (freq + 1)) + 1.0 for tid, freq in df.items()}


@torch.no_grad()
def encode_project_pool(
    texts: List[str],
    tokenizer,
    base_model,
    proj_head,
    device: torch.device,
    max_len: int,
    batch_size: int,
    pooling: str,
    idf_weights: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Supported pooling:
    - sentence_mean:
        encoder -> mean pool -> projection
    - token_mean:
        encoder -> project each token -> mean pool
    - token_idf:
        encoder -> project each token -> IDF-weighted pool
    """
    base_model.to(device).eval()
    proj_head.to(device).eval()

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

        outputs = base_model(**inputs, return_dict=True)
        last = outputs.last_hidden_state
        attn = inputs["attention_mask"].float()

        if pooling == "sentence_mean":
            mask = attn.unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            emb = proj_head(pooled)

        elif pooling in {"token_mean", "token_idf"}:
            bsz, seq_len, hidden = last.shape
            flat = last.reshape(bsz * seq_len, hidden)
            projected = proj_head(flat).reshape(bsz, seq_len, -1)

            if pooling == "token_idf":
                if idf_weights is None:
                    raise ValueError("pooling=token_idf requires idf_weights.")
                token_ids = inputs["input_ids"]
                weights = torch.ones_like(token_ids, dtype=torch.float32)
                for b in range(bsz):
                    for t in range(seq_len):
                        tid = int(token_ids[b, t].item())
                        weights[b, t] = float(idf_weights.get(tid, 1.0))
                weights = weights.to(device) * attn
            else:
                weights = attn

            weights = weights.unsqueeze(-1)
            emb = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)

        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")

        all_embs.append(emb.detach().cpu().numpy())
        print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embs).astype(np.float32)


def apply_kabsch(x: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ x.T).T + t


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print(f"Using device: {device}")
    print("Baseline: previous_pipeline_projection_procrustes")
    print(f"Source model: {args.src_model}")
    print(f"Target model: {args.tgt_model}")
    print(f"Projection dir: {args.proj_dir}")
    print(f"Alignment dir: {args.alignment_dir}")
    print(f"Pooling: {args.pooling}")

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

    src_tok = AutoTokenizer.from_pretrained(args.src_model)
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model)

    src_base = AutoModel.from_pretrained(args.src_model)
    tgt_base = AutoModel.from_pretrained(args.tgt_model)

    proj_dir = Path(args.proj_dir)

    if args.use_lora:
        print("Loading LoRA adapters...")
        src_base = maybe_load_lora(src_base, proj_dir / "src_adapters")
        tgt_base = maybe_load_lora(tgt_base, proj_dir / "tgt_adapters")

    src_hidden = get_hidden_size(src_base)
    tgt_hidden = get_hidden_size(tgt_base)

    src_head, src_proj_dim = load_projection_head(proj_dir / "src_proj.pt", src_hidden, device)
    tgt_head, tgt_proj_dim = load_projection_head(proj_dir / "tgt_proj.pt", tgt_hidden, device)

    if src_proj_dim != tgt_proj_dim:
        raise ValueError(f"Source and target projection dims differ: {src_proj_dim} vs {tgt_proj_dim}")

    proj_dim = src_proj_dim

    src_idf = None
    tgt_idf = None
    if args.pooling == "token_idf":
        print("Building source IDF weights...")
        src_idf = build_idf(src_tok, bah, max_len=args.src_max_len)
        print("Building target IDF weights...")
        tgt_idf = build_idf(tgt_tok, vn, max_len=args.tgt_max_len)

    print("[1/3] Encoding Bahnaric queries...")
    bah_emb = encode_project_pool(
        texts=bah,
        tokenizer=src_tok,
        base_model=src_base,
        proj_head=src_head,
        device=device,
        max_len=args.src_max_len,
        batch_size=args.batch_size,
        pooling=args.pooling,
        idf_weights=src_idf,
    )

    print("[2/3] Encoding Vietnamese candidates...")
    vn_emb = encode_project_pool(
        texts=vn,
        tokenizer=tgt_tok,
        base_model=tgt_base,
        proj_head=tgt_head,
        device=device,
        max_len=args.tgt_max_len,
        batch_size=args.batch_size,
        pooling=args.pooling,
        idf_weights=tgt_idf,
    )

    use_kabsch = (not args.no_kabsch) and (args.alignment_dir is not None)

    if use_kabsch:
        alignment_dir = Path(args.alignment_dir)
        R_path = alignment_dir / "R.npy"
        t_path = alignment_dir / "t.npy"

        if not R_path.exists() or not t_path.exists():
            raise FileNotFoundError(
                f"Missing Kabsch files in {alignment_dir}. Expected R.npy and t.npy."
            )

        R = np.load(R_path)
        t = np.load(t_path)

        if R.shape != (proj_dim, proj_dim):
            raise ValueError(f"R shape {R.shape} does not match projection dim {proj_dim}.")
        if t.shape[0] != proj_dim:
            raise ValueError(f"t shape {t.shape} does not match projection dim {proj_dim}.")

        print("[Kabsch] Applying R.npy and t.npy to Bahnaric embeddings...")
        bah_retrieval_emb = apply_kabsch(bah_emb, R, t)
        kabsch_used = True
    else:
        print("[Kabsch] Skipped. Using projected embeddings directly.")
        bah_retrieval_emb = bah_emb
        kabsch_used = False

    print(f"Bahnaric embedding shape: {bah_retrieval_emb.shape}")
    print(f"Vietnamese embedding shape: {vn_emb.shape}")

    print("[3/3] Retrieving candidates...")
    if args.use_csls:
        topk_idx, scores = csls_topk(
            query=bah_retrieval_emb,
            index=vn_emb,
            topk=args.topk_eval,
            csls_k=args.csls_k,
        )
        retrieval = "csls"
    else:
        topk_idx, scores = cosine_topk(
            query=bah_retrieval_emb,
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
            "method": "previous_pipeline_projection_procrustes",
            "pipeline": "encoder_pool_projection_optional_kabsch_retrieval",
            "test_csv": args.test_csv,
            "proj_dir": args.proj_dir,
            "alignment_dir": args.alignment_dir,
            "src_model": args.src_model,
            "tgt_model": args.tgt_model,
            "embedding_dim": int(proj_dim),
            "pooling": args.pooling,
            "retrieval": retrieval,
            "num_queries": int(len(df)),
            "candidate_pool_size": int(len(df)),
            "topk_eval": int(args.topk_eval),
            "eval_ks": eval_ks,
            "src_max_len": int(args.src_max_len),
            "tgt_max_len": int(args.tgt_max_len),
            "batch_size": int(args.batch_size),
            "kabsch_used": bool(kabsch_used),
            "use_csls": bool(args.use_csls),
            "csls_k": int(args.csls_k),
            "use_lora": bool(args.use_lora),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Previous pipeline baseline: encoder + projection + optional Kabsch + cosine/CSLS retrieval"
    )

    parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
    parser.add_argument("--proj_dir", required=True, help="Directory with src_proj.pt and tgt_proj.pt")
    parser.add_argument("--alignment_dir", default=None, help="Directory with R.npy and t.npy")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--src_model", default="xlm-roberta-base")
    parser.add_argument("--tgt_model", default="xlm-roberta-base")

    parser.add_argument(
        "--pooling",
        choices=["sentence_mean", "token_mean", "token_idf"],
        default="token_mean",
        help=(
            "sentence_mean: encoder -> mean pool -> projection; "
            "token_mean: encoder -> projection per token -> mean pool; "
            "token_idf: encoder -> projection per token -> IDF-weighted pool"
        ),
    )

    parser.add_argument("--src_max_len", type=int, default=256)
    parser.add_argument("--tgt_max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--no_kabsch", action="store_true", help="Do not apply R.npy/t.npy")
    parser.add_argument("--use_csls", action="store_true", help="Use CSLS instead of cosine")
    parser.add_argument("--csls_k", type=int, default=10)

    parser.add_argument("--use_lora", action="store_true", help="Load PEFT LoRA adapters from proj_dir/src_adapters and proj_dir/tgt_adapters")

    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")

    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    main(args)
