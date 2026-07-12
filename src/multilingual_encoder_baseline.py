# """
# Off-the-shelf multilingual encoder baseline for Bahnaric -> Vietnamese sentence retrieval.

# Baseline family:
# 5. Off-the-shelf multilingual encoders:
#    - mBERT
#    - XLM-R
#    - multilingual MiniLM
#    - LaBSE

# Task:
# Given a Bahnaric query sentence, retrieve the correct Vietnamese sentence from a candidate pool.
# For data/test.csv, the gold Vietnamese sentence is assumed to be on the same row as the Bahnaric query.

# This baseline does NOT train or fine-tune anything.
# It directly uses pretrained multilingual encoders.

# Retrieval:
# 1. Encode all Bahnaric test sentences.
# 2. Encode all Vietnamese candidate sentences.
# 3. Rank Vietnamese candidates by cosine similarity or CSLS.
# 4. Compute Top1 accuracy, MRR, Hit@K, Recall@K, Precision@K.

# Encoder backends:
# - transformers:
#   Uses AutoTokenizer + AutoModel with mean pooling.
#   Good for mBERT and XLM-R.

# - sentence_transformers:
#   Uses SentenceTransformer.encode().
#   Good for multilingual MiniLM and LaBSE.

# - auto:
#   Uses sentence_transformers for models whose names start with "sentence-transformers/",
#   otherwise uses transformers.

# Example:
# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name bert-base-multilingual-cased \
#   --encoder_name mbert \
#   --backend transformers \
#   --output_dir results/baselines/offtheshelf_mbert_cosine \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/multilingual_encoder_baseline.py \
#   --test_csv data/test.csv \
#   --model_name sentence-transformers/LaBSE \
#   --encoder_name labse \
#   --backend sentence_transformers \
#   --output_dir results/baselines/offtheshelf_labse_cosine \
#   --topk_eval 10 \
#   --eval_ks 1 5 10
# """

# import argparse
# import json
# import re
# import unicodedata
# from pathlib import Path
# from typing import Dict, List, Tuple

# import numpy as np
# import pandas as pd


# def normalize_text(
#     text: str,
#     lowercase: bool = False,
#     strip_accents: bool = False,
#     remove_punct: bool = False,
# ) -> str:
#     text = str(text)
#     text = unicodedata.normalize("NFC", text)

#     text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
#     text = text.replace("“", '"').replace("”", '"')
#     text = text.replace("–", "-").replace("—", "-")

#     if lowercase:
#         text = text.lower()

#     if strip_accents:
#         text = unicodedata.normalize("NFD", text)
#         text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
#         text = unicodedata.normalize("NFC", text)

#     if remove_punct:
#         text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(x, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     return x / norms


# def cosine_topk(query: np.ndarray, index: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
#     q = l2_normalize_rows(query)
#     z = l2_normalize_rows(index)
#     sims = q @ z.T

#     k = int(max(1, min(topk, index.shape[0])))
#     topk_idx = np.argsort(-sims, axis=1)[:, :k]
#     return topk_idx, sims


# def csls_topk(query: np.ndarray, index: np.ndarray, topk: int, csls_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
#     q = l2_normalize_rows(query)
#     z = l2_normalize_rows(index)
#     sims = q @ z.T

#     if q.shape[0] <= 1 or z.shape[0] <= 1:
#         return cosine_topk(query, index, topk)

#     k_csls = int(max(1, min(csls_k, q.shape[0] - 1, z.shape[0] - 1)))

#     rq = np.partition(sims, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
#     rz = np.partition(sims, -k_csls, axis=0)[-k_csls:, :].mean(axis=0)

#     csls = 2.0 * sims - rq[:, None] - rz[None, :]

#     k = int(max(1, min(topk, index.shape[0])))
#     topk_idx = np.argsort(-csls, axis=1)[:, :k]
#     return topk_idx, csls


# def ranking_metrics(
#     topk_idx: np.ndarray,
#     gold_idx: np.ndarray,
#     eval_ks: List[int],
# ) -> Tuple[Dict[str, float], np.ndarray]:
#     n_items = topk_idx.shape[0]
#     ranks = np.full(n_items, np.inf, dtype=np.float64)

#     for i in range(n_items):
#         hits = np.where(topk_idx[i] == gold_idx[i])[0]
#         if len(hits) > 0:
#             ranks[i] = float(hits[0] + 1)

#     metrics: Dict[str, float] = {}
#     metrics["MRR"] = float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)))
#     metrics["Top1_acc"] = float(np.mean(ranks == 1.0))

#     for k in eval_ks:
#         k_eff = int(max(1, min(k, topk_idx.shape[1])))
#         hit = float(np.mean(ranks <= k_eff))
#         metrics[f"Hit@{k_eff}"] = hit
#         metrics[f"Recall@{k_eff}"] = hit
#         metrics[f"Precision@{k_eff}"] = hit / float(k_eff)

#     return metrics, ranks


# def make_bucket_columns(df_out: pd.DataFrame) -> pd.DataFrame:
#     df_out = df_out.copy()

#     df_out["Bahnaric_len_chars"] = df_out["Bahnaric"].astype(str).str.len()
#     df_out["Vietnamese_len_chars"] = df_out["Gold_VN"].astype(str).str.len()
#     df_out["Vietnamese_len_words"] = df_out["Gold_VN"].astype(str).str.split().map(len)

#     def vn_len_bin(n: int) -> str:
#         if n <= 5:
#             return "1-5"
#         if n <= 15:
#             return "6-15"
#         if n <= 30:
#             return "16-30"
#         return ">30"

#     df_out["VN_len_bin"] = df_out["Vietnamese_len_words"].map(vn_len_bin)
#     return df_out


# def save_bucket_metrics(df_out: pd.DataFrame, output_dir: Path, eval_ks: List[int]) -> None:
#     metric_cols = ["P@1", "MRR"] + [f"Hit@{k}" for k in eval_ks if f"Hit@{k}" in df_out.columns]
#     available_cols = [col for col in metric_cols if col in df_out.columns]

#     grouped = df_out.groupby("VN_len_bin", dropna=False)
#     bucket_df = grouped[available_cols].mean().reset_index()
#     bucket_df["count"] = grouped.size().values

#     for col in available_cols:
#         bucket_df[col] = bucket_df[col].astype(float).round(4)

#     bucket_df.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


# def resolve_backend(model_name: str, backend: str) -> str:
#     if backend != "auto":
#         return backend

#     if model_name.startswith("sentence-transformers/"):
#         return "sentence_transformers"

#     return "transformers"


# def mean_pool(last_hidden, attention_mask):
#     import torch

#     mask = attention_mask.unsqueeze(-1).float()
#     summed = (last_hidden * mask).sum(dim=1)
#     lengths = mask.sum(dim=1).clamp(min=1e-9)
#     return summed / lengths


# def encode_with_transformers(
#     texts: List[str],
#     model_name: str,
#     batch_size: int,
#     max_len: int,
#     device: str,
# ) -> np.ndarray:
#     import torch
#     from transformers import AutoModel, AutoTokenizer

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#     model.to(device)
#     model.eval()

#     all_embs = []

#     with torch.no_grad():
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i : i + batch_size]
#             inputs = tokenizer(
#                 batch,
#                 padding=True,
#                 truncation=True,
#                 max_length=max_len,
#                 return_tensors="pt",
#             )
#             inputs = {k: v.to(device) for k, v in inputs.items()}

#             outputs = model(**inputs, return_dict=True)
#             emb = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
#             all_embs.append(emb.detach().cpu().numpy())

#             print(f"Encoded {min(i + batch_size, len(texts))}/{len(texts)} with transformers")

#     return np.vstack(all_embs).astype(np.float32)


# def encode_with_sentence_transformers(
#     texts: List[str],
#     model_name: str,
#     batch_size: int,
#     device: str,
# ) -> np.ndarray:
#     from sentence_transformers import SentenceTransformer

#     model = SentenceTransformer(model_name, device=device)
#     embs = model.encode(
#         texts,
#         batch_size=batch_size,
#         show_progress_bar=True,
#         convert_to_numpy=True,
#         normalize_embeddings=False,
#     )
#     return embs.astype(np.float32)


# def encode_texts(
#     texts: List[str],
#     model_name: str,
#     backend: str,
#     batch_size: int,
#     max_len: int,
#     device: str,
# ) -> np.ndarray:
#     if backend == "transformers":
#         return encode_with_transformers(
#             texts=texts,
#             model_name=model_name,
#             batch_size=batch_size,
#             max_len=max_len,
#             device=device,
#         )

#     if backend == "sentence_transformers":
#         return encode_with_sentence_transformers(
#             texts=texts,
#             model_name=model_name,
#             batch_size=batch_size,
#             device=device,
#         )

#     raise ValueError(f"Unknown backend: {backend}")


# def main(args: argparse.Namespace) -> None:
#     import torch

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     backend = resolve_backend(args.model_name, args.backend)

#     device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
#     print(f"Using device: {device}")
#     print(f"Model: {args.model_name}")
#     print(f"Encoder name: {args.encoder_name}")
#     print(f"Backend: {backend}")

#     df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

#     raw_bah = df["Bahnaric"].astype(str).tolist()
#     raw_vn = df["Vietnamese"].astype(str).tolist()

#     lowercase = bool(args.lowercase)

#     bah = [
#         normalize_text(
#             x,
#             lowercase=lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for x in raw_bah
#     ]

#     vn = [
#         normalize_text(
#             x,
#             lowercase=lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for x in raw_vn
#     ]

#     print("[1/3] Encoding Bahnaric queries...")
#     bah_emb = encode_texts(
#         texts=bah,
#         model_name=args.model_name,
#         backend=backend,
#         batch_size=args.batch_size,
#         max_len=args.max_len,
#         device=device,
#     )

#     print("[2/3] Encoding Vietnamese candidates...")
#     vn_emb = encode_texts(
#         texts=vn,
#         model_name=args.model_name,
#         backend=backend,
#         batch_size=args.batch_size,
#         max_len=args.max_len,
#         device=device,
#     )

#     print(f"Bahnaric embedding shape: {bah_emb.shape}")
#     print(f"Vietnamese embedding shape: {vn_emb.shape}")

#     print("[3/3] Retrieving candidates...")
#     if args.use_csls:
#         topk_idx, scores = csls_topk(
#             query=bah_emb,
#             index=vn_emb,
#             topk=args.topk_eval,
#             csls_k=args.csls_k,
#         )
#         retrieval = "csls"
#     else:
#         topk_idx, scores = cosine_topk(
#             query=bah_emb,
#             index=vn_emb,
#             topk=args.topk_eval,
#         )
#         retrieval = "cosine"

#     gold_idx = np.arange(len(df), dtype=np.int64)
#     eval_ks = [int(k) for k in args.eval_ks]

#     metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)

#     pred_top1_idx = topk_idx[:, 0]
#     preds = [raw_vn[i] for i in pred_top1_idx]
#     topk_preds = ["|".join([raw_vn[j] for j in row]) for row in topk_idx]

#     is_top1 = (pred_top1_idx == gold_idx).astype(np.float32)
#     rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).astype(np.float32)
#     gold_rank = [None if not np.isfinite(r) else int(r) for r in ranks.tolist()]

#     out = {
#         "Bahnaric": raw_bah,
#         "Bahnaric_normalized": bah,
#         "Predicted_VN": preds,
#         "Gold_VN": raw_vn,
#         "Gold_VN_normalized": vn,
#         "Gold_rank": gold_rank,
#         "TopK_Preds": topk_preds,
#         "P@1": is_top1.tolist(),
#         "MRR": rr.tolist(),
#         "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(df))],
#         "Gold_score": [float(scores[i, i]) for i in range(len(df))],
#     }

#     for k in eval_ks:
#         k_eff = int(max(1, min(k, topk_idx.shape[1])))
#         out[f"Hit@{k_eff}"] = (ranks <= float(k_eff)).astype(np.float32).tolist()

#     df_out = pd.DataFrame(out)
#     df_out = make_bucket_columns(df_out)
#     df_out.to_csv(output_dir / "sentence_predictions.csv", index=False)

#     save_bucket_metrics(df_out, output_dir, eval_ks)

#     rounded_metrics = {k: round(float(v), 4) for k, v in metrics.items()}
#     rounded_metrics.update(
#         {
#             "method": "off_the_shelf_multilingual_encoder",
#             "encoder_name": args.encoder_name,
#             "model_name": args.model_name,
#             "backend": backend,
#             "retrieval": retrieval,
#             "test_csv": args.test_csv,
#             "num_queries": int(len(df)),
#             "candidate_pool_size": int(len(df)),
#             "embedding_dim": int(bah_emb.shape[1]),
#             "max_len": int(args.max_len),
#             "batch_size": int(args.batch_size),
#             "strip_accents": bool(args.strip_accents),
#             "lowercase": lowercase,
#             "remove_punct": bool(args.remove_punct),
#             "use_csls": bool(args.use_csls),
#             "csls_k": int(args.csls_k),
#         }
#     )

#     with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
#         json.dump(rounded_metrics, f, ensure_ascii=False, indent=2)

#     print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
#     print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
#     print(f"Saved metrics to {output_dir / 'metrics.json'}")
#     print(f"Saved bucket metrics to {output_dir / 'bucket_metrics_vn_len.csv'}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Off-the-shelf multilingual encoder baseline for Bahnaric-Vietnamese sentence retrieval"
#     )

#     parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
#     parser.add_argument("--model_name", required=True, help="HuggingFace or sentence-transformers model name")
#     parser.add_argument("--encoder_name", required=True, help="Short label for output metrics, e.g. mbert/xlmr/minilm/labse")
#     parser.add_argument("--output_dir", required=True)

#     parser.add_argument(
#         "--backend",
#         choices=["auto", "transformers", "sentence_transformers"],
#         default="auto",
#         help="auto chooses sentence_transformers for sentence-transformers/* models, otherwise transformers",
#     )

#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--max_len", type=int, default=256)
#     parser.add_argument("--topk_eval", type=int, default=10)
#     parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

#     parser.add_argument("--use_csls", action="store_true")
#     parser.add_argument("--csls_k", type=int, default=10)

#     parser.add_argument("--lowercase", action="store_true")
#     parser.add_argument("--strip_accents", action="store_true")
#     parser.add_argument("--remove_punct", action="store_true")

#     parser.add_argument("--no_cuda", action="store_true")

#     args = parser.parse_args()
#     main(args)

#!/usr/bin/env python3
"""Reviewer-compliant off-the-shelf multilingual encoder retrieval baseline.

All encoder, preprocessing, and retrieval variants must be compared on a
held-out development pool. Only the configuration recorded in
``results/dev_selection/selected_configs.json`` may be evaluated on the test
pool.

This family does not train or fine-tune a model. A "configuration" selects the
pretrained encoder, text preprocessing, pooling backend, and cosine/CSLS
retrieval rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

FAMILY = "off_the_shelf"
EVALUATOR_SCRIPT = "src/multilingual_encoder_baseline.py"
REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")


def normalize_text(
    text: str,
    lowercase: bool = False,
    strip_accents: bool = False,
    remove_punct: bool = False,
) -> str:
    text = unicodedata.normalize("NFC", str(text))
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

    return re.sub(r"\s+", " ", text).strip()


def read_parallel_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def topk_from_scores(scores: np.ndarray, topk: int) -> np.ndarray:
    """Return score-sorted top-k indices without sorting every candidate."""
    k = int(max(1, min(topk, scores.shape[1])))
    if k == scores.shape[1]:
        return np.argsort(-scores, axis=1)
    unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    selected_scores = np.take_along_axis(scores, unsorted, axis=1)
    order = np.argsort(-selected_scores, axis=1)
    return np.take_along_axis(unsorted, order, axis=1)


def cosine_topk(query: np.ndarray, index: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    q = l2_normalize_rows(query)
    z = l2_normalize_rows(index)
    scores = q @ z.T
    return topk_from_scores(scores, topk), scores


def csls_topk(
    query: np.ndarray,
    index: np.ndarray,
    topk: int,
    csls_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    q = l2_normalize_rows(query)
    z = l2_normalize_rows(index)
    cosine = q @ z.T

    if q.shape[0] <= 1 or z.shape[0] <= 1:
        return topk_from_scores(cosine, topk), cosine

    k_csls = int(max(1, min(csls_k, q.shape[0] - 1, z.shape[0] - 1)))
    rq = np.partition(cosine, -k_csls, axis=1)[:, -k_csls:].mean(axis=1)
    rz = np.partition(cosine, -k_csls, axis=0)[-k_csls:, :].mean(axis=0)
    scores = 2.0 * cosine - rq[:, None] - rz[None, :]
    return topk_from_scores(scores, topk), scores


def ranking_metrics(
    topk_idx: np.ndarray,
    gold_idx: np.ndarray,
    eval_ks: List[int],
) -> Tuple[Dict[str, float], np.ndarray]:
    ranks = np.full(topk_idx.shape[0], np.inf, dtype=np.float64)
    for row in range(topk_idx.shape[0]):
        hits = np.where(topk_idx[row] == gold_idx[row])[0]
        if len(hits):
            ranks[row] = float(hits[0] + 1)

    metrics: Dict[str, float] = {
        "MRR": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
        "Top1_acc": float(np.mean(ranks == 1.0)),
    }
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
    metric_cols = ["P@1", "MRR"] + [
        f"Hit@{k}" for k in eval_ks if f"Hit@{k}" in df_out.columns
    ]
    available = [column for column in metric_cols if column in df_out.columns]
    grouped = df_out.groupby("VN_len_bin", dropna=False)
    bucket_df = grouped[available].mean().reset_index()
    bucket_df["count"] = grouped.size().values
    for column in available:
        bucket_df[column] = bucket_df[column].astype(float).round(4)
    bucket_df.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


def resolve_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend
    return (
        "sentence_transformers"
        if model_name.startswith("sentence-transformers/")
        else "transformers"
    )


def mean_pool(last_hidden: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-9)
    return summed / lengths


def encode_with_transformers(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_len: int,
    device: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embs: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs, return_dict=True)
            emb = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_embs.append(emb.detach().cpu().numpy())
            print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embs).astype(np.float32)


def encode_with_sentence_transformers(
    texts: List[str],
    model_name: str,
    batch_size: int,
    max_len: int,
    device: str,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_len
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(embs, dtype=np.float32)


def encode_texts(
    texts: List[str],
    model_name: str,
    backend: str,
    batch_size: int,
    max_len: int,
    device: str,
) -> np.ndarray:
    if backend == "transformers":
        return encode_with_transformers(texts, model_name, batch_size, max_len, device)
    if backend == "sentence_transformers":
        return encode_with_sentence_transformers(
            texts, model_name, batch_size, max_len, device
        )
    raise ValueError(f"Unknown backend: {backend}")


def build_configuration(args: argparse.Namespace, backend: str) -> Dict[str, Any]:
    return {
        "model_name": args.model_name,
        "encoder_name": args.encoder_name,
        "backend": backend,
        "retrieval": "csls" if args.use_csls else "cosine",
        "lowercase": bool(args.lowercase),
        "strip_accents": bool(args.strip_accents),
        "remove_punct": bool(args.remove_punct),
        "max_len": int(args.max_len),
        "use_csls": bool(args.use_csls),
        "csls_k": int(args.csls_k),
        "topk_eval": int(args.topk_eval),
        "eval_ks": [int(k) for k in args.eval_ks],
    }


def build_evaluation_cli_args(args: argparse.Namespace, backend: str) -> List[str]:
    cli = [
        "--model_name",
        args.model_name,
        "--encoder_name",
        args.encoder_name,
        "--backend",
        backend,
        "--batch_size",
        str(args.batch_size),
        "--max_len",
        str(args.max_len),
        "--topk_eval",
        str(args.topk_eval),
        "--eval_ks",
        *[str(k) for k in args.eval_ks],
        "--csls_k",
        str(args.csls_k),
    ]
    if args.use_csls:
        cli.append("--use_csls")
    if args.lowercase:
        cli.append("--lowercase")
    if args.strip_accents:
        cli.append("--strip_accents")
    if args.remove_punct:
        cli.append("--remove_punct")
    if args.embedding_cache_dir:
        cli.extend(["--embedding_cache_dir", args.embedding_cache_dir])
    return cli


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_test_authorization(
    args: argparse.Namespace,
    configuration: Dict[str, Any],
) -> None:
    if args.split_name != "test":
        return
    if not args.selection_manifest:
        raise ValueError(
            "Test evaluation requires --selection_manifest. Select the configuration on dev first."
        )

    path = Path(args.selection_manifest)
    if not path.is_file():
        raise FileNotFoundError(f"Selection manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if FAMILY not in manifest:
        raise ValueError(f"Selection manifest has no {FAMILY!r} entry")

    selected = manifest[FAMILY]
    expected_name = selected.get("configuration_name")
    expected_config = selected.get("configuration")
    if args.configuration_name != expected_name:
        raise ValueError(
            f"Unauthorized test configuration {args.configuration_name!r}; "
            f"dev selected {expected_name!r}."
        )
    if canonical_json(configuration) != canonical_json(expected_config):
        raise ValueError(
            "Test arguments do not match the exact development-selected configuration."
        )


def cache_key(
    input_path: Path,
    model_name: str,
    backend: str,
    lowercase: bool,
    strip_accents: bool,
    remove_punct: bool,
    max_len: int,
) -> str:
    payload = {
        "input_sha256": sha256_file(input_path),
        "model_name": model_name,
        "backend": backend,
        "lowercase": lowercase,
        "strip_accents": strip_accents,
        "remove_punct": remove_punct,
        "max_len": max_len,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:24]


def load_or_encode_embeddings(
    args: argparse.Namespace,
    input_path: Path,
    backend: str,
    bah: List[str],
    vn: List[str],
    device: str,
) -> Tuple[np.ndarray, np.ndarray, str]:
    cache_path: Path | None = None
    key = cache_key(
        input_path=input_path,
        model_name=args.model_name,
        backend=backend,
        lowercase=bool(args.lowercase),
        strip_accents=bool(args.strip_accents),
        remove_punct=bool(args.remove_punct),
        max_len=int(args.max_len),
    )

    if args.embedding_cache_dir:
        cache_dir = Path(args.embedding_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{key}.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path) as data:
                    query = data["query"]
                    candidate = data["candidate"]
                if query.shape[0] == len(bah) and candidate.shape[0] == len(vn):
                    print(f"Loaded cached embeddings: {cache_path}")
                    return (
                        np.asarray(query, dtype=np.float32),
                        np.asarray(candidate, dtype=np.float32),
                        str(cache_path),
                    )
                print(f"Ignoring cache with incompatible row counts: {cache_path}")
            except (OSError, ValueError, KeyError) as exc:
                print(f"Ignoring unreadable embedding cache {cache_path}: {exc}")

    print("[1/3] Encoding Bahnaric queries...")
    query = encode_texts(
        bah,
        args.model_name,
        backend,
        args.batch_size,
        args.max_len,
        device,
    )
    print("[2/3] Encoding Vietnamese candidates...")
    candidate = encode_texts(
        vn,
        args.model_name,
        backend,
        args.batch_size,
        args.max_len,
        device,
    )

    if cache_path is not None:
        temp_path = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temp_path, query=query, candidate=candidate)
        os.replace(temp_path, cache_path)
        print(f"Saved embedding cache: {cache_path}")

    return query, candidate, str(cache_path) if cache_path is not None else ""


def main(args: argparse.Namespace) -> None:
    import torch

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = resolve_backend(args.model_name, args.backend)
    configuration = build_configuration(args, backend)
    validate_test_authorization(args, configuration)

    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and not args.no_mps:
        device = "mps"
    else:
        device = "cpu"

    print(f"Split: {args.split_name}")
    print(f"Input: {input_path}")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Backend: {backend}")

    df = read_parallel_csv(input_path)
    raw_bah = df["Bahnaric"].astype(str).tolist()
    raw_vn = df["Vietnamese"].astype(str).tolist()
    bah = [
        normalize_text(
            text,
            lowercase=args.lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for text in raw_bah
    ]
    vn = [
        normalize_text(
            text,
            lowercase=args.lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for text in raw_vn
    ]

    bah_emb, vn_emb, cache_used = load_or_encode_embeddings(
        args=args,
        input_path=input_path,
        backend=backend,
        bah=bah,
        vn=vn,
        device=device,
    )

    print(f"Bahnaric embedding shape: {bah_emb.shape}")
    print(f"Vietnamese embedding shape: {vn_emb.shape}")
    print("[3/3] Retrieving candidates...")

    if args.use_csls:
        topk_idx, scores = csls_topk(
            bah_emb, vn_emb, topk=args.topk_eval, csls_k=args.csls_k
        )
        retrieval = "csls"
    else:
        topk_idx, scores = cosine_topk(bah_emb, vn_emb, topk=args.topk_eval)
        retrieval = "cosine"

    gold_idx = np.arange(len(df), dtype=np.int64)
    eval_ks = [int(k) for k in args.eval_ks]
    metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)

    pred_top1_idx = topk_idx[:, 0]
    predictions = [raw_vn[index] for index in pred_top1_idx]
    topk_predictions = [
        "|".join(raw_vn[index] for index in row) for row in topk_idx
    ]
    reciprocal_rank = np.where(
        np.isfinite(ranks), 1.0 / ranks, 0.0
    ).astype(np.float32)

    output: Dict[str, Any] = {
        "Bahnaric": raw_bah,
        "Bahnaric_normalized": bah,
        "Predicted_VN": predictions,
        "Gold_VN": raw_vn,
        "Gold_VN_normalized": vn,
        "Gold_rank": [None if not np.isfinite(r) else int(r) for r in ranks],
        "TopK_Preds": topk_predictions,
        "P@1": (pred_top1_idx == gold_idx).astype(np.float32).tolist(),
        "MRR": reciprocal_rank.tolist(),
        "Top1_score": [
            float(scores[row, pred_top1_idx[row]]) for row in range(len(df))
        ],
        "Gold_score": [float(scores[row, row]) for row in range(len(df))],
    }
    for k in eval_ks:
        k_eff = int(max(1, min(k, topk_idx.shape[1])))
        output[f"Hit@{k_eff}"] = (ranks <= k_eff).astype(np.float32).tolist()

    predictions_df = make_bucket_columns(pd.DataFrame(output))
    predictions_df.to_csv(output_dir / "sentence_predictions.csv", index=False)
    save_bucket_metrics(predictions_df, output_dir, eval_ks)

    rounded_metrics = {key: round(float(value), 4) for key, value in metrics.items()}
    rounded_metrics.update(
        {
            "schema_version": 1,
            "family": FAMILY,
            "split": args.split_name,
            "configuration_name": args.configuration_name,
            "configuration": configuration,
            "evaluator_script": EVALUATOR_SCRIPT,
            "evaluation_cli_args": build_evaluation_cli_args(args, backend),
            "method": "off_the_shelf_multilingual_encoder",
            "encoder_name": args.encoder_name,
            "model_name": args.model_name,
            "backend": backend,
            "retrieval": retrieval,
            "input_csv": str(input_path),
            "input_sha256": sha256_file(input_path),
            "num_queries": int(len(df)),
            "candidate_pool_size": int(len(df)),
            "embedding_dim": int(bah_emb.shape[1]),
            "max_len": int(args.max_len),
            "batch_size": int(args.batch_size),
            "device": device,
            "embedding_cache": cache_used,
            "selection_policy": "dev_only_then_single_test_evaluation",
            "training_policy": "off_the_shelf_no_training",
        }
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(rounded_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Development-selected off-the-shelf multilingual encoder baseline "
            "for Bahnaric-Vietnamese retrieval"
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_csv", help="Dev or test CSV")
    input_group.add_argument(
        "--test_csv",
        dest="input_csv",
        help="Deprecated alias for --input_csv",
    )
    parser.add_argument("--split_name", choices=["dev", "test"], required=True)
    parser.add_argument("--configuration_name", required=True)
    parser.add_argument("--selection_manifest", default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--model_name", required=True)
    parser.add_argument("--encoder_name", required=True)
    parser.add_argument(
        "--backend",
        choices=["auto", "transformers", "sentence_transformers"],
        default="auto",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--use_csls", action="store_true")
    parser.add_argument("--csls_k", type=int, default=10)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--embedding_cache_dir", default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--no_mps", action="store_true")
    main(parser.parse_args())
