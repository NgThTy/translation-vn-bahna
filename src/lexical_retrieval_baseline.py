# """
# Lexical retrieval baselines for Bahnaric -> Vietnamese sentence retrieval.

# Baselines:
# 1. TF-IDF character n-gram retrieval
# 2. BM25 character n-gram retrieval

# Task formulation:
# Given a Bahnaric query sentence, retrieve the correct Vietnamese sentence from
# a candidate pool. For test.csv, the gold Vietnamese sentence is assumed to be
# on the same row as the Bahnaric query.

# Expected CSV columns:
# - Bahnaric
# - Vietnamese

# Example:
# python src/lexical_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method tfidf_char \
#   --output_dir results/baselines/tfidf_char \
#   --ngram_min 2 \
#   --ngram_max 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/lexical_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method bm25_char \
#   --output_dir results/baselines/bm25_char \
#   --ngram_min 2 \
#   --ngram_max 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10
# """

# import argparse
# import json
# import math
# import re
# import unicodedata
# from collections import Counter
# from pathlib import Path
# from typing import Dict, List, Tuple

# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer


# def normalize_text(text: str, lowercase: bool = True, strip_accents: bool = False) -> str:
#     text = str(text)

#     # Normalize Unicode form.
#     text = unicodedata.normalize("NFC", text)

#     # Normalize apostrophe-like characters.
#     text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")

#     # Normalize quote/dash variants.
#     text = text.replace("“", '"').replace("”", '"')
#     text = text.replace("–", "-").replace("—", "-")

#     if lowercase:
#         text = text.lower()

#     if strip_accents:
#         # Remove combining marks after decomposition.
#         text = unicodedata.normalize("NFD", text)
#         text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
#         text = unicodedata.normalize("NFC", text)

#     # Collapse whitespace.
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def char_ngrams(text: str, ngram_min: int, ngram_max: int) -> List[str]:
#     text = f" {text} "
#     grams: List[str] = []
#     for n in range(ngram_min, ngram_max + 1):
#         if len(text) < n:
#             continue
#         grams.extend(text[i : i + n] for i in range(0, len(text) - n + 1))
#     return grams


# def ranking_metrics(topk_idx: np.ndarray, gold_idx: np.ndarray, eval_ks: List[int]) -> Tuple[Dict[str, float], np.ndarray]:
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


# def tfidf_char_retrieval(
#     queries: List[str],
#     candidates: List[str],
#     ngram_min: int,
#     ngram_max: int,
#     topk_eval: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     vectorizer = TfidfVectorizer(
#         analyzer="char",
#         ngram_range=(ngram_min, ngram_max),
#         lowercase=False,
#         norm="l2",
#         sublinear_tf=True,
#     )

#     candidate_matrix = vectorizer.fit_transform(candidates)
#     query_matrix = vectorizer.transform(queries)

#     # Cosine similarity because TF-IDF rows are L2-normalized.
#     sims = query_matrix @ candidate_matrix.T
#     sims = sims.toarray()

#     k = int(max(1, min(topk_eval, len(candidates))))
#     topk_idx = np.argsort(-sims, axis=1)[:, :k]
#     return topk_idx, sims


# def bm25_char_retrieval(
#     queries: List[str],
#     candidates: List[str],
#     ngram_min: int,
#     ngram_max: int,
#     topk_eval: int,
#     k1: float = 1.5,
#     b: float = 0.75,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     candidate_tokens = [char_ngrams(text, ngram_min, ngram_max) for text in candidates]
#     query_tokens = [char_ngrams(text, ngram_min, ngram_max) for text in queries]

#     n_docs = len(candidate_tokens)
#     doc_lens = np.array([len(toks) for toks in candidate_tokens], dtype=np.float64)
#     avgdl = float(doc_lens.mean()) if n_docs > 0 else 0.0

#     doc_tf = [Counter(toks) for toks in candidate_tokens]

#     df = Counter()
#     for toks in candidate_tokens:
#         df.update(set(toks))

#     idf = {
#         term: math.log(1.0 + ((n_docs - freq + 0.5) / (freq + 0.5)))
#         for term, freq in df.items()
#     }

#     scores = np.zeros((len(queries), n_docs), dtype=np.float64)

#     for qi, q_toks in enumerate(query_tokens):
#         q_terms = Counter(q_toks)
#         for term, q_count in q_terms.items():
#             if term not in idf:
#                 continue

#             term_idf = idf[term]

#             for di, tf_counter in enumerate(doc_tf):
#                 f = tf_counter.get(term, 0)
#                 if f == 0:
#                     continue

#                 denom = f + k1 * (1.0 - b + b * (doc_lens[di] / max(avgdl, 1e-9)))
#                 scores[qi, di] += term_idf * ((f * (k1 + 1.0)) / max(denom, 1e-9))

#     k = int(max(1, min(topk_eval, n_docs)))
#     topk_idx = np.argsort(-scores, axis=1)[:, :k]
#     return topk_idx, scores


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


# def main(args: argparse.Namespace) -> None:
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

#     raw_bah = df["Bahnaric"].astype(str).tolist()
#     raw_vn = df["Vietnamese"].astype(str).tolist()

#     queries = [
#         normalize_text(x, lowercase=not args.no_lowercase, strip_accents=args.strip_accents)
#         for x in raw_bah
#     ]
#     candidates = [
#         normalize_text(x, lowercase=not args.no_lowercase, strip_accents=args.strip_accents)
#         for x in raw_vn
#     ]

#     if args.method == "tfidf_char":
#         topk_idx, scores = tfidf_char_retrieval(
#             queries=queries,
#             candidates=candidates,
#             ngram_min=args.ngram_min,
#             ngram_max=args.ngram_max,
#             topk_eval=args.topk_eval,
#         )
#     elif args.method == "bm25_char":
#         topk_idx, scores = bm25_char_retrieval(
#             queries=queries,
#             candidates=candidates,
#             ngram_min=args.ngram_min,
#             ngram_max=args.ngram_max,
#             topk_eval=args.topk_eval,
#             k1=args.bm25_k1,
#             b=args.bm25_b,
#         )
#     else:
#         raise ValueError(f"Unknown method: {args.method}")

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
#         "Predicted_VN": preds,
#         "Gold_VN": raw_vn,
#         "Gold_rank": gold_rank,
#         "TopK_Preds": topk_preds,
#         "P@1": is_top1.tolist(),
#         "MRR": rr.tolist(),
#         "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(df))],
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
#             "method": args.method,
#             "test_csv": args.test_csv,
#             "num_queries": int(len(df)),
#             "candidate_pool_size": int(len(df)),
#             "ngram_min": int(args.ngram_min),
#             "ngram_max": int(args.ngram_max),
#             "strip_accents": bool(args.strip_accents),
#             "lowercase": not bool(args.no_lowercase),
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
#         description="BM25 / TF-IDF character n-gram baselines for Bahnaric-Vietnamese sentence retrieval"
#     )

#     parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
#     parser.add_argument(
#         "--method",
#         choices=["tfidf_char", "bm25_char"],
#         required=True,
#         help="Lexical retrieval method",
#     )
#     parser.add_argument("--output_dir", required=True)

#     parser.add_argument("--ngram_min", type=int, default=2)
#     parser.add_argument("--ngram_max", type=int, default=5)
#     parser.add_argument("--topk_eval", type=int, default=10)
#     parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

#     parser.add_argument("--strip_accents", action="store_true")
#     parser.add_argument("--no_lowercase", action="store_true")

#     parser.add_argument("--bm25_k1", type=float, default=1.5)
#     parser.add_argument("--bm25_b", type=float, default=0.75)

#     args = parser.parse_args()
#     main(args)

#!/usr/bin/env python3
"""Lexical retrieval baselines for Bahnaric -> Vietnamese sentence retrieval.

All configuration selection must be performed on a development split. A test
run is accepted only when the configuration matches the dev-selection manifest
written by ``select_dev_configs_and_evaluate_test.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

FAMILY = "lexical"
EVALUATOR_SCRIPT = "src/lexical_retrieval_baseline.py"
REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_accents: bool = False,
    remove_punct: bool = False,
    remove_spaces: bool = False,
) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

    if lowercase:
        text = text.lower()

    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = unicodedata.normalize("NFC", text)

    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    text = re.sub(r"\s+", " ", text).strip()
    if remove_spaces:
        text = re.sub(r"\s+", "", text)
    return text


def validate_dataframe(df: pd.DataFrame, input_csv: Path) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {missing}")

    cleaned = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(f"No valid bilingual pairs found in {input_csv}")
    return cleaned


def topk_from_score_block(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return top-k indices/scores, ordered by score then candidate index."""
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D matrix")

    n_candidates = scores.shape[1]
    k = max(1, min(k, n_candidates))

    if k == n_candidates:
        candidate_idx = np.broadcast_to(np.arange(n_candidates), scores.shape)
        order = np.lexsort((candidate_idx, -scores), axis=1)
        topk_idx = order[:, :k]
    else:
        # argpartition avoids a complete sort of every candidate pool. The
        # selected subset is then deterministically ordered.
        subset = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        subset_scores = np.take_along_axis(scores, subset, axis=1)
        topk_idx = np.empty_like(subset)
        for row in range(subset.shape[0]):
            order = np.lexsort((subset[row], -subset_scores[row]))
            topk_idx[row] = subset[row, order]

    topk_scores = np.take_along_axis(scores, topk_idx, axis=1)
    return topk_idx.astype(np.int64), topk_scores.astype(np.float64)


def retrieve_in_chunks(
    query_matrix: sparse.spmatrix,
    candidate_matrix: sparse.spmatrix,
    topk_eval: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_queries = query_matrix.shape[0]
    k = max(1, min(topk_eval, candidate_matrix.shape[0]))
    all_idx = np.empty((n_queries, k), dtype=np.int64)
    all_scores = np.empty((n_queries, k), dtype=np.float64)

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        score_block = query_matrix[start:end] @ candidate_matrix.T
        if sparse.issparse(score_block):
            score_block = score_block.toarray()
        else:
            score_block = np.asarray(score_block)
        idx, values = topk_from_score_block(score_block, k)
        all_idx[start:end] = idx
        all_scores[start:end] = values

    return all_idx, all_scores


def tfidf_char_retrieval(
    queries: List[str],
    candidates: List[str],
    ngram_min: int,
    ngram_max: int,
    topk_eval: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        lowercase=False,
        norm="l2",
        sublinear_tf=True,
        dtype=np.float64,
    )
    candidate_matrix = vectorizer.fit_transform(candidates)
    query_matrix = vectorizer.transform(queries)
    return retrieve_in_chunks(query_matrix, candidate_matrix, topk_eval, chunk_size)


def bm25_char_retrieval(
    queries: List[str],
    candidates: List[str],
    ngram_min: int,
    ngram_max: int,
    topk_eval: int,
    k1: float,
    b: float,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse BM25 over character n-grams.

    Query term frequency is binary, matching the prior implementation, where a
    repeated query n-gram contributed once.
    """
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        lowercase=False,
        dtype=np.float64,
    )
    candidate_counts = vectorizer.fit_transform(candidates).tocsr()
    query_counts = vectorizer.transform(queries).tocsr()
    query_counts.data[:] = 1.0

    n_docs = candidate_counts.shape[0]
    doc_lens = np.asarray(candidate_counts.sum(axis=1)).ravel().astype(np.float64)
    avgdl = float(doc_lens.mean()) if n_docs else 0.0

    binary_candidates = candidate_counts.copy()
    binary_candidates.data[:] = 1.0
    doc_freq = np.asarray(binary_candidates.sum(axis=0)).ravel().astype(np.float64)
    idf = np.log1p((n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    coo = candidate_counts.tocoo(copy=True)
    length_norm = 1.0 - b + b * (doc_lens / max(avgdl, 1e-12))
    denominator = coo.data + k1 * length_norm[coo.row]
    coo.data = idf[coo.col] * ((coo.data * (k1 + 1.0)) / np.maximum(denominator, 1e-12))
    weighted_candidates = coo.tocsr()

    return retrieve_in_chunks(query_counts, weighted_candidates, topk_eval, chunk_size)


def ranking_metrics(
    topk_idx: np.ndarray,
    gold_idx: np.ndarray,
    eval_ks: Iterable[int],
) -> Tuple[Dict[str, float], np.ndarray]:
    n_items = topk_idx.shape[0]
    ranks = np.full(n_items, np.inf, dtype=np.float64)

    for row in range(n_items):
        hits = np.flatnonzero(topk_idx[row] == gold_idx[row])
        if hits.size:
            ranks[row] = float(hits[0] + 1)

    metrics: Dict[str, float] = {
        "MRR": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
        "Top1_acc": float(np.mean(ranks == 1.0)),
    }
    for requested_k in eval_ks:
        effective_k = int(max(1, min(requested_k, topk_idx.shape[1])))
        hit = float(np.mean(ranks <= effective_k))
        metrics[f"Hit@{effective_k}"] = hit
        metrics[f"Recall@{effective_k}"] = hit
        metrics[f"Precision@{effective_k}"] = hit / float(effective_k)
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
    available_cols = [column for column in metric_cols if column in df_out.columns]
    grouped = df_out.groupby("VN_len_bin", dropna=False)
    bucket_df = grouped[available_cols].mean().reset_index()
    bucket_df["count"] = grouped.size().values
    for column in available_cols:
        bucket_df[column] = bucket_df[column].astype(float).round(4)
    bucket_df.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


def configuration_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "method": args.method,
        "ngram_min": int(args.ngram_min),
        "ngram_max": int(args.ngram_max),
        "strip_accents": bool(args.strip_accents),
        "lowercase": not bool(args.no_lowercase),
        "remove_punct": bool(args.remove_punct),
        "remove_spaces": bool(args.remove_spaces),
        "bm25_k1": float(args.bm25_k1),
        "bm25_b": float(args.bm25_b),
        "topk_eval": int(args.topk_eval),
        "eval_ks": [int(k) for k in args.eval_ks],
        "chunk_size": int(args.chunk_size),
    }


def evaluation_cli_args(config: Dict[str, Any]) -> List[str]:
    cli = [
        "--method", str(config["method"]),
        "--ngram_min", str(config["ngram_min"]),
        "--ngram_max", str(config["ngram_max"]),
        "--bm25_k1", str(config["bm25_k1"]),
        "--bm25_b", str(config["bm25_b"]),
        "--topk_eval", str(config["topk_eval"]),
        "--chunk_size", str(config["chunk_size"]),
        "--eval_ks", *[str(k) for k in config["eval_ks"]],
    ]
    if config["strip_accents"]:
        cli.append("--strip_accents")
    if not config["lowercase"]:
        cli.append("--no_lowercase")
    if config["remove_punct"]:
        cli.append("--remove_punct")
    if config["remove_spaces"]:
        cli.append("--remove_spaces")
    return cli


def authorize_test_run(selection_manifest: Path, configuration_name: str, config: Dict[str, Any]) -> None:
    if not selection_manifest.is_file():
        raise FileNotFoundError(
            "A test run requires --selection_manifest produced by "
            "src/select_dev_configs_and_evaluate_test.py"
        )
    manifest = json.loads(selection_manifest.read_text(encoding="utf-8"))
    selected = manifest.get(FAMILY)
    if not selected:
        raise ValueError(f"No dev-selected configuration for family {FAMILY!r} in {selection_manifest}")
    if selected.get("configuration_name") != configuration_name:
        raise ValueError(
            f"Test configuration {configuration_name!r} does not match the dev-selected "
            f"configuration {selected.get('configuration_name')!r}"
        )
    if selected.get("configuration") != config:
        raise ValueError("Test configuration parameters do not match the dev-selection manifest")


def main(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ngram_min < 1 or args.ngram_max < args.ngram_min:
        raise ValueError("Require 1 <= ngram_min <= ngram_max")
    if args.chunk_size < 1:
        raise ValueError("--chunk_size must be positive")

    config = configuration_from_args(args)
    if args.split_name == "test":
        if not args.selection_manifest:
            raise ValueError("--selection_manifest is required for held-out test evaluation")
        authorize_test_run(Path(args.selection_manifest), args.configuration_name, config)

    df = validate_dataframe(pd.read_csv(input_csv), input_csv)
    raw_bah = df["Bahnaric"].astype(str).tolist()
    raw_vn = df["Vietnamese"].astype(str).tolist()

    normalization_kwargs = {
        "lowercase": config["lowercase"],
        "strip_accents": config["strip_accents"],
        "remove_punct": config["remove_punct"],
        "remove_spaces": config["remove_spaces"],
    }
    queries = [normalize_text(text, **normalization_kwargs) for text in raw_bah]
    candidates = [normalize_text(text, **normalization_kwargs) for text in raw_vn]

    if args.method == "tfidf_char":
        topk_idx, topk_scores = tfidf_char_retrieval(
            queries,
            candidates,
            args.ngram_min,
            args.ngram_max,
            args.topk_eval,
            args.chunk_size,
        )
    elif args.method == "bm25_char":
        topk_idx, topk_scores = bm25_char_retrieval(
            queries,
            candidates,
            args.ngram_min,
            args.ngram_max,
            args.topk_eval,
            args.bm25_k1,
            args.bm25_b,
            args.chunk_size,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    gold_idx = np.arange(len(df), dtype=np.int64)
    eval_ks = [int(k) for k in args.eval_ks]
    metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)

    pred_top1_idx = topk_idx[:, 0]
    predictions = [raw_vn[index] for index in pred_top1_idx]
    topk_predictions = ["|".join(raw_vn[index] for index in row) for row in topk_idx]
    reciprocal_rank = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).astype(np.float32)

    output = {
        "Bahnaric": raw_bah,
        "Bahnaric_normalized": queries,
        "Predicted_VN": predictions,
        "Gold_VN": raw_vn,
        "Gold_VN_normalized": candidates,
        "Gold_rank": [None if not np.isfinite(rank) else int(rank) for rank in ranks],
        "TopK_Preds": topk_predictions,
        "P@1": (pred_top1_idx == gold_idx).astype(np.float32).tolist(),
        "MRR": reciprocal_rank.tolist(),
        "Top1_score": topk_scores[:, 0].astype(float).tolist(),
    }
    for requested_k in eval_ks:
        effective_k = int(max(1, min(requested_k, topk_idx.shape[1])))
        output[f"Hit@{effective_k}"] = (ranks <= effective_k).astype(np.float32).tolist()

    predictions_df = make_bucket_columns(pd.DataFrame(output))
    predictions_df.to_csv(output_dir / "sentence_predictions.csv", index=False)
    save_bucket_metrics(predictions_df, output_dir, eval_ks)

    rounded_metrics: Dict[str, Any] = {key: round(float(value), 6) for key, value in metrics.items()}
    rounded_metrics.update(
        {
            "schema_version": 2,
            "family": FAMILY,
            "configuration_name": args.configuration_name,
            "split": args.split_name,
            "input_csv": str(input_csv),
            "num_queries": int(len(df)),
            "candidate_pool_size": int(len(df)),
            "selection_metric": "Top1_acc",
            "configuration": config,
            "evaluator_script": EVALUATOR_SCRIPT,
            "evaluation_cli_args": evaluation_cli_args(config),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(rounded_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Saved bucket metrics to {output_dir / 'bucket_metrics_vn_len.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BM25 / TF-IDF character n-gram retrieval with dev-only configuration selection."
    )
    parser.add_argument("--input_csv", required=True, help="CSV with Bahnaric,Vietnamese columns")
    parser.add_argument("--split_name", choices=["dev", "test"], required=True)
    parser.add_argument("--configuration_name", required=True)
    parser.add_argument("--selection_manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["tfidf_char", "bm25_char"], required=True)
    parser.add_argument("--ngram_min", type=int, default=2)
    parser.add_argument("--ngram_max", type=int, default=5)
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--remove_spaces", action="store_true")
    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--bm25_b", type=float, default=0.75)
    main(parser.parse_args())
