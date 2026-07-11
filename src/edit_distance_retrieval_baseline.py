# """
# Levenshtein / normalized edit-distance baseline for Bahnaric -> Vietnamese sentence retrieval.

# Task:
# Given a Bahnaric query sentence, retrieve the correct Vietnamese sentence from a candidate pool.
# For data/test.csv, the gold Vietnamese sentence is assumed to be on the same row as the Bahnaric query.

# This baseline does NOT train anything.
# It only compares surface strings.

# Supported methods:
# 1. levenshtein_ratio
#    similarity = 1 - levenshtein_distance(a, b) / max(len(a), len(b))

# 2. levenshtein_distance
#    score = -levenshtein_distance(a, b)

# 3. token_jaccard
#    similarity = |tokens(a) intersection tokens(b)| / |tokens(a) union tokens(b)|

# Example:
# python src/edit_distance_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method levenshtein_ratio \
#   --output_dir results/baselines/levenshtein_ratio \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/edit_distance_retrieval_baseline.py \
#   --test_csv data/test.csv \
#   --method levenshtein_ratio \
#   --output_dir results/baselines/levenshtein_ratio_strip_accents \
#   --strip_accents \
#   --remove_spaces \
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
#     lowercase: bool = True,
#     strip_accents: bool = False,
#     remove_punct: bool = False,
#     remove_spaces: bool = False,
# ) -> str:
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
#         text = unicodedata.normalize("NFD", text)
#         text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
#         text = unicodedata.normalize("NFC", text)

#     if remove_punct:
#         # Keep letters, numbers, and whitespace.
#         text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

#     # Collapse whitespace.
#     text = re.sub(r"\s+", " ", text).strip()

#     if remove_spaces:
#         text = re.sub(r"\s+", "", text)

#     return text


# def levenshtein_distance(a: str, b: str) -> int:
#     """
#     Memory-efficient Levenshtein distance.
#     No external dependency required.
#     """
#     if a == b:
#         return 0

#     if len(a) < len(b):
#         a, b = b, a

#     # Now len(a) >= len(b)
#     previous = list(range(len(b) + 1))

#     for i, ca in enumerate(a, start=1):
#         current = [i]
#         for j, cb in enumerate(b, start=1):
#             insert_cost = current[j - 1] + 1
#             delete_cost = previous[j] + 1
#             replace_cost = previous[j - 1] + (0 if ca == cb else 1)
#             current.append(min(insert_cost, delete_cost, replace_cost))
#         previous = current

#     return previous[-1]


# def levenshtein_ratio(a: str, b: str) -> float:
#     """
#     Normalized edit-distance similarity.
#     1.0 means identical.
#     0.0 means maximally different under max-length normalization.
#     """
#     max_len = max(len(a), len(b))
#     if max_len == 0:
#         return 1.0
#     dist = levenshtein_distance(a, b)
#     return 1.0 - (float(dist) / float(max_len))


# def token_jaccard(a: str, b: str) -> float:
#     ta = set(a.split())
#     tb = set(b.split())

#     if not ta and not tb:
#         return 1.0
#     if not ta or not tb:
#         return 0.0

#     return len(ta & tb) / float(len(ta | tb))


# def score_pair(a: str, b: str, method: str) -> float:
#     if method == "levenshtein_ratio":
#         return levenshtein_ratio(a, b)
#     if method == "levenshtein_distance":
#         return -float(levenshtein_distance(a, b))
#     if method == "token_jaccard":
#         return token_jaccard(a, b)
#     raise ValueError(f"Unknown method: {method}")


# def edit_distance_retrieval(
#     queries: List[str],
#     candidates: List[str],
#     method: str,
#     topk_eval: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     n_queries = len(queries)
#     n_candidates = len(candidates)

#     scores = np.zeros((n_queries, n_candidates), dtype=np.float64)

#     for qi, query in enumerate(queries):
#         for ci, cand in enumerate(candidates):
#             scores[qi, ci] = score_pair(query, cand, method)

#     k = int(max(1, min(topk_eval, n_candidates)))
#     topk_idx = np.argsort(-scores, axis=1)[:, :k]
#     return topk_idx, scores


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


# def main(args: argparse.Namespace) -> None:
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

#     raw_bah = df["Bahnaric"].astype(str).tolist()
#     raw_vn = df["Vietnamese"].astype(str).tolist()

#     queries = [
#         normalize_text(
#             x,
#             lowercase=not args.no_lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#             remove_spaces=args.remove_spaces,
#         )
#         for x in raw_bah
#     ]

#     candidates = [
#         normalize_text(
#             x,
#             lowercase=not args.no_lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#             remove_spaces=args.remove_spaces,
#         )
#         for x in raw_vn
#     ]

#     topk_idx, scores = edit_distance_retrieval(
#         queries=queries,
#         candidates=candidates,
#         method=args.method,
#         topk_eval=args.topk_eval,
#     )

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
#         "Bahnaric_normalized": queries,
#         "Predicted_VN": preds,
#         "Gold_VN": raw_vn,
#         "Gold_VN_normalized": candidates,
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
#             "method": args.method,
#             "test_csv": args.test_csv,
#             "num_queries": int(len(df)),
#             "candidate_pool_size": int(len(df)),
#             "strip_accents": bool(args.strip_accents),
#             "lowercase": not bool(args.no_lowercase),
#             "remove_punct": bool(args.remove_punct),
#             "remove_spaces": bool(args.remove_spaces),
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
#         description="Levenshtein / normalized edit-distance baseline for Bahnaric-Vietnamese sentence retrieval"
#     )

#     parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
#     parser.add_argument(
#         "--method",
#         choices=["levenshtein_ratio", "levenshtein_distance", "token_jaccard"],
#         required=True,
#         help="Surface-similarity retrieval method",
#     )
#     parser.add_argument("--output_dir", required=True)

#     parser.add_argument("--topk_eval", type=int, default=10)
#     parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

#     parser.add_argument("--strip_accents", action="store_true")
#     parser.add_argument("--no_lowercase", action="store_true")
#     parser.add_argument("--remove_punct", action="store_true")
#     parser.add_argument("--remove_spaces", action="store_true")

#     args = parser.parse_args()
#     main(args)

#!/usr/bin/env python3
"""Surface-similarity retrieval baselines for Bahnaric -> Vietnamese.

All variant and preprocessing selection is performed on the development split.
A held-out test run is accepted only when its configuration matches the
selection manifest produced from development results.
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
from sklearn.feature_extraction.text import CountVectorizer

FAMILY = "edit_distance"
EVALUATOR_SCRIPT = "src/edit_distance_retrieval_baseline.py"
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
    n_candidates = scores.shape[1]
    k = max(1, min(k, n_candidates))
    if k == n_candidates:
        candidate_idx = np.broadcast_to(np.arange(n_candidates), scores.shape)
        order = np.lexsort((candidate_idx, -scores), axis=1)
        topk_idx = order[:, :k]
    else:
        subset = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        subset_scores = np.take_along_axis(scores, subset, axis=1)
        topk_idx = np.empty_like(subset)
        for row in range(subset.shape[0]):
            order = np.lexsort((subset[row], -subset_scores[row]))
            topk_idx[row] = subset[row, order]
    topk_scores = np.take_along_axis(scores, topk_idx, axis=1)
    return topk_idx.astype(np.int64), topk_scores.astype(np.float64)


def rapidfuzz_retrieval(
    queries: List[str],
    candidates: List[str],
    method: str,
    topk_eval: int,
    chunk_size: int,
    workers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein
    except ImportError as exc:
        raise RuntimeError(
            "Levenshtein evaluation on the dev pool requires RapidFuzz. Install it with: "
            "python -m pip install 'rapidfuzz>=3.9,<4'"
        ) from exc

    n_queries = len(queries)
    k = max(1, min(topk_eval, len(candidates)))
    all_idx = np.empty((n_queries, k), dtype=np.int64)
    all_scores = np.empty((n_queries, k), dtype=np.float64)
    gold_scores = np.empty(n_queries, dtype=np.float64)

    if method == "levenshtein_ratio":
        scorer = Levenshtein.normalized_similarity
        negate = False
    elif method == "levenshtein_distance":
        scorer = Levenshtein.distance
        negate = True
    else:
        raise ValueError(f"Unsupported RapidFuzz method: {method}")

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        block = process.cdist(
            queries[start:end],
            candidates,
            scorer=scorer,
            workers=workers,
            dtype=np.float32,
        )
        block = np.asarray(block, dtype=np.float64)
        if negate:
            block = -block

        idx, values = topk_from_score_block(block, k)
        all_idx[start:end] = idx
        all_scores[start:end] = values
        local_rows = np.arange(end - start)
        gold_scores[start:end] = block[local_rows, np.arange(start, end)]

    return all_idx, all_scores, gold_scores


def token_jaccard_retrieval(
    queries: List[str],
    candidates: List[str],
    topk_eval: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectorizer = CountVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        binary=True,
        dtype=np.float64,
    )
    candidate_matrix = vectorizer.fit_transform(candidates).tocsr()
    query_matrix = vectorizer.transform(queries).tocsr()

    candidate_sizes = np.asarray(candidate_matrix.sum(axis=1)).ravel()
    query_sizes = np.asarray(query_matrix.sum(axis=1)).ravel()

    n_queries = len(queries)
    k = max(1, min(topk_eval, len(candidates)))
    all_idx = np.empty((n_queries, k), dtype=np.int64)
    all_scores = np.empty((n_queries, k), dtype=np.float64)
    gold_scores = np.empty(n_queries, dtype=np.float64)

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)
        intersection = query_matrix[start:end] @ candidate_matrix.T
        if sparse.issparse(intersection):
            intersection = intersection.toarray()
        intersection = np.asarray(intersection, dtype=np.float64)

        union = query_sizes[start:end, None] + candidate_sizes[None, :] - intersection
        scores = np.divide(
            intersection,
            union,
            out=np.ones_like(intersection, dtype=np.float64),
            where=union > 0,
        )

        idx, values = topk_from_score_block(scores, k)
        all_idx[start:end] = idx
        all_scores[start:end] = values
        local_rows = np.arange(end - start)
        gold_scores[start:end] = scores[local_rows, np.arange(start, end)]

    return all_idx, all_scores, gold_scores


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
        "strip_accents": bool(args.strip_accents),
        "lowercase": not bool(args.no_lowercase),
        "remove_punct": bool(args.remove_punct),
        "remove_spaces": bool(args.remove_spaces),
        "topk_eval": int(args.topk_eval),
        "eval_ks": [int(k) for k in args.eval_ks],
        "chunk_size": int(args.chunk_size),
        "workers": int(args.workers),
    }


def evaluation_cli_args(config: Dict[str, Any]) -> List[str]:
    cli = [
        "--method", str(config["method"]),
        "--topk_eval", str(config["topk_eval"]),
        "--chunk_size", str(config["chunk_size"]),
        "--workers", str(config["workers"]),
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

    if args.chunk_size < 1:
        raise ValueError("--chunk_size must be positive")
    if args.workers == 0 or args.workers < -1:
        raise ValueError("--workers must be -1 or a positive integer")

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

    if args.method in {"levenshtein_ratio", "levenshtein_distance"}:
        topk_idx, topk_scores, gold_scores = rapidfuzz_retrieval(
            queries,
            candidates,
            args.method,
            args.topk_eval,
            args.chunk_size,
            args.workers,
        )
    elif args.method == "token_jaccard":
        topk_idx, topk_scores, gold_scores = token_jaccard_retrieval(
            queries,
            candidates,
            args.topk_eval,
            args.chunk_size,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    gold_idx = np.arange(len(df), dtype=np.int64)
    eval_ks = [int(k) for k in args.eval_ks]
    metrics, ranks = ranking_metrics(topk_idx, gold_idx, eval_ks)

    pred_top1_idx = topk_idx[:, 0]
    output = {
        "Bahnaric": raw_bah,
        "Bahnaric_normalized": queries,
        "Predicted_VN": [raw_vn[index] for index in pred_top1_idx],
        "Gold_VN": raw_vn,
        "Gold_VN_normalized": candidates,
        "Gold_rank": [None if not np.isfinite(rank) else int(rank) for rank in ranks],
        "TopK_Preds": ["|".join(raw_vn[index] for index in row) for row in topk_idx],
        "P@1": (pred_top1_idx == gold_idx).astype(np.float32).tolist(),
        "MRR": np.where(np.isfinite(ranks), 1.0 / ranks, 0.0).astype(np.float32).tolist(),
        "Top1_score": topk_scores[:, 0].astype(float).tolist(),
        "Gold_score": gold_scores.astype(float).tolist(),
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
        description="Levenshtein and token-Jaccard retrieval with dev-only configuration selection."
    )
    parser.add_argument("--input_csv", required=True, help="CSV with Bahnaric,Vietnamese columns")
    parser.add_argument("--split_name", choices=["dev", "test"], required=True)
    parser.add_argument("--configuration_name", required=True)
    parser.add_argument("--selection_manifest", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--method",
        choices=["levenshtein_ratio", "levenshtein_distance", "token_jaccard"],
        required=True,
    )
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument(
        "--workers",
        type=int,
        default=-1,
        help="RapidFuzz worker count; -1 uses all logical CPUs.",
    )
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--remove_spaces", action="store_true")
    main(parser.parse_args())
