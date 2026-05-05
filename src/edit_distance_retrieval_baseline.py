"""
Levenshtein / normalized edit-distance baseline for Bahnaric -> Vietnamese sentence retrieval.

Task:
Given a Bahnaric query sentence, retrieve the correct Vietnamese sentence from a candidate pool.
For data/test.csv, the gold Vietnamese sentence is assumed to be on the same row as the Bahnaric query.

This baseline does NOT train anything.
It only compares surface strings.

Supported methods:
1. levenshtein_ratio
   similarity = 1 - levenshtein_distance(a, b) / max(len(a), len(b))

2. levenshtein_distance
   score = -levenshtein_distance(a, b)

3. token_jaccard
   similarity = |tokens(a) intersection tokens(b)| / |tokens(a) union tokens(b)|

Example:
python src/edit_distance_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method levenshtein_ratio \
  --output_dir results/baselines/levenshtein_ratio \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/edit_distance_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method levenshtein_ratio \
  --output_dir results/baselines/levenshtein_ratio_strip_accents \
  --strip_accents \
  --remove_spaces \
  --topk_eval 10 \
  --eval_ks 1 5 10
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_accents: bool = False,
    remove_punct: bool = False,
    remove_spaces: bool = False,
) -> str:
    text = str(text)

    # Normalize Unicode form.
    text = unicodedata.normalize("NFC", text)

    # Normalize apostrophe-like characters.
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")

    # Normalize quote/dash variants.
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")

    if lowercase:
        text = text.lower()

    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = unicodedata.normalize("NFC", text)

    if remove_punct:
        # Keep letters, numbers, and whitespace.
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    if remove_spaces:
        text = re.sub(r"\s+", "", text)

    return text


def levenshtein_distance(a: str, b: str) -> int:
    """
    Memory-efficient Levenshtein distance.
    No external dependency required.
    """
    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    # Now len(a) >= len(b)
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current

    return previous[-1]


def levenshtein_ratio(a: str, b: str) -> float:
    """
    Normalized edit-distance similarity.
    1.0 means identical.
    0.0 means maximally different under max-length normalization.
    """
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(a, b)
    return 1.0 - (float(dist) / float(max_len))


def token_jaccard(a: str, b: str) -> float:
    ta = set(a.split())
    tb = set(b.split())

    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    return len(ta & tb) / float(len(ta | tb))


def score_pair(a: str, b: str, method: str) -> float:
    if method == "levenshtein_ratio":
        return levenshtein_ratio(a, b)
    if method == "levenshtein_distance":
        return -float(levenshtein_distance(a, b))
    if method == "token_jaccard":
        return token_jaccard(a, b)
    raise ValueError(f"Unknown method: {method}")


def edit_distance_retrieval(
    queries: List[str],
    candidates: List[str],
    method: str,
    topk_eval: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_queries = len(queries)
    n_candidates = len(candidates)

    scores = np.zeros((n_queries, n_candidates), dtype=np.float64)

    for qi, query in enumerate(queries):
        for ci, cand in enumerate(candidates):
            scores[qi, ci] = score_pair(query, cand, method)

    k = int(max(1, min(topk_eval, n_candidates)))
    topk_idx = np.argsort(-scores, axis=1)[:, :k]
    return topk_idx, scores


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


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

    raw_bah = df["Bahnaric"].astype(str).tolist()
    raw_vn = df["Vietnamese"].astype(str).tolist()

    queries = [
        normalize_text(
            x,
            lowercase=not args.no_lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
            remove_spaces=args.remove_spaces,
        )
        for x in raw_bah
    ]

    candidates = [
        normalize_text(
            x,
            lowercase=not args.no_lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
            remove_spaces=args.remove_spaces,
        )
        for x in raw_vn
    ]

    topk_idx, scores = edit_distance_retrieval(
        queries=queries,
        candidates=candidates,
        method=args.method,
        topk_eval=args.topk_eval,
    )

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
        "Bahnaric_normalized": queries,
        "Predicted_VN": preds,
        "Gold_VN": raw_vn,
        "Gold_VN_normalized": candidates,
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
            "method": args.method,
            "test_csv": args.test_csv,
            "num_queries": int(len(df)),
            "candidate_pool_size": int(len(df)),
            "strip_accents": bool(args.strip_accents),
            "lowercase": not bool(args.no_lowercase),
            "remove_punct": bool(args.remove_punct),
            "remove_spaces": bool(args.remove_spaces),
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
        description="Levenshtein / normalized edit-distance baseline for Bahnaric-Vietnamese sentence retrieval"
    )

    parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
    parser.add_argument(
        "--method",
        choices=["levenshtein_ratio", "levenshtein_distance", "token_jaccard"],
        required=True,
        help="Surface-similarity retrieval method",
    )
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--remove_spaces", action="store_true")

    args = parser.parse_args()
    main(args)
