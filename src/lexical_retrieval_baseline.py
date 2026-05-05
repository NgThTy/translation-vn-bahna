"""
Lexical retrieval baselines for Bahnaric -> Vietnamese sentence retrieval.

Baselines:
1. TF-IDF character n-gram retrieval
2. BM25 character n-gram retrieval

Task formulation:
Given a Bahnaric query sentence, retrieve the correct Vietnamese sentence from
a candidate pool. For test.csv, the gold Vietnamese sentence is assumed to be
on the same row as the Bahnaric query.

Expected CSV columns:
- Bahnaric
- Vietnamese

Example:
python src/lexical_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method tfidf_char \
  --output_dir results/baselines/tfidf_char \
  --ngram_min 2 \
  --ngram_max 5 \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/lexical_retrieval_baseline.py \
  --test_csv data/test.csv \
  --method bm25_char \
  --output_dir results/baselines/bm25_char \
  --ngram_min 2 \
  --ngram_max 5 \
  --topk_eval 10 \
  --eval_ks 1 5 10
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_text(text: str, lowercase: bool = True, strip_accents: bool = False) -> str:
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
        # Remove combining marks after decomposition.
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = unicodedata.normalize("NFC", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def char_ngrams(text: str, ngram_min: int, ngram_max: int) -> List[str]:
    text = f" {text} "
    grams: List[str] = []
    for n in range(ngram_min, ngram_max + 1):
        if len(text) < n:
            continue
        grams.extend(text[i : i + n] for i in range(0, len(text) - n + 1))
    return grams


def ranking_metrics(topk_idx: np.ndarray, gold_idx: np.ndarray, eval_ks: List[int]) -> Tuple[Dict[str, float], np.ndarray]:
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


def tfidf_char_retrieval(
    queries: List[str],
    candidates: List[str],
    ngram_min: int,
    ngram_max: int,
    topk_eval: int,
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        lowercase=False,
        norm="l2",
        sublinear_tf=True,
    )

    candidate_matrix = vectorizer.fit_transform(candidates)
    query_matrix = vectorizer.transform(queries)

    # Cosine similarity because TF-IDF rows are L2-normalized.
    sims = query_matrix @ candidate_matrix.T
    sims = sims.toarray()

    k = int(max(1, min(topk_eval, len(candidates))))
    topk_idx = np.argsort(-sims, axis=1)[:, :k]
    return topk_idx, sims


def bm25_char_retrieval(
    queries: List[str],
    candidates: List[str],
    ngram_min: int,
    ngram_max: int,
    topk_eval: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    candidate_tokens = [char_ngrams(text, ngram_min, ngram_max) for text in candidates]
    query_tokens = [char_ngrams(text, ngram_min, ngram_max) for text in queries]

    n_docs = len(candidate_tokens)
    doc_lens = np.array([len(toks) for toks in candidate_tokens], dtype=np.float64)
    avgdl = float(doc_lens.mean()) if n_docs > 0 else 0.0

    doc_tf = [Counter(toks) for toks in candidate_tokens]

    df = Counter()
    for toks in candidate_tokens:
        df.update(set(toks))

    idf = {
        term: math.log(1.0 + ((n_docs - freq + 0.5) / (freq + 0.5)))
        for term, freq in df.items()
    }

    scores = np.zeros((len(queries), n_docs), dtype=np.float64)

    for qi, q_toks in enumerate(query_tokens):
        q_terms = Counter(q_toks)
        for term, q_count in q_terms.items():
            if term not in idf:
                continue

            term_idf = idf[term]

            for di, tf_counter in enumerate(doc_tf):
                f = tf_counter.get(term, 0)
                if f == 0:
                    continue

                denom = f + k1 * (1.0 - b + b * (doc_lens[di] / max(avgdl, 1e-9)))
                scores[qi, di] += term_idf * ((f * (k1 + 1.0)) / max(denom, 1e-9))

    k = int(max(1, min(topk_eval, n_docs)))
    topk_idx = np.argsort(-scores, axis=1)[:, :k]
    return topk_idx, scores


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
        normalize_text(x, lowercase=not args.no_lowercase, strip_accents=args.strip_accents)
        for x in raw_bah
    ]
    candidates = [
        normalize_text(x, lowercase=not args.no_lowercase, strip_accents=args.strip_accents)
        for x in raw_vn
    ]

    if args.method == "tfidf_char":
        topk_idx, scores = tfidf_char_retrieval(
            queries=queries,
            candidates=candidates,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            topk_eval=args.topk_eval,
        )
    elif args.method == "bm25_char":
        topk_idx, scores = bm25_char_retrieval(
            queries=queries,
            candidates=candidates,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            topk_eval=args.topk_eval,
            k1=args.bm25_k1,
            b=args.bm25_b,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

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
        "Predicted_VN": preds,
        "Gold_VN": raw_vn,
        "Gold_rank": gold_rank,
        "TopK_Preds": topk_preds,
        "P@1": is_top1.tolist(),
        "MRR": rr.tolist(),
        "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(df))],
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
            "ngram_min": int(args.ngram_min),
            "ngram_max": int(args.ngram_max),
            "strip_accents": bool(args.strip_accents),
            "lowercase": not bool(args.no_lowercase),
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
        description="BM25 / TF-IDF character n-gram baselines for Bahnaric-Vietnamese sentence retrieval"
    )

    parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
    parser.add_argument(
        "--method",
        choices=["tfidf_char", "bm25_char"],
        required=True,
        help="Lexical retrieval method",
    )
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--ngram_min", type=int, default=2)
    parser.add_argument("--ngram_max", type=int, default=5)
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--no_lowercase", action="store_true")

    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--bm25_b", type=float, default=0.75)

    args = parser.parse_args()
    main(args)
