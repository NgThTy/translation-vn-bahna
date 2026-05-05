"""
FastAlign / GIZA++-style lexical word-alignment baseline for Bahnaric -> Vietnamese retrieval.

This script implements an IBM Model 1 lexical translation baseline in pure Python.

Why this baseline?
- FastAlign and GIZA++ are classical word alignment tools.
- They estimate word-level translation correspondences from parallel text.
- For this retrieval task, we use the learned lexical translation table p(vn_word | bah_word)
  to score each Vietnamese candidate sentence for a given Bahnaric query sentence.

Task:
Given a Bahnaric query sentence, retrieve the correct Vietnamese sentence from a candidate pool.
For data/test.csv, the gold Vietnamese sentence is assumed to be on the same row.

Methods:
1. ibm1
   Train p(tgt | src) with EM on train.csv.
   Score candidate by average log translation probability from Bahnaric tokens to Vietnamese tokens.

2. ibm1_sym
   Train both p(tgt | src) and p(src | tgt).
   Score candidate by average of forward and backward lexical scores.

No neural model is trained.
This is a classical lexical alignment baseline.
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def normalize_text(
    text: str,
    lowercase: bool = True,
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


def tokenize(text: str) -> List[str]:
    text = str(text).strip()
    if not text:
        return []
    return text.split()


def read_parallel_csv(
    path: str,
    lowercase: bool,
    strip_accents: bool,
    remove_punct: bool,
) -> Tuple[List[List[str]], List[List[str]], List[str], List[str]]:
    df = pd.read_csv(path).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

    if "Bahnaric" not in df.columns or "Vietnamese" not in df.columns:
        raise ValueError("CSV must have columns: Bahnaric,Vietnamese")

    raw_src = df["Bahnaric"].astype(str).tolist()
    raw_tgt = df["Vietnamese"].astype(str).tolist()

    src_tok = [
        tokenize(normalize_text(x, lowercase=lowercase, strip_accents=strip_accents, remove_punct=remove_punct))
        for x in raw_src
    ]
    tgt_tok = [
        tokenize(normalize_text(x, lowercase=lowercase, strip_accents=strip_accents, remove_punct=remove_punct))
        for x in raw_tgt
    ]

    return src_tok, tgt_tok, raw_src, raw_tgt


def build_candidate_sets(
    src_sents: List[List[str]],
    tgt_sents: List[List[str]],
    max_pairs_per_src_word: int = 200000,
) -> Dict[str, set]:
    """
    For IBM1 initialization, each source word can translate to any target word
    that co-occurs with it in a parallel sentence.

    This is much cheaper than full source_vocab x target_vocab initialization.
    """
    candidates: Dict[str, set] = defaultdict(set)

    for src, tgt in zip(src_sents, tgt_sents):
        src_types = set(src)
        tgt_types = set(tgt)
        for s in src_types:
            if len(candidates[s]) < max_pairs_per_src_word:
                candidates[s].update(tgt_types)

    return candidates


def initialize_uniform_t(
    candidates: Dict[str, set],
    add_null: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    t_table[src][tgt] = p(tgt | src)
    """
    t_table: Dict[str, Dict[str, float]] = {}

    for src_word, tgt_set in candidates.items():
        tgt_list = sorted(tgt_set)
        if add_null:
            tgt_list = ["<NULL>"] + tgt_list
        if not tgt_list:
            continue
        p = 1.0 / float(len(tgt_list))
        t_table[src_word] = {t: p for t in tgt_list}

    return t_table


def train_ibm1(
    src_sents: List[List[str]],
    tgt_sents: List[List[str]],
    num_iters: int = 5,
    min_prob: float = 1e-12,
) -> Dict[str, Dict[str, float]]:
    """
    Train IBM Model 1 p(tgt | src) using EM.

    Source = Bahnaric tokens.
    Target = Vietnamese tokens.
    """
    candidates = build_candidate_sets(src_sents, tgt_sents)
    t_table = initialize_uniform_t(candidates)

    print(f"IBM1 candidate source vocab size: {len(t_table)}")
    print(f"IBM1 iterations: {num_iters}")

    for it in range(1, num_iters + 1):
        count_st: DefaultDict[str, Counter] = defaultdict(Counter)
        total_s: Counter = Counter()
        log_likelihood = 0.0
        token_events = 0

        for src, tgt in zip(src_sents, tgt_sents):
            if not src or not tgt:
                continue

            src_types = list(dict.fromkeys(src))

            for t in tgt:
                denom = 0.0
                active_src = []

                for s in src_types:
                    prob = t_table.get(s, {}).get(t, 0.0)
                    if prob > 0.0:
                        denom += prob
                        active_src.append(s)

                if denom <= 0.0:
                    continue

                log_likelihood += math.log(max(denom, min_prob))
                token_events += 1

                for s in active_src:
                    delta = t_table[s].get(t, 0.0) / denom
                    count_st[s][t] += delta
                    total_s[s] += delta

        for s, counter in count_st.items():
            denom = float(total_s[s])
            if denom <= 0.0:
                continue
            for t, c in counter.items():
                t_table[s][t] = max(float(c) / denom, min_prob)

        avg_ll = log_likelihood / max(1, token_events)
        print(f"IBM1 iter {it}/{num_iters}: avg_log_likelihood={avg_ll:.6f}")

    return t_table


def lexical_score_forward(
    src_tokens: List[str],
    tgt_tokens: List[str],
    t_table: Dict[str, Dict[str, float]],
    smoothing: float = 1e-9,
    length_penalty: float = 0.0,
) -> float:
    """
    Score p(tgt_sentence | src_sentence) approximately.

    For each source token s, find the best translation probability among words in
    the candidate target sentence. This is a retrieval-friendly lexical coverage score.

    score = average_s log max_t p(t | s)
    """
    if not src_tokens or not tgt_tokens:
        return -1e9

    tgt_set = set(tgt_tokens)
    total = 0.0
    used = 0

    for s in src_tokens:
        trans = t_table.get(s)
        if not trans:
            total += math.log(smoothing)
            used += 1
            continue

        best = smoothing
        for t in tgt_set:
            p = trans.get(t, 0.0)
            if p > best:
                best = p

        total += math.log(max(best, smoothing))
        used += 1

    score = total / float(max(1, used))

    if length_penalty > 0.0:
        score -= length_penalty * abs(len(src_tokens) - len(tgt_tokens)) / float(max(len(src_tokens), len(tgt_tokens), 1))

    return score


def lexical_score_symmetric(
    src_tokens: List[str],
    tgt_tokens: List[str],
    fwd_table: Dict[str, Dict[str, float]],
    bwd_table: Dict[str, Dict[str, float]],
    smoothing: float = 1e-9,
    length_penalty: float = 0.0,
) -> float:
    fwd = lexical_score_forward(
        src_tokens,
        tgt_tokens,
        fwd_table,
        smoothing=smoothing,
        length_penalty=length_penalty,
    )
    bwd = lexical_score_forward(
        tgt_tokens,
        src_tokens,
        bwd_table,
        smoothing=smoothing,
        length_penalty=length_penalty,
    )
    return 0.5 * (fwd + bwd)


def retrieve_topk(
    src_queries: List[List[str]],
    tgt_candidates: List[List[str]],
    method: str,
    fwd_table: Dict[str, Dict[str, float]],
    bwd_table: Dict[str, Dict[str, float]],
    topk_eval: int,
    smoothing: float,
    length_penalty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_q = len(src_queries)
    n_c = len(tgt_candidates)

    scores = np.zeros((n_q, n_c), dtype=np.float64)

    for i, src in enumerate(src_queries):
        if i % 100 == 0:
            print(f"Scoring query {i}/{n_q}")

        for j, tgt in enumerate(tgt_candidates):
            if method == "ibm1":
                scores[i, j] = lexical_score_forward(
                    src,
                    tgt,
                    fwd_table,
                    smoothing=smoothing,
                    length_penalty=length_penalty,
                )
            elif method == "ibm1_sym":
                scores[i, j] = lexical_score_symmetric(
                    src,
                    tgt,
                    fwd_table,
                    bwd_table,
                    smoothing=smoothing,
                    length_penalty=length_penalty,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

    k = int(max(1, min(topk_eval, n_c)))
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


def save_translation_table(
    t_table: Dict[str, Dict[str, float]],
    output_path: Path,
    topn_per_source: int = 20,
) -> None:
    rows = []
    for s, trans in t_table.items():
        top_items = sorted(trans.items(), key=lambda x: -x[1])[:topn_per_source]
        for t, p in top_items:
            rows.append({"source": s, "target": t, "prob": float(p)})

    pd.DataFrame(rows).to_csv(output_path, index=False)


def export_fastalign_format(
    train_csv: str,
    output_path: Path,
    lowercase: bool,
    strip_accents: bool,
    remove_punct: bool,
) -> None:
    src_sents, tgt_sents, _, _ = read_parallel_csv(
        train_csv,
        lowercase=lowercase,
        strip_accents=strip_accents,
        remove_punct=remove_punct,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for src, tgt in zip(src_sents, tgt_sents):
            f.write(" ".join(src) + " ||| " + " ".join(tgt) + "\n")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lowercase = not args.no_lowercase

    print("[1/5] Reading training data...")
    train_src, train_tgt, _, _ = read_parallel_csv(
        args.train_csv,
        lowercase=lowercase,
        strip_accents=args.strip_accents,
        remove_punct=args.remove_punct,
    )

    print(f"Training sentence pairs: {len(train_src)}")

    if args.export_fastalign_input:
        fastalign_path = output_dir / "fastalign_input.txt"
        export_fastalign_format(
            args.train_csv,
            fastalign_path,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        print(f"Saved FastAlign-format training file to {fastalign_path}")

    print("[2/5] Training forward IBM1 p(Vietnamese | Bahnaric)...")
    fwd_table = train_ibm1(
        train_src,
        train_tgt,
        num_iters=args.ibm_iters,
        min_prob=args.smoothing,
    )

    bwd_table: Dict[str, Dict[str, float]] = {}
    if args.method == "ibm1_sym":
        print("[3/5] Training backward IBM1 p(Bahnaric | Vietnamese)...")
        bwd_table = train_ibm1(
            train_tgt,
            train_src,
            num_iters=args.ibm_iters,
            min_prob=args.smoothing,
        )
    else:
        print("[3/5] Skipping backward model because method=ibm1")

    print("[4/5] Reading test data...")
    test_src, test_tgt, raw_bah, raw_vn = read_parallel_csv(
        args.test_csv,
        lowercase=lowercase,
        strip_accents=args.strip_accents,
        remove_punct=args.remove_punct,
    )

    print(f"Test queries: {len(test_src)}")
    print(f"Candidate pool size: {len(test_tgt)}")

    print("[5/5] Scoring retrieval candidates...")
    topk_idx, scores = retrieve_topk(
        src_queries=test_src,
        tgt_candidates=test_tgt,
        method=args.method,
        fwd_table=fwd_table,
        bwd_table=bwd_table,
        topk_eval=args.topk_eval,
        smoothing=args.smoothing,
        length_penalty=args.length_penalty,
    )

    gold_idx = np.arange(len(test_src), dtype=np.int64)
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
        "Bahnaric_tokens": [" ".join(x) for x in test_src],
        "Predicted_VN": preds,
        "Gold_VN": raw_vn,
        "Gold_VN_tokens": [" ".join(x) for x in test_tgt],
        "Gold_rank": gold_rank,
        "TopK_Preds": topk_preds,
        "P@1": is_top1.tolist(),
        "MRR": rr.tolist(),
        "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(test_src))],
        "Gold_score": [float(scores[i, i]) for i in range(len(test_src))],
    }

    for k in eval_ks:
        k_eff = int(max(1, min(k, topk_idx.shape[1])))
        out[f"Hit@{k_eff}"] = (ranks <= float(k_eff)).astype(np.float32).tolist()

    df_out = pd.DataFrame(out)
    df_out = make_bucket_columns(df_out)
    df_out.to_csv(output_dir / "sentence_predictions.csv", index=False)

    save_bucket_metrics(df_out, output_dir, eval_ks)
    save_translation_table(fwd_table, output_dir / "translation_table_forward_top20.csv", topn_per_source=20)
    if bwd_table:
        save_translation_table(bwd_table, output_dir / "translation_table_backward_top20.csv", topn_per_source=20)

    rounded_metrics = {k: round(float(v), 4) for k, v in metrics.items()}
    rounded_metrics.update(
        {
            "method": args.method,
            "alignment_family": "fastalign_giza_ibm1",
            "test_csv": args.test_csv,
            "train_csv": args.train_csv,
            "num_train_pairs": int(len(train_src)),
            "num_queries": int(len(test_src)),
            "candidate_pool_size": int(len(test_tgt)),
            "ibm_iters": int(args.ibm_iters),
            "smoothing": float(args.smoothing),
            "length_penalty": float(args.length_penalty),
            "strip_accents": bool(args.strip_accents),
            "lowercase": lowercase,
            "remove_punct": bool(args.remove_punct),
        }
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rounded_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Saved bucket metrics to {output_dir / 'bucket_metrics_vn_len.csv'}")
    print(f"Saved forward translation table to {output_dir / 'translation_table_forward_top20.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastAlign/GIZA++-style IBM1 lexical alignment baseline for Bahnaric-Vietnamese retrieval"
    )

    parser.add_argument("--train_csv", required=True, help="Parallel training CSV with Bahnaric,Vietnamese columns")
    parser.add_argument("--test_csv", required=True, help="Parallel test CSV with Bahnaric,Vietnamese columns")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--method",
        choices=["ibm1", "ibm1_sym"],
        default="ibm1",
        help="ibm1 = forward lexical score; ibm1_sym = average forward/backward lexical scores",
    )

    parser.add_argument("--ibm_iters", type=int, default=5)
    parser.add_argument("--smoothing", type=float, default=1e-9)
    parser.add_argument("--length_penalty", type=float, default=0.0)

    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")

    parser.add_argument(
        "--export_fastalign_input",
        action="store_true",
        help="Also export train data in 'src ||| tgt' format for external fast_align.",
    )

    args = parser.parse_args()
    main(args)
