"""
Hybrid lexical-neural reranking for Bahnaric -> Vietnamese sentence retrieval.

Method 8:
Hybrid IBM1 + XLM-R LoRA reranking.

Pipeline:
1. Use an existing neural retrieval output, usually from:
   XLM-R LoRA + projection + no Kabsch + CSLS.

2. Read neural TopK candidates from sentence_predictions.csv.

3. Train IBM1 symmetric lexical translation models on train.csv:
   - p(Vietnamese | Bahnaric)
   - p(Bahnaric | Vietnamese)

4. For each query, rerank only the neural TopK candidates using:
   score = alpha * neural_score
         + beta  * ibm1_symmetric_score
         + gamma * levenshtein_score

5. Tune alpha/beta/gamma on dev if dev files are provided.
   Otherwise run fixed weights on test.

Expected use:
- Neural candidate generation should be done on GPU server.
- This reranker can run locally on CPU.
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Text utilities
# ============================================================

def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_accents: bool = False,
    remove_punct: bool = False,
    remove_spaces: bool = False,
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

    if remove_spaces:
        text = text.replace(" ", "")

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
) -> Tuple[List[List[str]], List[List[str]], List[str], List[str], pd.DataFrame]:
    df = pd.read_csv(path).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

    if "Bahnaric" not in df.columns or "Vietnamese" not in df.columns:
        raise ValueError("CSV must contain columns: Bahnaric,Vietnamese")

    raw_src = df["Bahnaric"].astype(str).tolist()
    raw_tgt = df["Vietnamese"].astype(str).tolist()

    src_tok = [
        tokenize(
            normalize_text(
                x,
                lowercase=lowercase,
                strip_accents=strip_accents,
                remove_punct=remove_punct,
            )
        )
        for x in raw_src
    ]

    tgt_tok = [
        tokenize(
            normalize_text(
                x,
                lowercase=lowercase,
                strip_accents=strip_accents,
                remove_punct=remove_punct,
            )
        )
        for x in raw_tgt
    ]

    return src_tok, tgt_tok, raw_src, raw_tgt, df


# ============================================================
# IBM1 training
# ============================================================

def build_candidate_sets(
    src_sents: List[List[str]],
    tgt_sents: List[List[str]],
    max_pairs_per_src_word: int = 200000,
) -> Dict[str, set]:
    candidates: Dict[str, set] = defaultdict(set)

    for src, tgt in zip(src_sents, tgt_sents):
        src_types = set(src)
        tgt_types = set(tgt)

        for s in src_types:
            if len(candidates[s]) < max_pairs_per_src_word:
                candidates[s].update(tgt_types)

    return candidates


def initialize_uniform_t(candidates: Dict[str, set]) -> Dict[str, Dict[str, float]]:
    t_table: Dict[str, Dict[str, float]] = {}

    for src_word, tgt_set in candidates.items():
        tgt_list = sorted(tgt_set)
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
        score -= length_penalty * abs(len(src_tokens) - len(tgt_tokens)) / float(
            max(len(src_tokens), len(tgt_tokens), 1)
        )

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


# ============================================================
# Neural TopK reading
# ============================================================

def split_pipe_cell(cell: str) -> List[str]:
    if pd.isna(cell):
        return []
    text = str(cell)
    if not text.strip():
        return []
    return [x.strip() for x in text.split("|") if x.strip()]


def parse_topk_scores(row: pd.Series, k: int) -> List[float]:
    """
    Supports optional TopK_Scores-style columns.
    If absent, fall back to rank-based score.
    """
    possible_cols = [
        "TopK_Scores",
        "TopK_scores",
        "TopK_scores_raw",
        "TopK_Sims",
        "TopK_similarities",
    ]

    for col in possible_cols:
        if col in row and not pd.isna(row[col]):
            vals = split_pipe_cell(row[col])
            try:
                scores = [float(x) for x in vals]
                if len(scores) >= k:
                    return scores[:k]
            except Exception:
                pass

    # Fallback: rank-based neural score.
    # Top-ranked candidate gets highest score.
    return [1.0 / float(i + 1) for i in range(k)]


def load_neural_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "TopK_Preds" not in df.columns:
        raise ValueError(
            f"{path} must contain TopK_Preds column. "
            "Please save neural top-K predictions from Baseline 6 first."
        )

    if "Bahnaric" not in df.columns:
        raise ValueError(f"{path} must contain Bahnaric column.")

    if "Gold_VN" not in df.columns and "Vietnamese" not in df.columns:
        raise ValueError(f"{path} must contain Gold_VN or Vietnamese column.")

    return df


# ============================================================
# Reranking
# ============================================================

def minmax_normalize(values: List[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return []

    lo = float(np.min(arr))
    hi = float(np.max(arr))

    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]

    return ((arr - lo) / (hi - lo)).astype(float).tolist()


def levenshtein_like_ratio(src_text: str, tgt_text: str) -> float:
    """
    Uses SequenceMatcher ratio as a lightweight lexical similarity score.
    This avoids requiring python-Levenshtein.
    """
    return float(SequenceMatcher(None, src_text, tgt_text).ratio())


def rerank_one_query(
    raw_src: str,
    norm_src_tokens: List[str],
    candidate_texts: List[str],
    neural_scores: List[float],
    fwd_table: Dict[str, Dict[str, float]],
    bwd_table: Dict[str, Dict[str, float]],
    lowercase: bool,
    strip_accents: bool,
    remove_punct: bool,
    smoothing: float,
    length_penalty: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    ibm_scores = []
    lev_scores = []

    norm_src_text_for_lev = normalize_text(
        raw_src,
        lowercase=lowercase,
        strip_accents=strip_accents,
        remove_punct=remove_punct,
        remove_spaces=False,
    )

    for cand in candidate_texts:
        norm_cand = normalize_text(
            cand,
            lowercase=lowercase,
            strip_accents=strip_accents,
            remove_punct=remove_punct,
            remove_spaces=False,
        )
        cand_tokens = tokenize(norm_cand)

        ibm = lexical_score_symmetric(
            norm_src_tokens,
            cand_tokens,
            fwd_table,
            bwd_table,
            smoothing=smoothing,
            length_penalty=length_penalty,
        )
        ibm_scores.append(ibm)

        lev = levenshtein_like_ratio(norm_src_text_for_lev, norm_cand)
        lev_scores.append(lev)

    neural_norm = minmax_normalize(neural_scores)
    ibm_norm = minmax_normalize(ibm_scores)
    lev_norm = minmax_normalize(lev_scores)

    final_scores = []
    for ns, iscore, lscore in zip(neural_norm, ibm_norm, lev_norm):
        final_scores.append(alpha * ns + beta * iscore + gamma * lscore)

    order = list(np.argsort(-np.asarray(final_scores, dtype=np.float64)))
    return order, final_scores, neural_norm, ibm_norm, lev_norm


def evaluate_reranking(
    neural_df: pd.DataFrame,
    test_raw_src: List[str],
    test_raw_tgt: List[str],
    fwd_table: Dict[str, Dict[str, float]],
    bwd_table: Dict[str, Dict[str, float]],
    lowercase: bool,
    strip_accents: bool,
    remove_punct: bool,
    smoothing: float,
    length_penalty: float,
    alpha: float,
    beta: float,
    gamma: float,
    output_predictions_path: str = None,
) -> Dict[str, float]:
    ranks = []
    rows = []

    if len(neural_df) != len(test_raw_src):
        raise ValueError(
            f"Neural predictions rows ({len(neural_df)}) != test rows ({len(test_raw_src)}). "
            "Make sure the neural prediction file was generated from the same split."
        )

    for i, row in neural_df.iterrows():
        raw_src = test_raw_src[i]
        gold = test_raw_tgt[i]

        norm_src = normalize_text(
            raw_src,
            lowercase=lowercase,
            strip_accents=strip_accents,
            remove_punct=remove_punct,
        )
        norm_src_tokens = tokenize(norm_src)

        candidate_texts = split_pipe_cell(row["TopK_Preds"])

        if not candidate_texts:
            ranks.append(np.inf)
            continue

        neural_scores = parse_topk_scores(row, len(candidate_texts))

        order, final_scores, neural_norm, ibm_norm, lev_norm = rerank_one_query(
            raw_src=raw_src,
            norm_src_tokens=norm_src_tokens,
            candidate_texts=candidate_texts,
            neural_scores=neural_scores,
            fwd_table=fwd_table,
            bwd_table=bwd_table,
            lowercase=lowercase,
            strip_accents=strip_accents,
            remove_punct=remove_punct,
            smoothing=smoothing,
            length_penalty=length_penalty,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        reranked_candidates = [candidate_texts[j] for j in order]
        reranked_scores = [final_scores[j] for j in order]

        gold_rank = np.inf
        for rank, cand in enumerate(reranked_candidates, start=1):
            if str(cand) == str(gold):
                gold_rank = float(rank)
                break

        ranks.append(gold_rank)

        rows.append(
            {
                "Bahnaric": raw_src,
                "Gold_VN": gold,
                "Predicted_VN": reranked_candidates[0] if reranked_candidates else "",
                "Gold_rank": None if not np.isfinite(gold_rank) else int(gold_rank),
                "TopK_Reranked": "|".join(reranked_candidates),
                "TopK_Reranked_Scores": "|".join([f"{x:.8f}" for x in reranked_scores]),
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            }
        )

    ranks_arr = np.asarray(ranks, dtype=np.float64)

    metrics = {}
    metrics["MRR"] = float(np.mean(np.where(np.isfinite(ranks_arr), 1.0 / ranks_arr, 0.0)))
    metrics["Top1_acc"] = float(np.mean(ranks_arr == 1.0))
    metrics["Hit@1"] = metrics["Top1_acc"]
    metrics["Recall@1"] = metrics["Top1_acc"]
    metrics["Precision@1"] = metrics["Top1_acc"]

    for k in [5, 10]:
        hit = float(np.mean(ranks_arr <= float(k)))
        metrics[f"Hit@{k}"] = hit
        metrics[f"Recall@{k}"] = hit
        metrics[f"Precision@{k}"] = hit / float(k)

    metrics["alpha"] = float(alpha)
    metrics["beta"] = float(beta)
    metrics["gamma"] = float(gamma)

    if output_predictions_path is not None:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(output_predictions_path, index=False)

    return metrics


def make_weight_grid(simple_alpha_grid: bool) -> List[Tuple[float, float, float]]:
    if simple_alpha_grid:
        grid = []
        for a in np.linspace(0.0, 1.0, 11):
            alpha = float(round(a, 2))
            beta = float(round(1.0 - alpha, 2))
            gamma = 0.0
            grid.append((alpha, beta, gamma))
        return grid

    alphas = [0.25, 0.5, 0.75, 1.0]
    betas = [0.25, 0.5, 0.75, 1.0]
    gammas = [0.0, 0.1, 0.25, 0.5]

    grid = []
    for a in alphas:
        for b in betas:
            for g in gammas:
                grid.append((a, b, g))

    return grid


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lowercase = not args.no_lowercase

    print("[1/5] Reading training data...")
    train_src, train_tgt, _, _, _ = read_parallel_csv(
        args.train_csv,
        lowercase=lowercase,
        strip_accents=args.strip_accents,
        remove_punct=args.remove_punct,
    )
    print(f"Training sentence pairs: {len(train_src)}")

    print("[2/5] Training forward IBM1 p(Vietnamese | Bahnaric)...")
    fwd_table = train_ibm1(
        train_src,
        train_tgt,
        num_iters=args.ibm_iters,
        min_prob=args.smoothing,
    )

    print("[3/5] Training backward IBM1 p(Bahnaric | Vietnamese)...")
    bwd_table = train_ibm1(
        train_tgt,
        train_src,
        num_iters=args.ibm_iters,
        min_prob=args.smoothing,
    )

    print("[4/5] Loading evaluation data and neural top-K predictions...")
    test_src_tok, test_tgt_tok, test_raw_src, test_raw_tgt, _ = read_parallel_csv(
        args.test_csv,
        lowercase=lowercase,
        strip_accents=args.strip_accents,
        remove_punct=args.remove_punct,
    )
    test_neural_df = load_neural_predictions(args.test_neural_predictions)

    print(f"Test queries: {len(test_raw_src)}")
    print(f"Loaded test neural predictions: {args.test_neural_predictions}")

    best_weights = (args.alpha, args.beta, args.gamma)
    dev_results = []

    if args.dev_csv and args.dev_neural_predictions:
        print("[5/5] Tuning weights on dev set...")
        _, _, dev_raw_src, dev_raw_tgt, _ = read_parallel_csv(
            args.dev_csv,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        dev_neural_df = load_neural_predictions(args.dev_neural_predictions)

        weight_grid = make_weight_grid(simple_alpha_grid=args.simple_alpha_grid)

        best_dev_mrr = -1.0
        best_dev_metrics = None

        for alpha, beta, gamma in weight_grid:
            metrics = evaluate_reranking(
                neural_df=dev_neural_df,
                test_raw_src=dev_raw_src,
                test_raw_tgt=dev_raw_tgt,
                fwd_table=fwd_table,
                bwd_table=bwd_table,
                lowercase=lowercase,
                strip_accents=args.strip_accents,
                remove_punct=args.remove_punct,
                smoothing=args.smoothing,
                length_penalty=args.length_penalty,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )
            dev_results.append(metrics)

            if metrics["MRR"] > best_dev_mrr:
                best_dev_mrr = metrics["MRR"]
                best_weights = (alpha, beta, gamma)
                best_dev_metrics = metrics

        pd.DataFrame(dev_results).sort_values(
            ["MRR", "Top1_acc", "Hit@10"],
            ascending=False,
        ).to_csv(output_dir / "dev_weight_grid.csv", index=False)

        print("Best dev metrics:")
        print(json.dumps({k: round(float(v), 4) for k, v in best_dev_metrics.items()}, indent=2))
        print(f"Selected weights: alpha={best_weights[0]}, beta={best_weights[1]}, gamma={best_weights[2]}")
    else:
        print("[5/5] No dev split provided. Using fixed weights from command line.")
        print(
            "For paper results, prefer dev tuning with "
            "--dev_csv and --dev_neural_predictions."
        )

    alpha, beta, gamma = best_weights

    print("Evaluating on test...")
    test_metrics = evaluate_reranking(
        neural_df=test_neural_df,
        test_raw_src=test_raw_src,
        test_raw_tgt=test_raw_tgt,
        fwd_table=fwd_table,
        bwd_table=bwd_table,
        lowercase=lowercase,
        strip_accents=args.strip_accents,
        remove_punct=args.remove_punct,
        smoothing=args.smoothing,
        length_penalty=args.length_penalty,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        output_predictions_path=str(output_dir / "sentence_predictions.csv"),
    )

    rounded = {k: round(float(v), 4) for k, v in test_metrics.items()}
    rounded.update(
        {
            "method": "hybrid_ibm1_neural_rerank",
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "test_neural_predictions": args.test_neural_predictions,
            "dev_csv": args.dev_csv,
            "dev_neural_predictions": args.dev_neural_predictions,
            "ibm_iters": int(args.ibm_iters),
            "smoothing": float(args.smoothing),
            "length_penalty": float(args.length_penalty),
            "strip_accents": bool(args.strip_accents),
            "lowercase": lowercase,
            "remove_punct": bool(args.remove_punct),
            "simple_alpha_grid": bool(args.simple_alpha_grid),
        }
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rounded, f, ensure_ascii=False, indent=2)

    print(json.dumps(rounded, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")

    if dev_results:
        print(f"Saved dev grid to {output_dir / 'dev_weight_grid.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid IBM1 + neural top-K reranking for Bahnaric-Vietnamese retrieval"
    )

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--test_neural_predictions", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--dev_csv", default=None)
    parser.add_argument("--dev_neural_predictions", default=None)

    parser.add_argument("--ibm_iters", type=int, default=5)
    parser.add_argument("--smoothing", type=float, default=1e-9)
    parser.add_argument("--length_penalty", type=float, default=0.1)

    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.0)

    parser.add_argument(
        "--simple_alpha_grid",
        action="store_true",
        help="Tune score = alpha * neural + (1-alpha) * IBM1, gamma=0.",
    )

    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    parser.add_argument("--no_lowercase", action="store_true")

    args = parser.parse_args()
    main(args)
