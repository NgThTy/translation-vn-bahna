"""
fastText + Procrustes / VecMap-style baseline for Bahnaric -> Vietnamese sentence retrieval.

Task:
Given a Bahnaric query sentence, retrieve the correct Vietnamese sentence from a candidate pool.
For data/test.csv, the gold Vietnamese sentence is assumed to be on the same row as the Bahnaric query.

Pipeline:
1. Train separate fastText embeddings:
   - Bahnaric fastText model from Bahnaric-side training text
   - Vietnamese fastText model from Vietnamese-side training text

2. Learn supervised Procrustes mapping:
   - Use lexicon_train.csv as seed dictionary
   - Compute Bahnaric word/phrase embeddings and Vietnamese word/phrase embeddings
   - Learn orthogonal map R from Bahnaric space to Vietnamese space

3. Sentence retrieval:
   - Embed each Bahnaric test sentence by averaging fastText word vectors
   - Map Bahnaric sentence embedding into Vietnamese space
   - Embed all Vietnamese candidate sentences
   - Retrieve nearest Vietnamese candidate using cosine or CSLS

This is a non-transformer baseline.
It tests whether classic word embeddings + linear cross-lingual mapping are competitive.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from gensim.models import FastText
except Exception as e:
    raise ImportError(
        "Could not import gensim FastText. Install dependencies with:\n"
        "python -m pip install 'gensim==4.3.3' 'scipy==1.12.0'"
    ) from e


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


def read_parallel_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)
    if "Bahnaric" not in df.columns or "Vietnamese" not in df.columns:
        raise ValueError("CSV must contain columns: Bahnaric,Vietnamese")
    return df


def build_training_sentences(
    csv_paths: List[str],
    side: str,
    lowercase: bool,
    strip_accents: bool,
    remove_punct: bool,
) -> List[List[str]]:
    """
    side must be either 'Bahnaric' or 'Vietnamese'.
    """
    all_sents: List[List[str]] = []

    for path in csv_paths:
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue

        df = read_parallel_csv(str(p))
        for text in df[side].astype(str).tolist():
            norm = normalize_text(
                text,
                lowercase=lowercase,
                strip_accents=strip_accents,
                remove_punct=remove_punct,
            )
            toks = tokenize(norm)
            if toks:
                all_sents.append(toks)

    if not all_sents:
        raise ValueError(f"No training sentences found for side={side} from csv_paths={csv_paths}")

    return all_sents


def train_fasttext(
    sentences: List[List[str]],
    vector_size: int,
    window: int,
    min_count: int,
    epochs: int,
    sg: int,
    workers: int,
    seed: int,
) -> FastText:
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=workers,
        seed=seed,
        min_n=3,
        max_n=6,
        bucket=200000,
    )
    model.build_vocab(corpus_iterable=sentences)
    model.train(
        corpus_iterable=sentences,
        total_examples=len(sentences),
        epochs=epochs,
    )
    return model


def phrase_embedding(model: FastText, text: str) -> np.ndarray:
    toks = tokenize(text)
    if not toks:
        return np.zeros(model.vector_size, dtype=np.float32)

    vecs = []
    for tok in toks:
        # Gensim FastText can infer vectors for OOV words through subwords.
        vecs.append(model.wv[tok])

    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)

    return np.mean(np.vstack(vecs), axis=0).astype(np.float32)


def sentence_embeddings(model: FastText, texts: List[str]) -> np.ndarray:
    embs = [phrase_embedding(model, text) for text in texts]
    return np.vstack(embs).astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def mean_center(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    return x - mu, mu


def kabsch_align(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Orthogonal Procrustes / Kabsch.
    Finds R such that x @ R is close to y.
    """
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same shape, got {x.shape} and {y.shape}")

    m = x.T @ y
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    return r.astype(np.float32)


def cosine_topk(query: np.ndarray, index: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    q = normalize_rows(query)
    z = normalize_rows(index)
    sims = q @ z.T
    k = int(max(1, min(topk, index.shape[0])))
    topk_idx = np.argsort(-sims, axis=1)[:, :k]
    return topk_idx, sims


def csls_topk(query: np.ndarray, index: np.ndarray, topk: int, csls_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    q = normalize_rows(query)
    z = normalize_rows(index)
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

    lowercase = not args.no_lowercase

    train_paths = [args.train_csv]
    if args.extra_train_csv:
        train_paths.extend(args.extra_train_csv)

    print("[1/5] Building monolingual training corpora...")
    src_sents = build_training_sentences(
        train_paths,
        side="Bahnaric",
        lowercase=lowercase,
        strip_accents=args.strip_accents,
        remove_punct=args.remove_punct,
    )
    tgt_sents = build_training_sentences(
        train_paths,
        side="Vietnamese",
        lowercase=lowercase,
        strip_accents=args.strip_accents,
        remove_punct=args.remove_punct,
    )

    print(f"Bahnaric training sentences: {len(src_sents)}")
    print(f"Vietnamese training sentences: {len(tgt_sents)}")

    print("[2/5] Training Bahnaric fastText model...")
    src_model = train_fasttext(
        src_sents,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        sg=args.sg,
        workers=args.workers,
        seed=args.seed,
    )

    print("[3/5] Training Vietnamese fastText model...")
    tgt_model = train_fasttext(
        tgt_sents,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        sg=args.sg,
        workers=args.workers,
        seed=args.seed,
    )

    print("[4/5] Learning supervised Procrustes mapping from lexicon_train.csv...")
    lex_df = read_parallel_csv(args.lexicon_train_csv)

    src_lex = [
        normalize_text(
            x,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for x in lex_df["Bahnaric"].astype(str).tolist()
    ]
    tgt_lex = [
        normalize_text(
            x,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for x in lex_df["Vietnamese"].astype(str).tolist()
    ]

    x = sentence_embeddings(src_model, src_lex)
    y = sentence_embeddings(tgt_model, tgt_lex)

    # VecMap-style preprocessing: length normalize + mean center + length normalize.
    if args.vecmap_normalize:
        x = normalize_rows(x)
        y = normalize_rows(y)
        x, src_mu = mean_center(x)
        y, tgt_mu = mean_center(y)
        x = normalize_rows(x)
        y = normalize_rows(y)
    else:
        src_mu = np.zeros((1, args.vector_size), dtype=np.float32)
        tgt_mu = np.zeros((1, args.vector_size), dtype=np.float32)

    r = kabsch_align(x, y)

    np.save(output_dir / "R.npy", r)
    np.save(output_dir / "src_mu.npy", src_mu)
    np.save(output_dir / "tgt_mu.npy", tgt_mu)

    print(f"Seed dictionary pairs used: {len(lex_df)}")

    print("[5/5] Evaluating sentence retrieval on test.csv...")
    test_df = read_parallel_csv(args.test_csv)

    raw_bah = test_df["Bahnaric"].astype(str).tolist()
    raw_vn = test_df["Vietnamese"].astype(str).tolist()

    norm_bah = [
        normalize_text(
            x,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for x in raw_bah
    ]
    norm_vn = [
        normalize_text(
            x,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for x in raw_vn
    ]

    src_test_emb = sentence_embeddings(src_model, norm_bah)
    tgt_test_emb = sentence_embeddings(tgt_model, norm_vn)

    if args.vecmap_normalize:
        src_test_emb = normalize_rows(src_test_emb)
        tgt_test_emb = normalize_rows(tgt_test_emb)
        src_test_emb = normalize_rows(src_test_emb - src_mu)
        tgt_test_emb = normalize_rows(tgt_test_emb - tgt_mu)

    src_mapped = src_test_emb @ r

    if args.use_csls:
        topk_idx, scores = csls_topk(src_mapped, tgt_test_emb, topk=args.topk_eval, csls_k=args.csls_k)
        retrieval = "csls"
    else:
        topk_idx, scores = cosine_topk(src_mapped, tgt_test_emb, topk=args.topk_eval)
        retrieval = "cosine"

    gold_idx = np.arange(len(test_df), dtype=np.int64)
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
        "Bahnaric_normalized": norm_bah,
        "Predicted_VN": preds,
        "Gold_VN": raw_vn,
        "Gold_VN_normalized": norm_vn,
        "Gold_rank": gold_rank,
        "TopK_Preds": topk_preds,
        "P@1": is_top1.tolist(),
        "MRR": rr.tolist(),
        "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(test_df))],
        "Gold_score": [float(scores[i, i]) for i in range(len(test_df))],
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
            "method": "fasttext_procrustes",
            "retrieval": retrieval,
            "test_csv": args.test_csv,
            "train_csv": args.train_csv,
            "lexicon_train_csv": args.lexicon_train_csv,
            "num_queries": int(len(test_df)),
            "candidate_pool_size": int(len(test_df)),
            "seed_dictionary_size": int(len(lex_df)),
            "vector_size": int(args.vector_size),
            "window": int(args.window),
            "min_count": int(args.min_count),
            "epochs": int(args.epochs),
            "sg": int(args.sg),
            "vecmap_normalize": bool(args.vecmap_normalize),
            "strip_accents": bool(args.strip_accents),
            "lowercase": lowercase,
            "remove_punct": bool(args.remove_punct),
            "use_csls": bool(args.use_csls),
            "csls_k": int(args.csls_k),
        }
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rounded_metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Saved bucket metrics to {output_dir / 'bucket_metrics_vn_len.csv'}")
    print(f"Saved mapping to {output_dir / 'R.npy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="fastText + supervised Procrustes / VecMap-style baseline for Bahnaric-Vietnamese sentence retrieval"
    )

    parser.add_argument("--train_csv", required=True, help="CSV with columns Bahnaric,Vietnamese for monolingual fastText training")
    parser.add_argument("--extra_train_csv", nargs="*", default=None, help="Optional extra CSV files for monolingual training")
    parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese for retrieval evaluation")
    parser.add_argument("--lexicon_train_csv", required=True, help="Seed dictionary CSV with columns Bahnaric,Vietnamese")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--vector_size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sg", type=int, default=1, help="1 = skip-gram, 0 = CBOW")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

    parser.add_argument("--vecmap_normalize", action="store_true", help="Apply VecMap-style normalize-center-normalize preprocessing")
    parser.add_argument("--use_csls", action="store_true", help="Use CSLS retrieval instead of cosine")
    parser.add_argument("--csls_k", type=int, default=10)

    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")

    args = parser.parse_args()
    main(args)
