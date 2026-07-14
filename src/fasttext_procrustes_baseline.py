# """
# Improved fastText + supervised Procrustes / VecMap-style baseline
# for Bahnaric -> Vietnamese sentence retrieval.

# Main improvements:
# 1. Supports mean and IDF-weighted aggregation.
# 2. Supports phrase-level or token-level Procrustes supervision.
# 3. Supports applying Procrustes before pooling or after pooling.
# 4. Keeps VecMap-style normalization and CSLS retrieval.
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

# try:
#     from gensim.models import FastText
# except Exception as e:
#     raise ImportError(
#         "Could not import gensim FastText. Install dependencies with:\n"
#         "python -m pip install 'gensim==4.3.3' 'scipy==1.12.0'"
#     ) from e


# def normalize_text(text: str, lowercase: bool = True, strip_accents: bool = False, remove_punct: bool = False) -> str:
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


# def tokenize(text: str) -> List[str]:
#     text = str(text).strip()
#     if not text:
#         return []
#     return text.split()


# def read_parallel_csv(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)
#     if "Bahnaric" not in df.columns or "Vietnamese" not in df.columns:
#         raise ValueError("CSV must contain columns: Bahnaric,Vietnamese")
#     return df


# def build_training_sentences(csv_paths: List[str], side: str, lowercase: bool, strip_accents: bool, remove_punct: bool) -> List[List[str]]:
#     all_sents = []

#     for path in csv_paths:
#         if not path:
#             continue
#         p = Path(path)
#         if not p.exists():
#             continue

#         df = read_parallel_csv(str(p))
#         for text in df[side].astype(str).tolist():
#             norm = normalize_text(
#                 text,
#                 lowercase=lowercase,
#                 strip_accents=strip_accents,
#                 remove_punct=remove_punct,
#             )
#             toks = tokenize(norm)
#             if toks:
#                 all_sents.append(toks)

#     if not all_sents:
#         raise ValueError(f"No training sentences found for side={side} from csv_paths={csv_paths}")

#     return all_sents


# def build_idf(sentences: List[List[str]]) -> Dict[str, float]:
#     df_counter = Counter()
#     n_docs = len(sentences)

#     for sent in sentences:
#         for tok in set(sent):
#             df_counter[tok] += 1

#     idf = {}
#     for tok, df in df_counter.items():
#         idf[tok] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0

#     return idf


# def train_fasttext(sentences: List[List[str]], vector_size: int, window: int, min_count: int, epochs: int, sg: int, workers: int, seed: int) -> FastText:
#     model = FastText(
#         vector_size=vector_size,
#         window=window,
#         min_count=min_count,
#         sg=sg,
#         workers=workers,
#         seed=seed,
#         min_n=3,
#         max_n=6,
#         bucket=200000,
#     )
#     model.build_vocab(corpus_iterable=sentences)
#     model.train(
#         corpus_iterable=sentences,
#         total_examples=len(sentences),
#         epochs=epochs,
#     )
#     return model


# def token_matrix(model: FastText, text: str) -> Tuple[List[str], np.ndarray]:
#     toks = tokenize(text)
#     if not toks:
#         return [], np.zeros((0, model.vector_size), dtype=np.float32)

#     vecs = [model.wv[tok] for tok in toks]
#     return toks, np.vstack(vecs).astype(np.float32)


# def aggregate_matrix(toks: List[str], mat: np.ndarray, pooling: str, idf: Dict[str, float] = None) -> np.ndarray:
#     if mat.shape[0] == 0:
#         return np.zeros(mat.shape[1], dtype=np.float32)

#     if pooling == "mean":
#         return mat.mean(axis=0).astype(np.float32)

#     if pooling == "idf":
#         weights = np.asarray([float(idf.get(tok, 1.0)) if idf else 1.0 for tok in toks], dtype=np.float32)
#         denom = float(weights.sum())
#         if denom <= 0.0:
#             return mat.mean(axis=0).astype(np.float32)
#         return ((mat * weights[:, None]).sum(axis=0) / denom).astype(np.float32)

#     raise ValueError(f"Unknown pooling: {pooling}")


# def embed_text(
#     model: FastText,
#     text: str,
#     pooling: str,
#     idf: Dict[str, float] = None,
#     r: np.ndarray = None,
#     map_before_pool: bool = False,
#     mu: np.ndarray = None,
#     vecmap_normalize: bool = False,
# ) -> np.ndarray:
#     toks, mat = token_matrix(model, text)

#     if mat.shape[0] == 0:
#         return np.zeros(model.vector_size, dtype=np.float32)

#     if map_before_pool and r is not None:
#         if vecmap_normalize:
#             mat = normalize_rows(mat)
#             if mu is not None:
#                 mat = normalize_rows(mat - mu)
#         mat = mat @ r

#     emb = aggregate_matrix(toks, mat, pooling=pooling, idf=idf)

#     return emb.astype(np.float32)


# def embed_texts(
#     model: FastText,
#     texts: List[str],
#     pooling: str,
#     idf: Dict[str, float] = None,
#     r: np.ndarray = None,
#     map_before_pool: bool = False,
#     mu: np.ndarray = None,
#     vecmap_normalize: bool = False,
# ) -> np.ndarray:
#     embs = [
#         embed_text(
#             model,
#             text,
#             pooling=pooling,
#             idf=idf,
#             r=r,
#             map_before_pool=map_before_pool,
#             mu=mu,
#             vecmap_normalize=vecmap_normalize,
#         )
#         for text in texts
#     ]
#     return np.vstack(embs).astype(np.float32)


# def make_alignment_pairs(
#     lex_df: pd.DataFrame,
#     lowercase: bool,
#     strip_accents: bool,
#     remove_punct: bool,
#     align_unit: str,
# ) -> Tuple[List[str], List[str]]:
#     src_items = []
#     tgt_items = []

#     for src_raw, tgt_raw in zip(lex_df["Bahnaric"].astype(str), lex_df["Vietnamese"].astype(str)):
#         src = normalize_text(src_raw, lowercase=lowercase, strip_accents=strip_accents, remove_punct=remove_punct)
#         tgt = normalize_text(tgt_raw, lowercase=lowercase, strip_accents=strip_accents, remove_punct=remove_punct)

#         src_toks = tokenize(src)
#         tgt_toks = tokenize(tgt)

#         if not src_toks or not tgt_toks:
#             continue

#         if align_unit == "phrase":
#             src_items.append(src)
#             tgt_items.append(tgt)
#         elif align_unit == "token":
#             if len(src_toks) == 1 and len(tgt_toks) == 1:
#                 src_items.append(src_toks[0])
#                 tgt_items.append(tgt_toks[0])
#         else:
#             raise ValueError(f"Unknown align_unit: {align_unit}")

#     if not src_items:
#         raise ValueError(f"No usable alignment pairs for align_unit={align_unit}")

#     return src_items, tgt_items


# def normalize_rows(x: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(x, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     return x / norms


# def mean_center(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     mu = x.mean(axis=0, keepdims=True)
#     return x - mu, mu


# def kabsch_align(x: np.ndarray, y: np.ndarray) -> np.ndarray:
#     if x.shape != y.shape:
#         raise ValueError(f"x and y must have same shape, got {x.shape} and {y.shape}")

#     m = x.T @ y
#     u, _, vt = np.linalg.svd(m)
#     r = u @ vt
#     return r.astype(np.float32)


# def cosine_topk(query: np.ndarray, index: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
#     q = normalize_rows(query)
#     z = normalize_rows(index)
#     sims = q @ z.T
#     k = int(max(1, min(topk, index.shape[0])))
#     topk_idx = np.argsort(-sims, axis=1)[:, :k]
#     return topk_idx, sims


# def csls_topk(query: np.ndarray, index: np.ndarray, topk: int, csls_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
#     q = normalize_rows(query)
#     z = normalize_rows(index)
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


# def ranking_metrics(topk_idx: np.ndarray, gold_idx: np.ndarray, eval_ks: List[int]) -> Tuple[Dict[str, float], np.ndarray]:
#     n_items = topk_idx.shape[0]
#     ranks = np.full(n_items, np.inf, dtype=np.float64)

#     for i in range(n_items):
#         hits = np.where(topk_idx[i] == gold_idx[i])[0]
#         if len(hits) > 0:
#             ranks[i] = float(hits[0] + 1)

#     metrics = {}
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

#     lowercase = not args.no_lowercase

#     train_paths = [args.train_csv]
#     if args.extra_train_csv:
#         train_paths.extend(args.extra_train_csv)

#     print("[1/5] Building monolingual training corpora...")
#     src_sents = build_training_sentences(train_paths, "Bahnaric", lowercase, args.strip_accents, args.remove_punct)
#     tgt_sents = build_training_sentences(train_paths, "Vietnamese", lowercase, args.strip_accents, args.remove_punct)

#     print(f"Bahnaric training sentences: {len(src_sents)}")
#     print(f"Vietnamese training sentences: {len(tgt_sents)}")

#     src_idf = build_idf(src_sents)
#     tgt_idf = build_idf(tgt_sents)

#     print("[2/5] Training Bahnaric fastText model...")
#     src_model = train_fasttext(src_sents, args.vector_size, args.window, args.min_count, args.epochs, args.sg, args.workers, args.seed)

#     print("[3/5] Training Vietnamese fastText model...")
#     tgt_model = train_fasttext(tgt_sents, args.vector_size, args.window, args.min_count, args.epochs, args.sg, args.workers, args.seed)

#     print("[4/5] Learning supervised Procrustes mapping from lexicon_train.csv...")
#     lex_df = read_parallel_csv(args.lexicon_train_csv)

#     src_lex, tgt_lex = make_alignment_pairs(
#         lex_df,
#         lowercase=lowercase,
#         strip_accents=args.strip_accents,
#         remove_punct=args.remove_punct,
#         align_unit=args.align_unit,
#     )

#     x = embed_texts(src_model, src_lex, pooling=args.pooling, idf=src_idf)
#     y = embed_texts(tgt_model, tgt_lex, pooling=args.pooling, idf=tgt_idf)

#     if args.vecmap_normalize:
#         x = normalize_rows(x)
#         y = normalize_rows(y)
#         x, src_mu = mean_center(x)
#         y, tgt_mu = mean_center(y)
#         x = normalize_rows(x)
#         y = normalize_rows(y)
#     else:
#         src_mu = np.zeros((1, args.vector_size), dtype=np.float32)
#         tgt_mu = np.zeros((1, args.vector_size), dtype=np.float32)

#     r = kabsch_align(x, y)

#     np.save(output_dir / "R.npy", r)
#     np.save(output_dir / "src_mu.npy", src_mu)
#     np.save(output_dir / "tgt_mu.npy", tgt_mu)

#     print(f"Lexicon rows read: {len(lex_df)}")
#     print(f"Alignment pairs used: {len(src_lex)}")
#     print(f"Alignment unit: {args.align_unit}")
#     print(f"Pooling: {args.pooling}")
#     print(f"Map before pool: {args.map_before_pool}")

#     print("[5/5] Evaluating sentence retrieval on test.csv...")
#     test_df = read_parallel_csv(args.test_csv)

#     raw_bah = test_df["Bahnaric"].astype(str).tolist()
#     raw_vn = test_df["Vietnamese"].astype(str).tolist()

#     norm_bah = [
#         normalize_text(x, lowercase=lowercase, strip_accents=args.strip_accents, remove_punct=args.remove_punct)
#         for x in raw_bah
#     ]
#     norm_vn = [
#         normalize_text(x, lowercase=lowercase, strip_accents=args.strip_accents, remove_punct=args.remove_punct)
#         for x in raw_vn
#     ]

#     if args.map_before_pool:
#         src_test_emb = embed_texts(
#             src_model,
#             norm_bah,
#             pooling=args.pooling,
#             idf=src_idf,
#             r=r,
#             map_before_pool=True,
#             mu=src_mu,
#             vecmap_normalize=args.vecmap_normalize,
#         )
#         tgt_test_emb = embed_texts(tgt_model, norm_vn, pooling=args.pooling, idf=tgt_idf)

#         if args.vecmap_normalize:
#             tgt_test_emb = normalize_rows(tgt_test_emb)
#             tgt_test_emb = normalize_rows(tgt_test_emb - tgt_mu)
#     else:
#         src_test_emb = embed_texts(src_model, norm_bah, pooling=args.pooling, idf=src_idf)
#         tgt_test_emb = embed_texts(tgt_model, norm_vn, pooling=args.pooling, idf=tgt_idf)

#         if args.vecmap_normalize:
#             src_test_emb = normalize_rows(src_test_emb)
#             tgt_test_emb = normalize_rows(tgt_test_emb)
#             src_test_emb = normalize_rows(src_test_emb - src_mu)
#             tgt_test_emb = normalize_rows(tgt_test_emb - tgt_mu)

#         src_test_emb = src_test_emb @ r

#     if args.use_csls:
#         topk_idx, scores = csls_topk(src_test_emb, tgt_test_emb, topk=args.topk_eval, csls_k=args.csls_k)
#         retrieval = "csls"
#     else:
#         topk_idx, scores = cosine_topk(src_test_emb, tgt_test_emb, topk=args.topk_eval)
#         retrieval = "cosine"

#     gold_idx = np.arange(len(test_df), dtype=np.int64)
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
#         "Bahnaric_normalized": norm_bah,
#         "Predicted_VN": preds,
#         "Gold_VN": raw_vn,
#         "Gold_VN_normalized": norm_vn,
#         "Gold_rank": gold_rank,
#         "TopK_Preds": topk_preds,
#         "P@1": is_top1.tolist(),
#         "MRR": rr.tolist(),
#         "Top1_score": [float(scores[i, pred_top1_idx[i]]) for i in range(len(test_df))],
#         "Gold_score": [float(scores[i, i]) for i in range(len(test_df))],
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
#             "method": "fasttext_procrustes",
#             "retrieval": retrieval,
#             "test_csv": args.test_csv,
#             "train_csv": args.train_csv,
#             "lexicon_train_csv": args.lexicon_train_csv,
#             "num_queries": int(len(test_df)),
#             "candidate_pool_size": int(len(test_df)),
#             "lexicon_rows": int(len(lex_df)),
#             "alignment_pairs_used": int(len(src_lex)),
#             "align_unit": args.align_unit,
#             "pooling": args.pooling,
#             "map_before_pool": bool(args.map_before_pool),
#             "vector_size": int(args.vector_size),
#             "window": int(args.window),
#             "min_count": int(args.min_count),
#             "epochs": int(args.epochs),
#             "sg": int(args.sg),
#             "vecmap_normalize": bool(args.vecmap_normalize),
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
#     print(f"Saved mapping to {output_dir / 'R.npy'}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Improved fastText + supervised Procrustes baseline for Bahnaric-Vietnamese retrieval"
#     )

#     parser.add_argument("--train_csv", required=True)
#     parser.add_argument("--extra_train_csv", nargs="*", default=None)
#     parser.add_argument("--test_csv", required=True)
#     parser.add_argument("--lexicon_train_csv", required=True)
#     parser.add_argument("--output_dir", required=True)

#     parser.add_argument("--vector_size", type=int, default=100)
#     parser.add_argument("--window", type=int, default=5)
#     parser.add_argument("--min_count", type=int, default=1)
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--sg", type=int, default=1)
#     parser.add_argument("--workers", type=int, default=2)
#     parser.add_argument("--seed", type=int, default=42)

#     parser.add_argument("--topk_eval", type=int, default=10)
#     parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])

#     parser.add_argument("--pooling", choices=["mean", "idf"], default="mean")
#     parser.add_argument("--align_unit", choices=["phrase", "token"], default="phrase")
#     parser.add_argument("--map_before_pool", action="store_true")

#     parser.add_argument("--vecmap_normalize", action="store_true")
#     parser.add_argument("--use_csls", action="store_true")
#     parser.add_argument("--csls_k", type=int, default=10)

#     parser.add_argument("--strip_accents", action="store_true")
#     parser.add_argument("--no_lowercase", action="store_true")
#     parser.add_argument("--remove_punct", action="store_true")

#     args = parser.parse_args()
#     main(args)

# #!/usr/bin/env python3
# """fastText + supervised Procrustes/VecMap retrieval baseline.

# All configuration, preprocessing, and hyperparameter selection must be carried
# out on ``data/dev.csv``. A held-out test run is accepted only when the supplied
# configuration exactly matches the development-selection manifest produced by
# ``src/select_dev_configs_and_evaluate_test.py``.

# The fitting policy is deliberately fixed and recorded in every metrics file:
# fastText models, IDF statistics, and the Procrustes mapping are fit only from
# ``data/train_fit.csv``. The development set is never added back for the final
# test run.
# """

# from __future__ import annotations

# import argparse
# import json
# import math
# import re
# import unicodedata
# from collections import Counter
# from pathlib import Path
# from typing import Any, Dict, Iterable, List, Optional, Tuple

# import numpy as np
# import pandas as pd

# try:
#     from gensim.models import FastText
# except Exception as exc:
#     raise ImportError(
#         "Could not import gensim FastText. Install dependencies with:\n"
#         "python -m pip install 'gensim==4.3.3' 'scipy==1.12.0'"
#     ) from exc

# FAMILY = "fasttext_procrustes"
# EVALUATOR_SCRIPT = "src/fasttext_procrustes_baseline.py"
# REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")
# REFIT_POLICY = "train_fit_only_no_dev_refit"


# def normalize_text(
#     text: str,
#     lowercase: bool = True,
#     strip_accents: bool = False,
#     remove_punct: bool = False,
# ) -> str:
#     text = unicodedata.normalize("NFC", str(text))
#     text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
#     text = text.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

#     if lowercase:
#         text = text.lower()

#     if strip_accents:
#         text = unicodedata.normalize("NFD", text)
#         text = "".join(char for char in text if unicodedata.category(char) != "Mn")
#         text = unicodedata.normalize("NFC", text)

#     if remove_punct:
#         text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

#     return re.sub(r"\s+", " ", text).strip()


# def tokenize(text: str) -> List[str]:
#     return str(text).strip().split() if str(text).strip() else []


# def read_parallel_csv(path: Path) -> pd.DataFrame:
#     if not path.is_file():
#         raise FileNotFoundError(f"CSV not found: {path}")

#     df = pd.read_csv(path)
#     missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
#     if missing:
#         raise ValueError(f"{path} is missing required columns: {missing}")

#     cleaned = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
#     if cleaned.empty:
#         raise ValueError(f"No valid bilingual pairs found in {path}")
#     return cleaned


# def build_training_sentences(
#     df: pd.DataFrame,
#     side: str,
#     lowercase: bool,
#     strip_accents: bool,
#     remove_punct: bool,
# ) -> List[List[str]]:
#     sentences: List[List[str]] = []
#     for text in df[side].astype(str):
#         normalized = normalize_text(
#             text,
#             lowercase=lowercase,
#             strip_accents=strip_accents,
#             remove_punct=remove_punct,
#         )
#         tokens = tokenize(normalized)
#         if tokens:
#             sentences.append(tokens)

#     if not sentences:
#         raise ValueError(f"No training sentences found for side={side!r}")
#     return sentences


# def build_idf(sentences: Iterable[List[str]]) -> Dict[str, float]:
#     sentence_list = list(sentences)
#     document_frequency: Counter[str] = Counter()
#     for sentence in sentence_list:
#         document_frequency.update(set(sentence))

#     n_documents = len(sentence_list)
#     return {
#         token: math.log((1.0 + n_documents) / (1.0 + frequency)) + 1.0
#         for token, frequency in document_frequency.items()
#     }


# def train_fasttext(
#     sentences: List[List[str]],
#     vector_size: int,
#     window: int,
#     min_count: int,
#     epochs: int,
#     sg: int,
#     workers: int,
#     seed: int,
# ) -> FastText:
#     model = FastText(
#         vector_size=vector_size,
#         window=window,
#         min_count=min_count,
#         sg=sg,
#         workers=workers,
#         seed=seed,
#         min_n=3,
#         max_n=6,
#         bucket=200000,
#     )
#     model.build_vocab(corpus_iterable=sentences)
#     model.train(
#         corpus_iterable=sentences,
#         total_examples=len(sentences),
#         epochs=epochs,
#     )
#     return model


# def token_matrix(model: FastText, text: str) -> Tuple[List[str], np.ndarray]:
#     tokens = tokenize(text)
#     if not tokens:
#         return [], np.zeros((0, model.vector_size), dtype=np.float32)
#     vectors = np.vstack([model.wv[token] for token in tokens]).astype(np.float32)
#     return tokens, vectors


# def aggregate_matrix(
#     tokens: List[str],
#     matrix: np.ndarray,
#     pooling: str,
#     idf: Optional[Dict[str, float]] = None,
# ) -> np.ndarray:
#     if matrix.shape[0] == 0:
#         return np.zeros(matrix.shape[1], dtype=np.float32)

#     if pooling == "mean":
#         return matrix.mean(axis=0).astype(np.float32)

#     if pooling == "idf":
#         weights = np.asarray(
#             [float(idf.get(token, 1.0)) if idf else 1.0 for token in tokens],
#             dtype=np.float32,
#         )
#         denominator = float(weights.sum())
#         if denominator <= 0.0:
#             return matrix.mean(axis=0).astype(np.float32)
#         return ((matrix * weights[:, None]).sum(axis=0) / denominator).astype(np.float32)

#     raise ValueError(f"Unknown pooling: {pooling}")


# def normalize_rows(matrix: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#     norms[norms == 0.0] = 1.0
#     return (matrix / norms).astype(np.float32)


# def mean_center(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     mean = matrix.mean(axis=0, keepdims=True).astype(np.float32)
#     return (matrix - mean).astype(np.float32), mean


# def embed_text(
#     model: FastText,
#     text: str,
#     pooling: str,
#     idf: Optional[Dict[str, float]] = None,
#     rotation: Optional[np.ndarray] = None,
#     map_before_pool: bool = False,
#     source_mean: Optional[np.ndarray] = None,
#     vecmap_normalize: bool = False,
# ) -> np.ndarray:
#     tokens, matrix = token_matrix(model, text)
#     if matrix.shape[0] == 0:
#         return np.zeros(model.vector_size, dtype=np.float32)

#     if map_before_pool and rotation is not None:
#         if vecmap_normalize:
#             matrix = normalize_rows(matrix)
#             if source_mean is not None:
#                 matrix = normalize_rows(matrix - source_mean)
#         matrix = matrix @ rotation

#     return aggregate_matrix(tokens, matrix, pooling=pooling, idf=idf)


# def embed_texts(
#     model: FastText,
#     texts: Iterable[str],
#     pooling: str,
#     idf: Optional[Dict[str, float]] = None,
#     rotation: Optional[np.ndarray] = None,
#     map_before_pool: bool = False,
#     source_mean: Optional[np.ndarray] = None,
#     vecmap_normalize: bool = False,
# ) -> np.ndarray:
#     embeddings = [
#         embed_text(
#             model=model,
#             text=text,
#             pooling=pooling,
#             idf=idf,
#             rotation=rotation,
#             map_before_pool=map_before_pool,
#             source_mean=source_mean,
#             vecmap_normalize=vecmap_normalize,
#         )
#         for text in texts
#     ]
#     return np.vstack(embeddings).astype(np.float32)


# def make_alignment_pairs(
#     alignment_df: pd.DataFrame,
#     lowercase: bool,
#     strip_accents: bool,
#     remove_punct: bool,
#     align_unit: str,
# ) -> Tuple[List[str], List[str]]:
#     source_items: List[str] = []
#     target_items: List[str] = []

#     for source_raw, target_raw in zip(
#         alignment_df["Bahnaric"].astype(str),
#         alignment_df["Vietnamese"].astype(str),
#     ):
#         source = normalize_text(
#             source_raw,
#             lowercase=lowercase,
#             strip_accents=strip_accents,
#             remove_punct=remove_punct,
#         )
#         target = normalize_text(
#             target_raw,
#             lowercase=lowercase,
#             strip_accents=strip_accents,
#             remove_punct=remove_punct,
#         )
#         source_tokens = tokenize(source)
#         target_tokens = tokenize(target)
#         if not source_tokens or not target_tokens:
#             continue

#         if align_unit == "phrase":
#             source_items.append(source)
#             target_items.append(target)
#         elif align_unit == "token":
#             if len(source_tokens) == 1 and len(target_tokens) == 1:
#                 source_items.append(source_tokens[0])
#                 target_items.append(target_tokens[0])
#         else:
#             raise ValueError(f"Unknown align_unit: {align_unit}")

#     if not source_items:
#         raise ValueError(
#             f"No usable alignment pairs for align_unit={align_unit!r} in the fitting data"
#         )
#     return source_items, target_items


# def kabsch_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
#     if source.shape != target.shape:
#         raise ValueError(
#             f"Source and target alignment matrices must have equal shapes, got "
#             f"{source.shape} and {target.shape}"
#         )
#     covariance = source.T @ target
#     left, _, right_transposed = np.linalg.svd(covariance)
#     return (left @ right_transposed).astype(np.float32)


# def cosine_topk(
#     query: np.ndarray,
#     index: np.ndarray,
#     topk: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     normalized_query = normalize_rows(query)
#     normalized_index = normalize_rows(index)
#     similarities = (normalized_query @ normalized_index.T).astype(np.float32)
#     k = int(max(1, min(topk, index.shape[0])))
#     topk_indices = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
#     return topk_indices.astype(np.int64), similarities


# def csls_topk(
#     query: np.ndarray,
#     index: np.ndarray,
#     topk: int,
#     csls_k: int,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     normalized_query = normalize_rows(query)
#     normalized_index = normalize_rows(index)
#     similarities = (normalized_query @ normalized_index.T).astype(np.float32)

#     if query.shape[0] <= 1 or index.shape[0] <= 1:
#         return cosine_topk(query, index, topk)

#     effective_csls_k = int(
#         max(1, min(csls_k, query.shape[0] - 1, index.shape[0] - 1))
#     )
#     query_neighborhood = np.partition(
#         similarities, -effective_csls_k, axis=1
#     )[:, -effective_csls_k:].mean(axis=1)
#     index_neighborhood = np.partition(
#         similarities, -effective_csls_k, axis=0
#     )[-effective_csls_k:, :].mean(axis=0)
#     csls_scores = (
#         2.0 * similarities
#         - query_neighborhood[:, None]
#         - index_neighborhood[None, :]
#     ).astype(np.float32)

#     k = int(max(1, min(topk, index.shape[0])))
#     topk_indices = np.argsort(-csls_scores, axis=1, kind="stable")[:, :k]
#     return topk_indices.astype(np.int64), csls_scores


# def ranking_metrics(
#     topk_indices: np.ndarray,
#     gold_indices: np.ndarray,
#     eval_ks: Iterable[int],
# ) -> Tuple[Dict[str, float], np.ndarray]:
#     ranks = np.full(topk_indices.shape[0], np.inf, dtype=np.float64)
#     for row in range(topk_indices.shape[0]):
#         hits = np.flatnonzero(topk_indices[row] == gold_indices[row])
#         if hits.size:
#             ranks[row] = float(hits[0] + 1)

#     metrics: Dict[str, float] = {
#         "MRR": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
#         "Top1_acc": float(np.mean(ranks == 1.0)),
#     }
#     for requested_k in eval_ks:
#         effective_k = int(max(1, min(requested_k, topk_indices.shape[1])))
#         hit = float(np.mean(ranks <= effective_k))
#         metrics[f"Hit@{effective_k}"] = hit
#         metrics[f"Recall@{effective_k}"] = hit
#         metrics[f"Precision@{effective_k}"] = hit / float(effective_k)
#     return metrics, ranks


# def make_bucket_columns(predictions: pd.DataFrame) -> pd.DataFrame:
#     predictions = predictions.copy()
#     predictions["Bahnaric_len_chars"] = predictions["Bahnaric"].astype(str).str.len()
#     predictions["Vietnamese_len_chars"] = predictions["Gold_VN"].astype(str).str.len()
#     predictions["Vietnamese_len_words"] = (
#         predictions["Gold_VN"].astype(str).str.split().map(len)
#     )

#     def length_bin(length: int) -> str:
#         if length <= 5:
#             return "1-5"
#         if length <= 15:
#             return "6-15"
#         if length <= 30:
#             return "16-30"
#         return ">30"

#     predictions["VN_len_bin"] = predictions["Vietnamese_len_words"].map(length_bin)
#     return predictions


# def save_bucket_metrics(
#     predictions: pd.DataFrame,
#     output_dir: Path,
#     eval_ks: Iterable[int],
# ) -> None:
#     metric_columns = ["P@1", "MRR"] + [
#         f"Hit@{k}" for k in eval_ks if f"Hit@{k}" in predictions.columns
#     ]
#     available_columns = [
#         column for column in metric_columns if column in predictions.columns
#     ]
#     grouped = predictions.groupby("VN_len_bin", dropna=False)
#     bucket_metrics = grouped[available_columns].mean().reset_index()
#     bucket_metrics["count"] = grouped.size().values
#     for column in available_columns:
#         bucket_metrics[column] = bucket_metrics[column].astype(float).round(6)
#     bucket_metrics.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


# def configuration_from_args(args: argparse.Namespace) -> Dict[str, Any]:
#     return {
#         "train_csv": str(Path(args.train_csv)),
#         "alignment_csv": str(Path(args.alignment_csv)),
#         "pooling": args.pooling,
#         "align_unit": args.align_unit,
#         "map_before_pool": bool(args.map_before_pool),
#         "vecmap_normalize": bool(args.vecmap_normalize),
#         "retrieval": "csls" if args.use_csls else "cosine",
#         "csls_k": int(args.csls_k),
#         "strip_accents": bool(args.strip_accents),
#         "lowercase": not bool(args.no_lowercase),
#         "remove_punct": bool(args.remove_punct),
#         "vector_size": int(args.vector_size),
#         "window": int(args.window),
#         "min_count": int(args.min_count),
#         "epochs": int(args.epochs),
#         "sg": int(args.sg),
#         "workers": int(args.workers),
#         "seed": int(args.seed),
#         "topk_eval": int(args.topk_eval),
#         "eval_ks": [int(value) for value in args.eval_ks],
#         "refit_policy": REFIT_POLICY,
#     }


# def evaluation_cli_args(configuration: Dict[str, Any]) -> List[str]:
#     cli = [
#         "--train_csv", str(configuration["train_csv"]),
#         "--alignment_csv", str(configuration["alignment_csv"]),
#         "--pooling", str(configuration["pooling"]),
#         "--align_unit", str(configuration["align_unit"]),
#         "--csls_k", str(configuration["csls_k"]),
#         "--vector_size", str(configuration["vector_size"]),
#         "--window", str(configuration["window"]),
#         "--min_count", str(configuration["min_count"]),
#         "--epochs", str(configuration["epochs"]),
#         "--sg", str(configuration["sg"]),
#         "--workers", str(configuration["workers"]),
#         "--seed", str(configuration["seed"]),
#         "--topk_eval", str(configuration["topk_eval"]),
#         "--eval_ks", *[str(value) for value in configuration["eval_ks"]],
#     ]
#     if configuration["map_before_pool"]:
#         cli.append("--map_before_pool")
#     if configuration["vecmap_normalize"]:
#         cli.append("--vecmap_normalize")
#     if configuration["retrieval"] == "csls":
#         cli.append("--use_csls")
#     if configuration["strip_accents"]:
#         cli.append("--strip_accents")
#     if not configuration["lowercase"]:
#         cli.append("--no_lowercase")
#     if configuration["remove_punct"]:
#         cli.append("--remove_punct")
#     return cli


# def authorize_test_run(
#     selection_manifest: Path,
#     configuration_name: str,
#     configuration: Dict[str, Any],
# ) -> None:
#     if not selection_manifest.is_file():
#         raise FileNotFoundError(
#             "A test run requires --selection_manifest produced by "
#             "src/select_dev_configs_and_evaluate_test.py"
#         )

#     manifest = json.loads(selection_manifest.read_text(encoding="utf-8"))
#     selected = manifest.get(FAMILY)
#     if not selected:
#         raise ValueError(
#             f"No development-selected configuration for family {FAMILY!r} "
#             f"in {selection_manifest}"
#         )
#     if selected.get("configuration_name") != configuration_name:
#         raise ValueError(
#             f"Test configuration {configuration_name!r} does not match the "
#             f"development-selected configuration "
#             f"{selected.get('configuration_name')!r}"
#         )
#     if selected.get("configuration") != configuration:
#         raise ValueError(
#             "Test configuration parameters do not match the development-selection manifest"
#         )


# def validate_arguments(args: argparse.Namespace) -> None:
#     if args.vector_size < 1:
#         raise ValueError("--vector_size must be positive")
#     if args.window < 1:
#         raise ValueError("--window must be positive")
#     if args.min_count < 1:
#         raise ValueError("--min_count must be positive")
#     if args.epochs < 1:
#         raise ValueError("--epochs must be positive")
#     if args.workers < 1:
#         raise ValueError("--workers must be positive")
#     if args.topk_eval < 1:
#         raise ValueError("--topk_eval must be positive")
#     if args.csls_k < 1:
#         raise ValueError("--csls_k must be positive")
#     if args.sg not in (0, 1):
#         raise ValueError("--sg must be either 0 (CBOW) or 1 (skip-gram)")


# def main(args: argparse.Namespace) -> None:
#     validate_arguments(args)
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     configuration = configuration_from_args(args)
#     if args.split_name == "test":
#         if not args.selection_manifest:
#             raise ValueError("--selection_manifest is required for held-out test evaluation")
#         authorize_test_run(
#             Path(args.selection_manifest),
#             args.configuration_name,
#             configuration,
#         )

#     train_path = Path(args.train_csv)
#     alignment_path = Path(args.alignment_csv)
#     input_path = Path(args.input_csv)

#     # The reviewer-compliant workflow requires all fitting inputs to be the
#     # train_fit split. Requiring identical paths prevents accidental use of the
#     # original full training pool or a lexicon derived from development data.
#     if train_path.resolve() != alignment_path.resolve():
#         raise ValueError(
#             "For the reviewer-compliant FastText/Procrustes experiment, "
#             "--train_csv and --alignment_csv must point to the same train_fit.csv file"
#         )

#     lowercase = not args.no_lowercase
#     train_df = read_parallel_csv(train_path)
#     evaluation_df = read_parallel_csv(input_path)

#     print("[1/5] Building fitting corpora from train_fit only...")
#     source_sentences = build_training_sentences(
#         train_df,
#         "Bahnaric",
#         lowercase,
#         args.strip_accents,
#         args.remove_punct,
#     )
#     target_sentences = build_training_sentences(
#         train_df,
#         "Vietnamese",
#         lowercase,
#         args.strip_accents,
#         args.remove_punct,
#     )
#     source_idf = build_idf(source_sentences)
#     target_idf = build_idf(target_sentences)

#     print("[2/5] Training Bahnaric fastText model...")
#     source_model = train_fasttext(
#         source_sentences,
#         args.vector_size,
#         args.window,
#         args.min_count,
#         args.epochs,
#         args.sg,
#         args.workers,
#         args.seed,
#     )

#     print("[3/5] Training Vietnamese fastText model...")
#     target_model = train_fasttext(
#         target_sentences,
#         args.vector_size,
#         args.window,
#         args.min_count,
#         args.epochs,
#         args.sg,
#         args.workers,
#         args.seed,
#     )

#     print("[4/5] Learning Procrustes mapping from train_fit only...")
#     source_alignment_items, target_alignment_items = make_alignment_pairs(
#         train_df,
#         lowercase=lowercase,
#         strip_accents=args.strip_accents,
#         remove_punct=args.remove_punct,
#         align_unit=args.align_unit,
#     )
#     source_alignment = embed_texts(
#         source_model,
#         source_alignment_items,
#         pooling=args.pooling,
#         idf=source_idf,
#     )
#     target_alignment = embed_texts(
#         target_model,
#         target_alignment_items,
#         pooling=args.pooling,
#         idf=target_idf,
#     )

#     if args.vecmap_normalize:
#         source_alignment = normalize_rows(source_alignment)
#         target_alignment = normalize_rows(target_alignment)
#         source_alignment, source_mean = mean_center(source_alignment)
#         target_alignment, target_mean = mean_center(target_alignment)
#         source_alignment = normalize_rows(source_alignment)
#         target_alignment = normalize_rows(target_alignment)
#     else:
#         source_mean = np.zeros((1, args.vector_size), dtype=np.float32)
#         target_mean = np.zeros((1, args.vector_size), dtype=np.float32)

#     rotation = kabsch_align(source_alignment, target_alignment)
#     np.save(output_dir / "R.npy", rotation)
#     np.save(output_dir / "src_mu.npy", source_mean)
#     np.save(output_dir / "tgt_mu.npy", target_mean)

#     print(f"[5/5] Evaluating on {args.split_name}: {input_path}")
#     raw_source = evaluation_df["Bahnaric"].astype(str).tolist()
#     raw_target = evaluation_df["Vietnamese"].astype(str).tolist()
#     normalized_source = [
#         normalize_text(
#             text,
#             lowercase=lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for text in raw_source
#     ]
#     normalized_target = [
#         normalize_text(
#             text,
#             lowercase=lowercase,
#             strip_accents=args.strip_accents,
#             remove_punct=args.remove_punct,
#         )
#         for text in raw_target
#     ]

#     if args.map_before_pool:
#         source_embeddings = embed_texts(
#             source_model,
#             normalized_source,
#             pooling=args.pooling,
#             idf=source_idf,
#             rotation=rotation,
#             map_before_pool=True,
#             source_mean=source_mean,
#             vecmap_normalize=args.vecmap_normalize,
#         )
#         target_embeddings = embed_texts(
#             target_model,
#             normalized_target,
#             pooling=args.pooling,
#             idf=target_idf,
#         )
#         if args.vecmap_normalize:
#             target_embeddings = normalize_rows(target_embeddings)
#             target_embeddings = normalize_rows(target_embeddings - target_mean)
#     else:
#         source_embeddings = embed_texts(
#             source_model,
#             normalized_source,
#             pooling=args.pooling,
#             idf=source_idf,
#         )
#         target_embeddings = embed_texts(
#             target_model,
#             normalized_target,
#             pooling=args.pooling,
#             idf=target_idf,
#         )
#         if args.vecmap_normalize:
#             source_embeddings = normalize_rows(source_embeddings)
#             target_embeddings = normalize_rows(target_embeddings)
#             source_embeddings = normalize_rows(source_embeddings - source_mean)
#             target_embeddings = normalize_rows(target_embeddings - target_mean)
#         source_embeddings = (source_embeddings @ rotation).astype(np.float32)

#     if args.use_csls:
#         topk_indices, scores = csls_topk(
#             source_embeddings,
#             target_embeddings,
#             topk=args.topk_eval,
#             csls_k=args.csls_k,
#         )
#     else:
#         topk_indices, scores = cosine_topk(
#             source_embeddings,
#             target_embeddings,
#             topk=args.topk_eval,
#         )

#     gold_indices = np.arange(len(evaluation_df), dtype=np.int64)
#     eval_ks = [int(value) for value in args.eval_ks]
#     metrics, ranks = ranking_metrics(topk_indices, gold_indices, eval_ks)

#     top1_indices = topk_indices[:, 0]
#     predictions = [raw_target[index] for index in top1_indices]
#     topk_predictions = [
#         "|".join(raw_target[index] for index in row) for row in topk_indices
#     ]
#     reciprocal_ranks = np.where(
#         np.isfinite(ranks), 1.0 / ranks, 0.0
#     ).astype(np.float32)

#     prediction_data: Dict[str, Any] = {
#         "Bahnaric": raw_source,
#         "Bahnaric_normalized": normalized_source,
#         "Predicted_VN": predictions,
#         "Gold_VN": raw_target,
#         "Gold_VN_normalized": normalized_target,
#         "Gold_rank": [
#             None if not np.isfinite(rank) else int(rank) for rank in ranks
#         ],
#         "TopK_Preds": topk_predictions,
#         "P@1": (top1_indices == gold_indices).astype(np.float32).tolist(),
#         "MRR": reciprocal_ranks.tolist(),
#         "Top1_score": [
#             float(scores[row, top1_indices[row]]) for row in range(len(evaluation_df))
#         ],
#         "Gold_score": [
#             float(scores[row, row]) for row in range(len(evaluation_df))
#         ],
#     }
#     for requested_k in eval_ks:
#         effective_k = int(max(1, min(requested_k, topk_indices.shape[1])))
#         prediction_data[f"Hit@{effective_k}"] = (
#             ranks <= effective_k
#         ).astype(np.float32).tolist()

#     predictions_df = make_bucket_columns(pd.DataFrame(prediction_data))
#     predictions_df.to_csv(output_dir / "sentence_predictions.csv", index=False)
#     save_bucket_metrics(predictions_df, output_dir, eval_ks)

#     rounded_metrics: Dict[str, Any] = {
#         key: round(float(value), 6) for key, value in metrics.items()
#     }
#     rounded_metrics.update(
#         {
#             "schema_version": 2,
#             "family": FAMILY,
#             "configuration_name": args.configuration_name,
#             "split": args.split_name,
#             "input_csv": str(input_path),
#             "train_csv": str(train_path),
#             "alignment_csv": str(alignment_path),
#             "num_queries": int(len(evaluation_df)),
#             "candidate_pool_size": int(len(evaluation_df)),
#             "train_rows": int(len(train_df)),
#             "alignment_pairs_used": int(len(source_alignment_items)),
#             "selection_metric": "Top1_acc",
#             "refit_policy": REFIT_POLICY,
#             "configuration": configuration,
#             "evaluator_script": EVALUATOR_SCRIPT,
#             "evaluation_cli_args": evaluation_cli_args(configuration),
#         }
#     )
#     (output_dir / "metrics.json").write_text(
#         json.dumps(rounded_metrics, ensure_ascii=False, indent=2) + "\n",
#         encoding="utf-8",
#     )

#     print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
#     print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
#     print(f"Saved metrics to {output_dir / 'metrics.json'}")
#     print(f"Saved mapping to {output_dir / 'R.npy'}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description=(
#             "fastText + Procrustes retrieval with development-only model and "
#             "variant selection"
#         )
#     )
#     parser.add_argument(
#         "--train_csv",
#         required=True,
#         help="Fitting split; use data/train_fit.csv",
#     )
#     parser.add_argument(
#         "--alignment_csv",
#         required=True,
#         help="Alignment supervision; must be the same data/train_fit.csv file",
#     )
#     parser.add_argument(
#         "--input_csv",
#         required=True,
#         help="Development or held-out test CSV",
#     )
#     parser.add_argument("--split_name", choices=["dev", "test"], required=True)
#     parser.add_argument("--configuration_name", required=True)
#     parser.add_argument("--selection_manifest", default=None)
#     parser.add_argument("--output_dir", required=True)

#     parser.add_argument("--vector_size", type=int, default=100)
#     parser.add_argument("--window", type=int, default=5)
#     parser.add_argument("--min_count", type=int, default=1)
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--sg", type=int, choices=[0, 1], default=1)
#     parser.add_argument(
#         "--workers",
#         type=int,
#         default=1,
#         help="Use 1 for reproducible selection; larger values may be nondeterministic",
#     )
#     parser.add_argument("--seed", type=int, default=42)

#     parser.add_argument("--topk_eval", type=int, default=10)
#     parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
#     parser.add_argument("--pooling", choices=["mean", "idf"], default="mean")
#     parser.add_argument("--align_unit", choices=["phrase", "token"], default="phrase")
#     parser.add_argument("--map_before_pool", action="store_true")
#     parser.add_argument("--vecmap_normalize", action="store_true")
#     parser.add_argument("--use_csls", action="store_true")
#     parser.add_argument("--csls_k", type=int, default=10)
#     parser.add_argument("--strip_accents", action="store_true")
#     parser.add_argument("--no_lowercase", action="store_true")
#     parser.add_argument("--remove_punct", action="store_true")
#     main(parser.parse_args())












#!/usr/bin/env python3
"""fastText + supervised Procrustes/VecMap retrieval baseline.

All configuration, preprocessing, and hyperparameter selection must be carried
out on ``data/dev.csv``. A held-out test run is accepted only when the supplied
configuration exactly matches the development-selection manifest produced by
``src/select_dev_configs_and_evaluate_test.py``.

The fitting policy is deliberately fixed and recorded in every metrics file:
fastText models, IDF statistics, and the Procrustes mapping are fit only from
``data/train_fit.csv``. The development set is never added back for the final
test run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from retrieval_scoring import VALID_RETRIEVALS, retrieve_topk

try:
    from gensim.models import FastText
except Exception as exc:
    raise ImportError(
        "Could not import gensim FastText. Install dependencies with:\n"
        "python -m pip install 'gensim==4.3.3' 'scipy==1.12.0'"
    ) from exc

FAMILY = "fasttext_procrustes"
EVALUATOR_SCRIPT = "src/fasttext_procrustes_baseline.py"
REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")
REFIT_POLICY = "train_fit_only_no_dev_refit"


def resolve_retrieval(args: argparse.Namespace) -> str:
    """Resolve the new retrieval flag while retaining --use_csls compatibility."""
    if args.retrieval is not None:
        if args.use_csls and args.retrieval != "csls":
            raise ValueError("--use_csls conflicts with --retrieval")
        return str(args.retrieval)
    return "csls" if args.use_csls else "cosine"


def resolve_neighborhood_k(args: argparse.Namespace) -> int:
    """Resolve --neighborhood_k and the deprecated --csls_k alias."""
    if args.csls_k is not None:
        if args.neighborhood_k != 10 and args.neighborhood_k != args.csls_k:
            raise ValueError("--csls_k conflicts with --neighborhood_k")
        return int(args.csls_k)
    return int(args.neighborhood_k)


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_accents: bool = False,
    remove_punct: bool = False,
) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")

    if lowercase:
        text = text.lower()

    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")
        text = unicodedata.normalize("NFC", text)

    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return str(text).strip().split() if str(text).strip() else []


def read_parallel_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    cleaned = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(f"No valid bilingual pairs found in {path}")
    return cleaned


def build_training_sentences(
    df: pd.DataFrame,
    side: str,
    lowercase: bool,
    strip_accents: bool,
    remove_punct: bool,
) -> List[List[str]]:
    sentences: List[List[str]] = []
    for text in df[side].astype(str):
        normalized = normalize_text(
            text,
            lowercase=lowercase,
            strip_accents=strip_accents,
            remove_punct=remove_punct,
        )
        tokens = tokenize(normalized)
        if tokens:
            sentences.append(tokens)

    if not sentences:
        raise ValueError(f"No training sentences found for side={side!r}")
    return sentences


def build_idf(sentences: Iterable[List[str]]) -> Dict[str, float]:
    sentence_list = list(sentences)
    document_frequency: Counter[str] = Counter()
    for sentence in sentence_list:
        document_frequency.update(set(sentence))

    n_documents = len(sentence_list)
    return {
        token: math.log((1.0 + n_documents) / (1.0 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }


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


def token_matrix(model: FastText, text: str) -> Tuple[List[str], np.ndarray]:
    tokens = tokenize(text)
    if not tokens:
        return [], np.zeros((0, model.vector_size), dtype=np.float32)
    vectors = np.vstack([model.wv[token] for token in tokens]).astype(np.float32)
    return tokens, vectors


def aggregate_matrix(
    tokens: List[str],
    matrix: np.ndarray,
    pooling: str,
    idf: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)

    if pooling == "mean":
        return matrix.mean(axis=0).astype(np.float32)

    if pooling == "idf":
        weights = np.asarray(
            [float(idf.get(token, 1.0)) if idf else 1.0 for token in tokens],
            dtype=np.float32,
        )
        denominator = float(weights.sum())
        if denominator <= 0.0:
            return matrix.mean(axis=0).astype(np.float32)
        return ((matrix * weights[:, None]).sum(axis=0) / denominator).astype(np.float32)

    raise ValueError(f"Unknown pooling: {pooling}")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def mean_center(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0, keepdims=True).astype(np.float32)
    return (matrix - mean).astype(np.float32), mean


def embed_text(
    model: FastText,
    text: str,
    pooling: str,
    idf: Optional[Dict[str, float]] = None,
    rotation: Optional[np.ndarray] = None,
    map_before_pool: bool = False,
    source_mean: Optional[np.ndarray] = None,
    vecmap_normalize: bool = False,
) -> np.ndarray:
    tokens, matrix = token_matrix(model, text)
    if matrix.shape[0] == 0:
        return np.zeros(model.vector_size, dtype=np.float32)

    if map_before_pool and rotation is not None:
        if vecmap_normalize:
            matrix = normalize_rows(matrix)
            if source_mean is not None:
                matrix = normalize_rows(matrix - source_mean)
        matrix = matrix @ rotation

    return aggregate_matrix(tokens, matrix, pooling=pooling, idf=idf)


def embed_texts(
    model: FastText,
    texts: Iterable[str],
    pooling: str,
    idf: Optional[Dict[str, float]] = None,
    rotation: Optional[np.ndarray] = None,
    map_before_pool: bool = False,
    source_mean: Optional[np.ndarray] = None,
    vecmap_normalize: bool = False,
) -> np.ndarray:
    embeddings = [
        embed_text(
            model=model,
            text=text,
            pooling=pooling,
            idf=idf,
            rotation=rotation,
            map_before_pool=map_before_pool,
            source_mean=source_mean,
            vecmap_normalize=vecmap_normalize,
        )
        for text in texts
    ]
    return np.vstack(embeddings).astype(np.float32)


def make_alignment_pairs(
    alignment_df: pd.DataFrame,
    lowercase: bool,
    strip_accents: bool,
    remove_punct: bool,
    align_unit: str,
) -> Tuple[List[str], List[str]]:
    source_items: List[str] = []
    target_items: List[str] = []

    for source_raw, target_raw in zip(
        alignment_df["Bahnaric"].astype(str),
        alignment_df["Vietnamese"].astype(str),
    ):
        source = normalize_text(
            source_raw,
            lowercase=lowercase,
            strip_accents=strip_accents,
            remove_punct=remove_punct,
        )
        target = normalize_text(
            target_raw,
            lowercase=lowercase,
            strip_accents=strip_accents,
            remove_punct=remove_punct,
        )
        source_tokens = tokenize(source)
        target_tokens = tokenize(target)
        if not source_tokens or not target_tokens:
            continue

        if align_unit == "phrase":
            source_items.append(source)
            target_items.append(target)
        elif align_unit == "token":
            if len(source_tokens) == 1 and len(target_tokens) == 1:
                source_items.append(source_tokens[0])
                target_items.append(target_tokens[0])
        else:
            raise ValueError(f"Unknown align_unit: {align_unit}")

    if not source_items:
        raise ValueError(
            f"No usable alignment pairs for align_unit={align_unit!r} in the fitting data"
        )
    return source_items, target_items


def kabsch_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape:
        raise ValueError(
            f"Source and target alignment matrices must have equal shapes, got "
            f"{source.shape} and {target.shape}"
        )
    covariance = source.T @ target
    left, _, right_transposed = np.linalg.svd(covariance)
    return (left @ right_transposed).astype(np.float32)


def ranking_metrics(
    topk_indices: np.ndarray,
    gold_indices: np.ndarray,
    eval_ks: Iterable[int],
) -> Tuple[Dict[str, float], np.ndarray]:
    ranks = np.full(topk_indices.shape[0], np.inf, dtype=np.float64)
    for row in range(topk_indices.shape[0]):
        hits = np.flatnonzero(topk_indices[row] == gold_indices[row])
        if hits.size:
            ranks[row] = float(hits[0] + 1)

    metrics: Dict[str, float] = {
        "MRR": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
        "Top1_acc": float(np.mean(ranks == 1.0)),
    }
    for requested_k in eval_ks:
        effective_k = int(max(1, min(requested_k, topk_indices.shape[1])))
        hit = float(np.mean(ranks <= effective_k))
        metrics[f"Hit@{effective_k}"] = hit
        metrics[f"Recall@{effective_k}"] = hit
        metrics[f"Precision@{effective_k}"] = hit / float(effective_k)
    return metrics, ranks


def make_bucket_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    predictions = predictions.copy()
    predictions["Bahnaric_len_chars"] = predictions["Bahnaric"].astype(str).str.len()
    predictions["Vietnamese_len_chars"] = predictions["Gold_VN"].astype(str).str.len()
    predictions["Vietnamese_len_words"] = (
        predictions["Gold_VN"].astype(str).str.split().map(len)
    )

    def length_bin(length: int) -> str:
        if length <= 5:
            return "1-5"
        if length <= 15:
            return "6-15"
        if length <= 30:
            return "16-30"
        return ">30"

    predictions["VN_len_bin"] = predictions["Vietnamese_len_words"].map(length_bin)
    return predictions


def save_bucket_metrics(
    predictions: pd.DataFrame,
    output_dir: Path,
    eval_ks: Iterable[int],
) -> None:
    metric_columns = ["P@1", "MRR"] + [
        f"Hit@{k}" for k in eval_ks if f"Hit@{k}" in predictions.columns
    ]
    available_columns = [
        column for column in metric_columns if column in predictions.columns
    ]
    grouped = predictions.groupby("VN_len_bin", dropna=False)
    bucket_metrics = grouped[available_columns].mean().reset_index()
    bucket_metrics["count"] = grouped.size().values
    for column in available_columns:
        bucket_metrics[column] = bucket_metrics[column].astype(float).round(6)
    bucket_metrics.to_csv(output_dir / "bucket_metrics_vn_len.csv", index=False)


def configuration_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "train_csv": str(Path(args.train_csv)),
        "alignment_csv": str(Path(args.alignment_csv)),
        "pooling": args.pooling,
        "align_unit": args.align_unit,
        "map_before_pool": bool(args.map_before_pool),
        "vecmap_normalize": bool(args.vecmap_normalize),
        "retrieval": resolve_retrieval(args),
        "neighborhood_k": resolve_neighborhood_k(args),
        "strip_accents": bool(args.strip_accents),
        "lowercase": not bool(args.no_lowercase),
        "remove_punct": bool(args.remove_punct),
        "vector_size": int(args.vector_size),
        "window": int(args.window),
        "min_count": int(args.min_count),
        "epochs": int(args.epochs),
        "sg": int(args.sg),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "topk_eval": int(args.topk_eval),
        "eval_ks": [int(value) for value in args.eval_ks],
        "refit_policy": REFIT_POLICY,
    }


def evaluation_cli_args(configuration: Dict[str, Any]) -> List[str]:
    cli = [
        "--train_csv", str(configuration["train_csv"]),
        "--alignment_csv", str(configuration["alignment_csv"]),
        "--pooling", str(configuration["pooling"]),
        "--align_unit", str(configuration["align_unit"]),
        "--retrieval", str(configuration["retrieval"]),
        "--neighborhood_k", str(configuration["neighborhood_k"]),
        "--vector_size", str(configuration["vector_size"]),
        "--window", str(configuration["window"]),
        "--min_count", str(configuration["min_count"]),
        "--epochs", str(configuration["epochs"]),
        "--sg", str(configuration["sg"]),
        "--workers", str(configuration["workers"]),
        "--seed", str(configuration["seed"]),
        "--topk_eval", str(configuration["topk_eval"]),
        "--eval_ks", *[str(value) for value in configuration["eval_ks"]],
    ]
    if configuration["map_before_pool"]:
        cli.append("--map_before_pool")
    if configuration["vecmap_normalize"]:
        cli.append("--vecmap_normalize")
    if configuration["strip_accents"]:
        cli.append("--strip_accents")
    if not configuration["lowercase"]:
        cli.append("--no_lowercase")
    if configuration["remove_punct"]:
        cli.append("--remove_punct")
    return cli


def authorize_test_run(
    selection_manifest: Path,
    configuration_name: str,
    configuration: Dict[str, Any],
) -> None:
    if not selection_manifest.is_file():
        raise FileNotFoundError(
            "A test run requires --selection_manifest produced by "
            "src/select_dev_configs_and_evaluate_test.py"
        )

    manifest = json.loads(selection_manifest.read_text(encoding="utf-8"))
    selected = manifest.get(FAMILY)
    if not selected:
        raise ValueError(
            f"No development-selected configuration for family {FAMILY!r} "
            f"in {selection_manifest}"
        )
    if selected.get("configuration_name") != configuration_name:
        raise ValueError(
            f"Test configuration {configuration_name!r} does not match the "
            f"development-selected configuration "
            f"{selected.get('configuration_name')!r}"
        )
    if selected.get("configuration") != configuration:
        raise ValueError(
            "Test configuration parameters do not match the development-selection manifest"
        )


def validate_arguments(args: argparse.Namespace) -> None:
    if args.vector_size < 1:
        raise ValueError("--vector_size must be positive")
    if args.window < 1:
        raise ValueError("--window must be positive")
    if args.min_count < 1:
        raise ValueError("--min_count must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.topk_eval < 1:
        raise ValueError("--topk_eval must be positive")
    if resolve_neighborhood_k(args) < 1:
        raise ValueError("--neighborhood_k must be positive")
    if args.sg not in (0, 1):
        raise ValueError("--sg must be either 0 (CBOW) or 1 (skip-gram)")


def main(args: argparse.Namespace) -> None:
    validate_arguments(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configuration = configuration_from_args(args)
    if args.split_name == "test":
        if not args.selection_manifest:
            raise ValueError("--selection_manifest is required for held-out test evaluation")
        authorize_test_run(
            Path(args.selection_manifest),
            args.configuration_name,
            configuration,
        )

    train_path = Path(args.train_csv)
    alignment_path = Path(args.alignment_csv)
    input_path = Path(args.input_csv)

    # The reviewer-compliant workflow requires all fitting inputs to be the
    # train_fit split. Requiring identical paths prevents accidental use of the
    # original full training pool or a lexicon derived from development data.
    if train_path.resolve() != alignment_path.resolve():
        raise ValueError(
            "For the reviewer-compliant FastText/Procrustes experiment, "
            "--train_csv and --alignment_csv must point to the same train_fit.csv file"
        )

    lowercase = not args.no_lowercase
    train_df = read_parallel_csv(train_path)
    evaluation_df = read_parallel_csv(input_path)

    print("[1/5] Building fitting corpora from train_fit only...")
    source_sentences = build_training_sentences(
        train_df,
        "Bahnaric",
        lowercase,
        args.strip_accents,
        args.remove_punct,
    )
    target_sentences = build_training_sentences(
        train_df,
        "Vietnamese",
        lowercase,
        args.strip_accents,
        args.remove_punct,
    )
    source_idf = build_idf(source_sentences)
    target_idf = build_idf(target_sentences)

    print("[2/5] Training Bahnaric fastText model...")
    source_model = train_fasttext(
        source_sentences,
        args.vector_size,
        args.window,
        args.min_count,
        args.epochs,
        args.sg,
        args.workers,
        args.seed,
    )

    print("[3/5] Training Vietnamese fastText model...")
    target_model = train_fasttext(
        target_sentences,
        args.vector_size,
        args.window,
        args.min_count,
        args.epochs,
        args.sg,
        args.workers,
        args.seed,
    )

    print("[4/5] Learning Procrustes mapping from train_fit only...")
    source_alignment_items, target_alignment_items = make_alignment_pairs(
        train_df,
        lowercase=lowercase,
        strip_accents=args.strip_accents,
        remove_punct=args.remove_punct,
        align_unit=args.align_unit,
    )
    source_alignment = embed_texts(
        source_model,
        source_alignment_items,
        pooling=args.pooling,
        idf=source_idf,
    )
    target_alignment = embed_texts(
        target_model,
        target_alignment_items,
        pooling=args.pooling,
        idf=target_idf,
    )

    if args.vecmap_normalize:
        source_alignment = normalize_rows(source_alignment)
        target_alignment = normalize_rows(target_alignment)
        source_alignment, source_mean = mean_center(source_alignment)
        target_alignment, target_mean = mean_center(target_alignment)
        source_alignment = normalize_rows(source_alignment)
        target_alignment = normalize_rows(target_alignment)
    else:
        source_mean = np.zeros((1, args.vector_size), dtype=np.float32)
        target_mean = np.zeros((1, args.vector_size), dtype=np.float32)

    rotation = kabsch_align(source_alignment, target_alignment)
    np.save(output_dir / "R.npy", rotation)
    np.save(output_dir / "src_mu.npy", source_mean)
    np.save(output_dir / "tgt_mu.npy", target_mean)

    print(f"[5/5] Evaluating on {args.split_name}: {input_path}")
    raw_source = evaluation_df["Bahnaric"].astype(str).tolist()
    raw_target = evaluation_df["Vietnamese"].astype(str).tolist()
    normalized_source = [
        normalize_text(
            text,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for text in raw_source
    ]
    normalized_target = [
        normalize_text(
            text,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for text in raw_target
    ]

    if args.map_before_pool:
        source_embeddings = embed_texts(
            source_model,
            normalized_source,
            pooling=args.pooling,
            idf=source_idf,
            rotation=rotation,
            map_before_pool=True,
            source_mean=source_mean,
            vecmap_normalize=args.vecmap_normalize,
        )
        target_embeddings = embed_texts(
            target_model,
            normalized_target,
            pooling=args.pooling,
            idf=target_idf,
        )
        if args.vecmap_normalize:
            target_embeddings = normalize_rows(target_embeddings)
            target_embeddings = normalize_rows(target_embeddings - target_mean)
    else:
        source_embeddings = embed_texts(
            source_model,
            normalized_source,
            pooling=args.pooling,
            idf=source_idf,
        )
        target_embeddings = embed_texts(
            target_model,
            normalized_target,
            pooling=args.pooling,
            idf=target_idf,
        )
        if args.vecmap_normalize:
            source_embeddings = normalize_rows(source_embeddings)
            target_embeddings = normalize_rows(target_embeddings)
            source_embeddings = normalize_rows(source_embeddings - source_mean)
            target_embeddings = normalize_rows(target_embeddings - target_mean)
        source_embeddings = (source_embeddings @ rotation).astype(np.float32)

    retrieval = resolve_retrieval(args)
    neighborhood_k = resolve_neighborhood_k(args)
    topk_indices, scores = retrieve_topk(
        source_embeddings,
        target_embeddings,
        retrieval=retrieval,
        topk=args.topk_eval,
        neighborhood_k=neighborhood_k,
    )

    gold_indices = np.arange(len(evaluation_df), dtype=np.int64)
    eval_ks = [int(value) for value in args.eval_ks]
    metrics, ranks = ranking_metrics(topk_indices, gold_indices, eval_ks)

    top1_indices = topk_indices[:, 0]
    predictions = [raw_target[index] for index in top1_indices]
    topk_predictions = [
        "|".join(raw_target[index] for index in row) for row in topk_indices
    ]
    reciprocal_ranks = np.where(
        np.isfinite(ranks), 1.0 / ranks, 0.0
    ).astype(np.float32)

    prediction_data: Dict[str, Any] = {
        "Bahnaric": raw_source,
        "Bahnaric_normalized": normalized_source,
        "Predicted_VN": predictions,
        "Gold_VN": raw_target,
        "Gold_VN_normalized": normalized_target,
        "Gold_rank": [
            None if not np.isfinite(rank) else int(rank) for rank in ranks
        ],
        "TopK_Preds": topk_predictions,
        "P@1": (top1_indices == gold_indices).astype(np.float32).tolist(),
        "MRR": reciprocal_ranks.tolist(),
        "Top1_score": [
            float(scores[row, top1_indices[row]]) for row in range(len(evaluation_df))
        ],
        "Gold_score": [
            float(scores[row, row]) for row in range(len(evaluation_df))
        ],
    }
    for requested_k in eval_ks:
        effective_k = int(max(1, min(requested_k, topk_indices.shape[1])))
        prediction_data[f"Hit@{effective_k}"] = (
            ranks <= effective_k
        ).astype(np.float32).tolist()

    predictions_df = make_bucket_columns(pd.DataFrame(prediction_data))
    predictions_df.to_csv(output_dir / "sentence_predictions.csv", index=False)
    save_bucket_metrics(predictions_df, output_dir, eval_ks)

    rounded_metrics: Dict[str, Any] = {
        key: round(float(value), 6) for key, value in metrics.items()
    }
    rounded_metrics.update(
        {
            "schema_version": 2,
            "family": FAMILY,
            "configuration_name": args.configuration_name,
            "split": args.split_name,
            "input_csv": str(input_path),
            "train_csv": str(train_path),
            "alignment_csv": str(alignment_path),
            "num_queries": int(len(evaluation_df)),
            "candidate_pool_size": int(len(evaluation_df)),
            "train_rows": int(len(train_df)),
            "alignment_pairs_used": int(len(source_alignment_items)),
            "selection_metric": "Top1_acc",
            "refit_policy": REFIT_POLICY,
            "configuration": configuration,
            "evaluator_script": EVALUATOR_SCRIPT,
            "evaluation_cli_args": evaluation_cli_args(configuration),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(rounded_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(rounded_metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions to {output_dir / 'sentence_predictions.csv'}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")
    print(f"Saved mapping to {output_dir / 'R.npy'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "fastText + Procrustes retrieval with development-only model and "
            "variant selection"
        )
    )
    parser.add_argument(
        "--train_csv",
        required=True,
        help="Fitting split; use data/train_fit.csv",
    )
    parser.add_argument(
        "--alignment_csv",
        required=True,
        help="Alignment supervision; must be the same data/train_fit.csv file",
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Development or held-out test CSV",
    )
    parser.add_argument("--split_name", choices=["dev", "test"], required=True)
    parser.add_argument("--configuration_name", required=True)
    parser.add_argument("--selection_manifest", default=None)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--vector_size", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sg", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Use 1 for reproducible selection; larger values may be nondeterministic",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--pooling", choices=["mean", "idf"], default="mean")
    parser.add_argument("--align_unit", choices=["phrase", "token"], default="phrase")
    parser.add_argument("--map_before_pool", action="store_true")
    parser.add_argument("--vecmap_normalize", action="store_true")
    parser.add_argument(
        "--retrieval",
        choices=list(VALID_RETRIEVALS),
        default=None,
        help=(
            "Retrieval criterion. New development runs should set this explicitly "
            "to cosine, csls, or margin_ratio."
        ),
    )
    parser.add_argument(
        "--neighborhood_k",
        type=int,
        default=10,
        help="Full-pool neighborhood size for CSLS and ratio-margin scoring",
    )
    parser.add_argument(
        "--use_csls",
        action="store_true",
        help="Deprecated alias for --retrieval csls",
    )
    parser.add_argument(
        "--csls_k",
        type=int,
        default=None,
        help="Deprecated alias for --neighborhood_k",
    )
    parser.add_argument("--strip_accents", action="store_true")
    parser.add_argument("--no_lowercase", action="store_true")
    parser.add_argument("--remove_punct", action="store_true")
    main(parser.parse_args())
