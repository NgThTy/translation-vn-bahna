"""
Off-the-shelf multilingual encoder baseline for Bahnaric -> Vietnamese sentence retrieval.

Baseline family:
5. Off-the-shelf multilingual encoders:
   - mBERT
   - XLM-R
   - multilingual MiniLM
   - LaBSE

Task:
Given a Bahnaric query sentence, retrieve the correct Vietnamese sentence from a candidate pool.
For data/test.csv, the gold Vietnamese sentence is assumed to be on the same row as the Bahnaric query.

This baseline does NOT train or fine-tune anything.
It directly uses pretrained multilingual encoders.

Retrieval:
1. Encode all Bahnaric test sentences.
2. Encode all Vietnamese candidate sentences.
3. Rank Vietnamese candidates by cosine similarity or CSLS.
4. Compute Top1 accuracy, MRR, Hit@K, Recall@K, Precision@K.

Encoder backends:
- transformers:
  Uses AutoTokenizer + AutoModel with mean pooling.
  Good for mBERT and XLM-R.

- sentence_transformers:
  Uses SentenceTransformer.encode().
  Good for multilingual MiniLM and LaBSE.

- auto:
  Uses sentence_transformers for models whose names start with "sentence-transformers/",
  otherwise uses transformers.

Example:
python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name bert-base-multilingual-cased \
  --encoder_name mbert \
  --backend transformers \
  --output_dir results/baselines/offtheshelf_mbert_cosine \
  --topk_eval 10 \
  --eval_ks 1 5 10

python src/multilingual_encoder_baseline.py \
  --test_csv data/test.csv \
  --model_name sentence-transformers/LaBSE \
  --encoder_name labse \
  --backend sentence_transformers \
  --output_dir results/baselines/offtheshelf_labse_cosine \
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


def resolve_backend(model_name: str, backend: str) -> str:
    if backend != "auto":
        return backend

    if model_name.startswith("sentence-transformers/"):
        return "sentence_transformers"

    return "transformers"


def mean_pool(last_hidden, attention_mask):
    import torch

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

    all_embs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, return_dict=True)
            emb = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            all_embs.append(emb.detach().cpu().numpy())

            print(f"Encoded {min(i + batch_size, len(texts))}/{len(texts)} with transformers")

    return np.vstack(all_embs).astype(np.float32)


def encode_with_sentence_transformers(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embs.astype(np.float32)


def encode_texts(
    texts: List[str],
    model_name: str,
    backend: str,
    batch_size: int,
    max_len: int,
    device: str,
) -> np.ndarray:
    if backend == "transformers":
        return encode_with_transformers(
            texts=texts,
            model_name=model_name,
            batch_size=batch_size,
            max_len=max_len,
            device=device,
        )

    if backend == "sentence_transformers":
        return encode_with_sentence_transformers(
            texts=texts,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )

    raise ValueError(f"Unknown backend: {backend}")


def main(args: argparse.Namespace) -> None:
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = resolve_backend(args.model_name, args.backend)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Encoder name: {args.encoder_name}")
    print(f"Backend: {backend}")

    df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

    raw_bah = df["Bahnaric"].astype(str).tolist()
    raw_vn = df["Vietnamese"].astype(str).tolist()

    lowercase = bool(args.lowercase)

    bah = [
        normalize_text(
            x,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for x in raw_bah
    ]

    vn = [
        normalize_text(
            x,
            lowercase=lowercase,
            strip_accents=args.strip_accents,
            remove_punct=args.remove_punct,
        )
        for x in raw_vn
    ]

    print("[1/3] Encoding Bahnaric queries...")
    bah_emb = encode_texts(
        texts=bah,
        model_name=args.model_name,
        backend=backend,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=device,
    )

    print("[2/3] Encoding Vietnamese candidates...")
    vn_emb = encode_texts(
        texts=vn,
        model_name=args.model_name,
        backend=backend,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=device,
    )

    print(f"Bahnaric embedding shape: {bah_emb.shape}")
    print(f"Vietnamese embedding shape: {vn_emb.shape}")

    print("[3/3] Retrieving candidates...")
    if args.use_csls:
        topk_idx, scores = csls_topk(
            query=bah_emb,
            index=vn_emb,
            topk=args.topk_eval,
            csls_k=args.csls_k,
        )
        retrieval = "csls"
    else:
        topk_idx, scores = cosine_topk(
            query=bah_emb,
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
            "method": "off_the_shelf_multilingual_encoder",
            "encoder_name": args.encoder_name,
            "model_name": args.model_name,
            "backend": backend,
            "retrieval": retrieval,
            "test_csv": args.test_csv,
            "num_queries": int(len(df)),
            "candidate_pool_size": int(len(df)),
            "embedding_dim": int(bah_emb.shape[1]),
            "max_len": int(args.max_len),
            "batch_size": int(args.batch_size),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Off-the-shelf multilingual encoder baseline for Bahnaric-Vietnamese sentence retrieval"
    )

    parser.add_argument("--test_csv", required=True, help="CSV with columns Bahnaric,Vietnamese")
    parser.add_argument("--model_name", required=True, help="HuggingFace or sentence-transformers model name")
    parser.add_argument("--encoder_name", required=True, help="Short label for output metrics, e.g. mbert/xlmr/minilm/labse")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--backend",
        choices=["auto", "transformers", "sentence_transformers"],
        default="auto",
        help="auto chooses sentence_transformers for sentence-transformers/* models, otherwise transformers",
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

    parser.add_argument("--no_cuda", action="store_true")

    args = parser.parse_args()
    main(args)
