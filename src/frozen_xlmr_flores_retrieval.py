"""
Frozen XLM-R retrieval for FLORES generalization pairs.

No training.
Pipeline:
  source sentence -> frozen XLM-R -> mean pooling
  Vietnamese sentence -> frozen XLM-R -> mean pooling
  retrieve using cosine or CSLS

Expected CSV columns:
  Bahnaric,Vietnamese
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def normalize_text(text: str, lowercase: bool = False) -> str:
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    if lowercase:
        text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, max_len, batch_size):
    model.eval()
    all_embs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", unit="batch"):
        batch = texts[start : start + batch_size]
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

    return np.vstack(all_embs).astype(np.float32)


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


def ranking_metrics(topk_idx: np.ndarray, gold_idx: np.ndarray, eval_ks: List[int]):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="xlm-roberta-base")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--topk_eval", type=int, default=10)
    parser.add_argument("--eval_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--use_csls", action="store_true")
    parser.add_argument("--csls_k", type=int, default=10)
    parser.add_argument("--lowercase", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)

    src_texts = [normalize_text(x, lowercase=args.lowercase) for x in df["Bahnaric"].astype(str).tolist()]
    tgt_texts = [normalize_text(x, lowercase=args.lowercase) for x in df["Vietnamese"].astype(str).tolist()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    print(f"Using device: {device}")
    print(f"Encoding {len(src_texts)} source queries...")
    src_emb = encode_texts(src_texts, tokenizer, model, device, args.max_len, args.batch_size)

    print(f"Encoding {len(tgt_texts)} Vietnamese candidates...")
    tgt_emb = encode_texts(tgt_texts, tokenizer, model, device, args.max_len, args.batch_size)

    if args.use_csls:
        topk_idx, scores = csls_topk(src_emb, tgt_emb, topk=args.topk_eval, csls_k=args.csls_k)
        retrieval = "csls"
    else:
        topk_idx, scores = cosine_topk(src_emb, tgt_emb, topk=args.topk_eval)
        retrieval = "cosine"

    gold_idx = np.arange(len(df), dtype=np.int64)
    metrics, ranks = ranking_metrics(topk_idx, gold_idx, args.eval_ks)

    rounded = {k: round(float(v), 4) for k, v in metrics.items()}
    rounded.update(
        {
            "method": "frozen_xlmr_retrieval",
            "pipeline": "frozen_xlmr_mean_pool_retrieval",
            "test_csv": args.test_csv,
            "model_name": args.model_name,
            "retrieval": retrieval,
            "num_queries": int(len(df)),
            "candidate_pool_size": int(len(df)),
            "embedding_dim": int(src_emb.shape[1]),
            "topk_eval": int(args.topk_eval),
            "eval_ks": [int(k) for k in args.eval_ks],
            "max_len": int(args.max_len),
            "batch_size": int(args.batch_size),
            "use_csls": bool(args.use_csls),
            "csls_k": int(args.csls_k),
        }
    )

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rounded, f, ensure_ascii=False, indent=2)

    pred_top1_idx = topk_idx[:, 0]
    out = pd.DataFrame(
        {
            "Source": df["Bahnaric"].astype(str).tolist(),
            "Gold_VN": df["Vietnamese"].astype(str).tolist(),
            "Predicted_VN": [df["Vietnamese"].iloc[i] for i in pred_top1_idx],
            "Gold_rank": [None if not np.isfinite(r) else int(r) for r in ranks.tolist()],
            "TopK_indices": ["|".join(map(str, row)) for row in topk_idx],
        }
    )
    out.to_csv(output_dir / "sentence_predictions.csv", index=False)

    print(json.dumps(rounded, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
