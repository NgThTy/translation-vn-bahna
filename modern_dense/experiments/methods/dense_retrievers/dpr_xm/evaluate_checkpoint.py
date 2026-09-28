#!/usr/bin/env python3
"""Evaluate a saved DPR-XM SentenceTransformer checkpoint on a fixed pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xm-root", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(args.xm_root))
    from src.utils.SentenceTransformer import SentenceTransformerCustom
    from src.utils.common import set_xmod_language

    frame = pd.read_csv(args.input_csv).dropna(
        subset=["Bahnaric", "Vietnamese"]
    ).reset_index(drop=True)
    model = SentenceTransformerCustom(str(args.model_dir))
    auto = model[0].auto_model
    if auto.__class__.__name__.lower().startswith("xmod"):
        set_xmod_language(auto, "bana")

    source = model.encode(
        frame["Bahnaric"].astype(str).tolist(),
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    target = model.encode(
        frame["Vietnamese"].astype(str).tolist(),
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    scores = source @ target.T
    k = min(args.topk, len(frame))
    top_scores, top_indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
    top_scores = top_scores.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    gold = np.arange(len(frame), dtype=np.int64)
    hits = top_indices == gold[:, None]
    ranks = np.full(len(frame), np.inf)
    found = hits.any(axis=1)
    ranks[found] = hits[found].argmax(axis=1) + 1

    model_file = args.model_dir / "model.safetensors"
    metrics = {
        "method": "DPR-XM supervised fine-tuning",
        "split": args.split,
        "num_queries": len(frame),
        "candidate_pool_size": len(frame),
        "acc@1": float(np.mean(ranks == 1)),
        "mrr@10": float(np.mean(np.where(np.isfinite(ranks), 1.0 / ranks, 0.0))),
        "recall@5": float(np.mean(ranks <= 5)),
        "recall@10": float(np.mean(ranks <= 10)),
        "model_dir": str(args.model_dir),
        "model_sha256": sha256(model_file),
        "input_csv": str(args.input_csv),
        "input_sha256": sha256(args.input_csv),
        "pooling": "mean",
        "similarity": "cosine",
    }
    output = frame[["Bahnaric", "Vietnamese"]].copy()
    output["gold_rank"] = ranks
    output["topk_indices"] = ["|".join(map(str, row)) for row in top_indices]
    output["topk_scores"] = ["|".join(f"{value:.8g}" for value in row) for row in top_scores]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
