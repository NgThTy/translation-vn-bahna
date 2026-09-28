#!/usr/bin/env python3
"""Prepare enough repeated Bahnar training triples for 50k ColBERT-X steps."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results" / "dense" / "dev_protocol" / "colbertx" / "data"
OUTPUT = ROOT / "artifacts" / "dense_support" / "colbert_20k_data"
TARGET_STEPS = 50_000
BATCH_SIZE = 16


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in (
        "collection.tsv",
        "queries.train.tsv",
        "queries.dev.tsv",
        "qrels.train.tsv",
        "qrels.dev.tsv",
        "triples.train.tsv",
    ):
        destination = OUTPUT / name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(SOURCE / name)

    source_triples = SOURCE / "triples.train.jsonl"
    lines = source_triples.read_text(encoding="utf-8").splitlines(keepends=True)
    repeats = math.ceil(TARGET_STEPS * BATCH_SIZE / len(lines))
    repeated_path = OUTPUT / "triples.train.jsonl"
    with repeated_path.open("w", encoding="utf-8") as handle:
        for _ in range(repeats):
            handle.writelines(lines)

    manifest = {
        "source": str(SOURCE),
        "source_triples": len(lines),
        "repeats": repeats,
        "output_triples": len(lines) * repeats,
        "target_steps": TARGET_STEPS,
        "steps_at_batch_16": (len(lines) * repeats) // 16,
        "steps_at_batch_8": (len(lines) * repeats) // 8,
        "purpose": "Match the released ColBERT-X v1 50k-step recipe; the original one-pass file ended at 2920 steps.",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
