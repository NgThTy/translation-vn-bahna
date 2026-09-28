#!/usr/bin/env python3
"""Build a split-local ColBERT query/collection/qrels pool from a parallel CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input_csv).dropna(
        subset=["Bahnaric", "Vietnamese"]
    ).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    collection = args.output_dir / "collection.tsv"
    queries = args.output_dir / "queries.dev.tsv"
    qrels = args.output_dir / "qrels.dev.tsv"
    collection.write_text(
        "".join(f"{i}\t{text}\n" for i, text in enumerate(frame["Vietnamese"].astype(str))),
        encoding="utf-8",
    )
    queries.write_text(
        "".join(f"{i}\t{text}\n" for i, text in enumerate(frame["Bahnaric"].astype(str))),
        encoding="utf-8",
    )
    qrels.write_text(
        "".join(f"{i}\t0\t{i}\t1\n" for i in range(len(frame))),
        encoding="utf-8",
    )
    manifest = {
        "split": args.split,
        "input_csv": str(args.input_csv),
        "input_sha256": sha256(args.input_csv),
        "num_queries": len(frame),
        "candidate_pool_size": len(frame),
        "gold_mapping": "row-aligned one-to-one qid=pid",
        "held_out_test_used_for_selection": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
