#!/usr/bin/env python3
"""Select one ColBERT checkpoint from development metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.eval_dir.glob("step_*/metrics.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["metrics_path"] = str(path)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No development metrics found under {args.eval_dir}")

    best = max(
        rows,
        key=lambda row: (row["acc@1"], row["mrr@10"], row["recall@5"]),
    )
    summary = {
        "selection_split": "development",
        "selection_rule": "Acc@1, then MRR@10, then Recall@5",
        "num_candidates": len(rows),
        "best": best,
        "candidates": rows,
    }
    output = args.eval_dir / "dev_selection_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
