#!/usr/bin/env python3
"""Create a deterministic train/dev split for Bahnaric-Vietnamese retrieval.

This script only reads the original training file. It never reads or modifies the
held-out test file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

REQUIRED_COLUMNS = ("Bahnaric", "Vietnamese")
AUTO_STRATIFY_COLUMNS = ("source", "Source", "dataset", "Dataset", "domain", "Domain")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def choose_strata(
    df: pd.DataFrame,
    requested_column: Optional[str],
    num_length_bins: int,
) -> tuple[Optional[pd.Series], str]:
    """Return robust stratification labels and a human-readable description."""
    if requested_column:
        if requested_column not in df.columns:
            raise ValueError(
                f"Requested stratification column {requested_column!r} is not present. "
                f"Available columns: {list(df.columns)}"
            )
        labels = df[requested_column].astype(str).fillna("<NA>")
        if labels.value_counts().min() >= 2:
            return labels, f"column:{requested_column}"
        return None, f"none (column {requested_column!r} contains a stratum with fewer than 2 rows)"

    for column in AUTO_STRATIFY_COLUMNS:
        if column in df.columns:
            labels = df[column].astype(str).fillna("<NA>")
            if labels.value_counts().min() >= 2:
                return labels, f"column:{column}"

    # Fall back to approximate sentence-length stratification. qcut may drop
    # duplicate boundaries when many sentences have the same length.
    vn_words = df["Vietnamese"].astype(str).str.split().map(len)
    n_unique = int(vn_words.nunique())
    q = min(max(2, num_length_bins), n_unique)
    if q < 2:
        return None, "none (insufficient sentence-length variation)"

    try:
        bins = pd.qcut(vn_words, q=q, labels=False, duplicates="drop")
    except ValueError:
        return None, "none (length-bin construction failed)"

    if bins.nunique() < 2 or bins.value_counts().min() < 2:
        return None, "none (length bins are too small for stratification)"
    return bins.astype(str), f"Vietnamese_word_length_qcut:{int(bins.nunique())}_bins"


def pair_groups(df: pd.DataFrame) -> pd.Series:
    # Exact pair grouping prevents duplicate bilingual pairs from appearing in
    # both train_fit and dev.
    return (
        df["Bahnaric"].astype(str)
        + "\u241f"
        + df["Vietnamese"].astype(str)
    )


def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input_csv)
    train_output = Path(args.train_output)
    dev_output = Path(args.dev_output)
    manifest_output = Path(args.manifest_output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Training file not found: {input_path}")

    for output_path in (train_output, dev_output, manifest_output):
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_path}. Pass --overwrite to replace existing split files."
            )

    df = pd.read_csv(input_path)
    validate_columns(df)

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        bad_rows = int(df[list(REQUIRED_COLUMNS)].isna().any(axis=1).sum())
        raise ValueError(
            f"Found {bad_rows} rows with missing Bahnaric/Vietnamese values. "
            "Clean the training data before creating the split."
        )

    if not 0.0 < args.dev_ratio < 1.0:
        raise ValueError("--dev_ratio must be strictly between 0 and 1")

    indices = np.arange(len(df), dtype=np.int64)
    groups = pair_groups(df)
    duplicate_pair_rows = int(groups.duplicated(keep=False).sum())

    stratification_description = "none"
    if duplicate_pair_rows > 0:
        # GroupShuffleSplit is used instead of row-level stratification so an
        # exact duplicate bilingual pair cannot leak across the split.
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=args.dev_ratio,
            random_state=args.seed,
        )
        train_idx, dev_idx = next(splitter.split(indices, groups=groups))
        split_strategy = "group_shuffle_by_exact_bilingual_pair"
        stratification_description = "disabled because duplicate-pair grouping took priority"
    else:
        strata, stratification_description = choose_strata(
            df=df,
            requested_column=args.stratify_column,
            num_length_bins=args.length_bins,
        )
        train_idx, dev_idx = train_test_split(
            indices,
            test_size=args.dev_ratio,
            random_state=args.seed,
            shuffle=True,
            stratify=strata,
        )
        split_strategy = "stratified_row_split" if strata is not None else "random_row_split"

    # Stable row order in output files makes diffs and reproduction easier.
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    dev_idx = np.sort(np.asarray(dev_idx, dtype=np.int64))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    dev_df = df.iloc[dev_idx].reset_index(drop=True)

    if len(train_df) + len(dev_df) != len(df):
        raise RuntimeError("Split row counts do not sum to the original training row count")
    if set(train_idx).intersection(set(dev_idx)):
        raise RuntimeError("Train/dev index overlap detected")

    train_output.parent.mkdir(parents=True, exist_ok=True)
    dev_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_output, index=False)
    dev_df.to_csv(dev_output, index=False)

    manifest = {
        "schema_version": 1,
        "input_csv": str(input_path),
        "input_sha256": sha256_file(input_path),
        "train_output": str(train_output),
        "dev_output": str(dev_output),
        "seed": int(args.seed),
        "dev_ratio_requested": float(args.dev_ratio),
        "original_rows": int(len(df)),
        "train_fit_rows": int(len(train_df)),
        "dev_rows": int(len(dev_df)),
        "actual_dev_ratio": float(len(dev_df) / len(df)),
        "split_strategy": split_strategy,
        "stratification": stratification_description,
        "duplicate_pair_rows_in_input": duplicate_pair_rows,
        "train_output_sha256": sha256_file(train_output),
        "dev_output_sha256": sha256_file(dev_output),
        "test_file_read_or_modified": False,
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a deterministic train_fit/dev split without touching the held-out test set."
    )
    parser.add_argument("--input_csv", default="data/train.csv")
    parser.add_argument("--train_output", default="data/train_fit.csv")
    parser.add_argument("--dev_output", default="data/dev.csv")
    parser.add_argument("--manifest_output", default="data/train_dev_split_manifest.json")
    parser.add_argument("--dev_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify_column",
        default=None,
        help="Optional metadata column. If omitted, source/domain is used when available, otherwise Vietnamese length bins.",
    )
    parser.add_argument("--length_bins", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
