"""
Prepare FLORES-200 low-resource generalization pairs.

Default pairs:
  khm_Khmr-vie_Latn  Khmer -> Vietnamese
  lao_Laoo-vie_Latn  Lao   -> Vietnamese

Output CSV columns:
  Bahnaric,Vietnamese

We keep the column name "Bahnaric" so existing retrieval/training
scripts can be reused without changing their dataset readers.

FLORES split mapping:
  dev     -> train.csv
  devtest -> test.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset


DEFAULT_PAIRS = [
    "khm_Khmr-vie_Latn",
    "lao_Laoo-vie_Latn",
]


def pair_to_slug(pair: str) -> str:
    return pair.replace("_", "").replace("-", "_").lower()


def load_pair_split(dataset_name: str, pair: str, split: str):
    return load_dataset(dataset_name, pair, split=split, trust_remote_code=True)


def get_sentence_columns(column_names, src_lang: str, tgt_lang: str):
    src_col = f"sentence_{src_lang}"
    tgt_col = f"sentence_{tgt_lang}"

    if src_col not in column_names:
        raise ValueError(
            f"Could not find source column {src_col}. "
            f"Available columns: {list(column_names)}"
        )

    if tgt_col not in column_names:
        raise ValueError(
            f"Could not find target column {tgt_col}. "
            f"Available columns: {list(column_names)}"
        )

    return src_col, tgt_col


def convert_split_to_csv(dataset_name: str, pair: str, split: str, out_csv: Path, max_rows=None):
    src_lang, tgt_lang = pair.split("-")

    print(f"Loading {dataset_name}, pair={pair}, split={split}")
    ds = load_pair_split(dataset_name=dataset_name, pair=pair, split=split)

    if len(ds) == 0:
        raise ValueError(f"Loaded empty dataset for pair={pair}, split={split}")

    src_col, tgt_col = get_sentence_columns(ds.column_names, src_lang, tgt_lang)

    rows = []
    limit = len(ds) if max_rows is None else min(len(ds), int(max_rows))

    for i in range(limit):
        ex = ds[i]
        rows.append(
            {
                "Bahnaric": str(ex[src_col]),
                "Vietnamese": str(ex[tgt_col]),
                "source_lang": src_lang,
                "target_lang": tgt_lang,
                "flores_pair": pair,
                "flores_split": split,
                "flores_id": ex.get("id", i + 1),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Wrote {len(rows)} rows to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="facebook/flores")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument("--output_root", default="data/flores_generalization")
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_test_rows", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)

    for pair in args.pairs:
        slug = pair_to_slug(pair)
        pair_dir = output_root / slug

        print("=" * 80)
        print(f"Preparing pair: {pair}")
        print(f"Output dir: {pair_dir}")

        convert_split_to_csv(
            dataset_name=args.dataset_name,
            pair=pair,
            split="dev",
            out_csv=pair_dir / "train.csv",
            max_rows=args.max_train_rows,
        )

        convert_split_to_csv(
            dataset_name=args.dataset_name,
            pair=pair,
            split="devtest",
            out_csv=pair_dir / "test.csv",
            max_rows=args.max_test_rows,
        )

    print("=" * 80)
    print("Finished preparing FLORES generalization data.")


if __name__ == "__main__":
    main()
