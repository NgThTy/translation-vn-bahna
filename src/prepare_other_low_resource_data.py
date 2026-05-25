import argparse
import json
from pathlib import Path

import pandas as pd


def convert_json_to_csv(input_json: Path, output_csv: Path):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for i, ex in enumerate(data):
        if "za" not in ex or "zh" not in ex:
            continue

        rows.append({
            "Bahnaric": str(ex["za"]).strip(),
            "Vietnamese": str(ex["zh"]).strip(),
            "source": ex.get("source", ""),
            "example_id": i,
            "source_lang": "za",
            "target_lang": "zh",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote {len(rows)} rows to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        default="data/other_low_resource_data",
    )
    parser.add_argument(
        "--output_root",
        default="data/other_low_resource_generalization/zhuang_chinese",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    convert_json_to_csv(
        input_root / "parallel_corpus.json",
        output_root / "train.csv",
    )

    convert_json_to_csv(
        input_root / "test_translation_set.json",
        output_root / "test.csv",
    )


if __name__ == "__main__":
    main()
