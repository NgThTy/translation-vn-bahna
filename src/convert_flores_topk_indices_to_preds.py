"""
Convert FLORES neural sentence_predictions.csv from TopK_indices format
to TopK_Preds format for Method 8 hybrid reranking.

Input:
  - test_csv with columns: Bahnaric,Vietnamese
  - neural_predictions_csv with column: TopK_indices

Output:
  - CSV with columns expected by hybrid_lexical_neural_rerank.py:
      Bahnaric
      Gold_VN
      TopK_Preds
      optional TopK_Scores if available
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_indices(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(x) for x in text.split("|") if x.strip() != ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--neural_predictions_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv).dropna(subset=["Bahnaric", "Vietnamese"]).reset_index(drop=True)
    pred_df = pd.read_csv(args.neural_predictions_csv)

    if len(test_df) != len(pred_df):
        raise ValueError(
            f"Length mismatch: test_csv has {len(test_df)} rows, "
            f"neural_predictions_csv has {len(pred_df)} rows"
        )

    if "TopK_indices" not in pred_df.columns:
        raise ValueError(
            f"Missing TopK_indices in {args.neural_predictions_csv}. "
            f"Available columns: {list(pred_df.columns)}"
        )

    vn_candidates = test_df["Vietnamese"].astype(str).tolist()

    rows = []
    for i in range(len(test_df)):
        topk_indices = parse_indices(pred_df.loc[i, "TopK_indices"])
        topk_preds = [vn_candidates[j] for j in topk_indices]

        row = {
            "Bahnaric": str(test_df.loc[i, "Bahnaric"]),
            "Gold_VN": str(test_df.loc[i, "Vietnamese"]),
            "TopK_Preds": "|".join(topk_preds),
        }

        if "TopK_scores" in pred_df.columns:
            row["TopK_Scores"] = pred_df.loc[i, "TopK_scores"]
        elif "TopK_scores" in pred_df.columns:
            row["TopK_Scores"] = pred_df.loc[i, "TopK_scores"]

        rows.append(row)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"Wrote converted hybrid input to {out_path}")


if __name__ == "__main__":
    main()
