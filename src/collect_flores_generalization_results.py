import json
from pathlib import Path

import pandas as pd


def main():
    root = Path("results/baselines")
    rows = []

    patterns = [
        "flores_*/metrics.json",
        "flores_*/test_cosine/metrics.json",
        "flores_*/test_csls/metrics.json",
    ]

    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)

            rows.append(
                {
                    "result_dir": str(path.parent),
                    "method": m.get("method"),
                    "pipeline": m.get("pipeline"),
                    "train_csv": m.get("train_csv"),
                    "test_csv": m.get("test_csv") or m.get("csv_path"),
                    "retrieval": m.get("retrieval"),
                    "Top1": m.get("Top1_acc"),
                    "MRR": m.get("MRR"),
                    "Hit@5": m.get("Hit@5"),
                    "Hit@10": m.get("Hit@10"),
                    "num_queries": m.get("num_queries"),
                    "embedding_dim": m.get("embedding_dim"),
                    "pooling": m.get("pooling"),
                    "epochs": m.get("epochs"),
                    "lr": m.get("lr"),
                    "kabsch_used": m.get("kabsch_used"),
                    "use_lora": m.get("use_lora"),
                }
            )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(["test_csv", "method", "retrieval"], na_position="last")

    out_path = Path("results/flores_generalization_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(out)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
