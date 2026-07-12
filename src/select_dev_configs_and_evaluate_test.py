#!/usr/bin/env python3
"""Select one configuration per family on dev and evaluate it once on test.

Expected dev metric files are produced by the updated baseline scripts and live
under ``results/dev/<family>/<configuration>/metrics.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_TIE_BREAKERS = ("MRR", "Recall@5")
SUPPORTED_EVALUATORS = {
    "lexical": "src/lexical_retrieval_baseline.py",
    "edit_distance": "src/edit_distance_retrieval_baseline.py",
    "fasttext_procrustes": "src/fasttext_procrustes_baseline.py",
    "ibm1": "src/word_alignment_baseline.py",
    "off_the_shelf": "src/multilingual_encoder_baseline.py",
    "xlmr_lora_projection": "src/previous_pipeline_baseline.py",
}


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read valid JSON from {path}: {exc}") from exc


def discover_dev_results(dev_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for metrics_path in sorted(dev_root.rglob("metrics.json")):
        metrics = load_json(metrics_path)
        if metrics.get("split") != "dev":
            continue
        family = metrics.get("family")
        configuration_name = metrics.get("configuration_name")
        if not family or not configuration_name:
            continue
        record = dict(metrics)
        record["metrics_path"] = str(metrics_path)
        records.append(record)
    return records


def metric_value(record: Dict[str, Any], metric: str) -> float:
    value = record.get(metric)
    if value is None:
        return float("-inf")
    return float(value)


def select_best(
    records: Iterable[Dict[str, Any]],
    selection_metric: str,
    tie_breakers: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["family"]), []).append(record)

    selected: Dict[str, Dict[str, Any]] = {}
    ordered_metrics = [selection_metric] + [
        metric for metric in tie_breakers if metric != selection_metric
    ]

    for family, candidates in grouped.items():
        def sort_key(record: Dict[str, Any]) -> tuple[Any, ...]:
            return tuple(-metric_value(record, metric) for metric in ordered_metrics) + (
                str(record["configuration_name"]),
            )

        winner = sorted(candidates, key=sort_key)[0]
        selected[family] = {
            "family": family,
            "configuration_name": winner["configuration_name"],
            "configuration": winner["configuration"],
            "development_metrics": {
                metric: winner.get(metric) for metric in ordered_metrics
            },
            "selection_metric": selection_metric,
            "tie_breakers": list(tie_breakers) + ["configuration_name_ascending"],
            "dev_metrics_path": winner["metrics_path"],
            "evaluator_script": winner.get("evaluator_script"),
            "evaluation_cli_args": winner.get("evaluation_cli_args"),
        }
    return selected


def flatten_dev_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "family": record.get("family"),
        "configuration_name": record.get("configuration_name"),
        "Top1_acc": record.get("Top1_acc"),
        "MRR": record.get("MRR"),
        "Recall@5": record.get("Recall@5"),
        "Recall@10": record.get("Recall@10"),
        "num_queries": record.get("num_queries"),
        "candidate_pool_size": record.get("candidate_pool_size"),
        "input_csv": record.get("input_csv"),
        "configuration": json.dumps(
            record.get("configuration", {}), ensure_ascii=False, sort_keys=True
        ),
        "metrics_path": record.get("metrics_path"),
    }


def write_all_dev_variants(records: List[Dict[str, Any]], output_path: Path) -> None:
    rows = [flatten_dev_record(record) for record in records]
    columns = [
        "family",
        "configuration_name",
        "Top1_acc",
        "MRR",
        "Recall@5",
        "Recall@10",
        "num_queries",
        "candidate_pool_size",
        "input_csv",
        "configuration",
        "metrics_path",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).sort_values(
        ["family", "Top1_acc", "MRR", "configuration_name"],
        ascending=[True, False, False, True],
        na_position="last",
    ).to_csv(output_path, index=False)


def evaluate_selected_family(
    family: str,
    selected: Dict[str, Dict[str, Any]],
    manifest_path: Path,
    test_csv: Path,
    test_root: Path,
    force_test: bool = False,
) -> Path:
    if family not in selected:
        raise ValueError(f"No dev results found for requested family {family!r}")
    if not test_csv.is_file():
        raise FileNotFoundError(f"Held-out test file not found: {test_csv}")

    choice = selected[family]
    evaluator_script = choice.get("evaluator_script") or SUPPORTED_EVALUATORS.get(family)
    if not evaluator_script:
        raise ValueError(f"No evaluator script is registered for family {family!r}")
    evaluator_path = Path(evaluator_script)
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"Evaluator script not found: {evaluator_path}")

    cli_args = choice.get("evaluation_cli_args")
    if not isinstance(cli_args, list):
        raise ValueError(
            f"Selected dev result for {family!r} does not contain evaluation_cli_args"
        )

    configuration_name = str(choice["configuration_name"])
    output_dir = test_root / family / configuration_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file() and not force_test:
        existing = load_json(metrics_path)
        if (
            existing.get("split") == "test"
            and existing.get("configuration_name") == configuration_name
        ):
            print(f"Skipping existing authorized test result: {metrics_path}")
            return metrics_path
        raise RuntimeError(
            f"Refusing to overwrite incompatible existing test result: {metrics_path}. "
            "Pass --force_test only after reviewing the mismatch."
        )

    command = [
        sys.executable,
        str(evaluator_path),
        "--input_csv",
        str(test_csv),
        "--split_name",
        "test",
        "--configuration_name",
        configuration_name,
        "--selection_manifest",
        str(manifest_path),
        "--output_dir",
        str(output_dir),
        *[str(value) for value in cli_args],
    ]

    print("Running the single dev-selected test evaluation:")
    print(" ".join(command))
    subprocess.run(command, check=True)
    return metrics_path


def collect_test_results(
    selected: Dict[str, Dict[str, Any]],
    test_root: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family, choice in sorted(selected.items()):
        configuration_name = str(choice["configuration_name"])
        metrics_path = test_root / family / configuration_name / "metrics.json"
        metrics: Dict[str, Any] = load_json(metrics_path) if metrics_path.is_file() else {}
        rows.append(
            {
                "family": family,
                "selected_configuration": configuration_name,
                "development_accuracy_at_1": choice.get("development_metrics", {}).get("Top1_acc"),
                "development_mrr": choice.get("development_metrics", {}).get("MRR"),
                "development_recall_at_5": choice.get("development_metrics", {}).get("Recall@5"),
                "test_accuracy_at_1": metrics.get("Top1_acc"),
                "test_mrr": metrics.get("MRR"),
                "test_recall_at_5": metrics.get("Recall@5"),
                "test_metrics_path": str(metrics_path) if metrics_path.is_file() else "",
            }
        )
    return pd.DataFrame(rows)


def read_old_table2(path: Path) -> pd.DataFrame:
    old_df = pd.read_csv(path)
    if "family" not in old_df.columns:
        raise ValueError(f"{path} must contain a 'family' column")

    if "old_table2_rank" not in old_df.columns:
        score_columns = [
            column
            for column in ("old_test_accuracy_at_1", "test_accuracy_at_1", "Top1_acc")
            if column in old_df.columns
        ]
        if not score_columns:
            raise ValueError(
                f"{path} must contain old_table2_rank or an old Accuracy@1 column"
            )
        score_column = score_columns[0]
        old_df["old_table2_rank"] = old_df[score_column].rank(
            method="min", ascending=False
        ).astype("Int64")
    return old_df


def write_ordering_comparison(
    test_results: pd.DataFrame,
    output_csv: Path,
    summary_json: Path,
    old_table2_csv: Optional[Path],
) -> None:
    comparison = test_results.copy()
    comparison["new_rank"] = pd.to_numeric(
        comparison["test_accuracy_at_1"], errors="coerce"
    ).rank(method="min", ascending=False).astype("Int64")

    summary: Dict[str, Any]
    if old_table2_csv is None:
        comparison["old_table2_rank"] = pd.NA
        comparison["rank_preserved"] = pd.NA
        summary = {
            "status": "not_computed_missing_old_table2_csv",
            "ordering_preserved": None,
            "message": "Provide --old_table2_csv after all families have dev-selected test results.",
        }
    else:
        old_df = read_old_table2(old_table2_csv)
        old_columns = ["family", "old_table2_rank"]
        old_score_column = next(
            (
                column
                for column in ("old_test_accuracy_at_1", "test_accuracy_at_1", "Top1_acc")
                if column in old_df.columns
            ),
            None,
        )
        if old_score_column is not None:
            old_columns.append(old_score_column)
        old_subset = old_df[old_columns].copy()
        if old_score_column is not None and old_score_column != "old_test_accuracy_at_1":
            old_subset = old_subset.rename(
                columns={old_score_column: "old_test_accuracy_at_1"}
            )
        comparison = comparison.merge(old_subset, on="family", how="left")
        comparison["rank_preserved"] = (
            comparison["old_table2_rank"].notna()
            & comparison["new_rank"].notna()
            & (
                comparison["old_table2_rank"].astype("Int64")
                == comparison["new_rank"].astype("Int64")
            )
        )

        old_families = set(old_df["family"].astype(str))
        evaluated_families = set(
            comparison.loc[
                comparison["test_accuracy_at_1"].notna(), "family"
            ].astype(str)
        )
        complete = old_families.issubset(evaluated_families)
        ordering_preserved = bool(comparison["rank_preserved"].all()) if complete else None
        summary = {
            "status": "complete" if complete else "not_computed_incomplete_family_coverage",
            "ordering_preserved": ordering_preserved,
            "old_table2_family_count": len(old_families),
            "families_with_dev_selected_test_results": len(evaluated_families),
            "missing_families": sorted(old_families - evaluated_families),
        }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(args: argparse.Namespace) -> None:
    dev_root = Path(args.dev_root)
    test_root = Path(args.test_root)
    output_root = Path(args.output_root)
    manifest_path = output_root / "selected_configs.json"

    records = discover_dev_results(dev_root)
    if not records:
        raise RuntimeError(f"No development metrics.json files found under {dev_root}")

    selected = select_best(records, args.selection_metric, args.tie_breakers)
    output_root.mkdir(parents=True, exist_ok=True)

    write_all_dev_variants(records, output_root / "all_dev_variants.csv")
    manifest_path.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.evaluate_test:
        families = args.family if args.family else sorted(selected)
        for family in families:
            evaluate_selected_family(
                family=family,
                selected=selected,
                manifest_path=manifest_path,
                test_csv=Path(args.test_csv),
                test_root=test_root,
                force_test=args.force_test,
            )

    test_results = collect_test_results(selected, test_root)
    test_results.to_csv(output_root / "dev_selected_test_results.csv", index=False)

    old_table2_path = Path(args.old_table2_csv) if args.old_table2_csv else None
    write_ordering_comparison(
        test_results=test_results,
        output_csv=output_root / "table2_ordering_comparison.csv",
        summary_json=output_root / "table2_ordering_summary.json",
        old_table2_csv=old_table2_path,
    )

    print(f"Saved all dev variants to {output_root / 'all_dev_variants.csv'}")
    print(f"Saved selected configurations to {manifest_path}")
    print(f"Saved dev-selected test results to {output_root / 'dev_selected_test_results.csv'}")
    print(f"Saved ordering comparison to {output_root / 'table2_ordering_comparison.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Select configurations on dev, evaluate one selected configuration "
            "per family on test, and compare Table 2 ordering."
        )
    )
    parser.add_argument("--dev_root", default="results/dev")
    parser.add_argument("--test_root", default="results/test")
    parser.add_argument("--output_root", default="results/dev_selection")
    parser.add_argument("--test_csv", default="data/test.csv")
    parser.add_argument("--selection_metric", default="Top1_acc")
    parser.add_argument("--tie_breakers", nargs="+", default=list(DEFAULT_TIE_BREAKERS))
    parser.add_argument(
        "--family",
        action="append",
        choices=sorted(SUPPORTED_EVALUATORS),
        help=(
            "Family to evaluate on test. Repeat for multiple families. Without "
            "this option, all discovered supported families are evaluated."
        ),
    )
    parser.add_argument("--evaluate_test", action="store_true")
    parser.add_argument(
        "--force_test",
        action="store_true",
        help="Overwrite an existing test result for the selected configuration.",
    )
    parser.add_argument(
        "--old_table2_csv",
        default=None,
        help="CSV containing family plus old_table2_rank or old Accuracy@1 values.",
    )
    main(parser.parse_args())

