# #!/usr/bin/env bash
# set -e

# # Method 8: Hybrid IBM1 + XLM-R LoRA reranking
# # Neural candidate generator:
# # XLM-R 50ep + LoRA + no Kabsch + CSLS

# NEURAL_TEST_PRED="results/baselines/previous_pipeline_xlmr_50ep_no_kabsch_csls_lora/sentence_predictions.csv"

# python src/hybrid_lexical_neural_rerank.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --test_neural_predictions "$NEURAL_TEST_PRED" \
#   --output_dir results/baselines/hybrid_ibm1_xlmr_lora_no_kabsch_csls_rerank_test_fixed \
#   --strip_accents \
#   --remove_punct \
#   --length_penalty 0.1 \
#   --ibm_iters 5 \
#   --alpha 0.5 \
#   --beta 0.5 \
#   --gamma 0.0

###################################
# #!/usr/bin/env bash
# set -euo pipefail

# TRAIN_CSV="${TRAIN_CSV:-data/train_fit.csv}"
# DEV_CSV="${DEV_CSV:-data/dev.csv}"
# TEST_CSV="${TEST_CSV:-data/test.csv}"
# DEV_ROOT="${HYBRID_DEV_ROOT:-results/dev/hybrid}"
# TEST_ROOT="${TEST_ROOT:-results/test}"
# SELECTION_ROOT="${SELECTION_ROOT:-results/dev_selection}"
# SELECTION_MANIFEST="$SELECTION_ROOT/selected_configs.json"
# IBM_CACHE_DIR="${IBM_CACHE_DIR:-results/cache/hybrid_ibm1}"
# ACTION="${HYBRID_ACTION:-all}"

# for required_file in \
#   "$TRAIN_CSV" \
#   "$DEV_CSV" \
#   "$TEST_CSV" \
#   src/hybrid_lexical_neural_rerank.py \
#   src/select_dev_configs_and_evaluate_test.py; do
#   if [[ ! -f "$required_file" ]]; then
#     echo "Missing required file: $required_file" >&2
#     exit 1
#   fi
# done

# mkdir -p "$DEV_ROOT" "$TEST_ROOT" "$SELECTION_ROOT" "$IBM_CACHE_DIR"

# refresh_selection_manifest() {
#   python src/select_dev_configs_and_evaluate_test.py \
#     --dev_root results/dev \
#     --test_root "$TEST_ROOT" \
#     --output_root "$SELECTION_ROOT" \
#     --test_csv "$TEST_CSV"
# }

# check_base_components() {
#   python - "$SELECTION_MANIFEST" <<'PY'
# import json
# import sys
# from pathlib import Path

# path = Path(sys.argv[1])
# data = json.loads(path.read_text(encoding="utf-8"))
# missing = [family for family in ("ibm1", "xlmr_lora_projection") if family not in data]
# if missing:
#     raise SystemExit(
#         "Missing development-selected base families: " + ", ".join(missing)
#         + ". Complete their dev experiments and rerun the central selector first."
#     )
# print("Selected IBM1 component:", data["ibm1"]["configuration_name"])
# print("Selected XLM-R component:", data["xlmr_lora_projection"]["configuration_name"])
# PY
# }

# run_dev_search() {
#   echo "Refreshing base-family selections from results/dev ..."
#   refresh_selection_manifest
#   check_base_components

#   python src/hybrid_lexical_neural_rerank.py \
#     --mode tune_dev \
#     --train_csv "$TRAIN_CSV" \
#     --input_csv "$DEV_CSV" \
#     --split_name dev \
#     --selection_manifest "$SELECTION_MANIFEST" \
#     --output_dir "$DEV_ROOT" \
#     --test_root "$TEST_ROOT" \
#     --ibm_cache_dir "$IBM_CACHE_DIR" \
#     --candidate_topks 5 10 \
#     --neural_score_modes raw reciprocal_rank \
#     --normalizations minmax rank \
#     --formulas weighted_sum weighted_product \
#     --length_penalties 0.0 0.1 \
#     --weights \
#       1.0,0.0,0.0 \
#       0.75,0.25,0.0 \
#       0.5,0.5,0.0 \
#       0.25,0.75,0.0 \
#       0.0,1.0,0.0 \
#       0.5,0.4,0.1 \
#     --eval_ks 1 5 10 \
#     --prune_stale

#   # Rebuild the shared manifest so it now also contains the hybrid winner.
#   refresh_selection_manifest
# }

# ensure_selected_neural_test_predictions() {
#   local force_flag=""

#   # An existing selected neural test result can be reused. If the hybrid winner
#   # needs raw neural scores but the old prediction file lacks TopK_Scores, rerun
#   # the same already-selected neural configuration with the updated evaluator.
#   if python - "$SELECTION_MANIFEST" "$TEST_ROOT" <<'PY'
# import json
# import sys
# from pathlib import Path
# import pandas as pd

# manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
# test_root = Path(sys.argv[2])
# hybrid = manifest.get("hybrid", {})
# neural = manifest.get("xlmr_lora_projection", {})
# needs_raw = hybrid.get("configuration", {}).get("neural_score_mode") == "raw"
# name = neural.get("configuration_name")
# if not name:
#     raise SystemExit(2)
# predictions = test_root / "xlmr_lora_projection" / name / "sentence_predictions.csv"
# if not predictions.is_file():
#     raise SystemExit(1)
# if needs_raw:
#     columns = pd.read_csv(predictions, nrows=1).columns
#     if "TopK_Scores" not in columns:
#         raise SystemExit(1)
# raise SystemExit(0)
# PY
#   then
#     echo "Selected neural test predictions are already usable."
#   else
#     force_flag="--force_test"
#     echo "Generating or refreshing the selected neural test predictions."
#   fi

#   python src/select_dev_configs_and_evaluate_test.py \
#     --dev_root results/dev \
#     --test_root "$TEST_ROOT" \
#     --output_root "$SELECTION_ROOT" \
#     --test_csv "$TEST_CSV" \
#     --evaluate_test \
#     --family xlmr_lora_projection \
#     $force_flag
# }

# run_selected_test() {
#   refresh_selection_manifest
#   check_base_components

#   python - "$SELECTION_MANIFEST" <<'PY'
# import json
# import sys
# from pathlib import Path

# data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
# if "hybrid" not in data:
#     raise SystemExit(
#         "No development-selected hybrid configuration exists. Run with "
#         "HYBRID_ACTION=dev first."
#     )
# print("Selected hybrid configuration:", data["hybrid"]["configuration_name"])
# PY

#   ensure_selected_neural_test_predictions

#   # The preceding selector call rewrites the shared manifest; rebuild once more
#   # so all current dev families, including hybrid, are represented consistently.
#   refresh_selection_manifest

#   python src/select_dev_configs_and_evaluate_test.py \
#     --dev_root results/dev \
#     --test_root "$TEST_ROOT" \
#     --output_root "$SELECTION_ROOT" \
#     --test_csv "$TEST_CSV" \
#     --evaluate_test \
#     --family hybrid
# }

# case "$ACTION" in
#   dev)
#     run_dev_search
#     ;;
#   test)
#     run_selected_test
#     ;;
#   all)
#     run_dev_search
#     run_selected_test
#     ;;
#   *)
#     echo "Unknown HYBRID_ACTION=$ACTION; expected dev, test, or all." >&2
#     exit 2
#     ;;
# esac

# echo "Hybrid reranking workflow completed: $ACTION"

####################################
#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV="${TRAIN_CSV:-data/train_fit.csv}"
DEV_CSV="${DEV_CSV:-data/dev.csv}"
TEST_CSV="${TEST_CSV:-data/test.csv}"
DEV_ROOT="${HYBRID_DEV_ROOT:-results/dev/hybrid}"
TEST_ROOT="${TEST_ROOT:-results/test}"
SELECTION_ROOT="${SELECTION_ROOT:-results/dev_selection}"
SELECTION_MANIFEST="$SELECTION_ROOT/selected_configs.json"
IBM_CACHE_DIR="${IBM_CACHE_DIR:-results/cache/hybrid_ibm1}"
NEURAL_SCORE_CACHE_DIR="${NEURAL_SCORE_CACHE_DIR:-results/cache/hybrid_neural_scores}"
ACTION="${HYBRID_ACTION:-all}"

for required_file in \
  "$TRAIN_CSV" \
  "$DEV_CSV" \
  "$TEST_CSV" \
  src/hybrid_lexical_neural_rerank.py \
  src/word_alignment_baseline.py \
  src/previous_pipeline_baseline.py \
  src/select_dev_configs_and_evaluate_test.py; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

mkdir -p \
  "$DEV_ROOT" \
  "$TEST_ROOT" \
  "$SELECTION_ROOT" \
  "$IBM_CACHE_DIR" \
  "$NEURAL_SCORE_CACHE_DIR"

refresh_selection_manifest() {
  python src/select_dev_configs_and_evaluate_test.py \
    --dev_root results/dev \
    --test_root "$TEST_ROOT" \
    --output_root "$SELECTION_ROOT" \
    --test_csv "$TEST_CSV"
}

check_base_components() {
  python - "$SELECTION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
missing = [
    family
    for family in ("ibm1", "xlmr_lora_projection")
    if family not in data
]
if missing:
    raise SystemExit(
        "Missing development-selected base families: "
        + ", ".join(missing)
        + ". Complete their development experiments and rebuild the selection manifest."
    )
print("Selected IBM1 component:", data["ibm1"]["configuration_name"])
print("Selected XLM-R component:", data["xlmr_lora_projection"]["configuration_name"])
PY
}

# Regenerate only the already-selected development configurations when their
# legacy artifacts lack candidate indices/scores or retrieval embeddings. This
# does not reopen model selection and does not evaluate any new variant.
ensure_selected_dev_artifacts() {
  python - "$SELECTION_MANIFEST" "$DEV_CSV" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

manifest_path = Path(sys.argv[1])
dev_csv = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

requirements = {
    "ibm1": {
        "columns": {"TopK_indices", "TopK_Scores"},
        "files": set(),
    },
    "xlmr_lora_projection": {
        "columns": {"TopK_indices", "TopK_Scores"},
        "files": {"retrieval_embeddings.npz"},
    },
}

for family, requirement in requirements.items():
    choice = manifest[family]
    metrics_path = Path(choice["dev_metrics_path"])
    output_dir = metrics_path.parent
    predictions_path = output_dir / "sentence_predictions.csv"

    missing = []
    if not predictions_path.is_file():
        missing.append(str(predictions_path))
    else:
        columns = set(pd.read_csv(predictions_path, nrows=1).columns)
        missing_columns = sorted(requirement["columns"] - columns)
        if missing_columns:
            missing.append(
                f"{predictions_path} columns: {', '.join(missing_columns)}"
            )
    for filename in requirement["files"]:
        path = output_dir / filename
        if not path.is_file():
            missing.append(str(path))

    if not missing:
        print(f"Selected {family} development artifacts are usable.")
        continue

    evaluator = choice.get("evaluator_script")
    cli_args = choice.get("evaluation_cli_args")
    if not evaluator or not isinstance(cli_args, list):
        raise SystemExit(
            f"Cannot regenerate selected {family} development artifacts: "
            "manifest lacks evaluator_script or evaluation_cli_args"
        )

    command = [
        sys.executable,
        str(evaluator),
        "--input_csv",
        str(dev_csv),
        "--split_name",
        "dev",
        "--configuration_name",
        str(choice["configuration_name"]),
        "--output_dir",
        str(output_dir),
        *[str(value) for value in cli_args],
    ]
    print(f"Refreshing selected {family} development artifacts because:")
    for item in missing:
        print("  -", item)
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)
PY
}

selected_test_artifacts_ok() {
  local family="$1"
  python - "$SELECTION_MANIFEST" "$TEST_ROOT" "$family" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
test_root = Path(sys.argv[2])
family = sys.argv[3]
choice = manifest.get(family, {})
name = choice.get("configuration_name")
if not name:
    raise SystemExit(1)
output_dir = test_root / family / name
predictions = output_dir / "sentence_predictions.csv"
metrics = output_dir / "metrics.json"
if not predictions.is_file() or not metrics.is_file():
    raise SystemExit(1)
columns = set(pd.read_csv(predictions, nrows=1).columns)
if not {"TopK_indices", "TopK_Scores"}.issubset(columns):
    raise SystemExit(1)
if family == "xlmr_lora_projection" and not (
    output_dir / "retrieval_embeddings.npz"
).is_file():
    raise SystemExit(1)
raise SystemExit(0)
PY
}

ensure_selected_test_artifacts() {
  local family
  local force_flag
  for family in ibm1 xlmr_lora_projection; do
    force_flag=""
    if selected_test_artifacts_ok "$family"; then
      echo "Selected $family test artifacts are already usable."
    else
      force_flag="--force_test"
      echo "Generating or refreshing selected $family test artifacts."
    fi

    python src/select_dev_configs_and_evaluate_test.py \
      --dev_root results/dev \
      --test_root "$TEST_ROOT" \
      --output_root "$SELECTION_ROOT" \
      --test_csv "$TEST_CSV" \
      --evaluate_test \
      --family "$family" \
      $force_flag
  done
}

run_dev_search() {
  echo "Refreshing development-selected base components."
  refresh_selection_manifest
  check_base_components
  ensure_selected_dev_artifacts

  # Regenerated artifacts preserve the same configurations and development
  # metrics, but rebuild the manifest to keep all paths current.
  refresh_selection_manifest
  check_base_components

  python src/hybrid_lexical_neural_rerank.py \
    --mode tune_dev \
    --train_csv "$TRAIN_CSV" \
    --input_csv "$DEV_CSV" \
    --split_name dev \
    --selection_manifest "$SELECTION_MANIFEST" \
    --output_dir "$DEV_ROOT" \
    --test_root "$TEST_ROOT" \
    --ibm_cache_dir "$IBM_CACHE_DIR" \
    --neural_score_cache_dir "$NEURAL_SCORE_CACHE_DIR" \
    --candidate_generators neural_first ibm1_first \
    --candidate_topks 5 10 \
    --neural_score_modes raw reciprocal_rank \
    --normalizations minmax rank \
    --formulas weighted_sum weighted_product \
    --length_penalties 0.0 0.1 \
    --weights \
      1.0,0.0,0.0 \
      0.75,0.25,0.0 \
      0.5,0.5,0.0 \
      0.25,0.75,0.0 \
      0.0,1.0,0.0 \
      0.5,0.4,0.1 \
    --eval_ks 1 5 10 \
    --prune_stale

  refresh_selection_manifest
}

run_selected_test() {
  refresh_selection_manifest
  check_base_components

  python - "$SELECTION_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
hybrid = data.get("hybrid")
if not isinstance(hybrid, dict):
    raise SystemExit(
        "No development-selected hybrid configuration exists. "
        "Run HYBRID_ACTION=dev first."
    )
configuration = hybrid["configuration"]
print("Selected hybrid configuration:", hybrid["configuration_name"])
print("Selected candidate generator:", configuration["candidate_generator"])
PY

  # Both selected base families receive their own single authorized test run.
  # The selected hybrid order then reuses those frozen artifacts.
  ensure_selected_test_artifacts
  refresh_selection_manifest

  python src/select_dev_configs_and_evaluate_test.py \
    --dev_root results/dev \
    --test_root "$TEST_ROOT" \
    --output_root "$SELECTION_ROOT" \
    --test_csv "$TEST_CSV" \
    --evaluate_test \
    --family hybrid
}

case "$ACTION" in
  dev)
    run_dev_search
    ;;
  test)
    run_selected_test
    ;;
  all)
    run_dev_search
    run_selected_test
    ;;
  *)
    echo "Unknown HYBRID_ACTION=$ACTION; expected dev, test, or all." >&2
    exit 2
    ;;
esac

echo "Hybrid stage-order workflow completed: $ACTION"
