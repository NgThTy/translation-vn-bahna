# #!/usr/bin/env bash
# set -e

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1 \
#   --output_dir results/baselines/ibm1_raw \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10 \
#   --export_fastalign_input

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1 \
#   --output_dir results/baselines/ibm1_strip_accents \
#   --strip_accents \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1 \
#   --output_dir results/baselines/ibm1_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1_sym \
#   --output_dir results/baselines/ibm1_sym_strip_accents_no_punct \
#   --strip_accents \
#   --remove_punct \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

# python src/word_alignment_baseline.py \
#   --train_csv data/train.csv \
#   --test_csv data/test.csv \
#   --method ibm1_sym \
#   --output_dir results/baselines/ibm1_sym_strip_accents_no_punct_lenpen \
#   --strip_accents \
#   --remove_punct \
#   --length_penalty 0.1 \
#   --ibm_iters 5 \
#   --topk_eval 10 \
#   --eval_ks 1 5 10

######################################

# #!/usr/bin/env bash
# set -euo pipefail

# TRAIN_CSV="data/train_fit.csv"
# DEV_CSV="data/dev.csv"
# TEST_CSV="data/test.csv"
# DEV_ROOT="results/dev/ibm1"
# TEST_ROOT="results/test/ibm1"
# SELECTION_ROOT="results/dev_selection"

# for required_file in "$TRAIN_CSV" "$DEV_CSV" "$TEST_CSV"; do
#   if [[ ! -f "$required_file" ]]; then
#     echo "Missing required file: $required_file" >&2
#     exit 1
#   fi
# done

# # Remove stale IBM1-family outputs so only the five declared variants
# # participate in this development selection run.
# rm -rf "$DEV_ROOT" "$TEST_ROOT"
# mkdir -p "$DEV_ROOT" "$TEST_ROOT" "$SELECTION_ROOT"

# COMMON_ARGS=(
#   --train_csv "$TRAIN_CSV"
#   --input_csv "$DEV_CSV"
#   --split_name dev
#   --ibm_iters 5
#   --topk_eval 10
#   --eval_ks 1 5 10
# )

# python src/word_alignment_baseline.py \
#   "${COMMON_ARGS[@]}" \
#   --method ibm1 \
#   --configuration_name ibm1_raw \
#   --output_dir "$DEV_ROOT/ibm1_raw" \
#   --export_fastalign_input

# python src/word_alignment_baseline.py \
#   "${COMMON_ARGS[@]}" \
#   --method ibm1 \
#   --strip_accents \
#   --configuration_name ibm1_strip_accents \
#   --output_dir "$DEV_ROOT/ibm1_strip_accents"

# python src/word_alignment_baseline.py \
#   "${COMMON_ARGS[@]}" \
#   --method ibm1 \
#   --strip_accents \
#   --remove_punct \
#   --configuration_name ibm1_strip_accents_no_punct \
#   --output_dir "$DEV_ROOT/ibm1_strip_accents_no_punct"

# python src/word_alignment_baseline.py \
#   "${COMMON_ARGS[@]}" \
#   --method ibm1_sym \
#   --strip_accents \
#   --remove_punct \
#   --configuration_name ibm1_sym_strip_accents_no_punct \
#   --output_dir "$DEV_ROOT/ibm1_sym_strip_accents_no_punct"

# python src/word_alignment_baseline.py \
#   "${COMMON_ARGS[@]}" \
#   --method ibm1_sym \
#   --strip_accents \
#   --remove_punct \
#   --length_penalty 0.1 \
#   --configuration_name ibm1_sym_strip_accents_no_punct_lenpen \
#   --output_dir "$DEV_ROOT/ibm1_sym_strip_accents_no_punct_lenpen"

# # The central selector scans every family under results/dev, rewrites the shared
# # consolidated manifest, and evaluates only the dev-selected IBM1 variant on test.
# python src/select_dev_configs_and_evaluate_test.py \
#   --dev_root results/dev \
#   --test_root results/test \
#   --output_root "$SELECTION_ROOT" \
#   --test_csv "$TEST_CSV" \
#   --evaluate_test \
#   --family ibm1

######################################
# #!/usr/bin/env bash
# set -euo pipefail

# TRAIN_CSV="data/train_fit.csv"
# DEV_CSV="data/dev.csv"
# TEST_CSV="data/test.csv"
# DEV_ROOT="results/dev/ibm1"
# TEST_ROOT="results/test/ibm1"
# SELECTION_ROOT="results/dev_selection"

# VARIANTS=(
#   ibm1_raw
#   ibm1_strip_accents
#   ibm1_strip_accents_no_punct
#   ibm1_sym_strip_accents_no_punct
#   ibm1_sym_strip_accents_no_punct_lenpen
# )

# for required_file in "$TRAIN_CSV" "$DEV_CSV" "$TEST_CSV"; do
#   if [[ ! -f "$required_file" ]]; then
#     echo "Missing required file: $required_file" >&2
#     exit 1
#   fi
# done

# # Do not delete previous completed results.
# mkdir -p "$DEV_ROOT" "$TEST_ROOT" "$SELECTION_ROOT"

# COMMON_ARGS=(
#   --train_csv "$TRAIN_CSV"
#   --input_csv "$DEV_CSV"
#   --split_name dev
#   --ibm_iters 5
#   --topk_eval 10
#   --eval_ks 1 5 10
# )

# run_variant() {
#   local name="$1"
#   shift

#   local output_dir="$DEV_ROOT/$name"
#   local metrics_file="$output_dir/metrics.json"

#   if [[ -s "$metrics_file" ]]; then
#     echo "[SKIP] Completed variant: $name"
#     return
#   fi

#   # Remove only an incomplete output for this particular variant.
#   if [[ -d "$output_dir" ]]; then
#     echo "[CLEAN] Removing incomplete output: $output_dir"
#     rm -rf "$output_dir"
#   fi

#   echo "[RUN] $name"

#   python src/word_alignment_baseline.py \
#     "${COMMON_ARGS[@]}" \
#     --configuration_name "$name" \
#     --output_dir "$output_dir" \
#     "$@"
# }

# run_named_variant() {
#   local name="$1"

#   case "$name" in
#     ibm1_raw)
#       run_variant "$name" \
#         --method ibm1 \
#         --export_fastalign_input
#       ;;

#     ibm1_strip_accents)
#       run_variant "$name" \
#         --method ibm1 \
#         --strip_accents
#       ;;

#     ibm1_strip_accents_no_punct)
#       run_variant "$name" \
#         --method ibm1 \
#         --strip_accents \
#         --remove_punct
#       ;;

#     ibm1_sym_strip_accents_no_punct)
#       run_variant "$name" \
#         --method ibm1_sym \
#         --strip_accents \
#         --remove_punct
#       ;;

#     ibm1_sym_strip_accents_no_punct_lenpen)
#       run_variant "$name" \
#         --method ibm1_sym \
#         --strip_accents \
#         --remove_punct \
#         --length_penalty 0.1
#       ;;

#     *)
#       echo "Unknown IBM1 variant: $name" >&2
#       exit 1
#       ;;
#   esac
# }

# # Run one specific variant:
# # IBM1_VARIANT=ibm1_sym_strip_accents_no_punct \
# #   bash run_word_alignment_baselines.sh
# if [[ -n "${IBM1_VARIANT:-}" ]]; then
#   run_named_variant "$IBM1_VARIANT"
#   exit 0
# fi

# # Without IBM1_VARIANT, run only variants that are still incomplete.
# for variant in "${VARIANTS[@]}"; do
#   run_named_variant "$variant"
# done

# # Never select from an incomplete set of variants.
# missing=0

# for variant in "${VARIANTS[@]}"; do
#   metrics_file="$DEV_ROOT/$variant/metrics.json"

#   if [[ ! -s "$metrics_file" ]]; then
#     echo "Missing completed result: $metrics_file" >&2
#     missing=1
#   fi
# done

# if [[ "$missing" -ne 0 ]]; then
#   echo "IBM1 development experiments are incomplete." >&2
#   echo "Selection and test evaluation will not run." >&2
#   exit 2
# fi

# echo "All five IBM1 development variants are complete."

# python src/select_dev_configs_and_evaluate_test.py \
#   --dev_root results/dev \
#   --test_root results/test \
#   --output_root "$SELECTION_ROOT" \
#   --test_csv "$TEST_CSV" \
#   --evaluate_test \
#   --family ibm1

#######################################
#!/usr/bin/env bash
set -euo pipefail

TRAIN_CSV="${TRAIN_CSV:-data/train_fit.csv}"
DEV_CSV="${DEV_CSV:-data/dev.csv}"
TEST_CSV="${TEST_CSV:-data/test.csv}"
DEV_ROOT="${IBM1_DEV_ROOT:-results/dev/ibm1}"
TEST_ROOT="${TEST_ROOT:-results/test}"
SELECTION_ROOT="${SELECTION_ROOT:-results/dev_selection}"

VARIANTS=(
  ibm1_raw
  ibm1_strip_accents
  ibm1_strip_accents_no_punct
  ibm1_sym_strip_accents_no_punct
  ibm1_sym_strip_accents_no_punct_lenpen
)

for required_file in \
  "$TRAIN_CSV" \
  "$DEV_CSV" \
  "$TEST_CSV" \
  src/word_alignment_baseline.py \
  src/select_dev_configs_and_evaluate_test.py; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 1
  fi
done

mkdir -p "$DEV_ROOT" "$TEST_ROOT" "$SELECTION_ROOT"

COMMON_ARGS=(
  --train_csv "$TRAIN_CSV"
  --input_csv "$DEV_CSV"
  --split_name dev
  --ibm_iters 5
  --topk_eval 10
  --eval_ks 1 5 10
)

run_variant() {
  local name="$1"
  shift
  local output_dir="$DEV_ROOT/$name"
  local metrics_file="$output_dir/metrics.json"

  if [[ -s "$metrics_file" ]]; then
    echo "[SKIP] Completed IBM1 variant: $name"
    return
  fi

  if [[ -d "$output_dir" ]]; then
    echo "[CLEAN] Removing incomplete output: $output_dir"
    rm -rf "$output_dir"
  fi

  echo "[RUN] IBM1 variant: $name"
  python src/word_alignment_baseline.py \
    "${COMMON_ARGS[@]}" \
    --configuration_name "$name" \
    --output_dir "$output_dir" \
    "$@"
}

run_named_variant() {
  local name="$1"
  case "$name" in
    ibm1_raw)
      run_variant "$name" --method ibm1 --export_fastalign_input
      ;;
    ibm1_strip_accents)
      run_variant "$name" --method ibm1 --strip_accents
      ;;
    ibm1_strip_accents_no_punct)
      run_variant "$name" --method ibm1 --strip_accents --remove_punct
      ;;
    ibm1_sym_strip_accents_no_punct)
      run_variant "$name" --method ibm1_sym --strip_accents --remove_punct
      ;;
    ibm1_sym_strip_accents_no_punct_lenpen)
      run_variant "$name" \
        --method ibm1_sym \
        --strip_accents \
        --remove_punct \
        --length_penalty 0.1
      ;;
    *)
      echo "Unknown IBM1 variant: $name" >&2
      exit 2
      ;;
  esac
}

# Use one variant per short server allocation, for example:
# IBM1_VARIANT=ibm1_sym_strip_accents_no_punct bash run_word_alignment_baselines.sh
if [[ -n "${IBM1_VARIANT:-}" ]]; then
  run_named_variant "$IBM1_VARIANT"
  exit 0
fi

for variant in "${VARIANTS[@]}"; do
  run_named_variant "$variant"
done

missing=0
for variant in "${VARIANTS[@]}"; do
  if [[ ! -s "$DEV_ROOT/$variant/metrics.json" ]]; then
    echo "Missing completed result: $DEV_ROOT/$variant/metrics.json" >&2
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo "IBM1 development experiments are incomplete; selection will not run." >&2
  exit 2
fi

python src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root "$TEST_ROOT" \
  --output_root "$SELECTION_ROOT" \
  --test_csv "$TEST_CSV" \
  --evaluate_test \
  --family ibm1
