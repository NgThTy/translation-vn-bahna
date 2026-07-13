# Bahnar--Vietnamese Cross-Lingual Sentence Retrieval

This repository contains code, data-processing scripts, and experiment outputs for the paper:

**Bahnar--Vietnamese Cross-Lingual Sentence Retrieval: A Low-Resource Case Study with Lexical, Neural, and Hybrid Methods**

The project studies **Bahnar--Vietnamese cross-lingual sentence retrieval** as a candidate-ranking task. Given a Bahnar query sentence, the system ranks Vietnamese candidate sentences and aims to place the gold Vietnamese counterpart as high as possible.

We compare lexical, edit-distance, static embedding alignment, word-alignment, multilingual encoder, dense retrieval, neural adaptation, full fine-tuning, and hybrid reranking methods. Our main proposed method is a **hybrid IBM1--XLM-R LoRA reranker**, where an XLM-R LoRA retriever first generates a top-$K$ candidate list and IBM1 lexical alignment scores are then used to rerank the candidates.

## Main Methods

The repository supports experiments for:

- Character n-gram TF-IDF and BM25 retrieval
- Levenshtein/edit-distance and token-overlap retrieval
- fastText with supervised Procrustes/VecMap-style alignment
- IBM Model 1 lexical alignment and sentence-level scoring
- Off-the-shelf multilingual encoders such as mBERT, XLM-R, MiniLM, and LaBSE
- Projection-based XLM-R adaptation with LoRA and projection heads
- Full-encoder XLM-R contrastive fine-tuning
- Hybrid IBM1--XLM-R LoRA reranking
- Transfer experiments for Khmer--Vietnamese, Lao--Vietnamese, and Zhuang--Chinese retrieval

## Task Definition

For each source-language query sentence `q`, the system ranks a candidate pool of target-language sentences `C = {c_1, ..., c_n}`.

For the main Bahnar--Vietnamese task:

- Each Bahnar sentence in the held-out test set is used as a query.
- All Vietnamese sentences in the same test set form the candidate pool.
- Each query has exactly one gold Vietnamese counterpart.
- Systems are evaluated with:
  - Accuracy@1
  - Recall@5
  - Recall@10
  - MRR

Although some older script names still use the word `translate`, the paper evaluates **sentence retrieval**, not machine translation.

## Repository Layout

The current repository is organized as follows:

```text
.
├── data/
│   ├── other_low_resource_generalization/
│   ├── trial/
│   ├── lexicon.csv
│   ├── lexicon_train.csv
│   ├── lexicon_test.csv
│   ├── src_vocab.csv
│   ├── tgt_vocab.csv
│   ├── train.csv
│   └── test.csv
├── results/
│   ├── alignment/
│   ├── baselines/
│   │   ├── bm25_char_2_5/
│   │   ├── bm25_char_3_6/
│   │   ├── tfidf_char_2_5/
│   │   ├── tfidf_char_3_6/
│   │   ├── fasttext_procrustes_*/
│   │   ├── previous_pipeline_xlmr_*/
│   │   ├── flores_*/
│   │   ├── other_zhuang_chinese_*/
│   │   └── token_jaccard_strip_accents_no_punct/
│   ├── models/
│   ├── sent_eval/
│   ├── sent_eval_rank/
│   ├── trial/
│   └── flores_generalization_summary.csv
├── src/
│   ├── align_embeddings.py
│   ├── collect_flores_generalization_results.py
│   ├── convert_flores_topk_indices_to_preds.py
│   ├── edit_distance_retrieval_baseline.py
│   ├── fasttext_procrustes_baseline.py
│   ├── frozen_xlmr_flores_retrieval.py
│   ├── full_encoder_contrastive_finetune_baseline.py
│   ├── hybrid_lexical_neural_rerank.py
│   ├── lexical_retrieval_baseline.py
│   ├── lora_projection_generalization_train_eval.py
│   ├── multilingual_encoder_baseline.py
│   ├── prepare_flores_generalization_pairs.py
│   ├── prepare_other_low_resource_data.py
│   ├── previous_pipeline_baseline.py
│   ├── split_lexicon_by_source.py
│   ├── test_embeddings.py
│   ├── train_embeddings.py
│   ├── translate_and_eval_sentences.py
│   └── word_alignment_baseline.py
├── run_edit_distance_baselines.sh
├── run_fasttext_procrustes_baselines.sh
├── run_flores_generalization_reviewer_response.sh
├── run_flores_ibm1_hybrid_rerank.sh
├── requirements.txt
├── README.md
└── .gitignore

# Development-based model selection for Bahnaric-Vietnamese retrieval

This patch removes test-set model selection for the lexical and edit-distance
families. All method, preprocessing, and parameter variants are compared on a
development split carved from `data/train.csv`. Only the development-selected
configuration is then evaluated on `data/test.csv`.

## Split policy

Create the split once:

```bash
python src/prepare_train_dev_split.py \
  --input_csv data/train.csv \
  --train_output data/train_fit.csv \
  --dev_output data/dev.csv \
  --manifest_output data/train_dev_split_manifest.json \
  --dev_ratio 0.10 \
  --seed 42
```

For 51,930 original training pairs and no duplicate-pair grouping adjustment,
this produces:

- `data/train_fit.csv`: 46,737 pairs
- `data/dev.csv`: 5,193 pairs
- `data/test.csv`: unchanged, 2,001 pairs

The split script never reads or modifies `data/test.csv`. It preserves each
Bahnaric-Vietnamese pair as one row. If exact duplicate bilingual pairs exist,
they are grouped so duplicates cannot appear in both train and development.
Otherwise, the script stratifies by source/domain metadata when available and
falls back to Vietnamese sentence-length bins. The manifest records the seed,
strategy, row counts, and SHA-256 checksums.

## Selection rule

The predefined primary selection metric is development `Top1_acc`
(Accuracy@1). Ties are resolved in this order:

1. higher development MRR;
2. higher development Recall@5;
3. lexicographically smaller configuration name.

The test set is not consulted during any tie-breaking or configuration choice.
The baseline evaluators reject a test run unless its parameters match
`results/dev_selection/selected_configs.json`.

## Lexical family

```bash
bash run_lexical_baselines.sh
```

This evaluates the declared TF-IDF/BM25, character n-gram, and preprocessing
variants on `data/dev.csv`, selects one configuration, and performs one test
run. Outputs are written to:

```text
results/dev/lexical/<configuration>/
results/test/lexical/<selected-configuration>/
results/dev_selection/
```

## Edit-distance family

RapidFuzz is used to make the full development candidate-pool evaluation
practical:

```bash
python -m pip install "rapidfuzz>=3.9,<4"
bash run_edit_distance_baselines.sh
```

Outputs are written to:

```text
results/dev/edit_distance/<configuration>/
results/test/edit_distance/<selected-configuration>/
results/dev_selection/
```

## Retraining policy

Lexical and edit-distance baselines have no learned model parameters, so no
retraining on `train_fit + dev` is performed after selection. Their vectorizers
or candidate statistics are constructed independently for the candidate pool
being evaluated, as required by the retrieval task. For learned families added
later, the manuscript and code must explicitly state whether the selected
configuration is retrained on `train_fit + dev` before its single test run.

## Table 2 ordering comparison

After every Table 2 family has a dev-selected test result, prepare a CSV such as:

```csv
family,old_table2_rank,old_test_accuracy_at_1
lexical,8,0.1234
edit_distance,7,0.1456
```

Then run:

```bash
python src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root results/dev_selection \
  --old_table2_csv results/dev_selection/old_table2.csv
```

The script writes:

```text
results/dev_selection/all_dev_variants.csv
results/dev_selection/selected_configs.json
results/dev_selection/dev_selected_test_results.csv
results/dev_selection/table2_ordering_comparison.csv
results/dev_selection/table2_ordering_summary.json
```

`table2_ordering_summary.json` states whether the original ordering is
preserved once complete results for all old Table 2 families are available.

## FastText + Procrustes development selection

The FastText/Procrustes family follows the same held-out selection protocol as
the lexical and edit-distance families.

### Fitting and refitting policy

- `data/train_fit.csv` is used to train both monolingual FastText models, build
  IDF weights, and learn the supervised Procrustes mapping.
- `data/dev.csv` is used only to compare preprocessing, pooling, alignment, and
  retrieval variants.
- The selected configuration is retrained from scratch on `data/train_fit.csv`
  only and evaluated once on `data/test.csv`.
- The development set is not merged back into training for the final run. This
  policy is recorded as `train_fit_only_no_dev_refit` in every metrics file.
- `--workers 1` and seed `42` are used to make FastText training reproducible.

The selected family member is chosen by:

1. highest development `Top1_acc`;
2. highest development `MRR`;
3. highest development `Recall@5`;
4. ascending configuration name as a deterministic final tie-breaker.

### Run

Create the split first if it does not already exist:

```bash
python src/prepare_train_dev_split.py \
  --input_csv data/train.csv \
  --train_output data/train_fit.csv \
  --dev_output data/dev.csv \
  --manifest_output data/train_dev_split_manifest.json \
  --dev_ratio 0.10 \
  --seed 42
```

Run all FastText/Procrustes variants on development data and then run exactly
one development-selected test evaluation:

```bash
bash run_fasttext_procrustes_baselines.sh
```

Development variants are written under:

```text
results/dev/fasttext_procrustes/<configuration>/
```

The single final test result is written under:

```text
results/test/fasttext_procrustes/<selected-configuration>/
```

Selection summaries are updated in:

```text
results/dev_selection/all_dev_variants.csv
results/dev_selection/selected_configs.json
results/dev_selection/dev_selected_test_results.csv
results/dev_selection/table2_ordering_comparison.csv
```

Do not run the FastText/Procrustes evaluator directly on `data/test.csv`. A test
run requires the selection manifest and rejects any configuration that does not
exactly match the development-selected configuration.

# IBM1 development-selection workflow

This patch converts the IBM1 word-alignment family to the same reviewer-compliant
protocol used by the other Bahnaric-Vietnamese families.

## Protocol

- Training data: `data/train_fit.csv` only.
- Selection data: `data/dev.csv`.
- Held-out evaluation data: `data/test.csv`.
- Selection metric: Accuracy@1 (`Top1_acc`).
- Tie-breakers: MRR, Recall@5, then configuration name in ascending order.
- Refit policy: `train_fit_only_no_dev_refit`.
- Test policy: exactly one development-selected IBM1 configuration is evaluated.

The five development variants are:

1. `ibm1_raw`
2. `ibm1_strip_accents`
3. `ibm1_strip_accents_no_punct`
4. `ibm1_sym_strip_accents_no_punct`
5. `ibm1_sym_strip_accents_no_punct_lenpen`

## Updated files

- `src/word_alignment_baseline.py`
- `run_word_alignment_baselines.sh`
- `src/select_dev_configs_and_evaluate_test.py`

The central selector now registers the `ibm1` family.

## Run

Make sure these files already exist:

```text
data/train_fit.csv
data/dev.csv
data/test.csv
```

Then run:

```bash
bash run_word_alignment_baselines.sh
```

The script clears only stale IBM1-family outputs, evaluates the five variants on
development data, rebuilds the consolidated selection manifest from every family
currently under `results/dev`, and evaluates only the selected IBM1 variant on
the held-out test set.

## Outputs

```text
results/dev/ibm1/<configuration>/metrics.json
results/test/ibm1/<selected-configuration>/metrics.json
results/dev_selection/all_dev_variants.csv
results/dev_selection/selected_configs.json
results/dev_selection/dev_selected_test_results.csv
results/dev_selection/table2_ordering_comparison.csv
results/dev_selection/table2_ordering_summary.json
```

`selected_configs.json` is rewritten as a consolidated manifest. Existing
lexical, edit-distance, and FastText entries remain present when their valid
`split: dev` metric files still exist under `results/dev`.

Inspect the selected IBM1 configuration with:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(
    Path("results/dev_selection/selected_configs.json").read_text(encoding="utf-8")
)
print(json.dumps(manifest["ibm1"], indent=2))
PY
```

The full Table 2 ordering should be interpreted only after every reported method
family has been converted to the same development-selection protocol.

# Off-the-shelf encoder development selection

The ten declared encoder/preprocessing/retrieval variants are evaluated only on
`data/dev.csv`. The central selector chooses one configuration using `Top1_acc`,
then `MRR`, then `Recall@5`, then the ascending configuration name. Only that
configuration is evaluated on `data/test.csv`.

This family performs no fitting. Its policy is `off_the_shelf_no_training`.

## Resumable execution

A completed variant is identified by a nonempty
`results/dev/off_the_shelf/<variant>/metrics.json`. Existing completed runs are
not deleted.

Run one variant per short server allocation:

```bash
OFFSHELF_VARIANT=offtheshelf_mbert_cosine_raw \
  bash run_off_the_shelf_encoder_baselines.sh
```

After all ten variants are complete, run the script without an environment
variable. It skips completed variants, rebuilds the consolidated selection
manifest, and evaluates the selected off-the-shelf configuration on test:

```bash
bash run_off_the_shelf_encoder_baselines.sh
```

Embedding caches are stored under
`results/cache/off_the_shelf_embeddings/`. Cosine and CSLS variants with the
same model and preprocessing share the same cached embeddings. Cache files are
runtime artifacts and should not be committed.

Add this line to `.gitignore`:

```gitignore
results/cache/
```

Expected result locations:

```text
results/dev/off_the_shelf/<all-ten-configurations>/
results/test/off_the_shelf/<one-selected-configuration>/
results/dev_selection/selected_configs.json
results/dev_selection/dev_selected_test_results.csv
```

# XLM-R LoRA projection: development-selected evaluation

This patch converts the XLM-R LoRA/projection/Kabsch experiments from
**test-set model selection** to **development-set model selection**.

## Data policy

- `data/train_fit.csv`: train LoRA adapters and projection heads; fit any learned alignment artifacts.
- `data/dev.csv`: compare all architecture, preprocessing, pooling, Kabsch, retrieval, epoch, sample-size, and hyperparameter variants.
- `data/test.csv`: evaluate only the exact development-selected configuration.
- Final-model policy: `train_fit_only_no_dev_refit`.

The updated training script writes `training_manifest.json`. The evaluator
requires this manifest by default, so a legacy checkpoint trained on the original
`data/train.csv` cannot silently be reported as reviewer-compliant.

## Files

- `src/previous_pipeline_baseline.py`: dev/test evaluator with exact test authorization.
- `src/lora_projection_generalization_train_eval.py`: train on `train_fit`, save provenance and resumable epoch checkpoints, evaluate on dev only.
- `src/select_dev_configs_and_evaluate_test.py`: registers the `xlmr_lora_projection` family.
- `run_previous_pipeline_variant.sh`: shared resumable variant runner.
- `run_previous_pipeline_baselines.sh`: six core variants.
- `run_previous_pipeline_additional_experiments.sh`: four no-Kabsch pooling variants.
- `run_previous_pipeline_clarification_experiments.sh`: two sentence-mean Kabsch variants.
- `run_previous_pipeline_token_level_kabsch_experiments.sh`: two token-level Kabsch variants.
- `run_previous_pipeline_dev_selection.sh`: verifies the 14-variant sweep, selects on dev, and performs one test evaluation.

## The 14 unique default variants

The old clarification script duplicated two token-mean variants already present
in the main baseline script. The new sweep keeps 14 unique configurations:

1. sentence-level Kabsch, token mean, cosine
2. sentence-level Kabsch, token mean, CSLS
3. no Kabsch, token mean, cosine
4. no Kabsch, token mean, CSLS
5. sentence-level Kabsch, token IDF, cosine
6. sentence-level Kabsch, token IDF, CSLS
7. no Kabsch, token IDF, cosine
8. no Kabsch, token IDF, CSLS
9. sentence-level Kabsch, sentence mean, cosine
10. sentence-level Kabsch, sentence mean, CSLS
11. no Kabsch, sentence mean, cosine
12. no Kabsch, sentence mean, CSLS
13. token-level Kabsch, token mean, CSLS
14. token-level Kabsch, token IDF, CSLS

All are written under:

```text
results/dev/xlmr_lora_projection/<configuration>/
```

Only the development winner is written under:

```text
results/test/xlmr_lora_projection/<selected-configuration>/
```

## 1. Retrain checkpoints on train_fit

A checkpoint trained on the original `data/train.csv` has seen rows now assigned
to `data/dev.csv`, so it should not be used for the reviewer-response result.
Train a replacement checkpoint:

```bash
python src/lora_projection_generalization_train_eval.py \
  --train_csv data/train_fit.csv \
  --dev_csv data/dev.csv \
  --checkpoint_dir results/models/reviewer_xlmr_50ep \
  --dev_output_root results/dev/xlmr_lora_projection \
  --configuration_prefix trainfit_xlmr_50ep_token_mean_no_kabsch \
  --pooling token_mean \
  --epochs 50 \
  --batch_size 16 \
  --eval_batch_size 8 \
  --grad_accum_steps 2 \
  --eval_retrievals both \
  --embedding_cache_dir results/cache/xlmr_lora_projection_embeddings
```

The final checkpoint is:

```text
results/models/reviewer_xlmr_50ep/checkpoint_final/
```

It contains:

```text
src_proj.pt
tgt_proj.pt
src_adapters/
tgt_adapters/
training_manifest.json
```

Set the runner environment to that directory:

```bash
export PROJ_DIR=results/models/reviewer_xlmr_50ep/checkpoint_final
```

### Selecting epoch count, sample size, and LoRA hyperparameters

Run additional training configurations on the same `train_fit/dev` split. For
example, compare 10 and 20 epochs:

```bash
python src/lora_projection_generalization_train_eval.py \
  --train_csv data/train_fit.csv \
  --dev_csv data/dev.csv \
  --checkpoint_dir results/models/reviewer_xlmr_10ep \
  --configuration_prefix trainfit_xlmr_10ep_token_mean_no_kabsch \
  --epochs 10 \
  --pooling token_mean \
  --eval_retrievals both

python src/lora_projection_generalization_train_eval.py \
  --train_csv data/train_fit.csv \
  --dev_csv data/dev.csv \
  --checkpoint_dir results/models/reviewer_xlmr_20ep \
  --configuration_prefix trainfit_xlmr_20ep_token_mean_no_kabsch \
  --epochs 20 \
  --pooling token_mean \
  --eval_retrievals both
```

Use `--train_sample_size`, `--lora_r`, `--lora_alpha`, `--projection_dim`,
`--lr`, or `--temperature` for additional declared variants. Their development
metrics are automatically discovered by the central selector.

### Resume after a wall-time limit

Each completed epoch is saved as:

```text
results/models/reviewer_xlmr_50ep/checkpoint_epoch_<N>/
```

Resume from an epoch checkpoint:

```bash
python src/lora_projection_generalization_train_eval.py \
  ...same arguments... \
  --resume_from results/models/reviewer_xlmr_50ep/checkpoint_epoch_12
```

## 2. Refit Kabsch without dev/test leakage

`R.npy` and `t.npy` must be learned from `train_fit.csv` or from an independent
training lexicon that contains no development or test pairs. Use the repository's
alignment-training code to rebuild them. Place an `alignment_manifest.json` in
the alignment directory with at least:

```json
{
  "fit_policy": "train_fit_only",
  "fit_csv": "data/train_fit.csv",
  "fit_csv_sha256": "<sha256>",
  "alignment_sample_size": 10000
}
```

Then set:

```bash
export ALIGN_DIR=results/alignment/reviewer_xlmr_10K_50ep_train_fit
export REQUIRE_ALIGNMENT_MANIFEST=1
```

Without an alignment manifest, the evaluator records hashes of `R.npy` and
`t.npy` but warns that provenance could not be verified.

## 3. Run development variants

The scripts are resumable: a configuration is skipped only when its
`metrics.json` exists and is nonempty.

```bash
bash run_previous_pipeline_baselines.sh
bash run_previous_pipeline_additional_experiments.sh
bash run_previous_pipeline_clarification_experiments.sh
bash run_previous_pipeline_token_level_kabsch_experiments.sh
```

For short server allocations, run one variant per job:

```bash
XLMR_VARIANT=previous_pipeline_xlmr_50ep_10K_kabsch_token_mean_csls_lora \
  bash run_previous_pipeline_baselines.sh
```

Do not remove `results/dev/xlmr_lora_projection` between jobs.

## 4. Select on dev and evaluate once on test

After all 14 default variants are complete:

```bash
bash run_previous_pipeline_dev_selection.sh
```

This command:

1. verifies the 14 required development results;
2. scans all valid families under `results/dev`;
3. selects the XLM-R LoRA winner by `Top1_acc`, then `MRR`, then `Recall@5`, then configuration name;
4. rewrites the shared consolidated `selected_configs.json`; and
5. evaluates exactly the selected XLM-R LoRA configuration on test.

An existing compatible test result is skipped. To deliberately rerun it:

```bash
FORCE_TEST=1 bash run_previous_pipeline_dev_selection.sh
```

## 5. Inspect results

```bash
python - <<'PY'
import json
from pathlib import Path

selection = json.loads(
    Path("results/dev_selection/selected_configs.json").read_text()
)
print(json.dumps(selection["xlmr_lora_projection"], indent=2))
PY

find results/test/xlmr_lora_projection -name metrics.json -print
```

## Important limitation

The patch validates syntax and the selection/authorization structure. It cannot
run the complete 14-variant GPU experiment without your checkpoints, adapters,
alignment matrices, model downloads, and hardware. You must execute the sweep
and inspect the produced metrics on your server.

# Full-encoder family: development-only selection

This patch changes the full-encoder family so that the held-out test set is not
used for model, checkpoint, preprocessing, or retrieval selection.

## Declared search space

The runner defines six training configurations covering:

- sentence-mean and token-mean pooling;
- MLP, linear, and identity projection choices;
- projection dimensions 128 and 256;
- symmetric and one-way InfoNCE;
- temperatures 0.05 and 0.07;
- learning rates 5e-6 and 1e-5.

Each training run is continued to epoch 10 and evaluated on `data/dev.csv` at
epochs 3, 5, and 10 with cosine and CSLS. Therefore, six expensive training
runs produce 36 development-result configurations without retraining separately
for every checkpoint/retrieval combination.

The training policy is fixed as:

```
train_fit_only_no_dev_refit
```

`data/dev.csv` is used only for configuration/checkpoint selection. The final
selected model is not refit on dev.

## Two-hour jobs and two GPUs

Submit the array job:

```bash
sbatch submit_full_encoder_contrastive_baseline.slurm
```

The array has six tasks and `%2` allows two tasks to run concurrently, one GPU
per task. Re-submit the same array job until all tasks are complete. Each task
uses `checkpoint_latest` to resume and skips completed development variants.

A single local/server run can target one configuration:

```bash
FULL_ENCODER_CONFIG=token_mlp256_sym_t007_lr1e5 \
  bash run_full_encoder_contrastive_baselines.sh
```

Check completion:

```bash
find results/dev/full_encoder -mindepth 2 -maxdepth 2 \
  -name metrics.json -size +0c | wc -l
```

The declared grid produces 36 files.

## Selection and one test evaluation

After all development results exist:

```bash
sbatch submit_full_encoder_selection.slurm
```

or:

```bash
FULL_ENCODER_ACTION=select bash run_full_encoder_contrastive_baselines.sh
```

The shared selector rebuilds `results/dev_selection/selected_configs.json` from
all families under `results/dev`, selects the full-encoder configuration by
Top1 accuracy, MRR, Recall@5, and deterministic name tie-break, and evaluates
only that selected configuration on `data/test.csv`.

The evaluation script refuses a test command when the configuration name,
checkpoint, or arguments differ from the selection manifest.

# Hybrid IBM1 + XLM-R LoRA development selection

This patch converts the hybrid reranking family to the same reviewer-compliant
protocol as the other families:

1. IBM1 is selected using the IBM1 family results on `data/dev.csv`.
2. The XLM-R LoRA candidate generator is selected using the XLM-R family
   results on `data/dev.csv`.
3. Only those two development-selected components are used to build the hybrid.
4. Hybrid-only choices are tuned on `data/dev.csv`.
5. Exactly one selected hybrid configuration is evaluated on `data/test.csv`.

The old directory

```text
results/baselines/hybrid_ibm1_xlmr_lora_no_kabsch_csls_rerank_test_fixed
```

must not be treated as final reviewer-response evidence when its components or
weights were chosen using test performance.

## Files

```text
src/hybrid_lexical_neural_rerank.py
src/select_dev_configs_and_evaluate_test.py
src/previous_pipeline_baseline.py
run_hybrid_rerank.sh
```

`previous_pipeline_baseline.py` now writes `TopK_indices` and `TopK_Scores` in
addition to `TopK_Preds`. Candidate indices avoid ambiguity when Vietnamese
sentences are duplicated, while raw neural scores permit direct score
interpolation. Older neural prediction files remain usable through rank-based
neural scores.

## Prerequisites

Complete development selection for both component families first:

```text
results/dev/ibm1/<configuration>/metrics.json
results/dev/xlmr_lora_projection/<configuration>/metrics.json
```

The selected XLM-R development directory must also contain:

```text
sentence_predictions.csv
```

The hybrid script reads the selected component names and configurations from:

```text
results/dev_selection/selected_configs.json
```

It does not accept arbitrary IBM1 or neural components.

## Apply the patch

```bash
unzip hybrid_dev_selection_patch.zip

cp hybrid_dev_selection_patch/src/hybrid_lexical_neural_rerank.py src/
cp hybrid_dev_selection_patch/src/select_dev_configs_and_evaluate_test.py src/
cp hybrid_dev_selection_patch/src/previous_pipeline_baseline.py src/
cp hybrid_dev_selection_patch/run_hybrid_rerank.sh .
```

Validate syntax:

```bash
python -m py_compile \
  src/hybrid_lexical_neural_rerank.py \
  src/select_dev_configs_and_evaluate_test.py \
  src/previous_pipeline_baseline.py

bash -n run_hybrid_rerank.sh
```

The IBM cache should normally be ignored by Git:

```gitignore
results/cache/hybrid_ibm1/
```

## Run the complete hybrid workflow

```bash
bash run_hybrid_rerank.sh
```

This performs the following sequence:

1. Rebuild the shared selection manifest from all current development results.
2. Confirm that `ibm1` and `xlmr_lora_projection` have development-selected
   configurations.
3. Train or load the selected IBM1 model using only `data/train_fit.csv`.
4. Load development candidates from the selected XLM-R configuration.
5. Evaluate the declared hybrid grid on `data/dev.csv`.
6. Rebuild the shared manifest so it contains the selected hybrid.
7. Ensure that the already-selected XLM-R configuration has produced its one
   authorized test candidate file.
8. Evaluate exactly one selected hybrid configuration on `data/test.csv`.

Run only development tuning:

```bash
HYBRID_ACTION=dev bash run_hybrid_rerank.sh
```

Run only the final selected test evaluation after development tuning:

```bash
HYBRID_ACTION=test bash run_hybrid_rerank.sh
```

## Default hybrid search space

The base component choices are fixed to the development winners:

```text
IBM1 component: selected_configs.json["ibm1"]
Neural component: selected_configs.json["xlmr_lora_projection"]
```

The hybrid-only grid covers:

```text
candidate top-k:       5, 10
neural score:          raw score, reciprocal rank
normalization:         min-max, rank
reranking formula:     weighted sum, weighted product
length penalty:        0.0, 0.1
weights (a,b,g):       (1,0,0)
                       (0.75,0.25,0)
                       (0.5,0.5,0)
                       (0.25,0.75,0)
                       (0,1,0)
                       (0.5,0.4,0.1)
```

Here:

```text
final score = neural contribution + IBM1 contribution + optional string contribution
alpha = neural weight
beta  = IBM1 weight
gamma = SequenceMatcher string-similarity weight
```

When `TopK_Scores` is available, the default grid contains 192 configurations.
When an older neural prediction file has no raw scores, the script excludes the
raw-score mode and evaluates 96 rank-based configurations. This behavior is
recorded in `search_manifest.json` and every `metrics.json`.

The central selector chooses with:

```text
Top1_acc
→ MRR
→ Recall@5
→ configuration name
```

## Result structure

Development grid:

```text
results/dev/hybrid/
├── hybrid_dev_grid.csv
├── search_manifest.json
├── hybrid_k5_.../
│   └── metrics.json
└── hybrid_k10_.../
    └── metrics.json
```

The development winner also receives a prediction file for auditing:

```text
results/dev/hybrid/<best-configuration>/sentence_predictions.csv
```

Final test output contains exactly one selected configuration:

```text
results/test/hybrid/<selected-configuration>/
├── metrics.json
└── sentence_predictions.csv
```

Consolidated selection files remain under:

```text
results/dev_selection/
├── selected_configs.json
├── all_dev_variants.csv
├── dev_selected_test_results.csv
├── table2_ordering_comparison.csv
└── table2_ordering_summary.json
```

## IBM1 fitting policy

The policy is fixed as:

```text
train_fit_only_no_dev_refit
```

IBM1 is trained only on `data/train_fit.csv`. `data/dev.csv` is used only for
component selection and hybrid parameter tuning. `data/test.csv` is read only
after the hybrid configuration has been frozen in the shared manifest.

The complete IBM translation tables are cached under:

```text
results/cache/hybrid_ibm1/
```

The cache key includes the SHA-256 hash of `train_fit.csv` and the selected IBM1
configuration, so stale models are not silently reused after data or
preprocessing changes.

## Test guard

The final test evaluator verifies all of the following against
`selected_configs.json`:

- hybrid configuration name;
- selected IBM1 component and its preprocessing/method;
- selected XLM-R component;
- candidate top-k;
- neural score representation;
- normalization;
- reranking formula;
- interpolation weights;
- length penalty.

A direct test command with different arguments is rejected.

## Raw neural score support

After replacing `previous_pipeline_baseline.py`, newly generated XLM-R
prediction files contain:

```text
TopK_Preds
TopK_indices
TopK_Scores
```

If the selected development prediction file was created before this patch, the
hybrid script prints a warning and tunes only rank-based neural scores. To also
compare raw-score interpolation, rerun the same already-selected XLM-R
development configuration with the updated evaluator. This does not change the
selected neural configuration; it only regenerates its prediction artifact.

# Hybrid IBM1–XLM-R LoRA stage-order patch

This patch turns the stage order into a development-selected hybrid choice.
It compares:

1. `neural_first`: XLM-R LoRA generates top-k candidates, then IBM1 and neural
   scores rerank them.
2. `ibm1_first`: IBM1 generates top-k candidates from the full candidate pool,
   then XLM-R LoRA and IBM1 scores rerank them.

The held-out test set is used only once, for the single hybrid configuration
selected on development data.

## Files

```text
src/hybrid_lexical_neural_rerank.py
src/previous_pipeline_baseline.py
src/word_alignment_baseline.py
src/select_dev_configs_and_evaluate_test.py
run_hybrid_rerank.sh
run_word_alignment_baselines.sh
```

## Base-artifact changes

`word_alignment_baseline.py` now saves:

```text
TopK_Preds
TopK_indices
TopK_Scores
```

`previous_pipeline_baseline.py` now saves the same top-k fields plus:

```text
retrieval_embeddings.npz
```

The embedding archive contains the exact source and target embeddings used by
the selected XLM-R retrieval configuration. The hybrid evaluator uses them to
score IBM1-first candidates that may not appear in XLM-R's original top-k.

## Apply

From the repository root:

```bash
cp hybrid_stage_order_dev_selection_patch/src/hybrid_lexical_neural_rerank.py src/
cp hybrid_stage_order_dev_selection_patch/src/previous_pipeline_baseline.py src/
cp hybrid_stage_order_dev_selection_patch/src/word_alignment_baseline.py src/
cp hybrid_stage_order_dev_selection_patch/src/select_dev_configs_and_evaluate_test.py src/
cp hybrid_stage_order_dev_selection_patch/run_hybrid_rerank.sh .
cp hybrid_stage_order_dev_selection_patch/run_word_alignment_baselines.sh .
```

Validate:

```bash
python -m py_compile \
  src/hybrid_lexical_neural_rerank.py \
  src/previous_pipeline_baseline.py \
  src/word_alignment_baseline.py \
  src/select_dev_configs_and_evaluate_test.py

bash -n run_hybrid_rerank.sh
bash -n run_word_alignment_baselines.sh
```

## Run development selection

The selected IBM1 and XLM-R development configurations must already exist in:

```text
results/dev_selection/selected_configs.json
```

Then run:

```bash
HYBRID_ACTION=dev bash run_hybrid_rerank.sh
```

The runner checks the selected base artifacts. When an old selected result lacks
`TopK_indices`, `TopK_Scores`, or `retrieval_embeddings.npz`, it reruns only
that already-selected development configuration. It does not rerun all IBM1 or
XLM-R variants and does not change base-family selection.

The default search has:

```text
2 stage orders
× 2 candidate top-k values
× 2 neural score modes
× 2 normalization modes
× 2 formulas
× 2 length penalties
× 6 weight settings
= 384 development configurations
```

These configurations reuse cached IBM1 tables and a cached full XLM-R score
matrix.

## Run selected test evaluation

After development selection:

```bash
HYBRID_ACTION=test bash run_hybrid_rerank.sh
```

The runner first ensures that the single development-selected IBM1 and XLM-R
base configurations have authorized test artifacts. It then evaluates only the
single selected hybrid stage order and parameter configuration.

The complete workflow is:

```bash
bash run_hybrid_rerank.sh
```

## Main outputs

```text
results/dev/hybrid/hybrid_dev_grid.csv
results/dev/hybrid/stage_order_comparison.csv
results/dev/hybrid/search_manifest.json
results/dev/hybrid/<configuration>/metrics.json
results/test/hybrid/<selected-configuration>/metrics.json
results/test/hybrid/<selected-configuration>/sentence_predictions.csv
```

`stage_order_comparison.csv` keeps the best development configuration for each
stage order and candidate top-k. It reports first-stage recall/accuracy and
final reranked accuracy/MRR, which directly supports the reviewer response.

Each hybrid `metrics.json` retains top-level `Top1_acc`, `MRR`, and `Recall@5`
for the central selector, and also records:

```text
candidate_generator
first_stage_metrics
final_metrics
reranking_gain
first_stage_recall_at_k
rerank_top1_gain
rerank_mrr_gain
```

## Two-hour server limit

The IBM1 runner is resumable. Run one missing IBM1 variant per allocation:

```bash
IBM1_VARIANT=ibm1_sym_strip_accents_no_punct \
  bash run_word_alignment_baselines.sh
```

Completed variants are skipped and are never deleted.

The hybrid grid itself does not train XLM-R. Its expensive reusable artifacts
are built once and cached under:

```text
results/cache/hybrid_ibm1/
results/cache/hybrid_neural_scores/
```

Add both directories to `.gitignore` when caches should not be versioned.

## Suggested reviewer explanation

> We use the XLM-R LoRA model as the first-stage candidate generator because
> the first stage is recall-oriented. The dense encoder captures sentence-level
> semantic similarity despite differences in surface form, word order, or
> incomplete lexical overlap, while IBM1 provides a complementary lexical
> consistency signal among retained candidates. To test this rationale rather
> than assume it, we additionally evaluate the reverse IBM1-first pipeline on
> the development set with identical candidate sizes and reranking choices. The
> stage order and all hybrid hyperparameters are selected only on development
> data, followed by one evaluation of the selected pipeline on the held-out test
> set.

