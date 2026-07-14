# Bahnar–Vietnamese Cross-Lingual Sentence Retrieval

Code and experiment workflows for:

> **Bahnar–Vietnamese Cross-Lingual Sentence Retrieval: A Low-Resource Case Study with Lexical, Neural, and Hybrid Methods**

This repository studies **cross-lingual sentence retrieval** between Bahnar and Vietnamese. Given a Bahnar query sentence, a system ranks a pool of Vietnamese candidate sentences and should place the aligned Vietnamese sentence as high as possible.

The repository includes lexical, edit-distance, word-alignment, multilingual-encoder, adaptation, full fine-tuning, and hybrid reranking baselines. The main hybrid system combines **XLM-R LoRA sentence-level scores** with **IBM Model 1 lexical evidence** over a top-\(K\) candidate list.

> Although some legacy script names contain `translate`, the evaluated task is sentence retrieval, not machine translation.

---

## Highlights

- Reviewer-compliant **train/dev/test separation**
- Development-only model and hyperparameter selection
- One authorized test evaluation per method family
- Cosine, CSLS, and Artetxe–Schwenk ratio-margin retrieval
- Resumable LoRA and full-encoder training
- Reproducible manifests with configuration and file hashes
- Neural-first and IBM1-first hybrid stage-order comparison
- Transfer experiments for other low-resource language pairs

---

## Task

For each Bahnar query \(q_i\), the system ranks all Vietnamese candidates:

```text
C = {c_1, c_2, ..., c_n}
```

Each query has one gold Vietnamese counterpart. The main evaluation reports:

- Accuracy@1 (`Top1_acc`)
- Mean Reciprocal Rank (`MRR`)
- Recall@5
- Recall@10

The candidate pool contains all Vietnamese sentences in the evaluated split.

---

## Experimental Protocol

### Data splits

The reviewer-compliant protocol uses:

| Split | Purpose | Pairs |
|---|---|---:|
| `data/train_fit.csv` | Model fitting and learned alignment | 46,736 |
| `data/dev.csv` | Variant and hyperparameter selection | 5,194 |
| `data/test.csv` | One final evaluation per family | 2,001 |

The split is derived from the original 51,930 training pairs with seed `42`. Exact duplicate bilingual pairs are grouped so that duplicates cannot appear in both training and development data.

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

The split script does not read or modify `data/test.csv`.

### Selection rule

All families use the same development-selection rule:

1. higher development Accuracy@1;
2. higher development MRR;
3. higher development Recall@5;
4. lexicographically smaller configuration name.

The selected model is not refit on `train_fit + dev`. The fixed policy is:

```text
train_fit_only_no_dev_refit
```

### Test-set guard

The central selector writes:

```text
results/dev_selection/selected_configs.json
```

Test evaluators verify that the requested configuration and arguments exactly match the development-selected entry. Arbitrary test configurations are rejected.

---

## Retrieval Criteria

Single-vector retrieval families compare three criteria:

### Cosine similarity

```text
score(x, y) = cosine(x, y)
```

### CSLS

```text
score(x, y) = 2 * cosine(x, y) - r_x - r_y
```

### Ratio margin

Following Artetxe and Schwenk (2019):

```text
score(x, y) = cosine(x, y) / (0.5 * (r_x + r_y))
```

Here, `r_x` and `r_y` are mean similarities to the nearest `k` neighbors. The default is:

```text
neighborhood_k = 10
```

Neighborhood statistics are computed from the **complete source–target similarity matrix**, before top-\(K\) truncation.

The shared implementation is:

```text
src/retrieval_scoring.py
```

---

## Methods

The repository supports:

- character n-gram TF-IDF and BM25;
- Levenshtein distance and token-overlap retrieval;
- fastText with supervised Procrustes alignment;
- IBM Model 1 lexical alignment;
- off-the-shelf mBERT, XLM-R, MiniLM, and LaBSE encoders;
- XLM-R LoRA with projection and optional Kabsch alignment;
- full-encoder XLM-R contrastive fine-tuning;
- IBM1–XLM-R LoRA hybrid reranking;
- low-resource transfer experiments.

### Hybrid positioning

The hybrid follows the standard **sparse/lexical–dense interpolation reranking** pattern:

1. one component retrieves a top-\(K\) candidate list;
2. neural and lexical evidence are normalized;
3. their scores are interpolated to rerank the retained candidates.

The interpolation recipe itself is not claimed as novel. The task-specific design uses IBM1 bilingual translation probabilities as lexical evidence and XLM-R LoRA representations as dense semantic evidence.

Two stage orders are compared on development data:

- `neural_first`: XLM-R retrieves candidates, then IBM1 and neural scores rerank them;
- `ibm1_first`: IBM1 retrieves candidates, then IBM1 and neural scores rerank them.

Stage order and all hybrid hyperparameters are selected on `data/dev.csv`.

---

## Repository Layout

```text
.
├── data/
│   ├── train.csv
│   ├── train_fit.csv
│   ├── dev.csv
│   ├── test.csv
│   ├── train_dev_split_manifest.json
│   └── other_low_resource_generalization/
├── results/
│   ├── alignment/
│   ├── cache/
│   ├── dev/
│   ├── dev_selection/
│   ├── models/
│   └── test/
├── src/
│   ├── prepare_train_dev_split.py
│   ├── retrieval_scoring.py
│   ├── lexical_retrieval_baseline.py
│   ├── edit_distance_retrieval_baseline.py
│   ├── fasttext_procrustes_baseline.py
│   ├── word_alignment_baseline.py
│   ├── multilingual_encoder_baseline.py
│   ├── previous_pipeline_baseline.py
│   ├── lora_projection_generalization_train_eval.py
│   ├── full_encoder_contrastive_finetune_baseline.py
│   ├── hybrid_lexical_neural_rerank.py
│   └── select_dev_configs_and_evaluate_test.py
├── run_lexical_baselines.sh
├── run_edit_distance_baselines.sh
├── run_fasttext_procrustes_baselines.sh
├── run_word_alignment_baselines.sh
├── run_off_the_shelf_encoder_baselines.sh
├── run_previous_pipeline_baselines.sh
├── run_previous_pipeline_additional_experiments.sh
├── run_previous_pipeline_clarification_experiments.sh
├── run_previous_pipeline_token_level_kabsch_experiments.sh
├── run_full_encoder_contrastive_baselines.sh
├── run_hybrid_rerank.sh
├── requirements.txt
└── README.md
```

Large checkpoints, downloaded models, and caches are runtime artifacts and should not be committed.

---

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone <repository-url>
cd translation-vn-bahna

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the edit-distance baseline:

```bash
python -m pip install "rapidfuzz>=3.9,<4"
```

GPU-based methods require a compatible PyTorch, CUDA, and Transformers setup.

---

## Quick Start

### 1. Prepare the split

```bash
python src/prepare_train_dev_split.py \
  --input_csv data/train.csv \
  --train_output data/train_fit.csv \
  --dev_output data/dev.csv \
  --manifest_output data/train_dev_split_manifest.json \
  --dev_ratio 0.10 \
  --seed 42
```

### 2. Run development experiments

```bash
bash run_lexical_baselines.sh
bash run_edit_distance_baselines.sh
bash run_fasttext_procrustes_baselines.sh
bash run_word_alignment_baselines.sh
bash run_off_the_shelf_encoder_baselines.sh
```

Run the XLM-R LoRA development variants:

```bash
bash run_previous_pipeline_baselines.sh
bash run_previous_pipeline_additional_experiments.sh
bash run_previous_pipeline_clarification_experiments.sh
bash run_previous_pipeline_token_level_kabsch_experiments.sh
```

Run or resume full-encoder training:

```bash
bash run_full_encoder_contrastive_baselines.sh
```

### 3. Select configurations on development data

```bash
python src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root results/dev_selection \
  --test_csv data/test.csv
```

This command rebuilds the selection manifest but does not evaluate the test set.

### 4. Evaluate one selected configuration per family

Example:

```bash
python src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root results/dev_selection \
  --test_csv data/test.csv \
  --evaluate_test \
  --family off_the_shelf
```

Repeat only for the families to be reported.

### 5. Run hybrid development selection

The IBM1 and XLM-R LoRA families must already have valid development selections.

```bash
HYBRID_ACTION=dev bash run_hybrid_rerank.sh
```

Rebuild the selection manifest:

```bash
python src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root results/dev_selection \
  --test_csv data/test.csv
```

Evaluate the selected hybrid once:

```bash
HYBRID_ACTION=test bash run_hybrid_rerank.sh
```

---

## Family Workflows

### Lexical retrieval

```bash
bash run_lexical_baselines.sh
```

Outputs:

```text
results/dev/lexical/<configuration>/
results/test/lexical/<selected-configuration>/
```

### Edit distance

```bash
bash run_edit_distance_baselines.sh
```

Outputs:

```text
results/dev/edit_distance/<configuration>/
results/test/edit_distance/<selected-configuration>/
```

### fastText + Procrustes

```bash
bash run_fasttext_procrustes_baselines.sh
```

The default search evaluates seven base configurations with cosine, CSLS, and ratio margin:

```text
7 base configurations × 3 criteria = 21 development variants
```

The models, IDF statistics, and alignment are fitted only on `data/train_fit.csv`.

### IBM Model 1

```bash
bash run_word_alignment_baselines.sh
```

The declared development variants are:

```text
ibm1_raw
ibm1_strip_accents
ibm1_strip_accents_no_punct
ibm1_sym_strip_accents_no_punct
ibm1_sym_strip_accents_no_punct_lenpen
```

IBM1 is trained only on `data/train_fit.csv`.

### Off-the-shelf encoders

```bash
bash run_off_the_shelf_encoder_baselines.sh
```

The default grid evaluates:

```text
4 encoders × 2 preprocessing settings × 3 criteria
= 24 development variants
```

Run one variant in a short allocation:

```bash
OFFSHELF_VARIANT=offtheshelf_labse_margin_ratio_strip_accents_no_punct \
  bash run_off_the_shelf_encoder_baselines.sh
```

No model fitting is performed for this family.

### XLM-R LoRA projection

Run the four development runners:

```bash
bash run_previous_pipeline_baselines.sh
bash run_previous_pipeline_additional_experiments.sh
bash run_previous_pipeline_clarification_experiments.sh
bash run_previous_pipeline_token_level_kabsch_experiments.sh
```

The current declared search contains 24 unique variants spanning:

- pooling strategy;
- Kabsch usage and level;
- token weighting;
- cosine, CSLS, and ratio-margin scoring.

Run one variant:

```bash
XLMR_VARIANT=previous_pipeline_xlmr_50ep_no_kabsch_token_mean_margin_ratio_lora \
  bash run_previous_pipeline_baselines.sh
```

Reviewer-compliant checkpoints must have been trained only on `data/train_fit.csv` and must include provenance manifests.

### Full-encoder contrastive fine-tuning

The runner defines six training configurations. Each is trained once to epoch 10 and saved at epochs 3, 5, and 10.

Each checkpoint is evaluated with cosine, CSLS, and ratio margin:

```text
6 training configurations
× 3 checkpoints
× 3 retrieval criteria
= 54 development variants
```

Train or resume one configuration:

```bash
FULL_ENCODER_CONFIG=sent_mlp256_sym_t007_lr1e5 \
  bash run_full_encoder_contrastive_baselines.sh
```

Recover missing development evaluations from saved checkpoints:

```bash
FULL_ENCODER_ACTION=dev_eval_existing \
  bash run_full_encoder_contrastive_baselines.sh
```

Select on development only:

```bash
FULL_ENCODER_ACTION=select \
  bash run_full_encoder_contrastive_baselines.sh
```

Evaluate the selected configuration once:

```bash
FULL_ENCODER_ACTION=test \
  bash run_full_encoder_contrastive_baselines.sh
```

Expected artifacts:

```text
18 checkpoint directories
54 development metrics files
```

### Hybrid IBM1 + XLM-R LoRA

Run development tuning:

```bash
HYBRID_ACTION=dev bash run_hybrid_rerank.sh
```

Run the selected test configuration:

```bash
HYBRID_ACTION=test bash run_hybrid_rerank.sh
```

The hybrid consumes the development-selected IBM1 and XLM-R configurations. It supports XLM-R components selected with cosine, CSLS, or ratio margin.

---

## Outputs

### Development results

```text
results/dev/<family>/<configuration>/
├── metrics.json
└── sentence_predictions.csv
```

Some evaluators also write embeddings, caches, or family-specific summaries.

### Selection files

```text
results/dev_selection/
├── all_dev_variants.csv
├── selected_configs.json
├── dev_selected_test_results.csv
├── table2_ordering_comparison.csv
└── table2_ordering_summary.json
```

### Test results

```text
results/test/<family>/<selected-configuration>/
├── metrics.json
└── sentence_predictions.csv
```

Only development-selected configurations should appear in the final reported test table.

---

## Reproducibility

Result files record, where applicable:

- family and configuration name;
- split name;
- model and preprocessing settings;
- retrieval criterion;
- neighborhood size;
- training policy;
- evaluation command arguments;
- checkpoint paths;
- input file paths and hashes;
- selection policy.

Recommended checks:

```bash
python -m py_compile \
  src/retrieval_scoring.py \
  src/fasttext_procrustes_baseline.py \
  src/multilingual_encoder_baseline.py \
  src/previous_pipeline_baseline.py \
  src/full_encoder_contrastive_finetune_baseline.py \
  src/hybrid_lexical_neural_rerank.py
```

```bash
bash -n run_fasttext_procrustes_baselines.sh
bash -n run_off_the_shelf_encoder_baselines.sh
bash -n run_previous_pipeline_variant.sh
bash -n run_full_encoder_contrastive_baselines.sh
bash -n run_hybrid_rerank.sh
```

---

## Table 2 Ordering Comparison

After all reported families have one development-selected test result, provide the original ordering:

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

The selector writes a comparison table and a JSON summary indicating whether the original ordering is preserved.

---

## Transfer Experiments

The repository includes additional scripts and data preparation for:

- Khmer–Vietnamese;
- Lao–Vietnamese;
- Zhuang–Chinese;
- FLORES-based generalization experiments.

These experiments are separate from the main Bahnar–Vietnamese development-selection protocol. Their preprocessing, candidate pools, and model-selection policies should be reported independently.

---

## Git Hygiene

Do not commit generated caches, downloaded models, or local operating-system files.

Recommended `.gitignore` entries:

```gitignore
.DS_Store
__pycache__/
*.py[cod]
.venv/

results/cache/
results/models/
logs/
```

Keep lightweight final metrics and manifests only when they are part of the reproducibility release.

---

## References

- Mikel Artetxe and Holger Schwenk. 2019. *Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings*. ACL.
- Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong, and Francisco Guzmán. *WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs*.

---

## Citation

A final BibTeX entry should be added after the paper metadata is finalized.

```bibtex
@misc{bahnar_vietnamese_retrieval,
  title  = {Bahnar--Vietnamese Cross-Lingual Sentence Retrieval:
            A Low-Resource Case Study with Lexical, Neural, and Hybrid Methods},
  year   = {2026},
  note   = {Code repository}
}
```

---

## License

No software license is currently declared in this README. Add a `LICENSE` file before public release and update this section with the selected license.

---

## Research Status

This repository contains active research code. Before reporting or releasing results:

- verify that every family used development-only selection;
- verify checkpoint and data provenance;
- ensure all three retrieval criteria were included for the main single-vector models;
- report only one development-selected test result per family;
- separate historical test-selected outputs from reviewer-compliant outputs.
