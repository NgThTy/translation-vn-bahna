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
