# Bahnar–Vietnamese Parallel-Sentence Retrieval

Code, data splits, result artifacts, and camera-ready reproducibility materials for:

> **Bahnar–Vietnamese Parallel-Sentence Retrieval: A Low-Resource Case Study with Lexical, Neural, and Hybrid Methods**

This repository studies **Bahnar–Vietnamese parallel-sentence retrieval (bitext retrieval)** as a controlled candidate-ranking task. Given a Bahnar query sentence, a system ranks Vietnamese candidates and should place the documented Vietnamese counterpart as high as possible.

The camera-ready comparison covers surface lexical retrieval, edit distance, static cross-lingual embeddings, IBM Model 1, off-the-shelf multilingual encoders, supervised modern dense retrieval, XLM-R LoRA adaptation, full XLM-R fine-tuning, and hybrid IBM1–XLM-R reranking.

> Some legacy script names contain `translate`, but the evaluated task is retrieval, not free-form machine translation.

## Camera-ready release

The initial frozen camera-ready source release is tagged as:

```text
v1.0-camera-ready
```

Repository:

```text
https://github.com/NgThTy/translation-vn-bahna
```

The release includes the main Bahnar–Vietnamese data splits, split manifest, experiment code, selected result artifacts, data checksums, environment records, and the retained modern-dense experiment/evaluation materials.

The selected modern-dense **trained weight files are not included** because they were not present in the imported upstream source. Modern-dense checkpoint metadata and development-selection records are retained instead.

## Highlights

- Leakage-safe train/development/test protocol
- Duplicate-safe deterministic train/development split with seed `42`
- One development-selected primary test result per method family
- Cosine, CSLS, and Artetxe–Schwenk ratio-margin retrieval
- Surface, lexical-alignment, multilingual neural, adapted neural, and full-fine-tuning baselines
- In-domain comparison of DPR-XM, LAMIR, ColBERT-X, and ColBERT-XM
- Development-selected XLM-R-first IBM1 reranking
- Paired uncertainty, seed-variation, and candidate-size sensitivity analyses
- Pair-specific transfer experiments on Khmer–Vietnamese, Lao–Vietnamese, and Zhuang–Chinese
- Source provenance, data checksums, locked dependencies, and a frozen camera-ready tag

## Task

For a Bahnar query \(q_i\), the system ranks every Vietnamese candidate in the evaluated pool:

```text
C = {c_1, c_2, ..., c_n}
```

Each query has exactly one documented Vietnamese counterpart in the pool.

The main paper reports:

- Recall@1
- Mean Reciprocal Rank (MRR)
- Recall@5
- Recall@10

Because each query has exactly one documented counterpart, **Recall@1 is numerically identical to the Accuracy@1 measure used in the original submission**.

The benchmark is a controlled candidate-ranking setting. It does not model every source of noise encountered in open-web bitext mining, such as missing matches, multiple valid translations, or partially aligned candidate collections.

## Main Bahnar–Vietnamese data

| Resource | Train | Dev | Test / held-out |
|---|---:|---:|---:|
| Sentence pairs | 46,736 | 5,194 | 2,001 |
| Lexicon entries | 8,285 | — | 2,073 |

The sentence train/development split is derived from 51,930 original pairs. Exact duplicate bilingual pairs are grouped before a deterministic seed-42 split so that duplicates cannot cross the train/development boundary.

Main files:

```text
data/train.csv
data/train_fit.csv
data/dev.csv
data/test.csv
data/train_dev_split_manifest.json
```

Verify the released split files with:

```bash
shasum -a 256 -c DATA_SHA256SUMS.txt
```

The camera-ready release records SHA-256 hashes for:

```text
data/train_fit.csv
data/dev.csv
data/test.csv
data/train_dev_split_manifest.json
```

## Development-selection protocol

Trainable sentence-level components and IBM1 lexical probabilities use only `data/train_fit.csv`. Lexicon-supervised Procrustes/Kabsch components use only the lexicon training split.

Configuration choices are made on `data/dev.csv`. Within a method family, the primary ordering is:

1. higher development Recall@1;
2. higher development MRR;
3. higher development Recall@5;
4. lexicographically smaller configuration name as the deterministic final tie-breaker.

The selected sentence-trained model is not refit on `train_fit + dev` before the primary held-out test evaluation.

The paper reports one development-selected primary test configuration per family. Later test-set uncertainty and sensitivity analyses are explicitly treated as **post-hoc diagnostics** and are not used to reselect the primary configuration.

## Main camera-ready results

Held-out Bahnar–Vietnamese results for one development-selected configuration per family:

| Group | Development-selected method | R@1 | Reciprocal-rank metric | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| Surface lexical | BM25 character n-gram, 2–5 | 0.3723 | 0.4004 MRR | 0.4393 | 0.4548 |
| Edit distance | Levenshtein ratio, strip accents | 0.4248 | 0.4447 MRR | 0.4703 | 0.4913 |
| Static embedding | fastText-Procrustes, ratio margin | 0.1654 | 0.2114 MRR | 0.2689 | 0.3298 |
| Word alignment | Symmetric IBM1, length penalty | 0.4933 | 0.5459 MRR | 0.6167 | 0.6662 |
| Off-the-shelf encoder | LaBSE, ratio margin | 0.2424 | 0.2734 MRR | 0.3153 | 0.3518 |
| Modern dense retrieval | Supervised DPR-XM | 0.4743 | **0.5506 MRR@10** | **0.6522** | **0.7221** |
| Task-adapted neural | XLM-R LoRA, token mean, ratio margin | 0.3648 | 0.4462 MRR | 0.5517 | 0.6402 |
| Full fine-tuning | Full XLM-R encoder, CSLS | 0.2224 | 0.2985 MRR | 0.3968 | 0.4808 |
| Hybrid | XLM-R-first + symmetric IBM1 reranking | **0.5162** | 0.5579 MRR | 0.6157 | 0.6402 |

### Important DPR-XM metric note

The retained DPR-XM evaluator retrieves the top 10 candidates before calculating reciprocal rank. Therefore, the reported `0.5506` is **MRR@10**, not full MRR, and should not be compared as though it were the same full-MRR quantity reported for the other main methods.

The corresponding artifact is:

```text
modern_dense/results/camera_ready_summary.json
```

## Key findings

Symmetric IBM1 is strong at the top rank, with Recall@1 of `0.4933`. The development-selected DPR-XM reaches Recall@1 of `0.4743` but provides the strongest deeper coverage, with Recall@5 of `0.6522` and Recall@10 of `0.7221`.

The development-selected XLM-R-first IBM1 hybrid gives the highest main-table Recall@1 point estimate, `0.5162`, with full MRR `0.5579`. Paired tests do not establish a significant difference between the hybrid and symmetric IBM1 at the 0.05 level, while reranking substantially improves over the XLM-R LoRA first stage.

Surface retrieval is not uniformly stronger than static cross-lingual alignment: the paper's post-hoc overlap analysis shows that BM25's aggregate strength is concentrated in high character-overlap pairs, while fastText-Procrustes is stronger in the two lowest character-overlap quartiles.

## Methods

The repository contains the following primary method families:

| Family | Representative implementation / runner |
|---|---|
| Character TF-IDF and BM25 | `src/lexical_retrieval_baseline.py`, `run_lexical_baselines.sh` |
| Levenshtein / token overlap | `src/edit_distance_retrieval_baseline.py`, `run_edit_distance_baselines.sh` |
| fastText + Procrustes | `src/fasttext_procrustes_baseline.py`, `run_fasttext_procrustes_baselines.sh` |
| IBM Model 1 | `src/word_alignment_baseline.py`, `run_word_alignment_baselines.sh` |
| Off-the-shelf multilingual encoders | `src/multilingual_encoder_baseline.py`, `run_off_the_shelf_encoder_baselines.sh` |
| XLM-R LoRA projection | `src/previous_pipeline_baseline.py` and the `run_previous_pipeline_*.sh` runners |
| Full XLM-R fine-tuning | `src/full_encoder_contrastive_finetune_baseline.py`, `run_full_encoder_contrastive_baselines.sh` |
| Hybrid IBM1 + XLM-R LoRA | `src/hybrid_lexical_neural_rerank.py`, `run_hybrid_rerank.sh` |
| Modern dense retrieval | `modern_dense/experiments/methods/dense_retrievers/` |

### Embedding retrieval criteria

Single-vector embedding families use cosine, CSLS, and ratio margin where applicable.

Cosine:

```text
score(x, y) = cosine(x, y)
```

CSLS:

```text
score(x, y) = 2 * cosine(x, y) - r_x - r_y
```

Ratio margin:

```text
score(x, y) = cosine(x, y) / (0.5 * (r_x + r_y))
```

For CSLS and ratio margin, `k = 10`. Neighborhood statistics are computed from the complete source–target similarity matrix before final ranking or hybrid top-\(K\) truncation.

Shared scoring implementation:

```text
src/retrieval_scoring.py
```

## Hybrid IBM1–XLM-R LoRA reranking

The selected hybrid uses XLM-R LoRA as the first-stage candidate retriever and symmetric IBM1 as complementary lexical evidence.

Development selection chooses:

```text
first stage:     XLM-R LoRA, ratio margin
K:               10
normalization:   min-max
combination:     weighted sum
neural weight:   0.25
IBM1 weight:     0.75
edit weight:     0
IBM1 length λ:   0.1
```

Development stage-order comparison:

| First stage | First-stage R@10 | Final R@1 | Final MRR | Final R@5 |
|---|---:|---:|---:|---:|
| Symmetric IBM1 | 0.6213 | 0.5477 | 0.5779 | 0.6176 |
| XLM-R LoRA, ratio margin | **0.7657** | **0.6103** | **0.6652** | **0.7397** |

This supports the recall-bottleneck rationale used in the paper: the neural first stage retains more documented counterparts for reranking, while IBM1 contributes complementary bilingual lexical evidence.

## Modern dense retrieval

Four in-domain dense retrievers are trained on the same 46,736 Bahnar–Vietnamese training pairs and selected on the 5,194-pair development set:

| Method | Selected checkpoint | Dev R@1 | Dev R@5 | Dev R@10 |
|---|---:|---:|---:|---:|
| DPR-XM | 20 epochs | **0.5766** | **0.7403** | **0.7869** |
| LAMIR | 50 epochs | 0.5381 | 0.7147 | 0.7711 |
| ColBERT-X | 50,000 steps | 0.4471 | 0.5941 | 0.6450 |
| ColBERT-XM | 60,000 steps | 0.4224 | 0.5668 | 0.6136 |

DPR-XM is selected as the modern-dense representative and is the only member of this family evaluated on the held-out 2,001-pair test set.

The retained public subset is intentionally focused. It contains experiment/evaluation scripts, selection metadata, the camera-ready result summary, provenance records, the upstream dependency list, and the upstream MIT license. It does **not** contain duplicate data, the upstream paper, old upstream result summaries, incomplete checkpoint directories, or the actual selected modern-dense model weights.

See:

```text
modern_dense/README.md
modern_dense/results/camera_ready_summary.json
modern_dense/provenance/
SOURCE_PROVENANCE.md
```

## Transfer experiments

The paper also reports supporting pair-specific transfer experiments.

### FLORES Khmer–Vietnamese and Lao–Vietnamese

For each pair:

- 997 aligned FLORES development pairs are used for pair-specific training.
- 1,012 FLORES `devtest` pairs form the retrieval evaluation pool.
- The LoRA projection model is trained separately for that language pair.

Best reported pair-specific LoRA + CSLS results:

| Pair | R@1 | MRR | R@5 | R@10 |
|---|---:|---:|---:|---:|
| Khmer → Vietnamese | 0.8014 | 0.8631 | 0.9407 | 0.9694 |
| Lao → Vietnamese | 0.8458 | 0.8982 | 0.9654 | 0.9802 |

### Zhuang–Chinese

The reported Zhuang–Chinese setup uses 4,944 training pairs and a separate 200-pair retrieval evaluation pool. Pair-specific LoRA + CSLS reaches:

```text
R@1  = 0.5700
MRR  = 0.6788
R@5  = 0.8400
R@10 = 0.8850
```

The raw and derived Zhuang–Chinese sentence data are **not redistributed in this repository**. After obtaining the source data cited in the paper under the applicable license/access terms, place them at:

```text
data/other_low_resource_data/parallel_corpus.json
data/other_low_resource_data/test_translation_set.json
```

Then run:

```bash
python src/prepare_other_low_resource_data.py
```

The generated files are:

```text
data/other_low_resource_generalization/zhuang_chinese/train.csv
data/other_low_resource_generalization/zhuang_chinese/test.csv
```

These generated third-party files are intentionally ignored by Git. See:

```text
data/other_low_resource_generalization/zhuang_chinese/README.md
```

The paper additionally reports a post-hoc Zhuang tokenization diagnostic. That diagnostic is not used to replace the primary transfer result.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── SOURCE_PROVENANCE.md
├── DATA_SHA256SUMS.txt
├── requirements.txt
├── requirements-lock.txt
├── environment.yml
│
├── data/
│   ├── train.csv
│   ├── train_fit.csv
│   ├── dev.csv
│   ├── test.csv
│   ├── train_dev_split_manifest.json
│   └── other_low_resource_generalization/
│       └── zhuang_chinese/
│           └── README.md
│
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
│
├── results/
│   └── ...
│
├── modern_dense/
│   ├── LICENSE
│   ├── README.md
│   ├── requirements-upstream.txt
│   ├── experiments/
│   │   └── methods/
│   │       └── dense_retrievers/
│   │           ├── colbert/
│   │           ├── dpr_xm/
│   │           └── lamir/
│   ├── provenance/
│   │   ├── dpr_xm/
│   │   └── lamir/
│   └── results/
│       └── camera_ready_summary.json
│
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
└── run_hybrid_rerank.sh
```

Large checkpoints, downloaded models, generated caches, local Python artifacts, and generated Zhuang–Chinese sentence files are not intended to be committed.

## Installation

### Option A: curated requirements

```bash
git clone https://github.com/NgThTy/translation-vn-bahna.git
cd translation-vn-bahna

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option B: recorded camera-ready environment

The repository also contains:

```text
environment.yml
requirements-lock.txt
```

These files record the environment used during camera-ready packaging. Depending on platform/CUDA availability, a fresh environment may require platform-specific PyTorch/CUDA adjustments.

The imported modern-dense component has its own upstream dependency record:

```text
modern_dense/requirements-upstream.txt
```

## Quick start

### 1. Verify the released data

```bash
shasum -a 256 -c DATA_SHA256SUMS.txt
```

### 2. Prepare the deterministic split if rebuilding from `data/train.csv`

```bash
python src/prepare_train_dev_split.py \
  --input_csv data/train.csv \
  --train_output data/train_fit.csv \
  --dev_output data/dev.csv \
  --manifest_output data/train_dev_split_manifest.json \
  --dev_ratio 0.10 \
  --seed 42
```

This command does not read or modify `data/test.csv`.

### 3. Run development experiments

```bash
bash run_lexical_baselines.sh
bash run_edit_distance_baselines.sh
bash run_fasttext_procrustes_baselines.sh
bash run_word_alignment_baselines.sh
bash run_off_the_shelf_encoder_baselines.sh
```

XLM-R LoRA development runners:

```bash
bash run_previous_pipeline_baselines.sh
bash run_previous_pipeline_additional_experiments.sh
bash run_previous_pipeline_clarification_experiments.sh
bash run_previous_pipeline_token_level_kabsch_experiments.sh
```

Full XLM-R:

```bash
bash run_full_encoder_contrastive_baselines.sh
```

### 4. Rebuild development selections

```bash
python src/select_dev_configs_and_evaluate_test.py \
  --dev_root results/dev \
  --test_root results/test \
  --output_root results/dev_selection \
  --test_csv data/test.csv
```

Without `--evaluate_test`, this rebuilds the selection manifest and does not run a test evaluation.

### 5. Run the selected hybrid workflow

Development tuning:

```bash
HYBRID_ACTION=dev bash run_hybrid_rerank.sh
```

Primary selected test evaluation:

```bash
HYBRID_ACTION=test bash run_hybrid_rerank.sh
```

## Reproducibility and provenance

The release includes several files intended to make the camera-ready artifact auditable:

```text
SOURCE_PROVENANCE.md
DATA_SHA256SUMS.txt
environment.yml
requirements-lock.txt
modern_dense/README.md
modern_dense/results/camera_ready_summary.json
modern_dense/provenance/
```

`SOURCE_PROVENANCE.md` records the origin of the primary repository and the imported modern-dense material. The modern-dense subset was imported from the upstream `xlmr-ibm1` repository at the recorded branch/commit and was pruned to the components relevant to the camera-ready in-domain dense-retrieval comparison.

No claim is made that missing trained modern-dense weight files are included.

## Data and third-party materials

The main Bahnar–Vietnamese split files used by the paper are present in this repository and can be verified with `DATA_SHA256SUMS.txt`.

Third-party code and data remain subject to their respective licenses and access conditions. In particular, the raw and generated Zhuang–Chinese sentence data are not redistributed here; the repository provides the preparation path and retained camera-ready result artifacts instead.

For provenance details, see:

```text
SOURCE_PROVENANCE.md
data/other_low_resource_generalization/zhuang_chinese/README.md
modern_dense/LICENSE
modern_dense/README.md
```

## Limitations of the released artifact

The public release should be interpreted together with the paper's limitations:

- the main benchmark focuses on Bahnar and should not be generalized to all Bahnaric languages;
- the benchmark uses one documented counterpart per query and is not a noisy open-web mining benchmark;
- some supporting transfer resources are third-party and are not fully redistributed;
- the actual selected modern-dense trained weight files are not included;
- post-hoc uncertainty, sensitivity, overlap, and tokenization analyses are diagnostic and do not redefine the primary development-selected test configurations.

## Citation

If you use this repository, please cite the paper:

```bibtex
@misc{bahnar_vietnamese_retrieval_2026,
  title  = {Bahnar--Vietnamese Parallel-Sentence Retrieval:
            A Low-Resource Case Study with Lexical, Neural, and Hybrid Methods},
  year   = {2026},
  note   = {Camera-ready code and data release:
            https://github.com/NgThTy/translation-vn-bahna}
}
```

Update this entry with the final proceedings metadata when it becomes available.

## License

The original code in this repository is released under the MIT License. See [`LICENSE`](LICENSE).

Third-party code and data remain subject to their respective licenses and access conditions. See [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md), [`modern_dense/LICENSE`](modern_dense/LICENSE), and the documentation accompanying third-party resources.
