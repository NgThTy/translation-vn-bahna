# Zhuang--Chinese transfer data

The raw and derived Zhuang--Chinese sentence data used for the supporting
transfer experiment are not redistributed in this repository.

The camera-ready experiment uses:

- 4,944 training pairs
- 200 held-out evaluation pairs

After obtaining the source data cited in the paper under its applicable
license and access terms, place the source files at:

- `data/other_low_resource_data/parallel_corpus.json`
- `data/other_low_resource_data/test_translation_set.json`

Then run `python src/prepare_other_low_resource_data.py`.

This generates:

- `train.csv`
- `test.csv`

in this directory.

Existing result artifacts are retained for reproducibility of the reported
camera-ready results.
