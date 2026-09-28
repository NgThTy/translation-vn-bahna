# Modern Dense Retrieval

This directory contains the modern-dense component used in the
camera-ready Bahnar--Vietnamese retrieval study.

## Systems

Four systems were trained on the same 46,736 Bahnar--Vietnamese
training pairs and compared on the 5,194-pair development set:

| Method | Selected checkpoint | Dev R@1 | Dev R@5 | Dev R@10 |
|---|---:|---:|---:|---:|
| DPR-XM | 20 epochs | 0.5766 | 0.7403 | 0.7869 |
| LAMIR | 50 epochs | 0.5381 | 0.7147 | 0.7711 |
| ColBERT-X | 50,000 steps | 0.4471 | 0.5941 | 0.6450 |
| ColBERT-XM | 60,000 steps | 0.4224 | 0.5668 | 0.6136 |

DPR-XM was selected by development Recall@1 and was the only modern-dense
architecture evaluated on the held-out 2,001-pair test set.

Test results:

- Recall@1: 0.4743
- Recall@5: 0.6522
- Recall@10: 0.7221
- MRR@10: 0.5506

## Metric note

The upstream DPR-XM and LAMIR evaluation code retrieved only the top 10
candidates before calculating reciprocal rank. Consequently, the upstream
field `mrr@10` is genuinely MRR@10 rather than full mean reciprocal rank.
It is retained under that name and is not relabeled as full MRR.

## Checkpoints

The upstream repository contains checkpoint metadata but does not contain
the actual selected modern-dense model weight files. This public release
therefore does not claim that those trained weight files are included.

## Provenance

Modern-dense code was imported from:

https://github.com/longstnguyen/xlmr-ibm1

Branch: xlmr-ibm1-final

Commit: 4b38db4ee70f6db3d18c82033cb1cfeff2770128

See ../SOURCE_PROVENANCE.md for details.
