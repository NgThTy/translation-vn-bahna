# Source Provenance

This repository is the unified source release corresponding to the
camera-ready paper.

## Primary Bahnar--Vietnamese source

Base repository:

https://github.com/NgThTy/translation-vn-bahna

The public camera-ready release is based on the verified
`camera-ready-integration` branch. The previous local working tree was
compared with `origin/main`; no substantive source-code differences were
found before camera-ready integration.

The local-only reviewer metrics archive was also checked: all 510 files
were already present identically in the repository and therefore were not
included separately.

## Modern dense retrieval

Modern-dense implementation components were imported from:

https://github.com/longstnguyen/xlmr-ibm1

Branch: `xlmr-ibm1-final`

Commit:

`4b38db4ee70f6db3d18c82033cb1cfeff2770128`

Only components relevant to the in-domain LAMIR, DPR-XM, ColBERT-X, and
ColBERT-XM experiments were imported.

The train, development, test, and duplicate-safe split-manifest files were
verified to be byte-identical between the two source repositories.

Historical README, RESULTS, paper, reviewer-response material, duplicate
data, unrelated method implementations, and incomplete checkpoint
directories from the upstream repository were not imported.

The original DPR-XM/LAMIR evaluator computed MRR@10. The camera-ready
release preserves that metric name rather than silently relabeling it as
full MRR.

The actual selected modern-dense weight files were not present in the
upstream repository and are therefore not claimed as part of this release.
