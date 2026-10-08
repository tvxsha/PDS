# New-corpus handling: clean-only calibration and few-shot adaptation (train + validation only)

The model is trained on the OTHER source groups. For the held-out corpus we then use only what a deployer would have. FPR/TPR below are measured on held-out documents that were NOT used for calibration or adaptation.

Target false-alarm rate: 5%. Compare with the fixed-threshold results in sentence_loso (BIPIA: 59% of clean flagged; PoisonedRAG: nothing flagged).

## Part A: clean-only calibration (no poisoned document from the new corpus is used)

### bipia (218 poisoned / 102 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.46 | 0.052 | 0.000 - 0.122 |
| sentence classifier | 20 | 0.05 | 0.046 | 0.000 - 0.122 |
| LR + sentence classifier | 20 | 0.25 | 0.045 | 0.000 - 0.110 |
| LR on the 3 current scores | 40 | 0.47 | 0.050 | 0.000 - 0.113 |
| sentence classifier | 40 | 0.05 | 0.046 | 0.000 - 0.097 |
| LR + sentence classifier | 40 | 0.28 | 0.048 | 0.000 - 0.097 |
| LR on the 3 current scores | 60 | 0.47 | 0.049 | 0.000 - 0.098 |
| sentence classifier | 60 | 0.05 | 0.046 | 0.000 - 0.095 |
| LR + sentence classifier | 60 | 0.29 | 0.047 | 0.000 - 0.095 |

### poisonedrag (160 poisoned / 120 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.04 | 0.044 | 0.000 - 0.110 |
| sentence classifier | 20 | 0.12 | 0.043 | 0.000 - 0.110 |
| LR + sentence classifier | 20 | 0.09 | 0.041 | 0.000 - 0.100 |
| LR on the 3 current scores | 40 | 0.04 | 0.048 | 0.000 - 0.100 |
| sentence classifier | 40 | 0.13 | 0.046 | 0.000 - 0.100 |
| LR + sentence classifier | 40 | 0.10 | 0.050 | 0.000 - 0.100 |
| LR on the 3 current scores | 60 | 0.05 | 0.049 | 0.000 - 0.100 |
| sentence classifier | 60 | 0.14 | 0.052 | 0.000 - 0.102 |
| LR + sentence classifier | 60 | 0.10 | 0.052 | 0.000 - 0.117 |

### pds (56 poisoned / 40 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.42 | 0.056 | 0.000 - 0.150 |
| sentence classifier | 20 | 0.08 | 0.057 | 0.000 - 0.150 |
| LR + sentence classifier | 20 | 0.38 | 0.057 | 0.000 - 0.150 |

## Part B: calibration plus few-shot adaptation

Training data = other sources + 20 clean documents of the new corpus + about k labelled attack documents of the new corpus (whole groups, so the actual count can be a little above k). A further, separate 20 clean documents set the threshold.
Means over 8 random draws; AUC is threshold-free.

### bipia

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.79 | 0.48 | 0.065 |
| 0 | sentence classifier | 0.67 | 0.06 | 0.046 |
| 0 | LR + sentence classifier | 0.76 | 0.13 | 0.050 |
| 5 | LR on the 3 current scores | 0.79 | 0.49 | 0.076 |
| 5 | sentence classifier | 0.71 | 0.08 | 0.044 |
| 5 | LR + sentence classifier | 0.80 | 0.29 | 0.034 |
| 10 | LR on the 3 current scores | 0.78 | 0.49 | 0.076 |
| 10 | sentence classifier | 0.73 | 0.16 | 0.044 |
| 10 | LR + sentence classifier | 0.82 | 0.36 | 0.044 |
| 20 | LR on the 3 current scores | 0.77 | 0.49 | 0.071 |
| 20 | sentence classifier | 0.73 | 0.16 | 0.033 |
| 20 | LR + sentence classifier | 0.83 | 0.41 | 0.050 |
| 40 | LR on the 3 current scores | 0.75 | 0.46 | 0.060 |
| 40 | sentence classifier | 0.79 | 0.25 | 0.031 |
| 40 | LR + sentence classifier | 0.85 | 0.52 | 0.043 |

### poisonedrag

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.54 | 0.01 | 0.019 |
| 0 | sentence classifier | 0.75 | 0.11 | 0.033 |
| 0 | LR + sentence classifier | 0.75 | 0.09 | 0.033 |
| 5 | LR on the 3 current scores | 0.55 | 0.01 | 0.019 |
| 5 | sentence classifier | 0.76 | 0.14 | 0.042 |
| 5 | LR + sentence classifier | 0.76 | 0.11 | 0.039 |
| 10 | LR on the 3 current scores | 0.55 | 0.01 | 0.017 |
| 10 | sentence classifier | 0.77 | 0.14 | 0.034 |
| 10 | LR + sentence classifier | 0.77 | 0.13 | 0.039 |
| 20 | LR on the 3 current scores | 0.55 | 0.02 | 0.014 |
| 20 | sentence classifier | 0.79 | 0.23 | 0.048 |
| 20 | LR + sentence classifier | 0.79 | 0.19 | 0.048 |
| 40 | LR on the 3 current scores | 0.55 | 0.04 | 0.030 |
| 40 | sentence classifier | 0.82 | 0.34 | 0.052 |
| 40 | LR + sentence classifier | 0.82 | 0.31 | 0.055 |

### pds

Skipped: too few clean documents to hold out calibration, adaptation and evaluation sets.

Caveats:

- Only three source groups exist, and the clean calibration sets are small (20 to 60 documents), so every number is noisy.
- Repetitions share the same underlying documents, so they do not give independent evidence; treat differences of a few points as noise.
- The test split has not been used.
