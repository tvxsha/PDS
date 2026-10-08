# New-corpus handling: clean-only calibration and few-shot adaptation (train + validation only)

The model is trained on the OTHER source groups. For the held-out corpus we then use only what a deployer would have. FPR/TPR below are measured on held-out documents that were NOT used for calibration or adaptation.

Target false-alarm rate: 5%. Compare with the fixed-threshold results in sentence_loso (BIPIA: 59% of clean flagged; PoisonedRAG: nothing flagged).

## Part A: clean-only calibration (no poisoned document from the new corpus is used)

### bipia (218 poisoned / 102 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.46 | 0.050 | 0.000 - 0.134 |
| sentence classifier | 20 | 0.13 | 0.049 | 0.000 - 0.110 |
| LR + sentence classifier | 20 | 0.41 | 0.052 | 0.000 - 0.122 |
| LR on the 3 current scores | 40 | 0.47 | 0.047 | 0.000 - 0.097 |
| sentence classifier | 40 | 0.13 | 0.047 | 0.000 - 0.113 |
| LR + sentence classifier | 40 | 0.44 | 0.046 | 0.000 - 0.097 |
| LR on the 3 current scores | 60 | 0.49 | 0.050 | 0.000 - 0.119 |
| sentence classifier | 60 | 0.16 | 0.052 | 0.000 - 0.095 |
| LR + sentence classifier | 60 | 0.46 | 0.051 | 0.000 - 0.095 |

### poisonedrag (160 poisoned / 120 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.06 | 0.048 | 0.000 - 0.110 |
| sentence classifier | 20 | 0.22 | 0.049 | 0.000 - 0.111 |
| LR + sentence classifier | 20 | 0.22 | 0.049 | 0.000 - 0.110 |
| LR on the 3 current scores | 40 | 0.05 | 0.047 | 0.000 - 0.100 |
| sentence classifier | 40 | 0.24 | 0.051 | 0.013 - 0.113 |
| LR + sentence classifier | 40 | 0.23 | 0.048 | 0.000 - 0.100 |
| LR on the 3 current scores | 60 | 0.05 | 0.050 | 0.000 - 0.100 |
| sentence classifier | 60 | 0.26 | 0.050 | 0.000 - 0.100 |
| LR + sentence classifier | 60 | 0.27 | 0.053 | 0.000 - 0.117 |

### pds (56 poisoned / 40 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.40 | 0.044 | 0.000 - 0.100 |
| sentence classifier | 20 | 0.08 | 0.054 | 0.000 - 0.150 |
| LR + sentence classifier | 20 | 0.40 | 0.054 | 0.000 - 0.150 |

## Part B: calibration plus few-shot adaptation

Training data = other sources + 20 clean documents of the new corpus + about k labelled attack documents of the new corpus (whole groups, so the actual count can be a little above k). A further, separate 20 clean documents set the threshold.
Means over 8 random draws; AUC is threshold-free.

### bipia

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.79 | 0.48 | 0.032 |
| 0 | sentence classifier | 0.71 | 0.04 | 0.034 |
| 0 | LR + sentence classifier | 0.80 | 0.14 | 0.034 |
| 5 | LR on the 3 current scores | 0.79 | 0.49 | 0.034 |
| 5 | sentence classifier | 0.76 | 0.09 | 0.025 |
| 5 | LR + sentence classifier | 0.83 | 0.39 | 0.044 |
| 10 | LR on the 3 current scores | 0.79 | 0.49 | 0.037 |
| 10 | sentence classifier | 0.77 | 0.23 | 0.047 |
| 10 | LR + sentence classifier | 0.84 | 0.50 | 0.045 |
| 20 | LR on the 3 current scores | 0.78 | 0.49 | 0.036 |
| 20 | sentence classifier | 0.80 | 0.35 | 0.049 |
| 20 | LR + sentence classifier | 0.86 | 0.58 | 0.061 |
| 40 | LR on the 3 current scores | 0.76 | 0.48 | 0.026 |
| 40 | sentence classifier | 0.82 | 0.35 | 0.045 |
| 40 | LR + sentence classifier | 0.87 | 0.60 | 0.054 |

### poisonedrag

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.56 | 0.01 | 0.031 |
| 0 | sentence classifier | 0.77 | 0.09 | 0.022 |
| 0 | LR + sentence classifier | 0.76 | 0.11 | 0.019 |
| 5 | LR on the 3 current scores | 0.56 | 0.02 | 0.033 |
| 5 | sentence classifier | 0.78 | 0.11 | 0.020 |
| 5 | LR + sentence classifier | 0.77 | 0.12 | 0.022 |
| 10 | LR on the 3 current scores | 0.57 | 0.02 | 0.033 |
| 10 | sentence classifier | 0.78 | 0.11 | 0.016 |
| 10 | LR + sentence classifier | 0.78 | 0.16 | 0.028 |
| 20 | LR on the 3 current scores | 0.59 | 0.04 | 0.036 |
| 20 | sentence classifier | 0.79 | 0.16 | 0.036 |
| 20 | LR + sentence classifier | 0.80 | 0.22 | 0.039 |
| 40 | LR on the 3 current scores | 0.63 | 0.09 | 0.041 |
| 40 | sentence classifier | 0.82 | 0.15 | 0.025 |
| 40 | LR + sentence classifier | 0.83 | 0.26 | 0.034 |

### pds

Skipped: too few clean documents to hold out calibration, adaptation and evaluation sets.

Caveats:

- Only three source groups exist, and the clean calibration sets are small (20 to 60 documents), so every number is noisy.
- Repetitions share the same underlying documents, so they do not give independent evidence; treat differences of a few points as noise.
- The test split has not been used.
