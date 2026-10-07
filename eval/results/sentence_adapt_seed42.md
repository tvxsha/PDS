# New-corpus handling: clean-only calibration and few-shot adaptation (train + validation only)

The model is trained on the OTHER source groups. For the held-out corpus we then use only what a deployer would have. FPR/TPR below are measured on held-out documents that were NOT used for calibration or adaptation.

Target false-alarm rate: 5%. Compare with the fixed-threshold results in sentence_loso (BIPIA: 59% of clean flagged; PoisonedRAG: nothing flagged).

## Part A: clean-only calibration (no poisoned document from the new corpus is used)

### bipia (218 poisoned / 102 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.33 | 0.047 | 0.000 - 0.110 |
| sentence classifier | 20 | 0.06 | 0.042 | 0.000 - 0.110 |
| LR + sentence classifier | 20 | 0.33 | 0.048 | 0.000 - 0.110 |
| LR on the 3 current scores | 40 | 0.33 | 0.049 | 0.000 - 0.113 |
| sentence classifier | 40 | 0.06 | 0.044 | 0.000 - 0.097 |
| LR + sentence classifier | 40 | 0.34 | 0.054 | 0.000 - 0.113 |
| LR on the 3 current scores | 60 | 0.33 | 0.049 | 0.000 - 0.095 |
| sentence classifier | 60 | 0.06 | 0.047 | 0.000 - 0.095 |
| LR + sentence classifier | 60 | 0.33 | 0.047 | 0.000 - 0.119 |

### poisonedrag (160 poisoned / 120 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.06 | 0.044 | 0.000 - 0.100 |
| sentence classifier | 20 | 0.15 | 0.052 | 0.000 - 0.120 |
| LR + sentence classifier | 20 | 0.13 | 0.052 | 0.000 - 0.120 |
| LR on the 3 current scores | 40 | 0.07 | 0.048 | 0.013 - 0.100 |
| sentence classifier | 40 | 0.13 | 0.049 | 0.011 - 0.100 |
| LR + sentence classifier | 40 | 0.11 | 0.049 | 0.013 - 0.100 |
| LR on the 3 current scores | 60 | 0.07 | 0.051 | 0.000 - 0.100 |
| sentence classifier | 60 | 0.12 | 0.047 | 0.000 - 0.100 |
| LR + sentence classifier | 60 | 0.10 | 0.047 | 0.000 - 0.100 |

### pds (56 poisoned / 40 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.41 | 0.049 | 0.000 - 0.150 |
| sentence classifier | 20 | 0.10 | 0.053 | 0.000 - 0.150 |
| LR + sentence classifier | 20 | 0.33 | 0.054 | 0.000 - 0.150 |

## Part B: calibration plus few-shot adaptation

Training data = other sources + 20 clean documents of the new corpus + about k labelled attack documents of the new corpus (whole groups, so the actual count can be a little above k). A further, separate 20 clean documents set the threshold.
Means over 8 random draws; AUC is threshold-free.

### bipia

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.74 | 0.41 | 0.044 |
| 0 | sentence classifier | 0.70 | 0.07 | 0.034 |
| 0 | LR + sentence classifier | 0.78 | 0.25 | 0.052 |
| 5 | LR on the 3 current scores | 0.74 | 0.43 | 0.044 |
| 5 | sentence classifier | 0.80 | 0.23 | 0.042 |
| 5 | LR + sentence classifier | 0.85 | 0.47 | 0.058 |
| 10 | LR on the 3 current scores | 0.73 | 0.43 | 0.043 |
| 10 | sentence classifier | 0.81 | 0.27 | 0.041 |
| 10 | LR + sentence classifier | 0.85 | 0.50 | 0.043 |
| 20 | LR on the 3 current scores | 0.73 | 0.44 | 0.052 |
| 20 | sentence classifier | 0.82 | 0.20 | 0.014 |
| 20 | LR + sentence classifier | 0.86 | 0.50 | 0.014 |
| 40 | LR on the 3 current scores | 0.71 | 0.43 | 0.045 |
| 40 | sentence classifier | 0.82 | 0.23 | 0.017 |
| 40 | LR + sentence classifier | 0.85 | 0.54 | 0.037 |

### poisonedrag

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.55 | 0.05 | 0.061 |
| 0 | sentence classifier | 0.72 | 0.08 | 0.025 |
| 0 | LR + sentence classifier | 0.72 | 0.09 | 0.041 |
| 5 | LR on the 3 current scores | 0.57 | 0.05 | 0.059 |
| 5 | sentence classifier | 0.73 | 0.08 | 0.020 |
| 5 | LR + sentence classifier | 0.74 | 0.09 | 0.033 |
| 10 | LR on the 3 current scores | 0.58 | 0.05 | 0.056 |
| 10 | sentence classifier | 0.75 | 0.10 | 0.030 |
| 10 | LR + sentence classifier | 0.76 | 0.12 | 0.045 |
| 20 | LR on the 3 current scores | 0.60 | 0.06 | 0.058 |
| 20 | sentence classifier | 0.77 | 0.08 | 0.025 |
| 20 | LR + sentence classifier | 0.79 | 0.10 | 0.038 |
| 40 | LR on the 3 current scores | 0.63 | 0.08 | 0.066 |
| 40 | sentence classifier | 0.82 | 0.13 | 0.025 |
| 40 | LR + sentence classifier | 0.83 | 0.11 | 0.027 |

### pds

Skipped: too few clean documents to hold out calibration, adaptation and evaluation sets.

Caveats:

- Only three source groups exist, and the clean calibration sets are small (20 to 60 documents), so every number is noisy.
- Repetitions share the same underlying documents, so they do not give independent evidence; treat differences of a few points as noise.
- The test split has not been used.
