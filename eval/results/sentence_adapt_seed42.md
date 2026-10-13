# New-corpus handling: clean-only calibration and few-shot adaptation (train + validation only)

The model is trained on the OTHER source groups. For the held-out corpus we then use only what a deployer would have. FPR/TPR below are measured on held-out documents that were NOT used for calibration or adaptation.

Target false-alarm rate: 5%. Compare with the fixed-threshold results in sentence_loso (BIPIA: 59% of clean flagged; PoisonedRAG: nothing flagged).

## Part A: clean-only calibration (no poisoned document from the new corpus is used)

### bipia (347 poisoned / 347 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.59 | 0.047 | 0.003 - 0.116 |
| sentence classifier | 20 | 0.05 | 0.045 | 0.003 - 0.107 |
| LR + sentence classifier | 20 | 0.33 | 0.047 | 0.003 - 0.107 |
| LR on the 3 current scores | 40 | 0.58 | 0.048 | 0.013 - 0.091 |
| sentence classifier | 40 | 0.05 | 0.044 | 0.010 - 0.091 |
| LR + sentence classifier | 40 | 0.37 | 0.047 | 0.010 - 0.094 |
| LR on the 3 current scores | 60 | 0.59 | 0.050 | 0.014 - 0.098 |
| sentence classifier | 60 | 0.05 | 0.050 | 0.017 - 0.094 |
| LR + sentence classifier | 60 | 0.40 | 0.051 | 0.014 - 0.087 |

### poisonedrag (400 poisoned / 299 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.11 | 0.049 | 0.007 - 0.111 |
| sentence classifier | 20 | 0.11 | 0.048 | 0.004 - 0.115 |
| LR + sentence classifier | 20 | 0.10 | 0.049 | 0.004 - 0.118 |
| LR on the 3 current scores | 40 | 0.11 | 0.047 | 0.008 - 0.093 |
| sentence classifier | 40 | 0.11 | 0.050 | 0.008 - 0.100 |
| LR + sentence classifier | 40 | 0.11 | 0.050 | 0.012 - 0.104 |
| LR on the 3 current scores | 60 | 0.11 | 0.045 | 0.013 - 0.088 |
| sentence classifier | 60 | 0.11 | 0.048 | 0.013 - 0.088 |
| LR + sentence classifier | 60 | 0.10 | 0.047 | 0.013 - 0.084 |

### pds (56 poisoned / 40 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.51 | 0.045 | 0.000 - 0.150 |
| sentence classifier | 20 | 0.10 | 0.045 | 0.000 - 0.150 |
| LR + sentence classifier | 20 | 0.29 | 0.045 | 0.000 - 0.150 |

## Part B: calibration plus few-shot adaptation

Training data = other sources + 20 clean documents of the new corpus + about k labelled attack documents of the new corpus (whole groups, so the actual count can be a little above k). A further, separate 20 clean documents set the threshold.
Means over 8 random draws; AUC is threshold-free.

### bipia

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.85 | 0.51 | 0.015 |
| 0 | sentence classifier | 0.61 | 0.04 | 0.035 |
| 0 | LR + sentence classifier | 0.75 | 0.11 | 0.038 |
| 5 | LR on the 3 current scores | 0.86 | 0.55 | 0.014 |
| 5 | sentence classifier | 0.70 | 0.08 | 0.038 |
| 5 | LR + sentence classifier | 0.81 | 0.17 | 0.031 |
| 10 | LR on the 3 current scores | 0.86 | 0.55 | 0.013 |
| 10 | sentence classifier | 0.74 | 0.14 | 0.043 |
| 10 | LR + sentence classifier | 0.83 | 0.23 | 0.033 |
| 20 | LR on the 3 current scores | 0.86 | 0.55 | 0.014 |
| 20 | sentence classifier | 0.78 | 0.18 | 0.039 |
| 20 | LR + sentence classifier | 0.85 | 0.36 | 0.031 |
| 40 | LR on the 3 current scores | 0.85 | 0.56 | 0.019 |
| 40 | sentence classifier | 0.80 | 0.27 | 0.042 |
| 40 | LR + sentence classifier | 0.87 | 0.53 | 0.025 |

### poisonedrag

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.59 | 0.14 | 0.077 |
| 0 | sentence classifier | 0.70 | 0.05 | 0.021 |
| 0 | LR + sentence classifier | 0.70 | 0.04 | 0.025 |
| 5 | LR on the 3 current scores | 0.60 | 0.14 | 0.074 |
| 5 | sentence classifier | 0.73 | 0.04 | 0.016 |
| 5 | LR + sentence classifier | 0.72 | 0.04 | 0.017 |
| 10 | LR on the 3 current scores | 0.60 | 0.15 | 0.074 |
| 10 | sentence classifier | 0.75 | 0.04 | 0.015 |
| 10 | LR + sentence classifier | 0.75 | 0.06 | 0.018 |
| 20 | LR on the 3 current scores | 0.60 | 0.15 | 0.072 |
| 20 | sentence classifier | 0.78 | 0.06 | 0.016 |
| 20 | LR + sentence classifier | 0.77 | 0.09 | 0.020 |
| 40 | LR on the 3 current scores | 0.61 | 0.15 | 0.069 |
| 40 | sentence classifier | 0.81 | 0.10 | 0.017 |
| 40 | LR + sentence classifier | 0.81 | 0.19 | 0.024 |

### pds

Skipped: too few clean documents to hold out calibration, adaptation and evaluation sets.

Caveats:

- Only three source groups exist, and the clean calibration sets are small (20 to 60 documents), so every number is noisy.
- Repetitions share the same underlying documents, so they do not give independent evidence; treat differences of a few points as noise.
- The test split has not been used.
