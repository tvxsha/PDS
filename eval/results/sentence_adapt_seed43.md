# New-corpus handling: clean-only calibration and few-shot adaptation (train + validation only)

The model is trained on the OTHER source groups. For the held-out corpus we then use only what a deployer would have. FPR/TPR below are measured on held-out documents that were NOT used for calibration or adaptation.

Target false-alarm rate: 5%. Compare with the fixed-threshold results in sentence_loso (BIPIA: 59% of clean flagged; PoisonedRAG: nothing flagged).

## Part A: clean-only calibration (no poisoned document from the new corpus is used)

### bipia (347 poisoned / 347 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.57 | 0.046 | 0.006 - 0.101 |
| sentence classifier | 20 | 0.06 | 0.042 | 0.003 - 0.095 |
| LR + sentence classifier | 20 | 0.37 | 0.042 | 0.003 - 0.095 |
| LR on the 3 current scores | 40 | 0.57 | 0.050 | 0.010 - 0.107 |
| sentence classifier | 40 | 0.07 | 0.046 | 0.010 - 0.098 |
| LR + sentence classifier | 40 | 0.41 | 0.047 | 0.010 - 0.094 |
| LR on the 3 current scores | 60 | 0.57 | 0.046 | 0.014 - 0.087 |
| sentence classifier | 60 | 0.07 | 0.044 | 0.010 - 0.084 |
| LR + sentence classifier | 60 | 0.43 | 0.050 | 0.014 - 0.094 |

### poisonedrag (400 poisoned / 299 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.09 | 0.052 | 0.004 - 0.122 |
| sentence classifier | 20 | 0.08 | 0.054 | 0.004 - 0.115 |
| LR + sentence classifier | 20 | 0.07 | 0.053 | 0.004 - 0.122 |
| LR on the 3 current scores | 40 | 0.09 | 0.047 | 0.008 - 0.100 |
| sentence classifier | 40 | 0.06 | 0.049 | 0.012 - 0.097 |
| LR + sentence classifier | 40 | 0.06 | 0.050 | 0.012 - 0.093 |
| LR on the 3 current scores | 60 | 0.10 | 0.047 | 0.013 - 0.080 |
| sentence classifier | 60 | 0.05 | 0.048 | 0.017 - 0.084 |
| LR + sentence classifier | 60 | 0.05 | 0.048 | 0.013 - 0.088 |

### pds (56 poisoned / 40 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.53 | 0.048 | 0.000 - 0.150 |
| sentence classifier | 20 | 0.08 | 0.047 | 0.000 - 0.150 |
| LR + sentence classifier | 20 | 0.21 | 0.043 | 0.000 - 0.100 |

## Part B: calibration plus few-shot adaptation

Training data = other sources + 20 clean documents of the new corpus + about k labelled attack documents of the new corpus (whole groups, so the actual count can be a little above k). A further, separate 20 clean documents set the threshold.
Means over 8 random draws; AUC is threshold-free.

### bipia

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.84 | 0.59 | 0.054 |
| 0 | sentence classifier | 0.66 | 0.07 | 0.044 |
| 0 | LR + sentence classifier | 0.77 | 0.16 | 0.047 |
| 5 | LR on the 3 current scores | 0.84 | 0.59 | 0.054 |
| 5 | sentence classifier | 0.76 | 0.19 | 0.039 |
| 5 | LR + sentence classifier | 0.84 | 0.38 | 0.032 |
| 10 | LR on the 3 current scores | 0.84 | 0.59 | 0.056 |
| 10 | sentence classifier | 0.78 | 0.18 | 0.022 |
| 10 | LR + sentence classifier | 0.85 | 0.46 | 0.036 |
| 20 | LR on the 3 current scores | 0.83 | 0.60 | 0.071 |
| 20 | sentence classifier | 0.80 | 0.24 | 0.024 |
| 20 | LR + sentence classifier | 0.86 | 0.53 | 0.040 |
| 40 | LR on the 3 current scores | 0.82 | 0.58 | 0.066 |
| 40 | sentence classifier | 0.82 | 0.33 | 0.024 |
| 40 | LR + sentence classifier | 0.87 | 0.59 | 0.033 |

### poisonedrag

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.58 | 0.09 | 0.054 |
| 0 | sentence classifier | 0.69 | 0.15 | 0.090 |
| 0 | LR + sentence classifier | 0.69 | 0.13 | 0.080 |
| 5 | LR on the 3 current scores | 0.58 | 0.09 | 0.053 |
| 5 | sentence classifier | 0.70 | 0.14 | 0.083 |
| 5 | LR + sentence classifier | 0.71 | 0.14 | 0.074 |
| 10 | LR on the 3 current scores | 0.58 | 0.09 | 0.053 |
| 10 | sentence classifier | 0.72 | 0.17 | 0.083 |
| 10 | LR + sentence classifier | 0.72 | 0.14 | 0.074 |
| 20 | LR on the 3 current scores | 0.58 | 0.09 | 0.051 |
| 20 | sentence classifier | 0.77 | 0.24 | 0.082 |
| 20 | LR + sentence classifier | 0.77 | 0.23 | 0.078 |
| 40 | LR on the 3 current scores | 0.59 | 0.09 | 0.050 |
| 40 | sentence classifier | 0.81 | 0.33 | 0.081 |
| 40 | LR + sentence classifier | 0.81 | 0.35 | 0.088 |

### pds

Skipped: too few clean documents to hold out calibration, adaptation and evaluation sets.

Caveats:

- Only three source groups exist, and the clean calibration sets are small (20 to 60 documents), so every number is noisy.
- Repetitions share the same underlying documents, so they do not give independent evidence; treat differences of a few points as noise.
- The test split has not been used.
