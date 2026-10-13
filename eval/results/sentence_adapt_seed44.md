# New-corpus handling: clean-only calibration and few-shot adaptation (train + validation only)

The model is trained on the OTHER source groups. For the held-out corpus we then use only what a deployer would have. FPR/TPR below are measured on held-out documents that were NOT used for calibration or adaptation.

Target false-alarm rate: 5%. Compare with the fixed-threshold results in sentence_loso (BIPIA: 59% of clean flagged; PoisonedRAG: nothing flagged).

## Part A: clean-only calibration (no poisoned document from the new corpus is used)

### bipia (347 poisoned / 347 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.57 | 0.048 | 0.003 - 0.113 |
| sentence classifier | 20 | 0.08 | 0.049 | 0.003 - 0.110 |
| LR + sentence classifier | 20 | 0.37 | 0.050 | 0.003 - 0.116 |
| LR on the 3 current scores | 40 | 0.57 | 0.048 | 0.010 - 0.092 |
| sentence classifier | 40 | 0.08 | 0.051 | 0.007 - 0.098 |
| LR + sentence classifier | 40 | 0.41 | 0.053 | 0.013 - 0.101 |
| LR on the 3 current scores | 60 | 0.57 | 0.050 | 0.014 - 0.091 |
| sentence classifier | 60 | 0.08 | 0.047 | 0.014 - 0.087 |
| LR + sentence classifier | 60 | 0.40 | 0.048 | 0.014 - 0.091 |

### poisonedrag (400 poisoned / 299 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.06 | 0.047 | 0.004 - 0.111 |
| sentence classifier | 20 | 0.08 | 0.049 | 0.004 - 0.108 |
| LR + sentence classifier | 20 | 0.08 | 0.049 | 0.004 - 0.111 |
| LR on the 3 current scores | 40 | 0.06 | 0.049 | 0.012 - 0.093 |
| sentence classifier | 40 | 0.07 | 0.048 | 0.012 - 0.097 |
| LR + sentence classifier | 40 | 0.07 | 0.048 | 0.012 - 0.093 |
| LR on the 3 current scores | 60 | 0.06 | 0.048 | 0.017 - 0.092 |
| sentence classifier | 60 | 0.07 | 0.051 | 0.017 - 0.092 |
| LR + sentence classifier | 60 | 0.07 | 0.051 | 0.017 - 0.092 |

### pds (56 poisoned / 40 clean)

| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |
|---|---|---|---|---|
| LR on the 3 current scores | 20 | 0.43 | 0.047 | 0.000 - 0.150 |
| sentence classifier | 20 | 0.06 | 0.048 | 0.000 - 0.150 |
| LR + sentence classifier | 20 | 0.23 | 0.053 | 0.000 - 0.150 |

## Part B: calibration plus few-shot adaptation

Training data = other sources + 20 clean documents of the new corpus + about k labelled attack documents of the new corpus (whole groups, so the actual count can be a little above k). A further, separate 20 clean documents set the threshold.
Means over 8 random draws; AUC is threshold-free.

### bipia

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.85 | 0.61 | 0.071 |
| 0 | sentence classifier | 0.68 | 0.14 | 0.075 |
| 0 | LR + sentence classifier | 0.78 | 0.28 | 0.070 |
| 5 | LR on the 3 current scores | 0.85 | 0.61 | 0.074 |
| 5 | sentence classifier | 0.75 | 0.28 | 0.077 |
| 5 | LR + sentence classifier | 0.83 | 0.46 | 0.071 |
| 10 | LR on the 3 current scores | 0.85 | 0.61 | 0.071 |
| 10 | sentence classifier | 0.79 | 0.30 | 0.064 |
| 10 | LR + sentence classifier | 0.85 | 0.51 | 0.069 |
| 20 | LR on the 3 current scores | 0.84 | 0.59 | 0.054 |
| 20 | sentence classifier | 0.80 | 0.35 | 0.051 |
| 20 | LR + sentence classifier | 0.86 | 0.56 | 0.060 |
| 40 | LR on the 3 current scores | 0.82 | 0.60 | 0.085 |
| 40 | sentence classifier | 0.80 | 0.36 | 0.040 |
| 40 | LR + sentence classifier | 0.87 | 0.57 | 0.041 |

### poisonedrag

| attacks added (k) | method | AUC | TPR | FPR |
|---|---|---|---|---|
| 0 | LR on the 3 current scores | 0.55 | 0.05 | 0.048 |
| 0 | sentence classifier | 0.69 | 0.09 | 0.053 |
| 0 | LR + sentence classifier | 0.69 | 0.13 | 0.078 |
| 5 | LR on the 3 current scores | 0.55 | 0.05 | 0.048 |
| 5 | sentence classifier | 0.71 | 0.12 | 0.058 |
| 5 | LR + sentence classifier | 0.71 | 0.16 | 0.077 |
| 10 | LR on the 3 current scores | 0.56 | 0.05 | 0.047 |
| 10 | sentence classifier | 0.73 | 0.13 | 0.050 |
| 10 | LR + sentence classifier | 0.73 | 0.16 | 0.062 |
| 20 | LR on the 3 current scores | 0.56 | 0.05 | 0.047 |
| 20 | sentence classifier | 0.75 | 0.17 | 0.051 |
| 20 | LR + sentence classifier | 0.75 | 0.20 | 0.062 |
| 40 | LR on the 3 current scores | 0.56 | 0.05 | 0.046 |
| 40 | sentence classifier | 0.79 | 0.15 | 0.033 |
| 40 | LR + sentence classifier | 0.79 | 0.20 | 0.050 |

### pds

Skipped: too few clean documents to hold out calibration, adaptation and evaluation sets.

Caveats:

- Only three source groups exist, and the clean calibration sets are small (20 to 60 documents), so every number is noisy.
- Repetitions share the same underlying documents, so they do not give independent evidence; treat differences of a few points as noise.
- The test split has not been used.
