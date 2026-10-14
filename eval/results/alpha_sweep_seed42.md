# False-alarm budget sweep, seed 42 (task 5.3, validation only)

Validation: 373 documents (201 poisoned, 172 clean). For every method and budget the threshold is the quantile of the out-of-fold TRAIN clean scores (1116 train documents), applied unchanged to validation; nothing here is tuned on validation. Intervals (95%) resample groups, not single documents.

**FPR resolution 0.58%.** One false alarm among the 172 clean validation documents is 0.58%. Budgets marked † (0.5%, 0.1%) allow less than one false alarm on validation, so the realised FPR cannot confirm them: where no false alarm was seen the upper end of the FPR interval is the rule-of-three bound 3/172 = 1.7%, not zero.

The fusion candidate is feature set F1 of task 4.10 (pattern, embedding, perplexity, sentence); **the candidate is provisional** (4.4 not merged yet), so rerun this script after task 4.10 is rerun.

## Recall (TPR) as the allowed false-alarm rate shrinks

| method | 10% | 5% | 2% | 1% | 0.5%† | 0.1%† |
|---|---|---|---|---|---|---|
| pattern bank | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 |
| sentence classifier | 0.76 | 0.64 | 0.39 | 0.31 | 0.30 | 0.27 |
| LR on the 3 current scores | 0.39 | 0.33 | 0.30 | 0.25 | 0.25 | 0.24 |
| LR + sentence classifier | 0.82 | 0.64 | 0.49 | 0.36 | 0.31 | 0.28 |
| fusion candidate | 0.82 | 0.64 | 0.49 | 0.36 | 0.31 | 0.28 |
| published classifier | 0.13 | 0.10 | 0.05 | 0.04 | 0.03 | 0.03 |

## Realised false-alarm rate (FPR) on validation

| method | target 10% | target 5% | target 2% | target 1% | target 0.5%† | target 0.1%† |
|---|---|---|---|---|---|---|
| pattern bank | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sentence classifier | 0.093 | 0.041 | 0.023 | 0.012 | 0.000 | 0.000 |
| LR on the 3 current scores | 0.134 | 0.076 | 0.070 | 0.035 | 0.029 | 0.012 |
| LR + sentence classifier | 0.087 | 0.041 | 0.029 | 0.006 | 0.006 | 0.000 |
| fusion candidate | 0.087 | 0.041 | 0.029 | 0.006 | 0.006 | 0.000 |
| published classifier | 0.145 | 0.064 | 0.035 | 0.017 | 0.012 | 0.012 |

A realised FPR above its target means the train threshold did not carry over to validation; such a method gets extra recall for free, so compare methods at similar realised FPR.

## Budget 10%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (51.4) | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.6228 | 52 of 514 (51.4) | 0.76 [0.68, 0.84] | 0.093 [0.056, 0.142] | 201 / 172 |
| LR on the 3 current scores | 0.5420 | 52 of 514 (51.4) | 0.39 [0.30, 0.48] | 0.134 [0.082, 0.185] | 201 / 172 |
| LR + sentence classifier | 0.5345 | 52 of 514 (51.4) | 0.82 [0.75, 0.88] | 0.087 [0.047, 0.132] | 201 / 172 |
| fusion candidate | 0.5345 | 52 of 514 (51.4) | 0.82 [0.75, 0.88] | 0.087 [0.047, 0.132] | 201 / 172 |
| published classifier | -0.7736 | 52 of 514 (51.4) | 0.13 [0.08, 0.19] | 0.145 [0.097, 0.201] | 201 / 172 |

## Budget 5%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (25.7) | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.6743 | 26 of 514 (25.7) | 0.64 [0.55, 0.74] | 0.041 [0.012, 0.077] | 201 / 172 |
| LR on the 3 current scores | 0.5546 | 26 of 514 (25.7) | 0.33 [0.25, 0.42] | 0.076 [0.035, 0.116] | 201 / 172 |
| LR + sentence classifier | 0.6532 | 26 of 514 (25.7) | 0.64 [0.56, 0.73] | 0.041 [0.016, 0.076] | 201 / 172 |
| fusion candidate | 0.6532 | 26 of 514 (25.7) | 0.64 [0.56, 0.73] | 0.041 [0.016, 0.076] | 201 / 172 |
| published classifier | 4.1244 | 26 of 514 (25.7) | 0.10 [0.06, 0.16] | 0.064 [0.029, 0.105] | 201 / 172 |

## Budget 2%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (10.3) | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.7762 | 11 of 514 (10.3) | 0.39 [0.31, 0.50] | 0.023 [0.006, 0.049] | 201 / 172 |
| LR on the 3 current scores | 0.5648 | 11 of 514 (10.3) | 0.30 [0.22, 0.40] | 0.070 [0.034, 0.109] | 201 / 172 |
| LR + sentence classifier | 0.7632 | 11 of 514 (10.3) | 0.49 [0.40, 0.59] | 0.029 [0.006, 0.059] | 201 / 172 |
| fusion candidate | 0.7632 | 11 of 514 (10.3) | 0.49 [0.40, 0.59] | 0.029 [0.006, 0.059] | 201 / 172 |
| published classifier | 8.9048 | 11 of 514 (10.3) | 0.05 [0.03, 0.09] | 0.035 [0.011, 0.065] | 201 / 172 |

## Budget 1%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (5.1) | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.8294 | 6 of 514 (5.1) | 0.31 [0.24, 0.41] | 0.012 [0.000, 0.030] | 201 / 172 |
| LR on the 3 current scores | 0.5770 | 6 of 514 (5.1) | 0.25 [0.18, 0.35] | 0.035 [0.006, 0.065] | 201 / 172 |
| LR + sentence classifier | 0.8597 | 6 of 514 (5.1) | 0.36 [0.29, 0.46] | 0.006 [0.000, 0.018] | 201 / 172 |
| fusion candidate | 0.8597 | 6 of 514 (5.1) | 0.36 [0.29, 0.46] | 0.006 [0.000, 0.018] | 201 / 172 |
| published classifier | 11.0720 | 6 of 514 (5.1) | 0.04 [0.02, 0.08] | 0.017 [0.000, 0.040] | 201 / 172 |

## Budget 0.5%† (not verifiable on validation)

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (2.6) | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.8536 | 2 of 514 (2.6) | 0.30 [0.22, 0.40] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| LR on the 3 current scores | 0.5822 | 3 of 514 (2.6) | 0.25 [0.18, 0.34] | 0.029 [0.006, 0.056] | 201 / 172 |
| LR + sentence classifier | 0.9005 | 3 of 514 (2.6) | 0.31 [0.24, 0.41] | 0.006 [0.000, 0.018] | 201 / 172 |
| fusion candidate | 0.9005 | 3 of 514 (2.6) | 0.31 [0.24, 0.41] | 0.006 [0.000, 0.018] | 201 / 172 |
| published classifier | 11.7530 | 3 of 514 (2.6) | 0.03 [0.01, 0.06] | 0.012 [0.000, 0.029] | 201 / 172 |

## Budget 0.1%† (not verifiable on validation)

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (0.5) | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.8843 | 1 of 514 (0.5) | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| LR on the 3 current scores | 0.5949 | 1 of 514 (0.5) | 0.24 [0.17, 0.33] | 0.012 [0.000, 0.029] | 201 / 172 |
| LR + sentence classifier | 0.9226 | 1 of 514 (0.5) | 0.28 [0.21, 0.38] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| fusion candidate | 0.9226 | 1 of 514 (0.5) | 0.28 [0.21, 0.38] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| published classifier | 12.3782 | 1 of 514 (0.5) | 0.03 [0.01, 0.06] | 0.012 [0.000, 0.029] | 201 / 172 |

## In words (generated from the tables above)

- pattern bank: recall 0.24 at 10%, 0.24 at 5%, 0.24 at 2%, 0.24 at 1%, 0.24 at 0.5%, 0.24 at 0.1%.
- sentence classifier: recall 0.76 at 10%, 0.64 at 5%, 0.39 at 2%, 0.31 at 1%, 0.30 at 0.5%, 0.27 at 0.1%.
- LR on the 3 current scores: recall 0.39 at 10%, 0.33 at 5%, 0.30 at 2%, 0.25 at 1%, 0.25 at 0.5%, 0.24 at 0.1%.
- LR + sentence classifier: recall 0.82 at 10%, 0.64 at 5%, 0.49 at 2%, 0.36 at 1%, 0.31 at 0.5%, 0.28 at 0.1%.
- fusion candidate: recall 0.82 at 10%, 0.64 at 5%, 0.49 at 2%, 0.36 at 1%, 0.31 at 0.5%, 0.28 at 0.1%.
- published classifier: recall 0.13 at 10%, 0.10 at 5%, 0.05 at 2%, 0.04 at 1%, 0.03 at 0.5%, 0.03 at 0.1%.

Only the budgets from 10% down to 1% can be checked on validation; the recall at 0.5%, 0.1% comes from a threshold whose false-alarm rate is not verified (FPR resolution 0.58%).

Caveats:

- The pattern bank gives clean documents a score of (almost) zero, so its threshold does not move with the budget and its row is flat.
- At the smallest budgets the TRAIN threshold itself rests on one or two clean documents (see the 'train clean above it' column), so it is noisy from seed to seed.
- The fusion candidate is a logistic regression; when it is feature set F1 it equals 'LR + sentence classifier' and the two rows are identical.
- The published classifier is not fitted on our data; the train split only fixes its cut-off.
- Validation numbers only; the held-out split was not used. A choice made after reading these tables makes them optimistic.
