# False-alarm budget sweep, seed 43 (task 5.3, validation only)

Validation: 373 documents (201 poisoned, 172 clean). For every method and budget the threshold is the quantile of the out-of-fold TRAIN clean scores (1116 train documents), applied unchanged to validation; nothing here is tuned on validation. Intervals (95%) resample groups, not single documents.

**FPR resolution 0.58%.** One false alarm among the 172 clean validation documents is 0.58%. Budgets marked † (0.5%, 0.1%) allow less than one false alarm on validation, so the realised FPR cannot confirm them: where no false alarm was seen the upper end of the FPR interval is the rule-of-three bound 3/172 = 1.7%, not zero.

The fusion candidate is feature set F1 of task 4.10 (pattern, embedding, perplexity, sentence); **the candidate is provisional** (4.4 not merged yet), so rerun this script after task 4.10 is rerun.

## Recall (TPR) as the allowed false-alarm rate shrinks

| method | 10% | 5% | 2% | 1% | 0.5%† | 0.1%† |
|---|---|---|---|---|---|---|
| pattern bank | 0.27 | 0.27 | 0.27 | 0.27 | 0.27 | 0.27 |
| sentence classifier | 0.81 | 0.66 | 0.40 | 0.32 | 0.31 | 0.30 |
| LR on the 3 current scores | 0.46 | 0.38 | 0.33 | 0.31 | 0.29 | 0.28 |
| LR + sentence classifier | 0.85 | 0.72 | 0.51 | 0.38 | 0.36 | 0.31 |
| fusion candidate | 0.85 | 0.72 | 0.51 | 0.38 | 0.36 | 0.31 |
| published classifier | 0.14 | 0.12 | 0.05 | 0.04 | 0.04 | 0.03 |

## Realised false-alarm rate (FPR) on validation

| method | target 10% | target 5% | target 2% | target 1% | target 0.5%† | target 0.1%† |
|---|---|---|---|---|---|---|
| pattern bank | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sentence classifier | 0.093 | 0.035 | 0.012 | 0.006 | 0.006 | 0.000 |
| LR on the 3 current scores | 0.140 | 0.099 | 0.052 | 0.023 | 0.012 | 0.006 |
| LR + sentence classifier | 0.081 | 0.047 | 0.006 | 0.006 | 0.000 | 0.000 |
| fusion candidate | 0.081 | 0.047 | 0.006 | 0.006 | 0.000 | 0.000 |
| published classifier | 0.122 | 0.087 | 0.035 | 0.012 | 0.006 | 0.000 |

A realised FPR above its target means the train threshold did not carry over to validation; such a method gets extra recall for free, so compare methods at similar realised FPR.

## Budget 10%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (51.4) | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.6197 | 52 of 514 (51.4) | 0.81 [0.74, 0.87] | 0.093 [0.052, 0.138] | 201 / 172 |
| LR on the 3 current scores | 0.5479 | 52 of 514 (51.4) | 0.46 [0.35, 0.56] | 0.140 [0.088, 0.192] | 201 / 172 |
| LR + sentence classifier | 0.5179 | 52 of 514 (51.4) | 0.85 [0.78, 0.91] | 0.081 [0.045, 0.124] | 201 / 172 |
| fusion candidate | 0.5179 | 52 of 514 (51.4) | 0.85 [0.78, 0.91] | 0.081 [0.045, 0.124] | 201 / 172 |
| published classifier | -1.0458 | 52 of 514 (51.4) | 0.14 [0.09, 0.20] | 0.122 [0.076, 0.173] | 201 / 172 |

## Budget 5%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (25.7) | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.6723 | 26 of 514 (25.7) | 0.66 [0.59, 0.74] | 0.035 [0.011, 0.065] | 201 / 172 |
| LR on the 3 current scores | 0.5592 | 26 of 514 (25.7) | 0.38 [0.30, 0.47] | 0.099 [0.054, 0.148] | 201 / 172 |
| LR + sentence classifier | 0.6509 | 26 of 514 (25.7) | 0.72 [0.64, 0.79] | 0.047 [0.017, 0.079] | 201 / 172 |
| fusion candidate | 0.6509 | 26 of 514 (25.7) | 0.72 [0.64, 0.79] | 0.047 [0.017, 0.079] | 201 / 172 |
| published classifier | 3.0888 | 26 of 514 (25.7) | 0.12 [0.08, 0.18] | 0.087 [0.048, 0.133] | 201 / 172 |

## Budget 2%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (10.3) | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.7788 | 11 of 514 (10.3) | 0.40 [0.32, 0.50] | 0.012 [0.000, 0.030] | 201 / 172 |
| LR on the 3 current scores | 0.5685 | 11 of 514 (10.3) | 0.33 [0.25, 0.42] | 0.052 [0.023, 0.086] | 201 / 172 |
| LR + sentence classifier | 0.7961 | 11 of 514 (10.3) | 0.51 [0.43, 0.60] | 0.006 [0.000, 0.018] | 201 / 172 |
| fusion candidate | 0.7961 | 11 of 514 (10.3) | 0.51 [0.43, 0.60] | 0.006 [0.000, 0.018] | 201 / 172 |
| published classifier | 8.6386 | 11 of 514 (10.3) | 0.05 [0.03, 0.09] | 0.035 [0.011, 0.065] | 201 / 172 |

## Budget 1%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (5.1) | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.8342 | 6 of 514 (5.1) | 0.32 [0.24, 0.42] | 0.006 [0.000, 0.018] | 201 / 172 |
| LR on the 3 current scores | 0.5738 | 6 of 514 (5.1) | 0.31 [0.24, 0.41] | 0.023 [0.006, 0.047] | 201 / 172 |
| LR + sentence classifier | 0.8775 | 6 of 514 (5.1) | 0.38 [0.30, 0.48] | 0.006 [0.000, 0.018] | 201 / 172 |
| fusion candidate | 0.8775 | 6 of 514 (5.1) | 0.38 [0.30, 0.48] | 0.006 [0.000, 0.018] | 201 / 172 |
| published classifier | 11.5329 | 6 of 514 (5.1) | 0.04 [0.02, 0.07] | 0.012 [0.000, 0.029] | 201 / 172 |

## Budget 0.5%† (not verifiable on validation)

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (2.6) | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.8457 | 3 of 514 (2.6) | 0.31 [0.24, 0.41] | 0.006 [0.000, 0.018] | 201 / 172 |
| LR on the 3 current scores | 0.5859 | 3 of 514 (2.6) | 0.29 [0.21, 0.38] | 0.012 [0.000, 0.030] | 201 / 172 |
| LR + sentence classifier | 0.9086 | 3 of 514 (2.6) | 0.36 [0.28, 0.46] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| fusion candidate | 0.9086 | 3 of 514 (2.6) | 0.36 [0.28, 0.46] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| published classifier | 11.9618 | 3 of 514 (2.6) | 0.04 [0.02, 0.07] | 0.006 [0.000, 0.018] | 201 / 172 |

## Budget 0.1%† (not verifiable on validation)

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (0.5) | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.8848 | 1 of 514 (0.5) | 0.30 [0.22, 0.40] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| LR on the 3 current scores | 0.5962 | 1 of 514 (0.5) | 0.28 [0.21, 0.38] | 0.006 [0.000, 0.018] | 201 / 172 |
| LR + sentence classifier | 0.9348 | 1 of 514 (0.5) | 0.31 [0.23, 0.40] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| fusion candidate | 0.9348 | 1 of 514 (0.5) | 0.31 [0.23, 0.40] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| published classifier | 12.8034 | 1 of 514 (0.5) | 0.03 [0.01, 0.06] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |

## In words (generated from the tables above)

- pattern bank: recall 0.27 at 10%, 0.27 at 5%, 0.27 at 2%, 0.27 at 1%, 0.27 at 0.5%, 0.27 at 0.1%.
- sentence classifier: recall 0.81 at 10%, 0.66 at 5%, 0.40 at 2%, 0.32 at 1%, 0.31 at 0.5%, 0.30 at 0.1%.
- LR on the 3 current scores: recall 0.46 at 10%, 0.38 at 5%, 0.33 at 2%, 0.31 at 1%, 0.29 at 0.5%, 0.28 at 0.1%.
- LR + sentence classifier: recall 0.85 at 10%, 0.72 at 5%, 0.51 at 2%, 0.38 at 1%, 0.36 at 0.5%, 0.31 at 0.1%.
- fusion candidate: recall 0.85 at 10%, 0.72 at 5%, 0.51 at 2%, 0.38 at 1%, 0.36 at 0.5%, 0.31 at 0.1%.
- published classifier: recall 0.14 at 10%, 0.12 at 5%, 0.05 at 2%, 0.04 at 1%, 0.04 at 0.5%, 0.03 at 0.1%.

Only the budgets from 10% down to 1% can be checked on validation; the recall at 0.5%, 0.1% comes from a threshold whose false-alarm rate is not verified (FPR resolution 0.58%).

Caveats:

- The pattern bank gives clean documents a score of (almost) zero, so its threshold does not move with the budget and its row is flat.
- At the smallest budgets the TRAIN threshold itself rests on one or two clean documents (see the 'train clean above it' column), so it is noisy from seed to seed.
- The fusion candidate is a logistic regression; when it is feature set F1 it equals 'LR + sentence classifier' and the two rows are identical.
- The published classifier is not fitted on our data; the train split only fixes its cut-off.
- Validation numbers only; the held-out split was not used. A choice made after reading these tables makes them optimistic.
