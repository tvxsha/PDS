# False-alarm budget sweep, seed 44 (task 5.3, validation only)

Validation: 373 documents (201 poisoned, 172 clean). For every method and budget the threshold is the quantile of the out-of-fold TRAIN clean scores (1116 train documents), applied unchanged to validation; nothing here is tuned on validation. Intervals (95%) resample groups, not single documents.

**FPR resolution 0.58%.** One false alarm among the 172 clean validation documents is 0.58%. Budgets marked † (0.5%, 0.1%) allow less than one false alarm on validation, so the realised FPR cannot confirm them: where no false alarm was seen the upper end of the FPR interval is the rule-of-three bound 3/172 = 1.7%, not zero.

The fusion candidate is feature set F1 of task 4.10 (pattern, embedding, perplexity, sentence); **the candidate is provisional** (4.4 not merged yet), so rerun this script after task 4.10 is rerun.

## Recall (TPR) as the allowed false-alarm rate shrinks

| method | 10% | 5% | 2% | 1% | 0.5%† | 0.1%† |
|---|---|---|---|---|---|---|
| pattern bank | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 | 0.26 |
| sentence classifier | 0.76 | 0.61 | 0.40 | 0.29 | 0.27 | 0.25 |
| LR on the 3 current scores | 0.39 | 0.32 | 0.29 | 0.27 | 0.26 | 0.26 |
| LR + sentence classifier | 0.81 | 0.69 | 0.43 | 0.32 | 0.28 | 0.28 |
| fusion candidate | 0.81 | 0.69 | 0.43 | 0.32 | 0.28 | 0.28 |
| published classifier | 0.14 | 0.11 | 0.07 | 0.06 | 0.05 | 0.04 |

## Realised false-alarm rate (FPR) on validation

| method | target 10% | target 5% | target 2% | target 1% | target 0.5%† | target 0.1%† |
|---|---|---|---|---|---|---|
| pattern bank | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| sentence classifier | 0.087 | 0.041 | 0.035 | 0.006 | 0.000 | 0.000 |
| LR on the 3 current scores | 0.105 | 0.064 | 0.047 | 0.017 | 0.012 | 0.000 |
| LR + sentence classifier | 0.081 | 0.058 | 0.023 | 0.000 | 0.000 | 0.000 |
| fusion candidate | 0.081 | 0.058 | 0.023 | 0.000 | 0.000 | 0.000 |
| published classifier | 0.140 | 0.105 | 0.058 | 0.035 | 0.029 | 0.017 |

A realised FPR above its target means the train threshold did not carry over to validation; such a method gets extra recall for free, so compare methods at similar realised FPR.

## Budget 10%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (51.4) | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.6326 | 52 of 514 (51.4) | 0.76 [0.69, 0.82] | 0.087 [0.048, 0.131] | 201 / 172 |
| LR on the 3 current scores | 0.5432 | 52 of 514 (51.4) | 0.39 [0.30, 0.49] | 0.105 [0.060, 0.155] | 201 / 172 |
| LR + sentence classifier | 0.5540 | 52 of 514 (51.4) | 0.81 [0.75, 0.86] | 0.081 [0.042, 0.125] | 201 / 172 |
| fusion candidate | 0.5540 | 52 of 514 (51.4) | 0.81 [0.75, 0.86] | 0.081 [0.042, 0.125] | 201 / 172 |
| published classifier | -0.6495 | 52 of 514 (51.4) | 0.14 [0.09, 0.20] | 0.140 [0.090, 0.193] | 201 / 172 |

## Budget 5%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (25.7) | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.6907 | 26 of 514 (25.7) | 0.61 [0.53, 0.70] | 0.041 [0.017, 0.072] | 201 / 172 |
| LR on the 3 current scores | 0.5530 | 26 of 514 (25.7) | 0.32 [0.25, 0.42] | 0.064 [0.028, 0.103] | 201 / 172 |
| LR + sentence classifier | 0.6411 | 26 of 514 (25.7) | 0.69 [0.61, 0.77] | 0.058 [0.028, 0.096] | 201 / 172 |
| fusion candidate | 0.6411 | 26 of 514 (25.7) | 0.69 [0.61, 0.77] | 0.058 [0.028, 0.096] | 201 / 172 |
| published classifier | 3.8625 | 26 of 514 (25.7) | 0.11 [0.07, 0.16] | 0.105 [0.061, 0.153] | 201 / 172 |

## Budget 2%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (10.3) | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.7620 | 11 of 514 (10.3) | 0.40 [0.32, 0.51] | 0.035 [0.012, 0.065] | 201 / 172 |
| LR on the 3 current scores | 0.5647 | 11 of 514 (10.3) | 0.29 [0.21, 0.38] | 0.047 [0.017, 0.080] | 201 / 172 |
| LR + sentence classifier | 0.8040 | 11 of 514 (10.3) | 0.43 [0.36, 0.53] | 0.023 [0.006, 0.048] | 201 / 172 |
| fusion candidate | 0.8040 | 11 of 514 (10.3) | 0.43 [0.36, 0.53] | 0.023 [0.006, 0.048] | 201 / 172 |
| published classifier | 7.6570 | 11 of 514 (10.3) | 0.07 [0.04, 0.12] | 0.058 [0.028, 0.097] | 201 / 172 |

## Budget 1%

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (5.1) | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.8383 | 6 of 514 (5.1) | 0.29 [0.21, 0.38] | 0.006 [0.000, 0.018] | 201 / 172 |
| LR on the 3 current scores | 0.5738 | 6 of 514 (5.1) | 0.27 [0.20, 0.36] | 0.017 [0.000, 0.040] | 201 / 172 |
| LR + sentence classifier | 0.8948 | 6 of 514 (5.1) | 0.32 [0.24, 0.42] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| fusion candidate | 0.8948 | 6 of 514 (5.1) | 0.32 [0.24, 0.42] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| published classifier | 9.9358 | 6 of 514 (5.1) | 0.06 [0.03, 0.10] | 0.035 [0.011, 0.065] | 201 / 172 |

## Budget 0.5%† (not verifiable on validation)

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (2.6) | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.8611 | 3 of 514 (2.6) | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| LR on the 3 current scores | 0.5817 | 3 of 514 (2.6) | 0.26 [0.19, 0.35] | 0.012 [0.000, 0.031] | 201 / 172 |
| LR + sentence classifier | 0.9152 | 3 of 514 (2.6) | 0.28 [0.21, 0.38] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| fusion candidate | 0.9152 | 3 of 514 (2.6) | 0.28 [0.21, 0.38] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| published classifier | 11.2347 | 3 of 514 (2.6) | 0.05 [0.03, 0.09] | 0.029 [0.006, 0.057] | 201 / 172 |

## Budget 0.1%† (not verifiable on validation)

| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |
|---|---|---|---|---|---|
| pattern bank | 0.0000 | 0 of 514 (0.5) | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| sentence classifier | 0.8924 | 1 of 514 (0.5) | 0.25 [0.18, 0.34] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| LR on the 3 current scores | 0.5891 | 1 of 514 (0.5) | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| LR + sentence classifier | 0.9350 | 1 of 514 (0.5) | 0.28 [0.20, 0.37] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| fusion candidate | 0.9350 | 1 of 514 (0.5) | 0.28 [0.20, 0.37] | 0.000 [0.000, 0.017] (rule of 3) | 201 / 172 |
| published classifier | 11.9656 | 1 of 514 (0.5) | 0.04 [0.02, 0.07] | 0.017 [0.000, 0.041] | 201 / 172 |

## In words (generated from the tables above)

- pattern bank: recall 0.26 at 10%, 0.26 at 5%, 0.26 at 2%, 0.26 at 1%, 0.26 at 0.5%, 0.26 at 0.1%.
- sentence classifier: recall 0.76 at 10%, 0.61 at 5%, 0.40 at 2%, 0.29 at 1%, 0.27 at 0.5%, 0.25 at 0.1%.
- LR on the 3 current scores: recall 0.39 at 10%, 0.32 at 5%, 0.29 at 2%, 0.27 at 1%, 0.26 at 0.5%, 0.26 at 0.1%.
- LR + sentence classifier: recall 0.81 at 10%, 0.69 at 5%, 0.43 at 2%, 0.32 at 1%, 0.28 at 0.5%, 0.28 at 0.1%.
- fusion candidate: recall 0.81 at 10%, 0.69 at 5%, 0.43 at 2%, 0.32 at 1%, 0.28 at 0.5%, 0.28 at 0.1%.
- published classifier: recall 0.14 at 10%, 0.11 at 5%, 0.07 at 2%, 0.06 at 1%, 0.05 at 0.5%, 0.04 at 0.1%.

Only the budgets from 10% down to 1% can be checked on validation; the recall at 0.5%, 0.1% comes from a threshold whose false-alarm rate is not verified (FPR resolution 0.58%).

Caveats:

- The pattern bank gives clean documents a score of (almost) zero, so its threshold does not move with the budget and its row is flat.
- At the smallest budgets the TRAIN threshold itself rests on one or two clean documents (see the 'train clean above it' column), so it is noisy from seed to seed.
- The fusion candidate is a logistic regression; when it is feature set F1 it equals 'LR + sentence classifier' and the two rows are identical.
- The published classifier is not fitted on our data; the train split only fixes its cut-off.
- Validation numbers only; the held-out split was not used. A choice made after reading these tables makes them optimistic.
