# Evaluation at a fixed false-alarm budget (validation only)

Validation: 373 documents (201 poisoned, 172 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. TPR is the share of poisoned documents flagged; FPR the share of clean documents flagged. Intervals (95%) resample groups. With only 172 clean validation documents, one false alarm is 0.6%, so FPR is coarse.

Flag-everything reference: TPR 1.00, FPR 1.00, precision 0.54, F1 0.70. That is the bar the F1 numbers in earlier reports have to beat.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.000] | 0.27 | 13.5% / 1.52% | 0 seen (up to ~174) |
| embedding | 0.13 [0.07, 0.20] | 0.064 [0.030, 0.102] | 0.11 | 2.0% / 0.20% | 640 |
| perplexity | 0.04 [0.02, 0.08] | 0.064 [0.030, 0.104] | 0.04 | 0.7% / 0.07% | 640 |
| naive average of the 3 | 0.26 [0.19, 0.34] | 0.058 [0.024, 0.095] | 0.24 | 4.3% / 0.44% | 581 |
| sentence classifier | 0.66 [0.59, 0.74] | 0.035 [0.011, 0.065] | 0.70 | 16.1% / 1.86% | 349 |
| LR on the 3 current scores | 0.38 [0.30, 0.47] | 0.099 [0.054, 0.148] | 0.32 | 3.8% / 0.39% | 988 |
| LR + sentence classifier | 0.72 [0.64, 0.79] | 0.047 [0.017, 0.079] | 0.74 | 13.5% / 1.53% | 465 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.000] | 0.27 | 13.5% / 1.52% | 0 seen (up to ~174) |
| embedding | 0.04 [0.01, 0.09] | 0.006 [0.000, 0.018] | 0.04 | 7.2% / 0.77% | 58 |
| perplexity | 0.02 [0.01, 0.05] | 0.012 [0.000, 0.030] | 0.02 | 2.1% / 0.21% | 116 |
| naive average of the 3 | 0.24 [0.17, 0.32] | 0.023 [0.006, 0.048] | 0.23 | 9.4% / 1.02% | 233 |
| sentence classifier | 0.32 [0.24, 0.42] | 0.006 [0.000, 0.018] | 0.35 | 36.0% / 5.27% | 58 |
| LR on the 3 current scores | 0.31 [0.24, 0.41] | 0.023 [0.006, 0.047] | 0.29 | 12.0% / 1.33% | 233 |
| LR + sentence classifier | 0.38 [0.30, 0.48] | 0.006 [0.000, 0.018] | 0.54 | 40.0% / 6.19% | 58 |

'best-case TPR' picks the threshold on validation itself (the optimistic number used in earlier reports). The first TPR column is the honest one: the threshold was fixed before validation was looked at. Compare methods at similar realised FPR: a method whose FPR overshoots the budget (for example 9% against 5%) gets extra recall for free.
Precision uses the observed FPR; where no false alarm was seen it uses the rule-of-three upper bound (3/172 = 1.7%) instead of zero, so zero-false-alarm methods are not flattered.

## By source (budget 5%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.55 / 0.00 | 0.00 / 0.00 | 0.43 / 0.00 |
| embedding | 0.00 / 0.00 | 0.16 / 0.15 | 0.71 / 0.00 |
| perplexity | 0.05 / 0.10 | 0.01 / 0.01 | 0.29 / 0.10 |
| naive average of the 3 | 0.51 / 0.09 | 0.01 / 0.01 | 0.50 / 0.10 |
| sentence classifier | 0.90 / 0.05 | 0.47 / 0.00 | 0.57 / 0.20 |
| LR on the 3 current scores | 0.55 / 0.00 | 0.20 / 0.20 | 0.64 / 0.20 |
| LR + sentence classifier | 0.82 / 0.03 | 0.66 / 0.03 | 0.57 / 0.30 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | pattern bank | LR on the 3 current scores | sentence classifier | LR + sentence classifier |
|---|---|---|---|---|---|
| bipia_code | 12 | 0.08 | 0.08 | 1.00 | 1.00 |
| bipia_text | 75 | 0.63 | 0.63 | 0.88 | 0.79 |
| pds_adversarial | 2 | 0.00 | 0.00 | 0.50 | 0.50 |
| pds_authority | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_corpus | 3 | 0.33 | 1.00 | 0.33 | 0.33 |
| pds_injection | 3 | 0.67 | 1.00 | 0.67 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.00 | 0.33 | 0.00 |
| poisonedrag_nq | 100 | 0.00 | 0.20 | 0.47 | 0.66 |

Caveats:

- Validation numbers only; the test split has not been used. Thresholds come from train, so this is a fair preview, but any design choice made after looking at these tables makes them optimistic again.
- Small cells (one attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so realised FPR on validation can sit a little off target.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- The calibrated average is not included here (its bounds come from the development set); it is in the protocol_eval table.
