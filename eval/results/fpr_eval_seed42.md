# Evaluation at a fixed false-alarm budget (validation only)

Validation: 373 documents (201 poisoned, 172 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. TPR is the share of poisoned documents flagged; FPR the share of clean documents flagged. Intervals (95%) resample groups. With only 172 clean validation documents, one false alarm is 0.6%, so FPR is coarse.

Flag-everything reference: TPR 1.00, FPR 1.00, precision 0.54, F1 0.70. That is the bar the F1 numbers in earlier reports have to beat.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.000] | 0.24 | 12.1% / 1.35% | 0 seen (up to ~174) |
| embedding | 0.08 [0.04, 0.14] | 0.081 [0.043, 0.122] | 0.04 | 1.0% / 0.10% | 814 |
| perplexity | 0.08 [0.04, 0.12] | 0.081 [0.042, 0.120] | 0.06 | 1.0% / 0.10% | 814 |
| naive average of the 3 | 0.22 [0.16, 0.30] | 0.047 [0.017, 0.079] | 0.22 | 4.6% / 0.48% | 465 |
| sentence classifier | 0.64 [0.55, 0.74] | 0.041 [0.012, 0.077] | 0.69 | 13.6% / 1.54% | 407 |
| LR on the 3 current scores | 0.33 [0.25, 0.42] | 0.076 [0.035, 0.116] | 0.27 | 4.2% / 0.43% | 756 |
| LR + sentence classifier | 0.64 [0.56, 0.73] | 0.041 [0.016, 0.076] | 0.75 | 13.7% / 1.55% | 407 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.000] | 0.24 | 12.1% / 1.35% | 0 seen (up to ~174) |
| embedding | 0.02 [0.01, 0.05] | 0.023 [0.005, 0.048] | 0.01 | 1.1% / 0.11% | 233 |
| perplexity | 0.02 [0.01, 0.05] | 0.012 [0.000, 0.029] | 0.01 | 2.1% / 0.21% | 116 |
| naive average of the 3 | 0.20 [0.14, 0.28] | 0.017 [0.000, 0.041] | 0.20 | 10.6% / 1.16% | 174 |
| sentence classifier | 0.31 [0.24, 0.41] | 0.012 [0.000, 0.030] | 0.31 | 21.4% / 2.63% | 116 |
| LR on the 3 current scores | 0.25 [0.18, 0.35] | 0.035 [0.006, 0.065] | 0.24 | 6.8% / 0.72% | 349 |
| LR + sentence classifier | 0.36 [0.29, 0.46] | 0.006 [0.000, 0.018] | 0.38 | 38.7% / 5.89% | 58 |

'best-case TPR' picks the threshold on validation itself (the optimistic number used in earlier reports). The first TPR column is the honest one: the threshold was fixed before validation was looked at. Compare methods at similar realised FPR: a method whose FPR overshoots the budget (for example 9% against 5%) gets extra recall for free.
Precision uses the observed FPR; where no false alarm was seen it uses the rule-of-three upper bound (3/172 = 1.7%) instead of zero, so zero-false-alarm methods are not flattered.

## By source (budget 5%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.51 / 0.00 | 0.00 / 0.00 | 0.29 / 0.00 |
| embedding | 0.00 / 0.00 | 0.09 / 0.16 | 0.57 / 0.20 |
| perplexity | 0.13 / 0.15 | 0.01 / 0.01 | 0.29 / 0.00 |
| naive average of the 3 | 0.44 / 0.08 | 0.01 / 0.01 | 0.43 / 0.00 |
| sentence classifier | 0.91 / 0.08 | 0.43 / 0.00 | 0.43 / 0.00 |
| LR on the 3 current scores | 0.51 / 0.00 | 0.13 / 0.15 | 0.64 / 0.20 |
| LR + sentence classifier | 0.75 / 0.07 | 0.57 / 0.01 | 0.50 / 0.00 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | pattern bank | LR on the 3 current scores | sentence classifier | LR + sentence classifier |
|---|---|---|---|---|---|
| bipia_code | 15 | 0.07 | 0.07 | 1.00 | 1.00 |
| bipia_text | 72 | 0.60 | 0.60 | 0.89 | 0.69 |
| pds_adversarial | 2 | 0.00 | 0.50 | 0.00 | 0.50 |
| pds_authority | 3 | 0.67 | 1.00 | 1.00 | 1.00 |
| pds_corpus | 3 | 0.00 | 0.67 | 0.00 | 0.00 |
| pds_injection | 3 | 0.67 | 1.00 | 0.67 | 0.67 |
| pds_perplexity | 3 | 0.00 | 0.00 | 0.33 | 0.33 |
| poisonedrag_nq | 100 | 0.00 | 0.13 | 0.43 | 0.57 |

Caveats:

- Validation numbers only; the test split has not been used. Thresholds come from train, so this is a fair preview, but any design choice made after looking at these tables makes them optimistic again.
- Small cells (one attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so realised FPR on validation can sit a little off target.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- The calibrated average is not included here (its bounds come from the development set); it is in the protocol_eval table.
