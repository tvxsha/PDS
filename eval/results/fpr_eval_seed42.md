# Evaluation at a fixed false-alarm budget (validation only)

Validation: 176 documents (110 poisoned, 66 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. TPR is the share of poisoned documents flagged; FPR the share of clean documents flagged. Intervals (95%) resample groups. With only 66 clean validation documents, one false alarm is 1.5%, so FPR is coarse.

Flag-everything reference: TPR 1.00, FPR 1.00, precision 0.62, F1 0.77. That is the bar the F1 numbers in earlier reports have to beat.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.29 [0.19, 0.41] | 0.000 [0.000, 0.000] | 0.29 | 6.1% / 0.64% | 0 seen (up to ~455) |
| embedding | 0.10 [0.03, 0.20] | 0.030 [0.000, 0.077] | 0.10 | 3.2% / 0.33% | 303 |
| perplexity | 0.14 [0.06, 0.24] | 0.091 [0.029, 0.161] | 0.07 | 1.5% / 0.15% | 909 |
| naive average of the 3 | 0.25 [0.15, 0.38] | 0.106 [0.033, 0.176] | 0.22 | 2.4% / 0.24% | 1061 |
| sentence classifier | 0.46 [0.34, 0.63] | 0.030 [0.000, 0.078] | 0.50 | 13.4% / 1.51% | 303 |
| LR on the 3 current scores | 0.43 [0.31, 0.56] | 0.091 [0.029, 0.169] | 0.38 | 4.5% / 0.47% | 909 |
| LR + sentence classifier | 0.51 [0.39, 0.66] | 0.030 [0.000, 0.078] | 0.57 | 14.5% / 1.65% | 303 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.29 [0.19, 0.41] | 0.000 [0.000, 0.000] | 0.29 | 6.1% / 0.64% | 0 seen (up to ~455) |
| embedding | 0.04 [0.00, 0.12] | 0.000 [0.000, 0.000] | 0.06 | 0.8% / 0.08% | 0 seen (up to ~455) |
| perplexity | 0.10 [0.03, 0.19] | 0.076 [0.016, 0.141] | 0.03 | 1.3% / 0.13% | 758 |
| naive average of the 3 | 0.25 [0.14, 0.36] | 0.061 [0.014, 0.125] | 0.19 | 3.9% / 0.40% | 606 |
| sentence classifier | 0.24 [0.15, 0.35] | 0.000 [0.000, 0.000] | 0.39 | 5.0% / 0.52% | 0 seen (up to ~455) |
| LR on the 3 current scores | 0.34 [0.22, 0.47] | 0.015 [0.000, 0.052] | 0.32 | 18.3% / 2.17% | 152 |
| LR + sentence classifier | 0.39 [0.28, 0.54] | 0.015 [0.000, 0.050] | 0.39 | 20.7% / 2.52% | 152 |

'best-case TPR' picks the threshold on validation itself (the optimistic number used in earlier reports). The first TPR column is the honest one: the threshold was fixed before validation was looked at. Compare methods at similar realised FPR: a method whose FPR overshoots the budget (for example 9% against 5%) gets extra recall for free.
Precision uses the observed FPR; where no false alarm was seen it uses the rule-of-three upper bound (3/66 = 4.5%) instead of zero, so zero-false-alarm methods are not flattered.

## By source (budget 5%): TPR / FPR

| method | bipia (56 pois / 26 clean) | poisonedrag (40 pois / 30 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.46 / 0.00 | 0.00 / 0.00 | 0.43 / 0.00 |
| embedding | 0.00 / 0.00 | 0.20 / 0.07 | 0.21 / 0.00 |
| perplexity | 0.21 / 0.23 | 0.00 / 0.00 | 0.21 / 0.00 |
| naive average of the 3 | 0.34 / 0.27 | 0.00 / 0.00 | 0.64 / 0.00 |
| sentence classifier | 0.80 / 0.04 | 0.00 / 0.00 | 0.43 / 0.10 |
| LR on the 3 current scores | 0.52 / 0.12 | 0.23 / 0.10 | 0.64 / 0.00 |
| LR + sentence classifier | 0.77 / 0.04 | 0.07 / 0.00 | 0.71 / 0.10 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | pattern bank | LR on the 3 current scores | sentence classifier | LR + sentence classifier |
|---|---|---|---|---|---|
| bipia_code | 12 | 0.08 | 0.08 | 1.00 | 1.00 |
| bipia_text | 44 | 0.57 | 0.64 | 0.75 | 0.70 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 |
| pds_authority | 3 | 0.67 | 0.67 | 0.67 | 0.67 |
| pds_corpus | 3 | 0.33 | 0.33 | 0.33 | 0.33 |
| pds_injection | 3 | 0.67 | 0.67 | 0.67 | 1.00 |
| pds_perplexity | 3 | 0.00 | 1.00 | 0.33 | 1.00 |
| poisonedrag_nq | 40 | 0.00 | 0.23 | 0.00 | 0.07 |

Caveats:

- Validation numbers only; the test split has not been used. Thresholds come from train, so this is a fair preview, but any design choice made after looking at these tables makes them optimistic again.
- Small cells (one attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so realised FPR on validation can sit a little off target.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- The calibrated average is not included here (its bounds come from the development set); it is in the protocol_eval table.
