# Evaluation at a fixed false-alarm budget (validation only)

Validation: 373 documents (201 poisoned, 172 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. TPR is the share of poisoned documents flagged; FPR the share of clean documents flagged. Intervals (95%) resample groups. With only 172 clean validation documents, one false alarm is 0.6%, so FPR is coarse.

Flag-everything reference: TPR 1.00, FPR 1.00, precision 0.54, F1 0.70. That is the bar the F1 numbers in earlier reports have to beat.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.000] | 0.26 | 13.2% / 1.49% | 0 seen (up to ~174) |
| embedding | 0.04 [0.01, 0.08] | 0.047 [0.017, 0.080] | 0.07 | 1.0% / 0.10% | 465 |
| perplexity | 0.05 [0.02, 0.08] | 0.052 [0.023, 0.087] | 0.05 | 1.0% / 0.10% | 523 |
| naive average of the 3 | 0.25 [0.18, 0.34] | 0.047 [0.017, 0.077] | 0.25 | 5.2% / 0.54% | 465 |
| sentence classifier | 0.61 [0.53, 0.70] | 0.041 [0.017, 0.072] | 0.65 | 13.2% / 1.48% | 407 |
| LR on the 3 current scores | 0.32 [0.25, 0.42] | 0.064 [0.028, 0.103] | 0.32 | 4.9% / 0.50% | 640 |
| LR + sentence classifier | 0.69 [0.61, 0.77] | 0.058 [0.028, 0.096] | 0.66 | 10.7% / 1.18% | 581 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.000] | 0.26 | 13.2% / 1.49% | 0 seen (up to ~174) |
| embedding | 0.00 [0.00, 0.00] | 0.006 [0.000, 0.018] | 0.00 | 0.0% / 0.00% | 58 |
| perplexity | 0.02 [0.00, 0.04] | 0.006 [0.000, 0.018] | 0.02 | 3.3% / 0.34% | 58 |
| naive average of the 3 | 0.23 [0.16, 0.32] | 0.012 [0.000, 0.029] | 0.23 | 16.6% / 1.93% | 116 |
| sentence classifier | 0.29 [0.21, 0.38] | 0.006 [0.000, 0.018] | 0.29 | 33.4% / 4.73% | 58 |
| LR on the 3 current scores | 0.27 [0.20, 0.36] | 0.017 [0.000, 0.040] | 0.26 | 13.5% / 1.52% | 174 |
| LR + sentence classifier | 0.32 [0.24, 0.42] | 0.000 [0.000, 0.000] | 0.33 | 15.8% / 1.82% | 0 seen (up to ~174) |

'best-case TPR' picks the threshold on validation itself (the optimistic number used in earlier reports). The first TPR column is the honest one: the threshold was fixed before validation was looked at. Compare methods at similar realised FPR: a method whose FPR overshoots the budget (for example 9% against 5%) gets extra recall for free.
Precision uses the observed FPR; where no false alarm was seen it uses the rule-of-three upper bound (3/172 = 1.7%) instead of zero, so zero-false-alarm methods are not flattered.

## By source (budget 5%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.53 / 0.00 | 0.00 / 0.00 | 0.50 / 0.00 |
| embedding | 0.00 / 0.00 | 0.06 / 0.09 | 0.21 / 0.10 |
| perplexity | 0.07 / 0.09 | 0.00 / 0.00 | 0.29 / 0.10 |
| naive average of the 3 | 0.49 / 0.08 | 0.00 / 0.00 | 0.57 / 0.10 |
| sentence classifier | 0.82 / 0.05 | 0.45 / 0.01 | 0.50 / 0.20 |
| LR on the 3 current scores | 0.53 / 0.00 | 0.11 / 0.12 | 0.57 / 0.20 |
| LR + sentence classifier | 0.77 / 0.05 | 0.64 / 0.05 | 0.57 / 0.20 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | pattern bank | LR on the 3 current scores | sentence classifier | LR + sentence classifier |
|---|---|---|---|---|---|
| bipia_code | 13 | 0.00 | 0.00 | 1.00 | 1.00 |
| bipia_text | 74 | 0.62 | 0.62 | 0.78 | 0.73 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 |
| pds_authority | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_corpus | 3 | 0.00 | 0.33 | 0.00 | 0.00 |
| pds_injection | 3 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.00 | 0.33 | 0.33 |
| poisonedrag_nq | 100 | 0.00 | 0.11 | 0.45 | 0.64 |

Caveats:

- Validation numbers only; the test split has not been used. Thresholds come from train, so this is a fair preview, but any design choice made after looking at these tables makes them optimistic again.
- Small cells (one attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so realised FPR on validation can sit a little off target.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- The calibrated average is not included here (its bounds come from the development set); it is in the protocol_eval table.
