# Dry run on the VALIDATION split (seed 44): rehearsal of the final test

This is the same code as the final test run, with validation documents in place of the held-out ones. It is not the result. Thresholds are the TRAIN clean-score quantile at the stated false-alarm budget (out-of-fold scores for fitted methods), applied unchanged to validation.

Split hash 4d1126b320. Evaluation documents: 373 (201 poisoned, 172 clean); 1116 train documents. Config sha256 4aad23b6b6ce. Intervals are 95% group-bootstrap intervals (1000 resamples of whole groups: the same email, question or twin pair). With only 172 clean documents, one false alarm is 0.6%, so FPR is coarse.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.000] | 13.2% / 1.49% | 0 seen (up to ~174) |
| embedding | 0.04 [0.01, 0.08] | 0.047 [0.017, 0.080] | 1.0% / 0.10% | 465 |
| perplexity | 0.05 [0.02, 0.08] | 0.052 [0.023, 0.087] | 1.0% / 0.10% | 523 |
| naive average of the 3 | 0.25 [0.18, 0.34] | 0.047 [0.017, 0.077] | 5.2% / 0.54% | 465 |
| sentence classifier | 0.61 [0.53, 0.70] | 0.041 [0.017, 0.072] | 13.2% / 1.48% | 407 |
| LR on the 3 current scores | 0.32 [0.25, 0.42] | 0.064 [0.028, 0.103] | 4.9% / 0.50% | 640 |
| LR + sentence classifier | 0.69 [0.61, 0.77] | 0.058 [0.028, 0.096] | 10.7% / 1.18% | 581 |
| published classifier | 0.11 [0.07, 0.16] | 0.105 [0.061, 0.153] | 1.0% / 0.10% | 1047 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.000] | 13.2% / 1.49% | 0 seen (up to ~174) |
| embedding | 0.00 [0.00, 0.00] | 0.006 [0.000, 0.018] | 0.0% / 0.00% | 58 |
| perplexity | 0.02 [0.00, 0.04] | 0.006 [0.000, 0.018] | 3.3% / 0.34% | 58 |
| naive average of the 3 | 0.23 [0.16, 0.32] | 0.012 [0.000, 0.029] | 16.6% / 1.93% | 116 |
| sentence classifier | 0.29 [0.21, 0.38] | 0.006 [0.000, 0.018] | 33.4% / 4.73% | 58 |
| LR on the 3 current scores | 0.27 [0.20, 0.36] | 0.017 [0.000, 0.040] | 13.5% / 1.52% | 174 |
| LR + sentence classifier | 0.32 [0.24, 0.42] | 0.000 [0.000, 0.000] | 15.8% / 1.82% | 0 seen (up to ~174) |
| published classifier | 0.06 [0.03, 0.10] | 0.035 [0.011, 0.065] | 1.7% / 0.17% | 349 |

Precision is computed from the TPR and FPR above for the stated shares of poisoned documents; it is not measured on a stream with that share. Where no false alarm was seen it uses the rule-of-three upper bound (3/172 = 1.7%) instead of zero, so zero-false-alarm methods are not flattered.

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
| published classifier | 0.16 / 0.21 | 0.01 / 0.00 | 0.50 / 0.00 |

## By attack type (budget 5%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 13 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| bipia_text | 74 | 0.62 | 0.00 | 0.08 | 0.58 | 0.78 | 0.62 | 0.73 | 0.19 |
| pds_adversarial | 2 | 0.50 | 0.00 | 0.50 | 1.00 | 0.00 | 0.50 | 0.50 | 0.00 |
| pds_authority | 3 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.33 |
| pds_corpus | 3 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 0.33 |
| pds_injection | 3 | 1.00 | 0.33 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.33 | 1.00 | 1.00 | 0.33 | 0.00 | 0.33 | 0.67 |
| poisonedrag_nq | 100 | 0.00 | 0.06 | 0.00 | 0.00 | 0.45 | 0.11 | 0.64 | 0.01 |

## By source (budget 1%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.53 / 0.00 | 0.00 / 0.00 | 0.50 / 0.00 |
| embedding | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.00 |
| perplexity | 0.00 / 0.00 | 0.00 / 0.00 | 0.29 / 0.10 |
| naive average of the 3 | 0.44 / 0.01 | 0.00 / 0.00 | 0.57 / 0.10 |
| sentence classifier | 0.66 / 0.01 | 0.00 / 0.00 | 0.07 / 0.00 |
| LR on the 3 current scores | 0.53 / 0.00 | 0.01 / 0.03 | 0.50 / 0.10 |
| LR + sentence classifier | 0.64 / 0.00 | 0.02 / 0.00 | 0.50 / 0.00 |
| published classifier | 0.07 / 0.07 | 0.00 / 0.00 | 0.43 / 0.00 |

## By attack type (budget 1%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.77 | 0.00 | 0.77 | 0.00 |
| bipia_text | 74 | 0.62 | 0.00 | 0.00 | 0.51 | 0.64 | 0.62 | 0.62 | 0.08 |
| pds_adversarial | 2 | 0.50 | 0.00 | 0.50 | 1.00 | 0.00 | 0.50 | 0.50 | 0.00 |
| pds_authority | 3 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.33 |
| pds_corpus | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pds_injection | 3 | 1.00 | 0.00 | 0.00 | 1.00 | 0.33 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| poisonedrag_nq | 100 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.02 | 0.00 |

Caveats:

- Validation numbers. Any design choice made after looking at them makes them optimistic; the final test run is the only unbiased use of the held-out documents.
- Nothing here is locked: this command can be repeated, and the files are overwritten.
- The published classifier is not fitted on our data; the train split only fixes its cut-off.
- Small cells (an attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so the realised FPR can sit a little off the target. Compare methods at similar realised FPR: a method that overshoots its budget gets extra recall for free.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- Hand-written adversarial and obfuscated sets are evaluated separately (stress_eval); they are not part of these tables.
