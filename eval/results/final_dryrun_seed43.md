# Dry run on the VALIDATION split (seed 43): rehearsal of the final test

This is the same code as the final test run, with validation documents in place of the held-out ones. It is not the result. Thresholds are the TRAIN clean-score quantile at the stated false-alarm budget (out-of-fold scores for fitted methods), applied unchanged to validation.

Split hash 93a61f0778. Evaluation documents: 373 (201 poisoned, 172 clean); 1116 train documents. Config sha256 2ee6766624a2. Intervals are 95% group-bootstrap intervals (1000 resamples of whole groups: the same email, question or twin pair). With only 172 clean documents, one false alarm is 0.6%, so FPR is coarse.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.000] | 13.5% / 1.52% | 0 seen (up to ~174) |
| embedding | 0.13 [0.07, 0.20] | 0.064 [0.030, 0.102] | 2.0% / 0.20% | 640 |
| perplexity | 0.04 [0.02, 0.08] | 0.064 [0.030, 0.104] | 0.7% / 0.07% | 640 |
| naive average of the 3 | 0.26 [0.19, 0.34] | 0.058 [0.024, 0.095] | 4.3% / 0.44% | 581 |
| sentence classifier | 0.66 [0.59, 0.74] | 0.035 [0.011, 0.065] | 16.1% / 1.86% | 349 |
| LR on the 3 current scores | 0.38 [0.30, 0.47] | 0.099 [0.054, 0.148] | 3.8% / 0.39% | 988 |
| LR + sentence classifier | 0.72 [0.64, 0.79] | 0.047 [0.017, 0.079] | 13.5% / 1.53% | 465 |
| published classifier | 0.12 [0.08, 0.18] | 0.087 [0.048, 0.133] | 1.4% / 0.14% | 872 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.000] | 13.5% / 1.52% | 0 seen (up to ~174) |
| embedding | 0.04 [0.01, 0.09] | 0.006 [0.000, 0.018] | 7.2% / 0.77% | 58 |
| perplexity | 0.02 [0.01, 0.05] | 0.012 [0.000, 0.030] | 2.1% / 0.21% | 116 |
| naive average of the 3 | 0.24 [0.17, 0.32] | 0.023 [0.006, 0.048] | 9.4% / 1.02% | 233 |
| sentence classifier | 0.32 [0.24, 0.42] | 0.006 [0.000, 0.018] | 36.0% / 5.27% | 58 |
| LR on the 3 current scores | 0.31 [0.24, 0.41] | 0.023 [0.006, 0.047] | 12.0% / 1.33% | 233 |
| LR + sentence classifier | 0.38 [0.30, 0.48] | 0.006 [0.000, 0.018] | 40.0% / 6.19% | 58 |
| published classifier | 0.04 [0.02, 0.07] | 0.012 [0.000, 0.029] | 3.3% / 0.34% | 116 |

Precision is computed from the TPR and FPR above for the stated shares of poisoned documents; it is not measured on a stream with that share. Where no false alarm was seen it uses the rule-of-three upper bound (3/172 = 1.7%) instead of zero, so zero-false-alarm methods are not flattered.

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
| published classifier | 0.16 / 0.17 | 0.02 / 0.00 | 0.64 / 0.00 |

## By attack type (budget 5%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 12 | 0.08 | 0.00 | 0.00 | 0.08 | 1.00 | 0.08 | 1.00 | 0.00 |
| bipia_text | 75 | 0.63 | 0.00 | 0.05 | 0.57 | 0.88 | 0.63 | 0.79 | 0.19 |
| pds_adversarial | 2 | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | 0.00 | 0.50 | 0.00 |
| pds_authority | 3 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.33 |
| pds_corpus | 3 | 0.33 | 1.00 | 0.00 | 0.33 | 0.33 | 1.00 | 0.33 | 0.67 |
| pds_injection | 3 | 0.67 | 0.67 | 0.00 | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.33 | 1.00 | 1.00 | 0.33 | 0.00 | 0.00 | 1.00 |
| poisonedrag_nq | 100 | 0.00 | 0.16 | 0.01 | 0.01 | 0.47 | 0.20 | 0.66 | 0.02 |

## By source (budget 1%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.55 / 0.00 | 0.00 / 0.00 | 0.43 / 0.00 |
| embedding | 0.00 / 0.00 | 0.04 / 0.01 | 0.36 / 0.00 |
| perplexity | 0.01 / 0.01 | 0.00 / 0.00 | 0.29 / 0.10 |
| naive average of the 3 | 0.46 / 0.03 | 0.01 / 0.00 | 0.50 / 0.10 |
| sentence classifier | 0.72 / 0.01 | 0.02 / 0.00 | 0.00 / 0.00 |
| LR on the 3 current scores | 0.55 / 0.00 | 0.07 / 0.05 | 0.57 / 0.00 |
| LR + sentence classifier | 0.70 / 0.01 | 0.10 / 0.00 | 0.43 / 0.00 |
| published classifier | 0.02 / 0.02 | 0.00 / 0.00 | 0.43 / 0.00 |

## By attack type (budget 1%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 12 | 0.08 | 0.00 | 0.00 | 0.08 | 1.00 | 0.08 | 1.00 | 0.00 |
| bipia_text | 75 | 0.63 | 0.00 | 0.01 | 0.52 | 0.68 | 0.63 | 0.65 | 0.03 |
| pds_adversarial | 2 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 |
| pds_authority | 3 | 1.00 | 0.33 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.33 |
| pds_corpus | 3 | 0.33 | 0.33 | 0.00 | 0.33 | 0.00 | 0.67 | 0.33 | 0.00 |
| pds_injection | 3 | 0.67 | 0.67 | 0.00 | 0.67 | 0.00 | 1.00 | 0.67 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.33 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| poisonedrag_nq | 100 | 0.00 | 0.04 | 0.00 | 0.01 | 0.02 | 0.07 | 0.10 | 0.00 |

Caveats:

- Validation numbers. Any design choice made after looking at them makes them optimistic; the final test run is the only unbiased use of the held-out documents.
- Nothing here is locked: this command can be repeated, and the files are overwritten.
- The published classifier is not fitted on our data; the train split only fixes its cut-off.
- Small cells (an attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so the realised FPR can sit a little off the target. Compare methods at similar realised FPR: a method that overshoots its budget gets extra recall for free.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- Hand-written adversarial and obfuscated sets are evaluated separately (stress_eval); they are not part of these tables.
