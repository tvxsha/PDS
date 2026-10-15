# Dry run on the VALIDATION split (seed 42): rehearsal of the final test

This is the same code as the final test run, with validation documents in place of the held-out ones. It is not the result. Thresholds are the TRAIN clean-score quantile at the stated false-alarm budget (out-of-fold scores for fitted methods), applied unchanged to validation.

Split hash 6bdb8ff98e. Evaluation documents: 373 (201 poisoned, 172 clean); 1116 train documents. Config sha256 2ee6766624a2. Intervals are 95% group-bootstrap intervals (1000 resamples of whole groups: the same email, question or twin pair). With only 172 clean documents, one false alarm is 0.6%, so FPR is coarse.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.000] | 12.1% / 1.35% | 0 seen (up to ~174) |
| embedding | 0.08 [0.04, 0.14] | 0.081 [0.043, 0.122] | 1.0% / 0.10% | 814 |
| perplexity | 0.08 [0.04, 0.12] | 0.081 [0.042, 0.120] | 1.0% / 0.10% | 814 |
| naive average of the 3 | 0.22 [0.16, 0.30] | 0.047 [0.017, 0.079] | 4.6% / 0.48% | 465 |
| sentence classifier | 0.64 [0.55, 0.74] | 0.041 [0.012, 0.077] | 13.6% / 1.54% | 407 |
| LR on the 3 current scores | 0.33 [0.25, 0.42] | 0.076 [0.035, 0.116] | 4.2% / 0.43% | 756 |
| LR + sentence classifier | 0.64 [0.56, 0.73] | 0.041 [0.016, 0.076] | 13.7% / 1.55% | 407 |
| published classifier | 0.10 [0.06, 0.16] | 0.064 [0.029, 0.105] | 1.6% / 0.16% | 640 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.000] | 12.1% / 1.35% | 0 seen (up to ~174) |
| embedding | 0.02 [0.01, 0.05] | 0.023 [0.005, 0.048] | 1.1% / 0.11% | 233 |
| perplexity | 0.02 [0.01, 0.05] | 0.012 [0.000, 0.029] | 2.1% / 0.21% | 116 |
| naive average of the 3 | 0.20 [0.14, 0.28] | 0.017 [0.000, 0.041] | 10.6% / 1.16% | 174 |
| sentence classifier | 0.31 [0.24, 0.41] | 0.012 [0.000, 0.030] | 21.4% / 2.63% | 116 |
| LR on the 3 current scores | 0.25 [0.18, 0.35] | 0.035 [0.006, 0.065] | 6.8% / 0.72% | 349 |
| LR + sentence classifier | 0.36 [0.29, 0.46] | 0.006 [0.000, 0.018] | 38.7% / 5.89% | 58 |
| published classifier | 0.04 [0.02, 0.08] | 0.017 [0.000, 0.040] | 2.5% / 0.26% | 174 |

Precision is computed from the TPR and FPR above for the stated shares of poisoned documents; it is not measured on a stream with that share. Where no false alarm was seen it uses the rule-of-three upper bound (3/172 = 1.7%) instead of zero, so zero-false-alarm methods are not flattered.

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
| published classifier | 0.14 / 0.13 | 0.01 / 0.00 | 0.57 / 0.00 |

## By attack type (budget 5%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 15 | 0.07 | 0.00 | 0.00 | 0.00 | 1.00 | 0.07 | 1.00 | 0.00 |
| bipia_text | 72 | 0.60 | 0.00 | 0.15 | 0.53 | 0.89 | 0.60 | 0.69 | 0.17 |
| pds_adversarial | 2 | 0.00 | 0.50 | 0.50 | 0.50 | 0.00 | 0.50 | 0.50 | 0.00 |
| pds_authority | 3 | 0.67 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 0.33 |
| pds_corpus | 3 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.67 |
| pds_injection | 3 | 0.67 | 0.67 | 0.00 | 0.67 | 0.67 | 1.00 | 0.67 | 1.00 |
| pds_perplexity | 3 | 0.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.00 | 0.33 | 0.67 |
| poisonedrag_nq | 100 | 0.00 | 0.09 | 0.01 | 0.01 | 0.43 | 0.13 | 0.57 | 0.01 |

## By source (budget 1%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.51 / 0.00 | 0.00 / 0.00 | 0.29 / 0.00 |
| embedding | 0.00 / 0.00 | 0.00 / 0.05 | 0.36 / 0.00 |
| perplexity | 0.01 / 0.01 | 0.00 / 0.01 | 0.29 / 0.00 |
| naive average of the 3 | 0.39 / 0.02 | 0.01 / 0.01 | 0.43 / 0.00 |
| sentence classifier | 0.68 / 0.02 | 0.03 / 0.00 | 0.07 / 0.00 |
| LR on the 3 current scores | 0.51 / 0.00 | 0.00 / 0.07 | 0.50 / 0.10 |
| LR + sentence classifier | 0.68 / 0.01 | 0.10 / 0.00 | 0.29 / 0.00 |
| published classifier | 0.05 / 0.03 | 0.00 / 0.00 | 0.36 / 0.00 |

## By attack type (budget 1%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 15 | 0.07 | 0.00 | 0.00 | 0.00 | 0.93 | 0.07 | 1.00 | 0.00 |
| bipia_text | 72 | 0.60 | 0.00 | 0.01 | 0.47 | 0.62 | 0.60 | 0.61 | 0.06 |
| pds_adversarial | 2 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 |
| pds_authority | 3 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 | 0.00 |
| pds_corpus | 3 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 0.00 | 0.00 |
| pds_injection | 3 | 0.67 | 0.67 | 0.00 | 0.67 | 0.33 | 0.67 | 0.67 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.33 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| poisonedrag_nq | 100 | 0.00 | 0.00 | 0.00 | 0.01 | 0.03 | 0.00 | 0.10 | 0.00 |

Caveats:

- Validation numbers. Any design choice made after looking at them makes them optimistic; the final test run is the only unbiased use of the held-out documents.
- Nothing here is locked: this command can be repeated, and the files are overwritten.
- The published classifier is not fitted on our data; the train split only fixes its cut-off.
- Small cells (an attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so the realised FPR can sit a little off the target. Compare methods at similar realised FPR: a method that overshoots its budget gets extra recall for free.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- Hand-written adversarial and obfuscated sets are evaluated separately (stress_eval); they are not part of these tables.
