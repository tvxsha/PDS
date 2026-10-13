# Our detectors versus a published classifier, at a fixed false-alarm budget (validation only)

Published classifier: `protectai/deberta-v3-base-prompt-injection-v2`, windows of 512 tokens (at most 8 per document), document score = most suspicious window, measured as a logit margin so that probabilities saturating at 0 or 1 do not hide the ranking.

Validation: 373 documents (201 poisoned, 172 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores for learned methods) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. The published classifier is not fitted on our data, so for it the train split only fixes the cut-off. Intervals (95%) resample groups. With only 172 clean validation documents, one false alarm is 0.6%.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.000] | 0.27 | 13.5% / 1.52% | 0 seen (up to ~174) |
| embedding | 0.13 [0.07, 0.20] | 0.064 [0.030, 0.102] | 0.11 | 2.0% / 0.20% | 640 |
| perplexity | 0.04 [0.02, 0.08] | 0.064 [0.030, 0.104] | 0.04 | 0.7% / 0.07% | 640 |
| LR on the 3 current scores | 0.38 [0.30, 0.47] | 0.099 [0.054, 0.148] | 0.32 | 3.8% / 0.39% | 988 |
| sentence classifier | 0.66 [0.59, 0.74] | 0.035 [0.011, 0.065] | 0.70 | 16.1% / 1.86% | 349 |
| LR + sentence classifier | 0.72 [0.64, 0.79] | 0.047 [0.017, 0.079] | 0.74 | 13.5% / 1.53% | 465 |
| published classifier | 0.12 [0.08, 0.18] | 0.087 [0.048, 0.133] | 0.07 | 1.4% / 0.14% | 872 |
| LR + published classifier | 0.41 [0.33, 0.51] | 0.116 [0.071, 0.167] | 0.33 | 3.5% / 0.35% | 1163 |
| LR + sentence + published | 0.72 [0.64, 0.79] | 0.047 [0.017, 0.079] | 0.74 | 13.5% / 1.53% | 465 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.27 [0.20, 0.36] | 0.000 [0.000, 0.000] | 0.27 | 13.5% / 1.52% | 0 seen (up to ~174) |
| embedding | 0.04 [0.01, 0.09] | 0.006 [0.000, 0.018] | 0.04 | 7.2% / 0.77% | 58 |
| perplexity | 0.02 [0.01, 0.05] | 0.012 [0.000, 0.030] | 0.02 | 2.1% / 0.21% | 116 |
| LR on the 3 current scores | 0.31 [0.24, 0.41] | 0.023 [0.006, 0.047] | 0.29 | 12.0% / 1.33% | 233 |
| sentence classifier | 0.32 [0.24, 0.42] | 0.006 [0.000, 0.018] | 0.35 | 36.0% / 5.27% | 58 |
| LR + sentence classifier | 0.38 [0.30, 0.48] | 0.006 [0.000, 0.018] | 0.54 | 40.0% / 6.19% | 58 |
| published classifier | 0.04 [0.02, 0.07] | 0.012 [0.000, 0.029] | 0.04 | 3.3% / 0.34% | 116 |
| LR + published classifier | 0.32 [0.24, 0.41] | 0.029 [0.006, 0.054] | 0.30 | 10.1% / 1.10% | 291 |
| LR + sentence + published | 0.38 [0.30, 0.48] | 0.006 [0.000, 0.018] | 0.54 | 40.0% / 6.19% | 58 |

## The published classifier as shipped (its own 0.5 cut-off, nothing tuned on our data)

TPR 0.13 [0.09, 0.19], FPR 0.110 [0.065, 0.159]. This is the cleanest number in the report: no threshold of ours is involved.

## By source (budget 5%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| LR on the 3 current scores | 0.55 / 0.00 | 0.20 / 0.20 | 0.64 / 0.20 |
| LR + sentence classifier | 0.82 / 0.03 | 0.66 / 0.03 | 0.57 / 0.30 |
| published classifier | 0.16 / 0.17 | 0.02 / 0.00 | 0.64 / 0.00 |
| LR + published classifier | 0.55 / 0.00 | 0.26 / 0.24 | 0.64 / 0.20 |
| LR + sentence + published | 0.82 / 0.03 | 0.66 / 0.03 | 0.57 / 0.30 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | LR on the 3 current scores | LR + sentence classifier | published classifier | LR + published classifier | LR + sentence + published |
|---|---|---|---|---|---|---|
| bipia_code | 12 | 0.08 | 1.00 | 0.00 | 0.08 | 1.00 |
| bipia_text | 75 | 0.63 | 0.79 | 0.19 | 0.63 | 0.79 |
| pds_adversarial | 2 | 0.00 | 0.50 | 0.00 | 0.00 | 0.50 |
| pds_authority | 3 | 1.00 | 1.00 | 0.33 | 1.00 | 1.00 |
| pds_corpus | 3 | 1.00 | 0.33 | 0.67 | 1.00 | 0.33 |
| pds_injection | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| poisonedrag_nq | 100 | 0.20 | 0.66 | 0.02 | 0.26 | 0.66 |

## Who catches what (budget 5%): our best method (LR + sentence classifier) against the published classifier

Poisoned validation documents (201): caught by both 17, only by the published classifier 8, only by ours 128, by neither 48.
Clean validation documents (172): falsely flagged by both 1, only by the published classifier 14, only by ours 7.
Flag a document if EITHER flags it: TPR 0.76, FPR 0.128 (the false-alarm rate adds up, which is why the learned combination is the fairer comparison).

## Latency per validation document (ms)

| detector | mean | median | 95th percentile |
|---|---|---|---|
| pattern bank | 0.4 | 0.2 | 1.4 |
| embedding | 77.3 | 41.8 | 152.7 |
| perplexity | 926.2 | 218.1 | 3225.5 |
| all three of ours | 1003.9 | 268.1 | 3369.5 |
| published classifier (protectai/deberta-v3-base-prompt-injection-v2) | 11178.4 | 258.4 | 1578.1 |

The published classifier used a mean of 1.0 windows per document; 0 of 1489 train and validation documents were longer than 8 windows and were cut off. Our detectors' times come from the protocol_eval run on the same documents, measured at a different time, so treat the comparison as indicative. Machine for the published classifier: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 18 logical CPUs, CPU only.

Caveats:

- Validation numbers only; the test split has not been used. Any design choice made after looking at these tables makes them optimistic again.
- Contamination is not checked (task 3.7). We do not know how much of BIPIA, PoisonedRAG or similar public text the classifier saw during its own training, so a high score on a public source must not be read as generalisation.
- The classifier was built for prompt-injection text. A low score on PoisonedRAG says it is the wrong tool for knowledge-corruption attacks, not that it is a bad classifier.
- Documents longer than the window cap are scored on their first windows only.
- 'LR + published' rows are fitted on train and use out-of-fold train scores for the threshold, exactly like the other learned rows.
- Small cells (a handful of documents per attack type, one false alarm = a whole percentage point) are noisy. Read the intervals.
