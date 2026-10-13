# Our detectors versus a published classifier, at a fixed false-alarm budget (validation only)

Published classifier: `protectai/deberta-v3-base-prompt-injection-v2`, windows of 512 tokens (at most 8 per document), document score = most suspicious window, measured as a logit margin so that probabilities saturating at 0 or 1 do not hide the ranking.

Validation: 373 documents (201 poisoned, 172 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores for learned methods) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. The published classifier is not fitted on our data, so for it the train split only fixes the cut-off. Intervals (95%) resample groups. With only 172 clean validation documents, one false alarm is 0.6%.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.000] | 0.24 | 12.1% / 1.35% | 0 seen (up to ~174) |
| embedding | 0.08 [0.04, 0.14] | 0.081 [0.043, 0.122] | 0.04 | 1.0% / 0.10% | 814 |
| perplexity | 0.08 [0.04, 0.12] | 0.081 [0.042, 0.120] | 0.06 | 1.0% / 0.10% | 814 |
| LR on the 3 current scores | 0.33 [0.25, 0.42] | 0.076 [0.035, 0.116] | 0.27 | 4.2% / 0.43% | 756 |
| sentence classifier | 0.64 [0.55, 0.74] | 0.041 [0.012, 0.077] | 0.69 | 13.6% / 1.54% | 407 |
| LR + sentence classifier | 0.64 [0.56, 0.73] | 0.041 [0.016, 0.076] | 0.75 | 13.7% / 1.55% | 407 |
| published classifier | 0.10 [0.06, 0.16] | 0.064 [0.029, 0.105] | 0.07 | 1.6% / 0.16% | 640 |
| LR + published classifier | 0.32 [0.24, 0.41] | 0.087 [0.045, 0.131] | 0.27 | 3.6% / 0.36% | 872 |
| LR + sentence + published | 0.65 [0.57, 0.73] | 0.041 [0.016, 0.076] | 0.75 | 13.8% / 1.57% | 407 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.24 [0.17, 0.33] | 0.000 [0.000, 0.000] | 0.24 | 12.1% / 1.35% | 0 seen (up to ~174) |
| embedding | 0.02 [0.01, 0.05] | 0.023 [0.005, 0.048] | 0.01 | 1.1% / 0.11% | 233 |
| perplexity | 0.02 [0.01, 0.05] | 0.012 [0.000, 0.029] | 0.01 | 2.1% / 0.21% | 116 |
| LR on the 3 current scores | 0.25 [0.18, 0.35] | 0.035 [0.006, 0.065] | 0.24 | 6.8% / 0.72% | 349 |
| sentence classifier | 0.31 [0.24, 0.41] | 0.012 [0.000, 0.030] | 0.31 | 21.4% / 2.63% | 116 |
| LR + sentence classifier | 0.36 [0.29, 0.46] | 0.006 [0.000, 0.018] | 0.38 | 38.7% / 5.89% | 58 |
| published classifier | 0.04 [0.02, 0.08] | 0.017 [0.000, 0.040] | 0.02 | 2.5% / 0.26% | 174 |
| LR + published classifier | 0.24 [0.17, 0.33] | 0.035 [0.006, 0.065] | 0.24 | 6.5% / 0.68% | 349 |
| LR + sentence + published | 0.36 [0.29, 0.46] | 0.006 [0.000, 0.018] | 0.38 | 38.7% / 5.89% | 58 |

## The published classifier as shipped (its own 0.5 cut-off, nothing tuned on our data)

TPR 0.13 [0.08, 0.19], FPR 0.128 [0.083, 0.182]. This is the cleanest number in the report: no threshold of ours is involved.

## By source (budget 5%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| LR on the 3 current scores | 0.51 / 0.00 | 0.13 / 0.15 | 0.64 / 0.20 |
| LR + sentence classifier | 0.75 / 0.07 | 0.57 / 0.01 | 0.50 / 0.00 |
| published classifier | 0.14 / 0.13 | 0.01 / 0.00 | 0.57 / 0.00 |
| LR + published classifier | 0.51 / 0.00 | 0.13 / 0.17 | 0.50 / 0.20 |
| LR + sentence + published | 0.75 / 0.07 | 0.58 / 0.01 | 0.50 / 0.00 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | LR on the 3 current scores | LR + sentence classifier | published classifier | LR + published classifier | LR + sentence + published |
|---|---|---|---|---|---|---|
| bipia_code | 15 | 0.07 | 1.00 | 0.00 | 0.07 | 1.00 |
| bipia_text | 72 | 0.60 | 0.69 | 0.17 | 0.60 | 0.69 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 | 0.50 |
| pds_authority | 3 | 1.00 | 1.00 | 0.33 | 0.67 | 1.00 |
| pds_corpus | 3 | 0.67 | 0.00 | 0.67 | 0.67 | 0.00 |
| pds_injection | 3 | 1.00 | 0.67 | 1.00 | 0.67 | 0.67 |
| pds_perplexity | 3 | 0.00 | 0.33 | 0.67 | 0.00 | 0.33 |
| poisonedrag_nq | 100 | 0.13 | 0.57 | 0.01 | 0.13 | 0.58 |

## Who catches what (budget 5%): our best method (LR + sentence classifier) against the published classifier

Poisoned validation documents (201): caught by both 15, only by the published classifier 6, only by ours 114, by neither 66.
Clean validation documents (172): falsely flagged by both 0, only by the published classifier 11, only by ours 7.
Flag a document if EITHER flags it: TPR 0.67, FPR 0.105 (the false-alarm rate adds up, which is why the learned combination is the fairer comparison).

## Latency per validation document (ms)

| detector | mean | median | 95th percentile |
|---|---|---|---|
| pattern bank | 0.4 | 0.1 | 1.4 |
| embedding | 52.0 | 35.4 | 117.7 |
| perplexity | 771.7 | 187.4 | 2944.1 |
| all three of ours | 824.0 | 223.5 | 3074.8 |
| published classifier (protectai/deberta-v3-base-prompt-injection-v2) | 1481.0 | 249.8 | 931.3 |

The published classifier used a mean of 1.0 windows per document; 0 of 1489 train and validation documents were longer than 8 windows and were cut off. Our detectors' times come from the protocol_eval run on the same documents, measured at a different time, so treat the comparison as indicative. Machine for the published classifier: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 18 logical CPUs, CPU only.

Caveats:

- Validation numbers only; the test split has not been used. Any design choice made after looking at these tables makes them optimistic again.
- Contamination is not checked (task 3.7). We do not know how much of BIPIA, PoisonedRAG or similar public text the classifier saw during its own training, so a high score on a public source must not be read as generalisation.
- The classifier was built for prompt-injection text. A low score on PoisonedRAG says it is the wrong tool for knowledge-corruption attacks, not that it is a bad classifier.
- Documents longer than the window cap are scored on their first windows only.
- 'LR + published' rows are fitted on train and use out-of-fold train scores for the threshold, exactly like the other learned rows.
- Small cells (a handful of documents per attack type, one false alarm = a whole percentage point) are noisy. Read the intervals.
