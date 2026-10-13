# Our detectors versus a published classifier, at a fixed false-alarm budget (validation only)

Published classifier: `protectai/deberta-v3-base-prompt-injection-v2`, windows of 512 tokens (at most 8 per document), document score = most suspicious window, measured as a logit margin so that probabilities saturating at 0 or 1 do not hide the ranking.

Validation: 373 documents (201 poisoned, 172 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores for learned methods) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. The published classifier is not fitted on our data, so for it the train split only fixes the cut-off. Intervals (95%) resample groups. With only 172 clean validation documents, one false alarm is 0.6%.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.000] | 0.26 | 13.2% / 1.49% | 0 seen (up to ~174) |
| embedding | 0.04 [0.01, 0.08] | 0.047 [0.017, 0.080] | 0.07 | 1.0% / 0.10% | 465 |
| perplexity | 0.05 [0.02, 0.08] | 0.052 [0.023, 0.087] | 0.05 | 1.0% / 0.10% | 523 |
| LR on the 3 current scores | 0.32 [0.25, 0.42] | 0.064 [0.028, 0.103] | 0.32 | 4.9% / 0.50% | 640 |
| sentence classifier | 0.61 [0.53, 0.70] | 0.041 [0.017, 0.072] | 0.65 | 13.2% / 1.48% | 407 |
| LR + sentence classifier | 0.69 [0.61, 0.77] | 0.058 [0.028, 0.096] | 0.66 | 10.7% / 1.18% | 581 |
| published classifier | 0.11 [0.07, 0.16] | 0.105 [0.061, 0.153] | 0.06 | 1.0% / 0.10% | 1047 |
| LR + published classifier | 0.32 [0.25, 0.42] | 0.058 [0.023, 0.096] | 0.31 | 5.3% / 0.55% | 581 |
| LR + sentence + published | 0.69 [0.61, 0.77] | 0.064 [0.033, 0.103] | 0.65 | 9.8% / 1.06% | 640 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.26 [0.19, 0.35] | 0.000 [0.000, 0.000] | 0.26 | 13.2% / 1.49% | 0 seen (up to ~174) |
| embedding | 0.00 [0.00, 0.00] | 0.006 [0.000, 0.018] | 0.00 | 0.0% / 0.00% | 58 |
| perplexity | 0.02 [0.00, 0.04] | 0.006 [0.000, 0.018] | 0.02 | 3.3% / 0.34% | 58 |
| LR on the 3 current scores | 0.27 [0.20, 0.36] | 0.017 [0.000, 0.040] | 0.26 | 13.5% / 1.52% | 174 |
| sentence classifier | 0.29 [0.21, 0.38] | 0.006 [0.000, 0.018] | 0.29 | 33.4% / 4.73% | 58 |
| LR + sentence classifier | 0.32 [0.24, 0.42] | 0.000 [0.000, 0.000] | 0.33 | 15.8% / 1.82% | 0 seen (up to ~174) |
| published classifier | 0.06 [0.03, 0.10] | 0.035 [0.011, 0.065] | 0.03 | 1.7% / 0.17% | 349 |
| LR + published classifier | 0.27 [0.20, 0.36] | 0.017 [0.000, 0.040] | 0.26 | 13.5% / 1.52% | 174 |
| LR + sentence + published | 0.31 [0.24, 0.41] | 0.000 [0.000, 0.000] | 0.32 | 15.4% / 1.77% | 0 seen (up to ~174) |

## The published classifier as shipped (its own 0.5 cut-off, nothing tuned on our data)

TPR 0.14 [0.09, 0.20], FPR 0.134 [0.084, 0.188]. This is the cleanest number in the report: no threshold of ours is involved.

## By source (budget 5%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| LR on the 3 current scores | 0.53 / 0.00 | 0.11 / 0.12 | 0.57 / 0.20 |
| LR + sentence classifier | 0.77 / 0.05 | 0.64 / 0.05 | 0.57 / 0.20 |
| published classifier | 0.16 / 0.21 | 0.01 / 0.00 | 0.50 / 0.00 |
| LR + published classifier | 0.53 / 0.00 | 0.11 / 0.11 | 0.57 / 0.20 |
| LR + sentence + published | 0.76 / 0.06 | 0.64 / 0.05 | 0.57 / 0.20 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | LR on the 3 current scores | LR + sentence classifier | published classifier | LR + published classifier | LR + sentence + published |
|---|---|---|---|---|---|---|
| bipia_code | 13 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 |
| bipia_text | 74 | 0.62 | 0.73 | 0.19 | 0.62 | 0.72 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 | 0.50 |
| pds_authority | 3 | 1.00 | 1.00 | 0.33 | 1.00 | 1.00 |
| pds_corpus | 3 | 0.33 | 0.00 | 0.33 | 0.33 | 0.00 |
| pds_injection | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.33 | 0.67 | 0.00 | 0.33 |
| poisonedrag_nq | 100 | 0.11 | 0.64 | 0.01 | 0.11 | 0.64 |

## Who catches what (budget 5%): our best method (LR + sentence classifier) against the published classifier

Poisoned validation documents (201): caught by both 15, only by the published classifier 7, only by ours 124, by neither 55.
Clean validation documents (172): falsely flagged by both 0, only by the published classifier 18, only by ours 10.
Flag a document if EITHER flags it: TPR 0.73, FPR 0.163 (the false-alarm rate adds up, which is why the learned combination is the fairer comparison).

## Latency per validation document (ms)

| detector | mean | median | 95th percentile |
|---|---|---|---|
| pattern bank | 0.5 | 0.2 | 1.8 |
| embedding | 71.8 | 58.8 | 135.0 |
| perplexity | 1025.8 | 266.6 | 3249.9 |
| all three of ours | 1098.1 | 329.5 | 3371.8 |
| published classifier (protectai/deberta-v3-base-prompt-injection-v2) | 399.6 | 256.4 | 1428.6 |

The published classifier used a mean of 1.0 windows per document; 0 of 1489 train and validation documents were longer than 8 windows and were cut off. Our detectors' times come from the protocol_eval run on the same documents, measured at a different time, so treat the comparison as indicative. Machine for the published classifier: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 18 logical CPUs, CPU only.

Caveats:

- Validation numbers only; the test split has not been used. Any design choice made after looking at these tables makes them optimistic again.
- Contamination is not checked (task 3.7). We do not know how much of BIPIA, PoisonedRAG or similar public text the classifier saw during its own training, so a high score on a public source must not be read as generalisation.
- The classifier was built for prompt-injection text. A low score on PoisonedRAG says it is the wrong tool for knowledge-corruption attacks, not that it is a bad classifier.
- Documents longer than the window cap are scored on their first windows only.
- 'LR + published' rows are fitted on train and use out-of-fold train scores for the threshold, exactly like the other learned rows.
- Small cells (a handful of documents per attack type, one false alarm = a whole percentage point) are noisy. Read the intervals.
