# Our detectors versus a published classifier, at a fixed false-alarm budget (validation only)

Published classifier: `protectai/deberta-v3-base-prompt-injection-v2`, windows of 512 tokens (at most 8 per document), document score = most suspicious window, measured as a logit margin so that probabilities saturating at 0 or 1 do not hide the ranking.

Validation: 176 documents (110 poisoned, 66 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores for learned methods) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. The published classifier is not fitted on our data, so for it the train split only fixes the cut-off. Intervals (95%) resample groups. With only 66 clean validation documents, one false alarm is 1.5%.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.29 [0.19, 0.41] | 0.000 [0.000, 0.000] | 0.29 | 6.1% / 0.64% | 0 seen (up to ~455) |
| embedding | 0.10 [0.03, 0.20] | 0.030 [0.000, 0.077] | 0.10 | 3.2% / 0.33% | 303 |
| perplexity | 0.14 [0.06, 0.24] | 0.091 [0.029, 0.161] | 0.07 | 1.5% / 0.15% | 909 |
| LR on the 3 current scores | 0.43 [0.31, 0.56] | 0.091 [0.029, 0.169] | 0.38 | 4.5% / 0.47% | 909 |
| sentence classifier | 0.46 [0.34, 0.63] | 0.030 [0.000, 0.078] | 0.50 | 13.4% / 1.51% | 303 |
| LR + sentence classifier | 0.51 [0.39, 0.66] | 0.030 [0.000, 0.078] | 0.57 | 14.5% / 1.65% | 303 |
| published classifier | 0.20 [0.11, 0.32] | 0.106 [0.033, 0.188] | 0.07 | 1.9% / 0.19% | 1061 |
| LR + published classifier | 0.35 [0.25, 0.50] | 0.061 [0.014, 0.121] | 0.35 | 5.6% / 0.58% | 606 |
| LR + sentence + published | 0.50 [0.37, 0.67] | 0.030 [0.000, 0.078] | 0.59 | 14.3% / 1.62% | 303 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.29 [0.19, 0.41] | 0.000 [0.000, 0.000] | 0.29 | 6.1% / 0.64% | 0 seen (up to ~455) |
| embedding | 0.04 [0.00, 0.12] | 0.000 [0.000, 0.000] | 0.06 | 0.8% / 0.08% | 0 seen (up to ~455) |
| perplexity | 0.10 [0.03, 0.19] | 0.076 [0.016, 0.141] | 0.03 | 1.3% / 0.13% | 758 |
| LR on the 3 current scores | 0.34 [0.22, 0.47] | 0.015 [0.000, 0.052] | 0.32 | 18.3% / 2.17% | 152 |
| sentence classifier | 0.24 [0.15, 0.35] | 0.000 [0.000, 0.000] | 0.39 | 5.0% / 0.52% | 0 seen (up to ~455) |
| LR + sentence classifier | 0.39 [0.28, 0.54] | 0.015 [0.000, 0.050] | 0.39 | 20.7% / 2.52% | 152 |
| published classifier | 0.11 [0.05, 0.19] | 0.061 [0.014, 0.121] | 0.05 | 1.8% / 0.18% | 606 |
| LR + published classifier | 0.35 [0.25, 0.50] | 0.015 [0.000, 0.048] | 0.34 | 19.1% / 2.29% | 152 |
| LR + sentence + published | 0.40 [0.29, 0.55] | 0.015 [0.000, 0.050] | 0.40 | 21.1% / 2.57% | 152 |

## The published classifier as shipped (its own 0.5 cut-off, nothing tuned on our data)

TPR 0.25 [0.15, 0.38], FPR 0.152 [0.066, 0.243]. This is the cleanest number in the report: no threshold of ours is involved.

## By source (budget 5%): TPR / FPR

| method | bipia (56 pois / 26 clean) | poisonedrag (40 pois / 30 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| LR on the 3 current scores | 0.52 / 0.12 | 0.23 / 0.10 | 0.64 / 0.00 |
| LR + sentence classifier | 0.77 / 0.04 | 0.07 / 0.00 | 0.71 / 0.10 |
| published classifier | 0.25 / 0.27 | 0.00 / 0.00 | 0.57 / 0.00 |
| LR + published classifier | 0.48 / 0.15 | 0.00 / 0.00 | 0.86 / 0.00 |
| LR + sentence + published | 0.77 / 0.04 | 0.03 / 0.00 | 0.79 / 0.10 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | LR on the 3 current scores | LR + sentence classifier | published classifier | LR + published classifier | LR + sentence + published |
|---|---|---|---|---|---|---|
| bipia_code | 12 | 0.08 | 1.00 | 0.00 | 0.08 | 1.00 |
| bipia_text | 44 | 0.64 | 0.70 | 0.32 | 0.59 | 0.70 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 | 0.50 |
| pds_authority | 3 | 0.67 | 0.67 | 0.67 | 1.00 | 1.00 |
| pds_corpus | 3 | 0.33 | 0.33 | 0.33 | 0.67 | 0.33 |
| pds_injection | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 |
| poisonedrag_nq | 40 | 0.23 | 0.07 | 0.00 | 0.00 | 0.03 |

## Who catches what (budget 5%): our best method (LR + sentence classifier) against the published classifier

Poisoned validation documents (110): caught by both 17, only by the published classifier 5, only by ours 39, by neither 49.
Clean validation documents (66): falsely flagged by both 0, only by the published classifier 7, only by ours 2.
Flag a document if EITHER flags it: TPR 0.55, FPR 0.136 (the false-alarm rate adds up, which is why the learned combination is the fairer comparison).

## Latency per validation document (ms)

| detector | mean | median | 95th percentile |
|---|---|---|---|
| pattern bank | 0.4 | 0.2 | 1.4 |
| embedding | 67.3 | 51.1 | 131.4 |
| perplexity | 782.2 | 283.4 | 3228.1 |
| all three of ours | 849.8 | 334.8 | 3359.8 |
| published classifier (protectai/deberta-v3-base-prompt-injection-v2) | 537.7 | 405.6 | 1247.3 |

The published classifier used a mean of 1.0 windows per document; 0 of 696 train and validation documents were longer than 8 windows and were cut off. Our detectors' times come from the protocol_eval run on the same documents, measured at a different time, so treat the comparison as indicative. Machine for the published classifier: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 18 logical CPUs, CPU only.

Caveats:

- Validation numbers only; the test split has not been used. Any design choice made after looking at these tables makes them optimistic again.
- Contamination is not checked (task 3.7). We do not know how much of BIPIA, PoisonedRAG or similar public text the classifier saw during its own training, so a high score on a public source must not be read as generalisation.
- The classifier was built for prompt-injection text. A low score on PoisonedRAG says it is the wrong tool for knowledge-corruption attacks, not that it is a bad classifier.
- Documents longer than the window cap are scored on their first windows only.
- 'LR + published' rows are fitted on train and use out-of-fold train scores for the threshold, exactly like the other learned rows.
- Small cells (a handful of documents per attack type, one false alarm = a whole percentage point) are noisy. Read the intervals.
