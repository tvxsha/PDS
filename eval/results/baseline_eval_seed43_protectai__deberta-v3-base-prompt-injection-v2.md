# Our detectors versus a published classifier, at a fixed false-alarm budget (validation only)

Published classifier: `protectai/deberta-v3-base-prompt-injection-v2`, windows of 512 tokens (at most 8 per document), document score = most suspicious window, measured as a logit margin so that probabilities saturating at 0 or 1 do not hide the ranking.

Validation: 176 documents (110 poisoned, 66 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores for learned methods) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. The published classifier is not fitted on our data, so for it the train split only fixes the cut-off. Intervals (95%) resample groups. With only 66 clean validation documents, one false alarm is 1.5%.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.25 [0.17, 0.37] | 0.000 [0.000, 0.000] | 0.25 | 5.4% / 0.56% | 0 seen (up to ~455) |
| embedding | 0.05 [0.00, 0.14] | 0.045 [0.000, 0.103] | 0.05 | 1.0% / 0.10% | 455 |
| perplexity | 0.13 [0.05, 0.22] | 0.106 [0.032, 0.177] | 0.07 | 1.2% / 0.12% | 1061 |
| LR on the 3 current scores | 0.30 [0.19, 0.42] | 0.061 [0.014, 0.121] | 0.30 | 4.8% / 0.49% | 606 |
| sentence classifier | 0.36 [0.26, 0.50] | 0.015 [0.000, 0.048] | 0.45 | 19.5% / 2.35% | 152 |
| LR + sentence classifier | 0.41 [0.30, 0.56] | 0.015 [0.000, 0.048] | 0.45 | 21.4% / 2.63% | 152 |
| published classifier | 0.16 [0.08, 0.27] | 0.061 [0.014, 0.123] | 0.13 | 2.7% / 0.27% | 606 |
| LR + published classifier | 0.30 [0.21, 0.42] | 0.015 [0.000, 0.048] | 0.41 | 16.7% / 1.94% | 152 |
| LR + sentence + published | 0.41 [0.30, 0.56] | 0.015 [0.000, 0.048] | 0.49 | 21.4% / 2.63% | 152 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.25 [0.17, 0.37] | 0.000 [0.000, 0.000] | 0.25 | 5.4% / 0.56% | 0 seen (up to ~455) |
| embedding | 0.03 [0.00, 0.08] | 0.015 [0.000, 0.048] | 0.03 | 1.8% / 0.18% | 152 |
| perplexity | 0.03 [0.00, 0.07] | 0.000 [0.000, 0.000] | 0.03 | 0.6% / 0.06% | 0 seen (up to ~455) |
| LR on the 3 current scores | 0.30 [0.19, 0.42] | 0.015 [0.000, 0.048] | 0.28 | 16.7% / 1.94% | 152 |
| sentence classifier | 0.29 [0.19, 0.42] | 0.000 [0.000, 0.000] | 0.34 | 6.1% / 0.64% | 0 seen (up to ~455) |
| LR + sentence classifier | 0.36 [0.26, 0.51] | 0.000 [0.000, 0.000] | 0.36 | 7.5% / 0.79% | 0 seen (up to ~455) |
| published classifier | 0.07 [0.03, 0.13] | 0.000 [0.000, 0.000] | 0.07 | 1.6% / 0.16% | 0 seen (up to ~455) |
| LR + published classifier | 0.30 [0.21, 0.42] | 0.000 [0.000, 0.000] | 0.30 | 6.2% / 0.66% | 0 seen (up to ~455) |
| LR + sentence + published | 0.36 [0.26, 0.51] | 0.000 [0.000, 0.000] | 0.36 | 7.5% / 0.79% | 0 seen (up to ~455) |

## The published classifier as shipped (its own 0.5 cut-off, nothing tuned on our data)

TPR 0.24 [0.14, 0.36], FPR 0.091 [0.030, 0.169]. This is the cleanest number in the report: no threshold of ours is involved.

## By source (budget 5%): TPR / FPR

| method | bipia (56 pois / 26 clean) | poisonedrag (40 pois / 30 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| LR on the 3 current scores | 0.39 / 0.00 | 0.12 / 0.13 | 0.43 / 0.00 |
| LR + sentence classifier | 0.68 / 0.00 | 0.03 / 0.00 | 0.43 / 0.10 |
| published classifier | 0.18 / 0.15 | 0.00 / 0.00 | 0.57 / 0.00 |
| LR + published classifier | 0.39 / 0.00 | 0.03 / 0.03 | 0.71 / 0.00 |
| LR + sentence + published | 0.68 / 0.00 | 0.03 / 0.00 | 0.43 / 0.10 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | LR on the 3 current scores | LR + sentence classifier | published classifier | LR + published classifier | LR + sentence + published |
|---|---|---|---|---|---|---|
| bipia_code | 16 | 0.06 | 1.00 | 0.00 | 0.06 | 1.00 |
| bipia_text | 40 | 0.53 | 0.55 | 0.25 | 0.53 | 0.55 |
| pds_adversarial | 2 | 0.00 | 0.00 | 0.50 | 0.50 | 0.00 |
| pds_authority | 3 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 |
| pds_corpus | 3 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 |
| pds_injection | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.00 | 0.67 | 0.67 | 0.00 |
| poisonedrag_nq | 40 | 0.12 | 0.03 | 0.00 | 0.03 | 0.03 |

## Who catches what (budget 5%): our best method (LR + sentence classifier) against the published classifier

Poisoned validation documents (110): caught by both 13, only by the published classifier 5, only by ours 32, by neither 60.
Clean validation documents (66): falsely flagged by both 0, only by the published classifier 4, only by ours 1.
Flag a document if EITHER flags it: TPR 0.45, FPR 0.076 (the false-alarm rate adds up, which is why the learned combination is the fairer comparison).

## Latency per validation document (ms)

| detector | mean | median | 95th percentile |
|---|---|---|---|
| pattern bank | 0.3 | 0.1 | 1.0 |
| embedding | 32.0 | 23.7 | 57.8 |
| perplexity | 368.5 | 114.4 | 1436.8 |
| all three of ours | 400.8 | 139.2 | 1495.5 |
| published classifier (protectai/deberta-v3-base-prompt-injection-v2) | 668.9 | 382.3 | 1873.6 |

The published classifier used a mean of 1.1 windows per document; 0 of 696 train and validation documents were longer than 8 windows and were cut off. Our detectors' times come from the protocol_eval run on the same documents, measured at a different time, so treat the comparison as indicative. Machine for the published classifier: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 18 logical CPUs, CPU only.

Caveats:

- Validation numbers only; the test split has not been used. Any design choice made after looking at these tables makes them optimistic again.
- Contamination is not checked (task 3.7). We do not know how much of BIPIA, PoisonedRAG or similar public text the classifier saw during its own training, so a high score on a public source must not be read as generalisation.
- The classifier was built for prompt-injection text. A low score on PoisonedRAG says it is the wrong tool for knowledge-corruption attacks, not that it is a bad classifier.
- Documents longer than the window cap are scored on their first windows only.
- 'LR + published' rows are fitted on train and use out-of-fold train scores for the threshold, exactly like the other learned rows.
- Small cells (a handful of documents per attack type, one false alarm = a whole percentage point) are noisy. Read the intervals.
