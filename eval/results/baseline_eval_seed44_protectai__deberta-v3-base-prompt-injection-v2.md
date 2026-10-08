# Our detectors versus a published classifier, at a fixed false-alarm budget (validation only)

Published classifier: `protectai/deberta-v3-base-prompt-injection-v2`, windows of 512 tokens (at most 8 per document), document score = most suspicious window, measured as a logit margin so that probabilities saturating at 0 or 1 do not hide the ranking.

Validation: 176 documents (110 poisoned, 66 clean). Each method's threshold is chosen on the TRAIN split (out-of-fold scores for learned methods) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. The published classifier is not fitted on our data, so for it the train split only fixes the cut-off. Intervals (95%) resample groups. With only 66 clean validation documents, one false alarm is 1.5%.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.23 [0.14, 0.34] | 0.000 [0.000, 0.000] | 0.23 | 4.8% / 0.50% | 0 seen (up to ~455) |
| embedding | 0.05 [0.01, 0.09] | 0.045 [0.000, 0.101] | 0.11 | 1.0% / 0.10% | 455 |
| perplexity | 0.08 [0.02, 0.16] | 0.061 [0.014, 0.121] | 0.08 | 1.3% / 0.13% | 606 |
| LR on the 3 current scores | 0.30 [0.20, 0.42] | 0.045 [0.000, 0.101] | 0.35 | 6.2% / 0.66% | 455 |
| sentence classifier | 0.38 [0.27, 0.52] | 0.015 [0.000, 0.050] | 0.48 | 20.3% / 2.46% | 152 |
| LR + sentence classifier | 0.41 [0.30, 0.55] | 0.030 [0.000, 0.077] | 0.54 | 12.0% / 1.33% | 303 |
| published classifier | 0.13 [0.06, 0.22] | 0.045 [0.000, 0.100] | 0.17 | 2.8% / 0.28% | 455 |
| LR + published classifier | 0.35 [0.25, 0.48] | 0.045 [0.000, 0.100] | 0.35 | 7.1% / 0.76% | 455 |
| LR + sentence + published | 0.44 [0.32, 0.58] | 0.015 [0.000, 0.050] | 0.53 | 22.5% / 2.80% | 152 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|---|
| pattern bank | 0.23 [0.14, 0.34] | 0.000 [0.000, 0.000] | 0.23 | 4.8% / 0.50% | 0 seen (up to ~455) |
| embedding | 0.01 [0.00, 0.03] | 0.000 [0.000, 0.000] | 0.01 | 0.2% / 0.02% | 0 seen (up to ~455) |
| perplexity | 0.03 [0.00, 0.07] | 0.000 [0.000, 0.000] | 0.03 | 0.6% / 0.06% | 0 seen (up to ~455) |
| LR on the 3 current scores | 0.27 [0.18, 0.39] | 0.015 [0.000, 0.053] | 0.26 | 15.4% / 1.77% | 152 |
| sentence classifier | 0.25 [0.17, 0.36] | 0.000 [0.000, 0.000] | 0.26 | 5.4% / 0.56% | 0 seen (up to ~455) |
| LR + sentence classifier | 0.30 [0.20, 0.43] | 0.000 [0.000, 0.000] | 0.31 | 6.2% / 0.66% | 0 seen (up to ~455) |
| published classifier | 0.08 [0.03, 0.15] | 0.015 [0.000, 0.047] | 0.05 | 5.2% / 0.54% | 152 |
| LR + published classifier | 0.30 [0.20, 0.42] | 0.000 [0.000, 0.000] | 0.31 | 6.2% / 0.66% | 0 seen (up to ~455) |
| LR + sentence + published | 0.29 [0.19, 0.42] | 0.000 [0.000, 0.000] | 0.31 | 6.1% / 0.64% | 0 seen (up to ~455) |

## The published classifier as shipped (its own 0.5 cut-off, nothing tuned on our data)

TPR 0.19 [0.10, 0.30], FPR 0.076 [0.027, 0.136]. This is the cleanest number in the report: no threshold of ours is involved.

## By source (budget 5%): TPR / FPR

| method | bipia (56 pois / 26 clean) | poisonedrag (40 pois / 30 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| LR on the 3 current scores | 0.36 / 0.00 | 0.03 / 0.10 | 0.86 / 0.00 |
| LR + sentence classifier | 0.66 / 0.04 | 0.05 / 0.00 | 0.43 / 0.10 |
| published classifier | 0.16 / 0.12 | 0.00 / 0.00 | 0.36 / 0.00 |
| LR + published classifier | 0.38 / 0.04 | 0.12 / 0.07 | 0.86 / 0.00 |
| LR + sentence + published | 0.68 / 0.04 | 0.07 / 0.00 | 0.50 / 0.00 |

## By attack type (budget 5%): TPR among poisoned validation documents

| attack type | n | LR on the 3 current scores | LR + sentence classifier | published classifier | LR + published classifier | LR + sentence + published |
|---|---|---|---|---|---|---|
| bipia_code | 18 | 0.11 | 1.00 | 0.00 | 0.11 | 1.00 |
| bipia_text | 38 | 0.47 | 0.50 | 0.24 | 0.50 | 0.53 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 | 0.50 |
| pds_authority | 3 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| pds_corpus | 3 | 1.00 | 0.00 | 0.00 | 0.67 | 0.00 |
| pds_injection | 3 | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 1.00 | 0.00 | 0.67 | 1.00 | 0.00 |
| poisonedrag_nq | 40 | 0.03 | 0.05 | 0.00 | 0.12 | 0.07 |

## Who catches what (budget 5%): our best method (LR + sentence classifier) against the published classifier

Poisoned validation documents (110): caught by both 8, only by the published classifier 6, only by ours 37, by neither 59.
Clean validation documents (66): falsely flagged by both 0, only by the published classifier 3, only by ours 2.
Flag a document if EITHER flags it: TPR 0.46, FPR 0.076 (the false-alarm rate adds up, which is why the learned combination is the fairer comparison).

## Latency per validation document (ms)

| detector | mean | median | 95th percentile |
|---|---|---|---|
| pattern bank | 0.2 | 0.1 | 0.8 |
| embedding | 28.1 | 19.0 | 52.4 |
| perplexity | 327.9 | 93.1 | 1379.9 |
| all three of ours | 356.2 | 110.6 | 1434.8 |
| published classifier (protectai/deberta-v3-base-prompt-injection-v2) | 695.2 | 411.4 | 1950.2 |

The published classifier used a mean of 1.1 windows per document; 0 of 696 train and validation documents were longer than 8 windows and were cut off. Our detectors' times come from the protocol_eval run on the same documents, measured at a different time, so treat the comparison as indicative. Machine for the published classifier: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel, 18 logical CPUs, CPU only.

Caveats:

- Validation numbers only; the test split has not been used. Any design choice made after looking at these tables makes them optimistic again.
- Contamination is not checked (task 3.7). We do not know how much of BIPIA, PoisonedRAG or similar public text the classifier saw during its own training, so a high score on a public source must not be read as generalisation.
- The classifier was built for prompt-injection text. A low score on PoisonedRAG says it is the wrong tool for knowledge-corruption attacks, not that it is a bad classifier.
- Documents longer than the window cap are scored on their first windows only.
- 'LR + published' rows are fitted on train and use out-of-fold train scores for the threshold, exactly like the other learned rows.
- Small cells (a handful of documents per attack type, one false alarm = a whole percentage point) are noisy. Read the intervals.
