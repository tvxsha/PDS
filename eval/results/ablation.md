# Ablation: what does the perplexity detector add? (development set)

77 documents (60 poisoned, 17 clean). Scores and timings are replayed from evaluation_raw.csv. Thresholds were chosen on these same documents, so every number is optimistic.

## 1. Cascade with and without the perplexity stage

| cascade | precision | recall | F1 | false alarms | misses | mean ms / document | time saved vs running all three |
|---|---|---|---|---|---|---|---|
| A -> B -> C (as published) | 0.94 | 1.00 | 0.97 | 4 | 0 | 105.3 | 70.5% |
| A -> B only | 0.94 | 1.00 | 0.97 | 4 | 0 | 55.7 | 84.4% |

Run all three on every document: 357.4 ms per document (A 0.12, B 74.56, C 282.73; medians 0.13 / 60.07 / 241.89).
Stage C ran on 13 documents and flagged 0 of them (0 poisoned). The two cascades disagree on 0 documents.
Mean timings include model warm-up on the first documents, which is why they sit above the medians.

Which detector clears each category on its own (raw score at or above that detector's cascade threshold):

| category | documents | A alone | B alone | C alone | C flags a document that A and B both miss |
|---|---|---|---|---|---|
| injection | 15 | 10 | 11 | 4 | 0 |
| authority | 15 | 11 | 14 | 0 | 0 |
| corpus | 15 | 2 | 15 | 0 | 0 |
| perplexity | 15 | 0 | 15 | 15 | 0 |
| clean (false alarms) | 17 | 0 | 4 | 0 | 0 |

## 2. Calibrated mean over every subset of detectors

Each detector is min-max scaled to its own observed range on this set, the chosen scores are averaged, and the threshold with the best F1 on a 0.001 grid is used. Same optimistic treatment for every subset.

| detectors | best threshold | precision | recall | F1 | false alarms | misses |
|---|---|---|---|---|---|---|
| A | 0.000 | 0.78 | 1.00 | 0.876 | 17 | 0 |
| B | 0.440 | 0.93 | 0.93 | 0.933 | 4 | 4 |
| C | 0.000 | 0.78 | 1.00 | 0.876 | 17 | 0 |
| A+B | 0.250 | 0.95 | 1.00 | 0.976 | 3 | 0 |
| A+C | 0.000 | 0.78 | 1.00 | 0.876 | 17 | 0 |
| B+C | 0.220 | 0.94 | 0.97 | 0.951 | 4 | 2 |
| A+B+C | 0.193 | 0.98 | 0.98 | 0.983 | 1 | 1 |

A+B scores F1 0.976 and A+B+C scores 0.983. With 77 documents a single changed decision moves F1 by roughly 0.01, so this gap is about one document.

Reading: on this set, perplexity adds no document that the other two stages miss, and removing it cuts the cascade's mean time per document by about half. That is a statement about this set only. The perplexity detector was designed for fluency-breaking attacks (the pds_perplexity category), and a set with more of them, or with attacks the embedding stage misses, could show a different picture.
