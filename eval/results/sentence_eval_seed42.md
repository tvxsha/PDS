# Sentence classifier evaluation (validation only, thresholds picked on validation: optimistic)

train 520 docs, validation 176 docs (110 poisoned, 66 clean). Recall is measured at a false-positive rate of at most 5%; with only 66 clean validation documents that is a handful of false alarms, so intervals are wide.

| method | AUC | recall @ FPR<=5% (95% CI) | bipia | poisonedrag | pds |
|---|---|---|---|---|---|
| pattern bank | 0.65 | 0.29 [0.21, 0.37] | 0.46 | 0.00 | 0.43 |
| embedding | 0.55 | 0.10 [0.05, 0.17] | 0.04 | 0.20 | 0.21 |
| perplexity | 0.52 | 0.07 [0.02, 0.19] | 0.04 | 0.05 | 0.29 |
| sentence classifier (4.6) | 0.81 | 0.50 [0.38, 0.64] | 0.80 | 0.20 | 0.14 |
| LR on the 3 current scores | 0.72 | 0.38 [0.28, 0.50] | 0.50 | 0.10 | 0.64 |
| LR + sentence classifier | 0.86 | 0.57 [0.38, 0.69] | 0.80 | 0.23 | 0.43 |
| LR + sentence + neighbour density (exploratory) | 0.87 | 0.58 [0.37, 0.72] | 0.79 | 0.57 | 0.43 |

Per-source ROC-AUC:

| method | bipia | poisonedrag | pds |
|---|---|---|---|
| pattern bank | 0.73 | 0.50 | 0.71 |
| embedding | 0.53 | 0.68 | 0.77 |
| perplexity | 0.47 | 0.46 | 0.64 |
| sentence classifier (4.6) | 0.92 | 0.82 | 0.89 |
| LR on the 3 current scores | 0.74 | 0.68 | 0.88 |
| LR + sentence classifier | 0.91 | 0.84 | 0.91 |
| LR + sentence + neighbour density (exploratory) | 0.89 | 0.92 | 0.91 |

Prevalence-adjusted precision at the same operating point (task 1.5):

| method | TPR | FPR | precision at 1% | at 0.1% | at 0.01% | false alarms per 10,000 docs |
|---|---|---|---|---|---|---|
| LR on the 3 current scores | 0.38 | 0.045 | 7.8% | 0.83% | 0.084% | 455 |
| LR + sentence classifier | 0.57 | 0.030 | 16.0% | 1.86% | 0.189% | 303 |

Caveats:

- Thresholds were picked on validation, so these numbers are optimistic. The test split has not been used.
- The sentence classifier learns attack wording from the train split. Its BIPIA gain is within-benchmark; task 4.6 still needs the leave-one-source-out check before any claim about other attack styles.
- Neighbour density is only meaningful if clean documents also have topical neighbours. In this pool the clean PoisonedRAG passages are random NQ passages while the poisoned ones come in groups of five, so its PoisonedRAG result is probably inflated. Treat it as a hint, not a result.
