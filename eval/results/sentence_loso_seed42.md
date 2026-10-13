# Leave-one-source-out check for the sentence classifier (train + validation only)

Each block trains on the other source groups and tests on the held-out one, so the held-out attack style, wording and topics are never seen. Nothing was tuned on the held-out data.

- AUC: threshold-free. 0.50 means no signal.
- best-case recall: recall at FPR <= 5% using the best threshold ON the held-out source (optimistic).
- transferred TPR / FPR: threshold set so that 5% of the TRAINING sources' clean documents are flagged, then applied unchanged to the held-out source. This is what a new deployment would actually see.

## Held out: bipia (347 poisoned / 347 clean)

| method | AUC (95% CI) | best-case recall | transferred TPR | transferred FPR |
|---|---|---|---|---|
| pattern bank | 0.77 [0.74, 0.80] | 0.54 | 0.54 | 0.00 |
| LR on the 3 current scores | 0.86 [0.83, 0.88] | 0.59 | 0.54 | 0.00 |
| sentence classifier | 0.58 [0.53, 0.62] | 0.05 | 0.86 | 0.78 |
| LR + sentence classifier | 0.79 [0.75, 0.82] | 0.41 | 0.86 | 0.58 |

## Held out: poisonedrag (400 poisoned / 299 clean)

| method | AUC (95% CI) | best-case recall | transferred TPR | transferred FPR |
|---|---|---|---|---|
| pattern bank | 0.50 [0.50, 0.50] | 0.00 | 0.00 | 0.00 |
| LR on the 3 current scores | 0.60 [0.56, 0.64] | 0.14 | 0.62 | 0.48 |
| sentence classifier | 0.71 [0.67, 0.74] | 0.12 | 0.01 | 0.00 |
| LR + sentence classifier | 0.70 [0.66, 0.73] | 0.12 | 0.01 | 0.00 |

## Held out: pds (56 poisoned / 40 clean)

| method | AUC (95% CI) | best-case recall | transferred TPR | transferred FPR |
|---|---|---|---|---|
| pattern bank | 0.69 [0.63, 0.75] | 0.38 | 0.38 | 0.00 |
| LR on the 3 current scores | 0.71 [0.61, 0.80] | 0.54 | 0.62 | 0.15 |
| sentence classifier | 0.60 [0.49, 0.72] | 0.12 | 0.23 | 0.07 |
| LR + sentence classifier | 0.69 [0.59, 0.79] | 0.41 | 0.50 | 0.15 |

Caveats:

- Only three source groups exist, so this is a coarse test of generalisation, not a guarantee about unseen attacks.
- Intervals come from resampling documents; documents from the same BIPIA email or PoisonedRAG question are related, so the true uncertainty is somewhat larger.
- The test split has not been used.
