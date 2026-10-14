# Domain-aware embedding variants (seed 42, train + validation only)

The original embedding score is (1 - max cosine to any baseline document) / 2. The variants average over more neighbours (emb_top3, emb_top5) or compare only with baseline documents of the same domain (emb_dom_max, emb_dom_z). For every score, higher means more anomalous. No variant uses a label; the baseline is the split's baseline role. Thresholds come from the TRAIN clean scores at the 5% budget and are applied unchanged to validation (201 poisoned, 172 clean validation documents; one false alarm = 0.6%).

Domain labels are metadata of the benchmark. In a deployment the domain would be the collection a document comes from (the mailbox, the wiki, the table store), which is usually known, but it would have to be recorded.

## Setup

- Baseline documents: 285. Domains with fewer than 5 baseline documents fall back to the whole baseline: hr_support_policy.
- Consistency check: the original score recomputed from these embeddings differs from the cached original by at most 1.1e-07 (batch and single encoding differ in the last digits; fine).
- The ms column of the cached variant signals is only the time after the embedding exists; add the original embedding's ms for the full cost.

### Domain x label (train)

| domain | clean | poisoned | baseline docs | note |
|---|---|---|---|---|
| code | 47 | 47 | 21 |  |
| email | 36 | 36 | 16 |  |
| hr_support_policy | 5 | 0 | 3 | 100% clean: shortcut risk; falls back to the whole baseline |
| software_docs | 25 | 42 | 14 |  |
| table | 177 | 177 | 107 |  |
| wikipedia_qa | 224 | 300 | 124 |  |

Shortcut risk: hr_support_policy has more than 90% of one label in train, so a score that depends on the domain could partly encode the label. Read those domains with care.

## Each signal alone (threshold at the 5% budget)

| signal | AUC train | AUC validation | AUC inside a domain, train / validation | TPR (95% CI) | FPR (95% CI) |
|---|---|---|---|---|---|
| original embedding | 0.62 | 0.58 | 0.66 / 0.63 | 0.08 [0.04, 0.14] | 0.081 [0.043, 0.122] |
| emb_top3 | 0.60 | 0.57 | 0.66 / 0.63 | 0.09 [0.05, 0.14] | 0.076 [0.039, 0.117] |
| emb_top5 | 0.60 | 0.56 | 0.65 / 0.62 | 0.05 [0.03, 0.09] | 0.070 [0.031, 0.110] |
| emb_dom_max | 0.62 | 0.58 | 0.67 / 0.65 | 0.10 [0.05, 0.16] | 0.076 [0.037, 0.115] |
| emb_dom_z | 0.62 | 0.59 | 0.67 / 0.65 | 0.11 [0.07, 0.16] | 0.064 [0.029, 0.102] |

AUC below 0.50 means poisoned documents look MORE like the baseline than clean ones do. 'Inside a domain' is the unweighted mean AUC over the domains with at least 5 documents of each label (5 domains in train, 5 in validation): it shows whether a variant separates better within a domain. If the overall AUC rises but this column does not, the gain comes from putting the domains on a common scale, not from a better look at each document. A false-alarm rate far above the budget means the train threshold did not carry over, and the recall is partly free.

## LR on (pattern, embedding signal, perplexity), the signal replacing the original embedding

AUC train is out-of-fold. The 'original embedding' row is the 'LR on the 3 current scores' method of `fpr_eval`.

| signal | AUC train | AUC validation | TPR (95% CI) | FPR (95% CI) |
|---|---|---|---|---|
| original embedding | 0.78 | 0.73 | 0.33 [0.25, 0.42] | 0.076 [0.035, 0.116] |
| emb_top3 | 0.78 | 0.73 | 0.34 [0.25, 0.44] | 0.105 [0.060, 0.153] |
| emb_top5 | 0.77 | 0.72 | 0.33 [0.25, 0.42] | 0.110 [0.066, 0.159] |
| emb_dom_max | 0.78 | 0.74 | 0.35 [0.27, 0.45] | 0.093 [0.049, 0.139] |
| emb_dom_z | 0.77 | 0.72 | 0.29 [0.22, 0.39] | 0.058 [0.024, 0.095] |

## AUC by source group (train / validation)

| signal | bipia | poisonedrag | pds |
|---|---|---|---|
| original embedding | 0.58 / 0.59 | 0.62 / 0.49 | 0.90 / 0.89 |
| emb_top3 | 0.58 / 0.58 | 0.61 / 0.46 | 0.91 / 0.90 |
| emb_top5 | 0.57 / 0.58 | 0.61 / 0.45 | 0.88 / 0.90 |
| emb_dom_max | 0.59 / 0.59 | 0.62 / 0.50 | 0.95 / 0.95 |
| emb_dom_z | 0.60 / 0.61 | 0.62 / 0.50 | 0.95 / 0.94 |

## Selection (rule fixed in advance, uses TRAIN AUC of the signal alone, all three seeds)

| signal | train AUC seed 42 | train AUC seed 43 | train AUC seed 44 | mean |
|---|---|---|---|---|
| original embedding | 0.615 | 0.607 | 0.589 | 0.604 |
| emb_top3 | 0.604 | 0.599 | 0.589 | 0.597 |
| emb_top5 | 0.599 | 0.593 | 0.585 | 0.592 |
| emb_dom_max | 0.618 | 0.607 | 0.588 | 0.604 |
| emb_dom_z | 0.625 | 0.610 | 0.595 | 0.610 |

Rule: highest mean TRAIN AUC over the seeds; the original stays unless the best variant beats it by 0.03.

**Selected: original.** No variant is clearly better than the original: the best one (emb_dom_z) gains +0.006 mean TRAIN AUC, below the required 0.03. The original embedding stays.

Caveats:

- Validation numbers are a preview only. The choice above was made on TRAIN AUC and not on anything shown for validation.
- Four variants were compared, so the best of them is optimistic even on train; that is why the rule asks for a margin.
- A negative result is a result: if the original stays, the paper says the domain-aware variants were tested and did not help.
