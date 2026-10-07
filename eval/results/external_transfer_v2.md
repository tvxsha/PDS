# Zero-shot external transfer (frozen development pipeline)

Embedding baseline, calibration bounds and thresholds are exactly those used for the 77-document development set. Nothing was tuned on these documents.

## Calibrated fusion at threshold 0.15

| source | n clean | n poisoned | recall | FPR | precision | F1 |
|---|---|---|---|---|---|---|
| bipia | 60 | 60 | 0.98 | 1.00 | 0.50 | 0.66 |
| poisonedrag | 60 | 60 | 1.00 | 1.00 | 0.50 | 0.67 |

## Calibrated fusion at threshold 0.2

| source | n clean | n poisoned | recall | FPR | precision | F1 |
|---|---|---|---|---|---|---|
| bipia | 60 | 60 | 0.93 | 0.97 | 0.49 | 0.64 |
| poisonedrag | 60 | 60 | 1.00 | 1.00 | 0.50 | 0.67 |

## Does each detector carry ANY signal here? (ROC-AUC, threshold-free)

0.50 = no better than a coin flip; below 0.50 = the detector points the wrong way.

| source | pattern | embedding | perplexity | calibrated fusion |
|---|---|---|---|---|
| bipia | 0.73 | 0.46 | 0.54 | 0.61 |
| poisonedrag | 0.50 | 0.64 | 0.44 | 0.52 |

## BIPIA: does attack position matter? (recall at threshold 0.15)

| position | n | recall | embedding mean (cal.) |
|---|---|---|---|
| start | 20 | 1.00 | 0.84 |
| middle | 23 | 0.96 | 0.83 |
| end | 17 | 1.00 | 0.88 |

## Per attack type: recall at 0.15

| attack type | n | recall |
|---|---|---|
| bipia_code | 17 | 0.94 |
| bipia_text | 43 | 1.00 |
| poisonedrag_nq | 60 | 1.00 |
