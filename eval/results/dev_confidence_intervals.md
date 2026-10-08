# Development-set fusion table with 95% bootstrap intervals

77 documents (60 poisoned, 17 clean). Documents are resampled with replacement (1000 resamples) and the thresholds are held fixed, so these intervals do NOT include the uncertainty from having tuned the thresholds on the same documents. Only 17 clean documents are present, so precision rests on very few false alarms and its interval is wide.

| method | threshold | P (95% CI) | R (95% CI) | F1 (95% CI) | FP | FN | F1 with threshold re-tuned on resamples, scored on left-out documents |
|---|---|---|---|---|---|---|---|
| Flag every document | - | 0.78 [0.69, 0.87] | 1.00 [1.00, 1.00] | 0.88 [0.82, 0.93] | 17 | 0 | n/a (no tuned threshold) |
| Naive mean | 0.30 (default) | 1.00 [1.00, 1.00] | 0.65 [0.53, 0.77] | 0.79 [0.69, 0.87] | 0 | 21 | n/a (no tuned threshold) |
| Naive mean, re-tuned | 0.102 (tuned) | 0.94 [0.87, 0.99] | 1.00 [1.00, 1.00] | 0.97 [0.93, 0.99] | 4 | 0 | 0.95 [0.89, 1.00] |
| Learned, raw scores | 0.65 (tuned) | 0.86 [0.77, 0.94] | 0.90 [0.82, 0.97] | 0.88 [0.81, 0.93] | 9 | 6 | 0.86 [0.76, 0.94] |
| Calibrated mean | 0.20 (tuned) | 0.98 [0.95, 1.00] | 0.97 [0.92, 1.00] | 0.97 [0.94, 1.00] | 1 | 2 | 0.96 [0.91, 1.00] |
| Learned, calibrated | 0.65 (tuned) | 0.97 [0.92, 1.00] | 0.98 [0.95, 1.00] | 0.98 [0.94, 1.00] | 2 | 1 | 0.96 [0.91, 1.00] |
| Staged cascade | 0.60/0.30/0.20 | 0.94 [0.87, 0.99] | 1.00 [1.00, 1.00] | 0.97 [0.93, 0.99] | 4 | 0 | n/a (thresholds fixed by hand earlier) |

## How fragile is the re-tuned naive threshold?

| naive threshold | precision | recall | F1 | FP | FN |
|---|---|---|---|---|---|
| 0.095 | 0.90 | 1.00 | 0.94 | 7 | 0 |
| 0.100 | 0.91 | 1.00 | 0.95 | 6 | 0 |
| 0.102 | 0.94 | 1.00 | 0.97 | 4 | 0 |
| 0.105 | 0.94 | 0.98 | 0.96 | 4 | 1 |
| 0.110 | 0.95 | 0.97 | 0.96 | 3 | 2 |
| 0.120 | 0.96 | 0.88 | 0.92 | 2 | 7 |

The best naive threshold (0.102) sits in a window less than a thousandth wide: moving it a few thousandths either way costs documents. That is the 'tuned on the evaluation set' problem in one table.

