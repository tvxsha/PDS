# Final evaluation on the held-out TEST split (seed 44)

The test documents were scored once for this seed, after the configuration was frozen. Fitting used TRAIN only (no refit on train + validation, a conservative choice). Each method's threshold is the TRAIN clean-score quantile at the stated false-alarm budget (out-of-fold scores for fitted methods), applied unchanged.

Split hash 4d1126b320. Evaluation documents: 373 (201 poisoned, 172 clean); 1116 train documents. Config sha256 b8cc4b57a7ff. Intervals are 95% group-bootstrap intervals (1000 resamples of whole groups: the same email, question or twin pair). With only 172 clean documents, one false alarm is 0.6%, so FPR is coarse.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.25 [0.18, 0.33] | 0.000 [0.000, 0.000] | 12.6% / 1.41% | 0 seen (up to ~174) |
| embedding | 0.07 [0.03, 0.14] | 0.052 [0.023, 0.087] | 1.4% / 0.14% | 523 |
| perplexity | 0.05 [0.02, 0.08] | 0.064 [0.029, 0.102] | 0.8% / 0.08% | 640 |
| naive average of the 3 | 0.23 [0.16, 0.32] | 0.105 [0.060, 0.153] | 2.2% / 0.22% | 1047 |
| sentence classifier | 0.51 [0.42, 0.60] | 0.023 [0.006, 0.050] | 18.2% / 2.16% | 233 |
| LR on the 3 current scores | 0.37 [0.28, 0.47] | 0.070 [0.035, 0.111] | 5.1% / 0.53% | 698 |
| LR + sentence classifier | 0.66 [0.58, 0.74] | 0.035 [0.011, 0.065] | 16.1% / 1.86% | 349 |
| published classifier | 0.07 [0.04, 0.12] | 0.052 [0.024, 0.088] | 1.4% / 0.14% | 523 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.25 [0.18, 0.33] | 0.000 [0.000, 0.000] | 12.6% / 1.41% | 0 seen (up to ~174) |
| embedding | 0.01 [0.00, 0.03] | 0.000 [0.000, 0.000] | 0.6% / 0.06% | 0 seen (up to ~174) |
| perplexity | 0.03 [0.01, 0.06] | 0.035 [0.012, 0.064] | 1.0% / 0.10% | 349 |
| naive average of the 3 | 0.21 [0.15, 0.30] | 0.041 [0.013, 0.072] | 5.0% / 0.52% | 407 |
| sentence classifier | 0.32 [0.24, 0.42] | 0.006 [0.000, 0.018] | 36.0% / 5.27% | 58 |
| LR on the 3 current scores | 0.27 [0.21, 0.37] | 0.012 [0.000, 0.030] | 19.2% / 2.30% | 116 |
| LR + sentence classifier | 0.32 [0.25, 0.42] | 0.006 [0.000, 0.018] | 36.0% / 5.27% | 58 |
| published classifier | 0.03 [0.01, 0.06] | 0.006 [0.000, 0.018] | 4.9% / 0.51% | 58 |

Precision is computed from the TPR and FPR above for the stated shares of poisoned documents; it is not measured on a stream with that share. Where no false alarm was seen it uses the rule-of-three upper bound (3/172 = 1.7%) instead of zero, so zero-false-alarm methods are not flattered.

## By source (budget 5%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.52 / 0.00 | 0.00 / 0.00 | 0.36 / 0.00 |
| embedding | 0.01 / 0.01 | 0.10 / 0.11 | 0.29 / 0.00 |
| perplexity | 0.08 / 0.10 | 0.00 / 0.03 | 0.21 / 0.00 |
| naive average of the 3 | 0.47 / 0.18 | 0.00 / 0.03 | 0.43 / 0.00 |
| sentence classifier | 0.82 / 0.05 | 0.28 / 0.00 | 0.29 / 0.00 |
| LR on the 3 current scores | 0.53 / 0.00 | 0.20 / 0.16 | 0.64 / 0.00 |
| LR + sentence classifier | 0.78 / 0.07 | 0.56 / 0.00 | 0.64 / 0.00 |
| published classifier | 0.08 / 0.10 | 0.00 / 0.00 | 0.57 / 0.00 |

## By attack type (budget 5%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 13 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.08 | 1.00 | 0.00 |
| bipia_text | 74 | 0.61 | 0.01 | 0.09 | 0.55 | 0.78 | 0.61 | 0.74 | 0.09 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 | 0.50 | 1.00 | 1.00 | 0.00 |
| pds_authority | 3 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 1.00 | 0.67 | 0.33 |
| pds_corpus | 3 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 | 0.33 |
| pds_injection | 3 | 0.67 | 0.33 | 0.00 | 0.67 | 0.33 | 0.67 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| poisonedrag_nq | 100 | 0.00 | 0.10 | 0.00 | 0.00 | 0.28 | 0.20 | 0.56 | 0.00 |

## By source (budget 1%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.52 / 0.00 | 0.00 / 0.00 | 0.36 / 0.00 |
| embedding | 0.00 / 0.00 | 0.01 / 0.00 | 0.07 / 0.00 |
| perplexity | 0.05 / 0.05 | 0.00 / 0.03 | 0.21 / 0.00 |
| naive average of the 3 | 0.44 / 0.06 | 0.00 / 0.03 | 0.36 / 0.00 |
| sentence classifier | 0.71 / 0.01 | 0.02 / 0.00 | 0.07 / 0.00 |
| LR on the 3 current scores | 0.52 / 0.00 | 0.05 / 0.03 | 0.36 / 0.00 |
| LR + sentence classifier | 0.68 / 0.01 | 0.02 / 0.00 | 0.29 / 0.00 |
| published classifier | 0.01 / 0.01 | 0.00 / 0.00 | 0.36 / 0.00 |

## By attack type (budget 1%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.85 | 0.00 | 0.92 | 0.00 |
| bipia_text | 74 | 0.61 | 0.00 | 0.05 | 0.51 | 0.69 | 0.61 | 0.64 | 0.01 |
| pds_adversarial | 2 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 |
| pds_authority | 3 | 0.67 | 0.00 | 0.00 | 0.00 | 0.00 | 0.67 | 0.67 | 0.00 |
| pds_corpus | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| pds_injection | 3 | 0.67 | 0.33 | 0.00 | 0.67 | 0.33 | 0.67 | 0.67 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.67 |
| poisonedrag_nq | 100 | 0.00 | 0.01 | 0.00 | 0.00 | 0.02 | 0.05 | 0.02 | 0.00 |

Caveats:

- This run is locked: methods, thresholds and configuration were fixed before the test documents were scored, and it was run once for this seed. A bug found later is reported as a limitation, not fixed by a re-run.
- Several methods are compared on the same documents; no correction for multiple comparisons is applied.
- The published classifier is not fitted on our data; the train split only fixes its cut-off.
- Small cells (an attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so the realised FPR can sit a little off the target. Compare methods at similar realised FPR: a method that overshoots its budget gets extra recall for free.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- Hand-written adversarial and obfuscated sets are evaluated separately (stress_eval); they are not part of these tables.
