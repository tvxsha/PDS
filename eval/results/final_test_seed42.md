# Final evaluation on the held-out TEST split (seed 42)

The test documents were scored once for this seed, after the configuration was frozen. Fitting used TRAIN only (no refit on train + validation, a conservative choice). Each method's threshold is the TRAIN clean-score quantile at the stated false-alarm budget (out-of-fold scores for fitted methods), applied unchanged.

Split hash 6bdb8ff98e. Evaluation documents: 373 (201 poisoned, 172 clean); 1116 train documents. Config sha256 b8cc4b57a7ff. Intervals are 95% group-bootstrap intervals (1000 resamples of whole groups: the same email, question or twin pair). With only 172 clean documents, one false alarm is 0.6%, so FPR is coarse.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.20 [0.14, 0.28] | 0.000 [0.000, 0.000] | 10.6% / 1.16% | 0 seen (up to ~174) |
| embedding | 0.15 [0.07, 0.25] | 0.081 [0.043, 0.123] | 1.9% / 0.19% | 814 |
| perplexity | 0.08 [0.05, 0.13] | 0.105 [0.062, 0.153] | 0.8% / 0.08% | 1047 |
| naive average of the 3 | 0.22 [0.16, 0.30] | 0.093 [0.053, 0.135] | 2.4% / 0.24% | 930 |
| sentence classifier | 0.61 [0.53, 0.70] | 0.041 [0.017, 0.075] | 13.2% / 1.48% | 407 |
| LR on the 3 current scores | 0.37 [0.27, 0.47] | 0.134 [0.087, 0.184] | 2.7% / 0.27% | 1337 |
| LR + sentence classifier | 0.67 [0.59, 0.75] | 0.052 [0.023, 0.090] | 11.5% / 1.27% | 523 |
| published classifier | 0.06 [0.03, 0.10] | 0.064 [0.030, 0.105] | 1.0% / 0.10% | 640 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.20 [0.14, 0.28] | 0.000 [0.000, 0.000] | 10.6% / 1.16% | 0 seen (up to ~174) |
| embedding | 0.04 [0.01, 0.07] | 0.017 [0.000, 0.039] | 2.3% / 0.23% | 174 |
| perplexity | 0.03 [0.01, 0.06] | 0.023 [0.006, 0.048] | 1.3% / 0.13% | 233 |
| naive average of the 3 | 0.19 [0.14, 0.27] | 0.017 [0.000, 0.037] | 10.1% / 1.10% | 174 |
| sentence classifier | 0.32 [0.25, 0.41] | 0.012 [0.000, 0.030] | 21.7% / 2.67% | 116 |
| LR on the 3 current scores | 0.29 [0.20, 0.38] | 0.041 [0.012, 0.073] | 6.7% / 0.70% | 407 |
| LR + sentence classifier | 0.39 [0.32, 0.48] | 0.012 [0.000, 0.030] | 25.5% / 3.27% | 116 |
| published classifier | 0.02 [0.01, 0.05] | 0.006 [0.000, 0.018] | 4.1% / 0.43% | 58 |

Precision is computed from the TPR and FPR above for the stated shares of poisoned documents; it is not measured on a stream with that share. Where no false alarm was seen it uses the rule-of-three upper bound (3/172 = 1.7%) instead of zero, so zero-false-alarm methods are not flattered.

## By source (budget 5%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.41 / 0.00 | 0.00 / 0.00 | 0.36 / 0.00 |
| embedding | 0.00 / 0.00 | 0.22 / 0.17 | 0.64 / 0.10 |
| perplexity | 0.16 / 0.20 | 0.00 / 0.01 | 0.21 / 0.00 |
| naive average of the 3 | 0.44 / 0.17 | 0.00 / 0.01 | 0.50 / 0.00 |
| sentence classifier | 0.79 / 0.07 | 0.45 / 0.01 | 0.64 / 0.00 |
| LR on the 3 current scores | 0.43 / 0.00 | 0.27 / 0.29 | 0.71 / 0.10 |
| LR + sentence classifier | 0.72 / 0.07 | 0.61 / 0.04 | 0.79 / 0.00 |
| published classifier | 0.05 / 0.11 | 0.01 / 0.01 | 0.57 / 0.00 |

## By attack type (budget 5%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 17 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.06 | 1.00 | 0.00 |
| bipia_text | 70 | 0.51 | 0.00 | 0.20 | 0.54 | 0.74 | 0.51 | 0.66 | 0.06 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 | 0.50 | 1.00 | 1.00 | 0.00 |
| pds_authority | 3 | 0.67 | 0.67 | 0.00 | 0.33 | 1.00 | 1.00 | 1.00 | 0.67 |
| pds_corpus | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.67 | 0.67 | 0.33 |
| pds_injection | 3 | 0.67 | 1.00 | 0.00 | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 1.00 | 1.00 | 1.00 | 0.67 | 0.00 | 0.33 | 0.67 |
| poisonedrag_nq | 100 | 0.00 | 0.22 | 0.00 | 0.00 | 0.45 | 0.27 | 0.61 | 0.01 |

## By source (budget 1%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.41 / 0.00 | 0.00 / 0.00 | 0.36 / 0.00 |
| embedding | 0.00 / 0.00 | 0.04 / 0.03 | 0.29 / 0.10 |
| perplexity | 0.03 / 0.03 | 0.00 / 0.01 | 0.21 / 0.00 |
| naive average of the 3 | 0.37 / 0.02 | 0.00 / 0.01 | 0.50 / 0.00 |
| sentence classifier | 0.67 / 0.02 | 0.06 / 0.00 | 0.00 / 0.00 |
| LR on the 3 current scores | 0.41 / 0.00 | 0.15 / 0.08 | 0.50 / 0.10 |
| LR + sentence classifier | 0.62 / 0.02 | 0.19 / 0.00 | 0.43 / 0.00 |
| published classifier | 0.01 / 0.01 | 0.00 / 0.00 | 0.29 / 0.00 |

## By attack type (budget 1%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 17 | 0.00 | 0.00 | 0.00 | 0.00 | 0.82 | 0.00 | 0.94 | 0.00 |
| bipia_text | 70 | 0.51 | 0.00 | 0.04 | 0.46 | 0.63 | 0.51 | 0.54 | 0.01 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 | 0.50 | 0.00 |
| pds_authority | 3 | 0.67 | 0.00 | 0.00 | 0.33 | 0.00 | 0.67 | 0.67 | 0.33 |
| pds_corpus | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 |
| pds_injection | 3 | 0.67 | 1.00 | 0.00 | 0.67 | 0.00 | 1.00 | 0.67 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| poisonedrag_nq | 100 | 0.00 | 0.04 | 0.00 | 0.00 | 0.06 | 0.15 | 0.19 | 0.00 |

Caveats:

- This run is locked: methods, thresholds and configuration were fixed before the test documents were scored, and it was run once for this seed. A bug found later is reported as a limitation, not fixed by a re-run.
- Several methods are compared on the same documents; no correction for multiple comparisons is applied.
- The published classifier is not fitted on our data; the train split only fixes its cut-off.
- Small cells (an attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so the realised FPR can sit a little off the target. Compare methods at similar realised FPR: a method that overshoots its budget gets extra recall for free.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- Hand-written adversarial and obfuscated sets are evaluated separately (stress_eval); they are not part of these tables.
