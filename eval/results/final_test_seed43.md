# Final evaluation on the held-out TEST split (seed 43)

The test documents were scored once for this seed, after the configuration was frozen. Fitting used TRAIN only (no refit on train + validation, a conservative choice). Each method's threshold is the TRAIN clean-score quantile at the stated false-alarm budget (out-of-fold scores for fitted methods), applied unchanged.

Split hash 93a61f0778. Evaluation documents: 373 (201 poisoned, 172 clean); 1116 train documents. Config sha256 b8cc4b57a7ff. Intervals are 95% group-bootstrap intervals (1000 resamples of whole groups: the same email, question or twin pair). With only 172 clean documents, one false alarm is 0.6%, so FPR is coarse.

## False-alarm budget 5%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.28 [0.20, 0.37] | 0.000 [0.000, 0.000] | 13.9% / 1.57% | 0 seen (up to ~174) |
| embedding | 0.12 [0.06, 0.20] | 0.076 [0.036, 0.116] | 1.6% / 0.16% | 756 |
| perplexity | 0.04 [0.02, 0.07] | 0.035 [0.011, 0.065] | 1.1% / 0.11% | 349 |
| naive average of the 3 | 0.27 [0.20, 0.36] | 0.023 [0.006, 0.047] | 10.4% / 1.14% | 233 |
| sentence classifier | 0.64 [0.56, 0.72] | 0.052 [0.023, 0.088] | 10.9% / 1.20% | 523 |
| LR on the 3 current scores | 0.41 [0.33, 0.51] | 0.099 [0.059, 0.145] | 4.0% / 0.41% | 988 |
| LR + sentence classifier | 0.71 [0.63, 0.78] | 0.052 [0.023, 0.088] | 12.0% / 1.33% | 523 |
| published classifier | 0.08 [0.04, 0.13] | 0.052 [0.023, 0.089] | 1.5% / 0.15% | 523 |

## False-alarm budget 1%

| method | TPR (95% CI) | FPR (95% CI) | precision at 1% / 0.1% poison | false alarms per 10,000 clean |
|---|---|---|---|---|
| pattern bank | 0.28 [0.20, 0.37] | 0.000 [0.000, 0.000] | 13.9% / 1.57% | 0 seen (up to ~174) |
| embedding | 0.04 [0.01, 0.08] | 0.023 [0.006, 0.048] | 1.7% / 0.17% | 233 |
| perplexity | 0.02 [0.00, 0.05] | 0.017 [0.000, 0.040] | 1.4% / 0.14% | 174 |
| naive average of the 3 | 0.25 [0.19, 0.34] | 0.006 [0.000, 0.019] | 30.6% / 4.19% | 58 |
| sentence classifier | 0.33 [0.26, 0.43] | 0.017 [0.000, 0.040] | 16.2% / 1.88% | 174 |
| LR on the 3 current scores | 0.34 [0.26, 0.44] | 0.041 [0.012, 0.073] | 7.9% / 0.84% | 407 |
| LR + sentence classifier | 0.38 [0.30, 0.48] | 0.017 [0.000, 0.040] | 18.0% / 2.12% | 174 |
| published classifier | 0.02 [0.01, 0.05] | 0.000 [0.000, 0.000] | 1.4% / 0.14% | 0 seen (up to ~174) |

Precision is computed from the TPR and FPR above for the stated shares of poisoned documents; it is not measured on a stream with that share. Where no false alarm was seen it uses the rule-of-three upper bound (3/172 = 1.7%) instead of zero, so zero-false-alarm methods are not flattered.

## By source (budget 5%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.57 / 0.00 | 0.00 / 0.00 | 0.43 / 0.00 |
| embedding | 0.00 / 0.00 | 0.16 / 0.16 | 0.64 / 0.10 |
| perplexity | 0.06 / 0.07 | 0.00 / 0.00 | 0.21 / 0.00 |
| naive average of the 3 | 0.52 / 0.05 | 0.00 / 0.00 | 0.64 / 0.00 |
| sentence classifier | 0.85 / 0.08 | 0.47 / 0.00 | 0.50 / 0.20 |
| LR on the 3 current scores | 0.57 / 0.01 | 0.21 / 0.20 | 0.79 / 0.10 |
| LR + sentence classifier | 0.76 / 0.07 | 0.66 / 0.01 | 0.71 / 0.20 |
| published classifier | 0.09 / 0.10 | 0.00 / 0.00 | 0.57 / 0.00 |

## By attack type (budget 5%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 13 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| bipia_text | 74 | 0.68 | 0.00 | 0.07 | 0.61 | 0.82 | 0.68 | 0.72 | 0.11 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 | 1.00 | 0.00 |
| pds_authority | 3 | 0.67 | 1.00 | 0.00 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_corpus | 3 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 | 1.00 | 0.33 | 0.00 |
| pds_injection | 3 | 1.00 | 0.00 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 1.00 | 1.00 | 1.00 | 0.33 | 0.00 | 0.33 | 0.67 |
| poisonedrag_nq | 100 | 0.00 | 0.16 | 0.00 | 0.00 | 0.47 | 0.21 | 0.66 | 0.00 |

## By source (budget 1%): TPR / FPR

| method | bipia (87 pois / 87 clean) | poisonedrag (100 pois / 75 clean) | pds (14 pois / 10 clean) |
|---|---|---|---|
| pattern bank | 0.57 / 0.00 | 0.00 / 0.00 | 0.43 / 0.00 |
| embedding | 0.00 / 0.00 | 0.05 / 0.04 | 0.21 / 0.10 |
| perplexity | 0.02 / 0.03 | 0.00 / 0.00 | 0.21 / 0.00 |
| naive average of the 3 | 0.49 / 0.01 | 0.00 / 0.00 | 0.57 / 0.00 |
| sentence classifier | 0.72 / 0.03 | 0.02 / 0.00 | 0.14 / 0.00 |
| LR on the 3 current scores | 0.57 / 0.00 | 0.12 / 0.08 | 0.50 / 0.10 |
| LR + sentence classifier | 0.74 / 0.03 | 0.07 / 0.00 | 0.36 / 0.00 |
| published classifier | 0.00 / 0.00 | 0.00 / 0.00 | 0.36 / 0.00 |

## By attack type (budget 1%): TPR among poisoned documents

| attack type | n | pattern bank | embedding | perplexity | naive average of the 3 | sentence classifier | LR on the 3 current scores | LR + sentence classifier | published classifier |
|---|---|---|---|---|---|---|---|---|---|
| bipia_code | 13 | 0.00 | 0.00 | 0.00 | 0.00 | 0.85 | 0.00 | 1.00 | 0.00 |
| bipia_text | 74 | 0.68 | 0.00 | 0.03 | 0.58 | 0.70 | 0.68 | 0.69 | 0.00 |
| pds_adversarial | 2 | 0.50 | 0.50 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.00 |
| pds_authority | 3 | 0.67 | 0.33 | 0.00 | 0.67 | 0.00 | 0.67 | 0.67 | 0.33 |
| pds_corpus | 3 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.33 | 0.00 | 0.00 |
| pds_injection | 3 | 1.00 | 0.00 | 0.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| pds_perplexity | 3 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.33 |
| poisonedrag_nq | 100 | 0.00 | 0.05 | 0.00 | 0.00 | 0.02 | 0.12 | 0.07 | 0.00 |

Caveats:

- This run is locked: methods, thresholds and configuration were fixed before the test documents were scored, and it was run once for this seed. A bug found later is reported as a limitation, not fixed by a re-run.
- Several methods are compared on the same documents; no correction for multiple comparisons is applied.
- The published classifier is not fitted on our data; the train split only fixes its cut-off.
- Small cells (an attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.
- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so the realised FPR can sit a little off the target. Compare methods at similar realised FPR: a method that overshoots its budget gets extra recall for free.
- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.
- Hand-written adversarial and obfuscated sets are evaluated separately (stress_eval); they are not part of these tables.
