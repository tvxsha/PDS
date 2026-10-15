# Fusion re-fit with the new signals, seed 42 (task 4.10, validation only)

Each feature set is a logistic regression (class-balanced, C = 1) fitted on TRAIN. Its threshold at each false-alarm budget is the quantile of the out-of-fold TRAIN scores of the clean documents, applied unchanged to validation (373 documents: 201 poisoned, 172 clean). The sentence-classifier feature is out-of-fold on train, as in eval.fpr_eval. F0 and F1 reproduce the two LR rows of fpr_eval (checked to 1e-9 against eval.apply_models).

## Inputs

- Task 4.5 kept the original embedding (eval/results/embedding_variants_selected.json): F2 is identical to F1 and is not run.
- **Task 4.4 was not delivered and is waived (--waive-44): F3 and F5 are not run, and the paper must say that the boilerplate-subtraction variants were not tested.** The candidate file is marked complete with this waiver recorded.
- The published classifier's margins come from the cache of eval.baseline_eval (F4).

## The rule (fixed before any validation number was looked at)

The feature set is chosen by highest mean out-of-fold TRAIN AUC over seeds 42 to 44; sets less than 0.005 below the best are tied and the one with fewer features and cheaper signals wins (F0, F1, F2, F3, F5, F4). Validation numbers below are shown for information; they play no part in the choice. If nothing beats F1, F1 stays.

## Out-of-fold TRAIN AUC (the numbers the choice used)

| feature set | features | seed 42 | seed 43 | seed 44 | mean |
|---|---|---|---|---|---|
| F0: pattern + embedding + perplexity | pattern, embedding, perplexity | 0.781 | 0.762 | 0.746 | 0.763 |
| F1: F0 + sentence classifier **<- selected** | pattern, embedding, perplexity, sentence | 0.913 | 0.924 | 0.910 | 0.916 |
| F4: F1 + published classifier | pattern, embedding, perplexity, sentence, published | 0.912 | 0.924 | 0.910 | 0.915 |

F1 (F0 + sentence classifier) has the highest mean out-of-fold TRAIN AUC, 0.916, and is selected. Nothing beat F1, so F1 stays.
The three seeds share most of their documents, so the mean over seeds is a steadier number than one seed, not three independent measurements.

Not run: F2, F3, F5 (see Inputs).

## False-alarm budget 5% (validation)

| feature set | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean | validation AUC |
|---|---|---|---|---|---|---|
| F0: pattern + embedding + perplexity | 0.33 [0.25, 0.42] | 0.076 [0.035, 0.116] | 0.27 | 4.2% / 0.43% | 756 | 0.735 |
| F1: F0 + sentence classifier | 0.64 [0.56, 0.73] | 0.041 [0.016, 0.076] | 0.75 | 13.7% / 1.55% | 407 | 0.940 |
| F4: F1 + published classifier | 0.65 [0.57, 0.73] | 0.041 [0.016, 0.076] | 0.75 | 13.8% / 1.57% | 407 | 0.940 |

## False-alarm budget 1% (validation)

| feature set | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean | validation AUC |
|---|---|---|---|---|---|---|
| F0: pattern + embedding + perplexity | 0.25 [0.18, 0.35] | 0.035 [0.006, 0.065] | 0.24 | 6.8% / 0.72% | 349 | 0.735 |
| F1: F0 + sentence classifier | 0.36 [0.29, 0.46] | 0.006 [0.000, 0.018] | 0.38 | 38.7% / 5.89% | 58 | 0.940 |
| F4: F1 + published classifier | 0.36 [0.29, 0.46] | 0.006 [0.000, 0.018] | 0.38 | 38.7% / 5.89% | 58 | 0.940 |

Caveats:

- With 172 clean validation documents one false alarm is 0.6%, so FPR is coarse. Differences of a few points between feature sets are usually inside the intervals.
- 'best-case TPR' picks the threshold on validation itself (optimistic); the first TPR column is the honest one.
- Precision uses the observed FPR, or the rule-of-three bound 3/n where no false alarm was seen.
- F5 replaces the one embedding column by two (one per variant); every other set has the columns listed in the table.
- The published margin comes from the shared cache made by eval.baseline_eval; the classifier is not fitted on our data.
- Validation was not used to choose anything here. Any later change made after reading these tables makes them optimistic.
