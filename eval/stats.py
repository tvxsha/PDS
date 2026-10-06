"""
Statistical helpers for the evaluation protocol: bootstrap confidence
intervals, McNemar's test for comparing two detectors on the same data,
and multi-seed aggregation into mean +/- std.

Every function here takes y_true / y_pred as plain lists or 1-D arrays of
0/1 ints -- the same labels already used in eval/results/*.csv's
true_label column ("0" clean / "1" poisoned; cast to int before calling).

Usage:
    from eval.stats import compute_metrics, bootstrap_ci, mcnemar_test, aggregate_seeds
"""

import numpy as np
from scipy.stats import chi2


def _confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def compute_metrics(y_true, y_pred):
    """
    Precision, recall, F1, and false-positive rate for one set of
    predictions. All four are 0.0 when their denominator is 0 (e.g. no
    predicted positives), rather than raising -- small resamples can hit
    this, and a bootstrap loop shouldn't crash on it.
    """
    tp, fp, fn, tn = _confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def bootstrap_ci(y_true, y_pred, n_resamples=1000, ci=0.95, seed=0):
    """
    Bootstrap confidence intervals for precision/recall/F1/FPR.

    Resamples (y_true, y_pred) PAIRS together, with replacement, n times
    (keeping each doc's true label matched to its own prediction), recomputes
    all four metrics each time, and returns the point estimate on the real
    data plus the ci*100% percentile interval from the resamples.

    Returns: {metric_name: {"point": x, "low": x, "high": x}}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        raise ValueError("y_true/y_pred must be non-empty")
    if len(y_pred) != n:
        raise ValueError("y_true and y_pred must be the same length")

    rng = np.random.default_rng(seed)
    point = compute_metrics(y_true, y_pred)

    resampled = {"precision": [], "recall": [], "f1": [], "fpr": []}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)  # sample doc indices with replacement
        m = compute_metrics(y_true[idx], y_pred[idx])
        for k in resampled:
            resampled[k].append(m[k])

    alpha = (1 - ci) / 2
    result = {}
    for k, values in resampled.items():
        low = float(np.percentile(values, alpha * 100))
        high = float(np.percentile(values, (1 - alpha) * 100))
        result[k] = {"point": point[k], "low": low, "high": high}
    return result


def mcnemar_test(y_true, y_pred_a, y_pred_b, correction=True):
    """
    McNemar's test for comparing two detectors' predictions on the SAME
    set of documents (a paired test -- use this, not two separate bootstrap
    CIs, when the real question is "is detector A actually better than B
    on this data, or could the gap just be noise").

    Only the docs where A and B disagree matter:
      b = A wrong, B right
      c = A right, B wrong
    Under the null hypothesis (A and B are equally accurate), b and c
    should come out roughly equal. Uses the chi-square approximation with
    Yates' continuity correction by default, which is the safer choice
    when b + c is small (our adversarial/cross-domain sets are small).

    Returns: {"b": b, "c": c, "statistic": chi2_stat, "p_value": p}
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)
    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("y_true, y_pred_a, y_pred_b must all be the same length")

    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true

    b = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    c = int(np.sum(a_correct & ~b_correct))  # A right, B wrong

    if b + c == 0:
        # A and B never disagree -- nothing to test, treat as no evidence of a difference
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    if correction:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        statistic = (b - c) ** 2 / (b + c)

    p_value = float(1 - chi2.cdf(statistic, df=1))
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": p_value}


def aggregate_seeds(results):
    """
    Aggregate one metric (or several) across multiple random seeds -- e.g.
    the same evaluation re-run with several split seeds -- into mean +/- std.

    `results` is a list of dicts, one per seed, all sharing the same keys
    mapped to numeric values, e.g.:
        [{"f1": 0.81, "precision": 0.90}, {"f1": 0.78, "precision": 0.88}, ...]

    Returns: {key: {"mean": x, "std": x, "n": n}}
    """
    if not results:
        raise ValueError("results must be a non-empty list")
    keys = results[0].keys()
    out = {}
    for k in keys:
        values = np.array([r[k] for r in results], dtype=float)
        out[k] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
    return out
