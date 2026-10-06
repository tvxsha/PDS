"""
Small synthetic-data checks for eval/stats.py. No pytest dependency yet
(none is used elsewhere in the repo) -- plain asserts, run directly:

    python -m eval.test_stats
"""

from eval.stats import compute_metrics, bootstrap_ci, mcnemar_test, aggregate_seeds


def test_compute_metrics_perfect():
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0]  # perfect predictions
    m = compute_metrics(y_true, y_pred)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["fpr"] == 0.0


def test_compute_metrics_known_confusion():
    # 2 TP, 1 FP, 1 FN, 2 TN by construction
    y_true = [1, 1, 1, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0]
    m = compute_metrics(y_true, y_pred)
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["fpr"] - 1 / 3) < 1e-9


def test_compute_metrics_no_predicted_positives():
    # denominator-zero case shouldn't raise
    y_true = [1, 0, 1, 0]
    y_pred = [0, 0, 0, 0]
    m = compute_metrics(y_true, y_pred)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


def test_bootstrap_ci_shape_and_bounds():
    y_true = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1] * 5
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0, 0, 1] * 5
    result = bootstrap_ci(y_true, y_pred, n_resamples=200, seed=42)
    for metric in ["precision", "recall", "f1", "fpr"]:
        assert metric in result
        low, point, high = result[metric]["low"], result[metric]["point"], result[metric]["high"]
        assert 0.0 <= low <= point <= high <= 1.0, f"{metric}: {low}, {point}, {high}"


def test_bootstrap_ci_deterministic_with_seed():
    y_true = [1, 0, 1, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
    r1 = bootstrap_ci(y_true, y_pred, n_resamples=100, seed=7)
    r2 = bootstrap_ci(y_true, y_pred, n_resamples=100, seed=7)
    assert r1 == r2, "same seed should give identical CIs"


def test_mcnemar_identical_predictors():
    y_true = [1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 0, 0, 1, 1, 1, 0]
    result = mcnemar_test(y_true, y_pred, y_pred)  # A vs itself
    assert result["b"] == 0 and result["c"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_detects_real_difference():
    # A is always correct, B is always wrong -- maximally different
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y_pred_a = y_true[:]
    y_pred_b = [1 - v for v in y_true]
    result = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert result["c"] == 10  # A right, B wrong, on every doc
    assert result["b"] == 0
    assert result["p_value"] < 0.01


def test_aggregate_seeds_mean_std():
    results = [{"f1": 0.80, "precision": 0.90}, {"f1": 0.90, "precision": 0.80}]
    agg = aggregate_seeds(results)
    assert abs(agg["f1"]["mean"] - 0.85) < 1e-9
    assert abs(agg["precision"]["mean"] - 0.85) < 1e-9
    assert agg["f1"]["n"] == 2
    assert agg["f1"]["std"] > 0  # the two values differ, std should be nonzero


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    run_all()
