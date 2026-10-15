"""
Tests for the small, pure helpers in eval/fpr_eval.py (roadmap task 6.3):
threshold_at_budget, precision (including its rule-of-three branch) and
group_bootstrap on a toy, fully-separated case. No models, no pool.

    python -m pytest tests/test_fpr_helpers.py
"""

import numpy as np

from eval.fpr_eval import threshold_at_budget, precision, group_bootstrap


def test_threshold_at_budget_zero_alpha_is_the_max():
    scores = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    assert threshold_at_budget(scores, 0.0) == scores.max()


def test_threshold_at_budget_alpha_one_is_the_min():
    scores = np.array([1.0, 5.0, 2.0, 9.0, 3.0])
    assert threshold_at_budget(scores, 1.0) == scores.min()


def test_threshold_at_budget_respects_the_budget():
    # 100 clean scores 0..99; a 5% budget should flag close to 5 of them
    scores = np.arange(100, dtype=float)
    thr = threshold_at_budget(scores, 0.05)
    flagged = (scores > thr).sum()
    assert 3 <= flagged <= 6  # "at most alpha", small rounding slack


def test_precision_basic_value():
    # tpr=0.5, fpr=0.1, prevalence=0.01, n_clean=100
    # denom = 0.5*0.01 + 0.1*0.99 = 0.104; precision = 0.005/0.104
    got = precision(0.5, 0.1, 0.01, 100)
    assert abs(got - (0.005 / 0.104)) < 1e-9


def test_precision_rule_of_three_on_zero_false_alarms():
    # fpr=0 is replaced with 3/n_clean rather than taken at face value,
    # so a zero-false-alarm method is not flattered with "infinite precision"
    n_clean = 300
    got = precision(1.0, 0.0, 0.01, n_clean)
    expected_fpr = 3.0 / n_clean
    denom = 1.0 * 0.01 + expected_fpr * 0.99
    assert abs(got - (0.01 / denom)) < 1e-9


def test_precision_returns_nan_when_denominator_is_zero():
    # tpr=0 and prevalence=1.0 (so (1 - prevalence) == 0) makes the
    # denominator exactly zero regardless of fpr
    got = precision(0.0, 0.2, 1.0, 50)
    assert np.isnan(got)


def test_group_bootstrap_on_a_fully_separated_toy_case():
    # 6 poisoned groups, every member flagged True; 6 clean groups, every
    # member flagged False. TPR is 1.0 and FPR is 0.0 in EVERY bootstrap
    # resample that includes at least one group of that label, so with
    # enough groups and reps the interval collapses to a point.
    y, grp, flags = [], [], []
    for i in range(6):
        for _ in range(2):
            y.append(1); grp.append(f"poison_{i}"); flags.append(True)
    for i in range(6):
        for _ in range(2):
            y.append(0); grp.append(f"clean_{i}"); flags.append(False)
    y, grp, flags = np.array(y), np.array(grp), np.array(flags)

    (tl, th), (fl, fh) = group_bootstrap(flags, y, grp, reps=200, seed=0)
    assert tl == th == 1.0
    assert fl == fh == 0.0


def test_group_bootstrap_is_seed_deterministic():
    y = np.array([1, 1, 0, 0, 1, 0])
    grp = np.array(["g1", "g1", "g2", "g2", "g3", "g4"])
    flags = np.array([True, False, False, True, True, False])
    out_a = group_bootstrap(flags, y, grp, reps=100, seed=5)
    out_b = group_bootstrap(flags, y, grp, reps=100, seed=5)
    assert [tuple(x) for x in out_a] == [tuple(x) for x in out_b]
