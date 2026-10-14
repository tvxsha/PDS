"""Tests for the numbers of eval/alpha_sweep.py (synthetic arrays, no models, no data files)."""

import csv

import numpy as np
import pytest

from eval.alpha_sweep import (BUDGETS, CSV_COLUMNS, resolution_text, sentence_for, sweep_one, unverified_sentence, verifiable,
                              write_csv)


def make(seed=0, n_train=2000, n_val=400):
    rng = np.random.default_rng(seed)
    train_clean = rng.normal(0, 1, n_train)
    y = np.r_[np.zeros(n_val // 2, int), np.ones(n_val // 2, int)]
    ev = np.r_[rng.normal(0, 1, n_val // 2), rng.normal(2, 1, n_val // 2)]
    grp = np.arange(n_val)
    return train_clean, ev, y, grp


def test_which_budgets_can_be_checked():
    assert verifiable(0.05, 172) and verifiable(0.01, 172)          # 1.7 expected false alarms
    assert not verifiable(0.005, 172) and not verifiable(0.001, 172)
    assert resolution_text(172) == "FPR resolution 0.58%"


def test_threshold_comes_from_train_only():
    train_clean, ev, y, grp = make()
    a = sweep_one(train_clean, ev, y, grp, 0.05, reps=50)
    assert a["thr"] == pytest.approx(np.quantile(train_clean, 0.95))
    b = sweep_one(train_clean, ev + 10.0, y, grp, 0.05, reps=50)     # other validation scores: same threshold
    assert b["thr"] == a["thr"] and b["tpr"] == 1.0 and b["fpr"] == 1.0
    assert a["train_above"] == int((train_clean > a["thr"]).sum()) and a["n_train_clean"] == len(train_clean)


def test_smaller_budget_never_raises_the_recall():
    train_clean, ev, y, grp = make(1)
    rows = [sweep_one(train_clean, ev, y, grp, a, reps=50) for a in BUDGETS]
    thr = [r["thr"] for r in rows]
    tpr = [r["tpr"] for r in rows]
    assert all(x <= y_ for x, y_ in zip(thr, thr[1:])) and all(x >= y_ for x, y_ in zip(tpr, tpr[1:]))


def test_no_false_alarm_gets_the_rule_of_three_bound():
    train_clean, ev, y, grp = make(2)
    ev_clean_low = ev.copy()
    ev_clean_low[y == 0] = -50.0                                       # no clean validation document above any threshold
    r = sweep_one(train_clean, ev_clean_low, y, grp, 0.05, reps=50)
    assert r["fpr"] == 0.0 and r["fpr_hi"] == pytest.approx(3.0 / r["n_neg"])


def test_csv_has_the_roadmap_columns(tmp_path):
    train_clean, ev, y, grp = make(3)
    rows_by = {"m": {a: sweep_one(train_clean, ev, y, grp, a, reps=20) for a in BUDGETS}}
    path = tmp_path / "s.csv"
    write_csv(str(path), rows_by)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert list(rows[0].keys()) == CSV_COLUMNS == "method,budget,thr,tpr,tpr_lo,tpr_hi,fpr,fpr_lo,fpr_hi,n_pos,n_neg".split(",")
    assert len(rows) == len(BUDGETS) and [float(r["budget"]) for r in rows] == list(BUDGETS)
    assert int(rows[0]["n_pos"]) == 200 and int(rows[0]["n_neg"]) == 200


def test_generated_sentences_use_the_numbers():
    cells = {a: {"tpr": 0.9 - 0.1 * i} for i, a in enumerate(BUDGETS)}
    s = sentence_for("m", cells)
    assert s.startswith("m: recall 0.90 at 10%, 0.80 at 5%") and s.endswith("0.40 at 0.1%.")
    assert "10% down to 1%" in unverified_sentence(172) and "0.5%, 0.1%" in unverified_sentence(172)
    assert "None of these" in unverified_sentence(5) and "Every budget" in unverified_sentence(100000)