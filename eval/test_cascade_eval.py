"""Tests for the pure parts of eval/cascade_eval.py (synthetic arrays and small JSON files, no models, no data files)."""

import json

import numpy as np
import pytest

from eval.cascade_eval import (COLUMNS, DESIGN_A, RUN_ALL, STAGE_COLUMNS, check_stages, choose_split, clear_thresholds,
                               compositions, conclusion, design_b, expected_ms, order_by_cost, read_latency, run_cascade, run_designs,
                               sentences, simulate, stage_of, stages_of, threshold_at_budget, to_frames)


def write_latency(tmp_path, detectors, hardware=None):
    p = tmp_path / "latency.json"
    p.write_text(json.dumps({"hardware": hardware or {"platform": "unit-test machine"}, "n": 10, "detectors": detectors}))
    return str(p)


def det(median, p95=None):
    return {"median_ms": median, "p95_ms": p95 if p95 is not None else median * 2, "mean_ms": median * 1.2}


# ---- latency file and stage order ------------------------------------------------------------------------------

def test_latency_keys_are_matched_by_name(tmp_path):
    path = write_latency(tmp_path, {"pattern": det(0.5), "Sentence_classifier": det(3.0), "embedding": det(25.0),
                                    "perplexity": det(100.0), "published classifier": det(200.0), "ours3": det(130.0)},
                         hardware={"platform": "Win", "logical_cores": 8, "nested": {"x": 1}})
    ms, hardware, p95, found = read_latency(path)
    assert ms == {"pattern": 0.5, "sentence": 3.0, "embedding": 25.0, "perplexity": 100.0, "published": 200.0}
    assert p95["sentence"] == 6.0 and "ours3" in found
    assert "platform: Win" in hardware and "logical_cores: 8" in hardware and "nested" not in hardware


def test_latency_problems_stop_with_a_message(tmp_path):
    with pytest.raises(SystemExit):
        read_latency(str(tmp_path / "nope.json"))
    with pytest.raises(SystemExit):
        read_latency(write_latency(tmp_path, {"pattern": {"median_ms": 0.0}}))            # not positive
    with pytest.raises(SystemExit):
        read_latency(write_latency(tmp_path, {"pattern": {"median_ms": "fast"}}))         # not a number
    ms, _, _, found = read_latency(write_latency(tmp_path, {"pattern": det(1.0), "other": det(2.0)}))
    with pytest.raises(SystemExit):
        check_stages(["pattern", "sentence"], ms, found)                                   # the sentence stage has no key


def test_stages_come_from_the_fusion_columns():
    assert stages_of(["pattern", "embedding", "perplexity", "sentence"]) == ["pattern", "embedding", "perplexity", "sentence"]
    assert stages_of(["pattern", "emb_abtt1", "perplexity", "sentence", "published"])[1] == "embedding"      # a variant is the embedding stage
    assert stage_of("emb_dom_z") == "embedding" and stage_of("published") == "published"
    with pytest.raises(SystemExit):
        stages_of(["pattern", "emb_dom_z", "emb_abtt1", "perplexity", "sentence"])         # two embedding columns


def test_stages_are_ordered_by_measured_cost_with_the_card_order_for_ties():
    ms = {"pattern": 0.5, "sentence": 3.0, "embedding": 25.0, "perplexity": 100.0, "published": 200.0}
    assert order_by_cost(list(ms), ms) == ["pattern", "sentence", "embedding", "perplexity", "published"]
    ms2 = dict(ms, embedding=1.0)                                                          # a cheap embedding runs before the sentence classifier
    assert order_by_cost(list(ms2), ms2)[:3] == ["pattern", "embedding", "sentence"]
    tied = {"sentence": 1.0, "pattern": 1.0, "perplexity": 1.0}
    assert order_by_cost(list(tied), tied) == ["pattern", "sentence", "perplexity"]


# ---- thresholds and the grid -----------------------------------------------------------------------------------

def test_threshold_rule_is_the_one_of_fpr_eval():
    from eval.fpr_eval import threshold_at_budget as reference
    x = np.random.default_rng(0).normal(size=500)
    for a in (0.0, 0.001, 0.01, 0.05, 0.1):
        assert threshold_at_budget(x, a) == reference(x, a)


def test_compositions_cover_every_split_once():
    got = list(compositions(10, 5))
    assert len(got) == 1001 and len(set(got)) == 1001 and all(sum(c) == 10 and min(c) >= 0 for c in got)
    assert list(compositions(3, 1)) == [(3,)] and len(list(compositions(4, 2))) == 5


def toy(seed=0, n=600):
    """Three stages. 'a' (cheap) separates a third of the poisoned documents perfectly; 'b' and 'c' are weaker."""
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    a = np.where((y == 1) & (rng.random(n) < 0.35), 5.0 + rng.random(n), rng.normal(0, 1, n))
    b = rng.normal(0, 1, n) + 1.2 * y
    c = rng.normal(0, 1, n) + 0.6 * y
    return {"a": a, "b": b, "c": c}, y


def test_the_chosen_split_has_the_highest_train_recall_of_all_splits():
    scores, y = toy()
    order, budget = ["a", "b", "c"], 0.05
    info = choose_split(scores, y, order, budget)
    assert sum(info["split"]) == 10 and info["share"].keys() == set(order)
    best = 0.0
    for split in compositions(10, 3):                                  # brute force, one document at a time
        thr = [threshold_at_budget(scores[s][y == 0], budget * u / 10) for s, u in zip(order, split)]
        hit = [any(scores[s][i] > t for s, t in zip(order, thr)) for i in range(len(y))]
        best = max(best, float(np.mean([h for h, yy in zip(hit, y) if yy == 1])))
    assert info["train_recall"] == pytest.approx(best)
    n_clean = int((y == 0).sum())
    assert info["train_fpr"] <= budget + len(order) / n_clean          # union bound, up to a document per stage


def test_ties_give_the_budget_to_the_cheaper_stage():
    x = np.r_[np.zeros(100), np.ones(100) * 2.0]                       # two identical stages that separate perfectly at zero FPR
    y = np.r_[np.zeros(100, int), np.ones(100, int)]
    info = choose_split({"first": x, "second": x.copy()}, y, ["first", "second"], 0.05)
    assert info["train_recall"] == 1.0 and info["split"] == (10, 0)    # same recall and FPR: the earlier stage keeps the budget


# ---- the cascade itself ----------------------------------------------------------------------------------------

def naive(scores, flag, clear, order, i):
    """One document, plain Python: (flagged, index of the last stage run)."""
    for k, s in enumerate(order):
        if scores[s][i] > flag[s]:
            return True, k
        if clear is not None and k < len(order) - 1 and scores[s][i] < clear[s]:
            return False, k
    return False, len(order) - 1


def test_simulate_agrees_with_a_document_by_document_loop():
    scores, y = toy(1)
    order = ["a", "b", "c"]
    flag = {"a": 3.0, "b": 1.5, "c": 1.0}
    clear = {"a": -0.5, "b": -0.2}
    for cl in (None, clear):
        flagged, exit_idx = simulate(scores, flag, cl, order)
        for i in range(len(y)):
            assert (bool(flagged[i]), int(exit_idx[i])) == naive(scores, flag, cl, order, i)


def test_a_flag_has_priority_over_a_clear():
    scores = {"a": np.array([0.5]), "b": np.array([9.0])}
    flagged, exit_idx = simulate(scores, {"a": 0.4, "b": 1.0}, {"a": 0.9}, ["a", "b"])      # clear threshold above the flag threshold
    assert flagged[0] and exit_idx[0] == 0


def test_clear_early_never_flags_more_than_the_or_cascade_and_nothing_cleared_is_the_same_cascade():
    scores, y = toy(2)
    order = ["a", "b", "c"]
    info = choose_split(scores, y, order, 0.05)
    a_flag, a_exit = simulate(scores, info["flag_thr"], None, order)
    for q in (0.01, 0.05, 0.3):
        clear = clear_thresholds(scores, y, order, q)
        b_flag, b_exit = simulate(scores, info["flag_thr"], clear, order)
        assert not (b_flag & ~a_flag).any()                           # B flags a subset of A
        assert (b_exit <= a_exit).all()                               # and never runs a stage that A skips
    never = {s: -np.inf for s in order[:-1]}
    b_flag, b_exit = simulate(scores, info["flag_thr"], never, order)
    assert (b_flag == a_flag).all() and (b_exit == a_exit).all()
    assert set(clear_thresholds(scores, y, order, 0.05)) == {"a", "b"}   # the last stage needs no clear threshold


def test_clear_thresholds_come_from_the_poisoned_train_scores():
    scores, y = toy(4)
    order = ["a", "b", "c"]
    for q in (0.01, 0.02, 0.05):
        clear = clear_thresholds(scores, y, order, q)
        for s in order[:-1]:
            assert clear[s] == pytest.approx(np.quantile(scores[s][y == 1], q))
            assert float((scores[s][y == 1] < clear[s]).mean()) <= q + 1e-9        # at most q of the poisoned documents are cleared by a stage


def test_expected_milliseconds_by_hand():
    order, ms = ["p", "s", "e"], {"p": 1.0, "s": 4.0, "e": 20.0}
    y = np.array([0, 0, 0, 0, 1, 1])
    exit_idx = np.array([0, 2, 2, 2, 1, 2])          # costs 1, 25, 25, 25 | 5, 25
    clean, pois, mix, total = expected_ms(exit_idx, y, order, ms, prevalences=(0.5, 0.01))
    assert clean == pytest.approx((1 + 25 + 25 + 25) / 4) and pois == pytest.approx(15.0) and total == 25.0
    assert mix[0.5] == pytest.approx(0.5 * clean + 0.5 * pois) and mix[0.01] == pytest.approx(0.99 * clean + 0.01 * pois)


# ---- the designs side by side ----------------------------------------------------------------------------------

def make_run(seed=3, reps=30):
    scores, y = toy(seed, 800)
    order = ["a", "b", "c"]
    ms = {"a": 1.0, "b": 20.0, "c": 100.0}
    ev, yv = toy(seed + 10, 400)
    grp = np.arange(len(yv))
    lr = (0.5 * scores["a"] + scores["b"], 0.5 * ev["a"] + ev["b"])
    rows, stage_rows = run_designs(scores, y, ev, yv, grp, order, ms, budgets=(0.05, 0.01), reps=reps, lr=lr)
    return rows, stage_rows, ms, yv


def test_every_design_and_budget_gets_a_row_with_the_documented_columns():
    rows, stage_rows, ms, yv = make_run()
    assert len(rows) == 2 * (1 + 1 + 3)                                          # reference, A and three B per budget
    assert [r["design"] for r in rows[:5]] == [RUN_ALL, DESIGN_A, design_b(0.01), design_b(0.02), design_b(0.05)]
    assert all(set(r) == set(COLUMNS) for r in rows) and all(set(r) == set(STAGE_COLUMNS) for r in stage_rows)
    assert len(stage_rows) == 2 * 5 * 3
    frames = to_frames({"rows": rows, "stage_rows": stage_rows})
    assert list(frames["cascade"].columns) == COLUMNS and list(frames["cascade_stages"].columns) == STAGE_COLUMNS


def test_the_or_cascade_saves_nothing_on_clean_traffic_beyond_what_it_flags_and_clearing_saves_more():
    rows, _, ms, yv = make_run()
    total = sum(ms.values())
    for budget in (0.05, 0.01):
        by = {r["design"]: r for r in rows if r["budget"] == budget}
        ref, a = by[RUN_ALL], by[DESIGN_A]
        assert ref["ms_clean"] == pytest.approx(total) and ref["ms_poisoned"] == pytest.approx(total) and ref["saving_clean"] == 0
        assert a["ms_clean"] >= total - a["clean_exit_early"] * total - 1e-9            # only early flags save anything
        assert a["saving_clean"] <= a["clean_exit_early"] + 1e-9
        for q in (0.01, 0.02, 0.05):
            b = by[design_b(q)]
            assert b["ms_clean"] <= a["ms_clean"] + 1e-9 and b["tpr"] <= a["tpr"] + 1e-9
            assert b["split"] == a["split"] and b["train_fpr"] <= a["train_fpr"] + 1e-9
        assert by[design_b(0.05)]["clean_exit_early"] >= by[design_b(0.01)]["clean_exit_early"]      # a larger q clears more


def test_stage_usage_is_consistent_with_the_rows():
    rows, stage_rows, ms, yv = make_run()
    for r in rows:
        stages = [s for s in stage_rows if s["design"] == r["design"] and s["budget"] == r["budget"]]
        assert stages[0]["reach_clean"] == 1.0 and stages[0]["reach_poisoned"] == 1.0         # everything meets the first stage
        reach = [s["reach_clean"] for s in stages]
        assert reach == sorted(reach, reverse=True)                                           # nobody reaches a later stage without an earlier one
        assert r["clean_exit_early"] == pytest.approx(1.0 - reach[-1])


def test_sentences_and_the_held_out_role_guard():
    rows, stage_rows, ms, yv = make_run()
    res = {"rows": rows}
    text = " ".join(sentences(res, 0.05))
    assert "Design A" in text and "B: clear-early, q = 5%" in text and "fusion candidate" in text and "saving" in text
    one = conclusion({"rows": rows, "budgets": (0.05, 0.01)})
    assert "\n" not in one and one.count(". ") == 0 and one.endswith("against design A.")      # one sentence
    a = next(r for r in rows if r["design"] == DESIGN_A and r["budget"] == 0.05)
    assert f"{a['ms_clean']:.1f} ms per clean document" in one and f"{a['ms_all']:.1f} ms of running every stage" in one
    assert one.startswith("At the 5% budget") and conclusion({"rows": rows, "budgets": (0.05, 0.01)}, 0.01).startswith("At the 1% budget")
    with pytest.raises(SystemExit):
        run_cascade(42, "held-out")                                                                # the held-out role is only run by final_test