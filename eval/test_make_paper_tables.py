"""Tests for eval.make_paper_tables (small hand-checkable files, no models, no data). Run:  python -m pytest eval/test_make_paper_tables.py"""

import json

import numpy as np
import pandas as pd
import pytest

from eval import cascade_eval, make_paper_tables as mpt
from eval.fpr_eval import precision

CFG = {"methods": ["pattern bank", "LR + sentence classifier"], "budgets": [0.05, 0.01], "prevalences": [0.01, 0.001],
       "bootstrap_reps": 50, "seeds": [1, 2]}


def write_seed(tmp_path, seed, flip=0):
    """10 poisoned + 10 clean documents in 10 groups of two. 'pattern bank': 6 of 10 poisoned and 1 of 10 clean above the
    threshold 0.5. The other method flags everything above 0.5 as well, but `flip` poisoned documents less."""
    rows = []
    for i in range(20):
        y = int(i < 10)
        grp = f"g{i % 10}"
        src = "bipia" if i % 2 == 0 else "pds_x"
        atk = "direct" if y else "clean"
        s = 0.9 if (y and i < 6) or (not y and i == 10) else 0.1
        s2 = 0.0 if (y and i < flip) else s
        rows += [(f"d{i}", y, src, atk, grp, "pattern bank", s), (f"d{i}", y, src, atk, grp, "LR + sentence classifier", s2)]
    df = pd.DataFrame(rows, columns=["id", "label", "source", "attack_type", "group", "method", "score"])
    df.to_csv(tmp_path / f"final_dryrun_seed{seed}_scores.csv", index=False)
    thr = {m: {"0.05": 0.5, "0.01": 0.5} for m in CFG["methods"]}
    (tmp_path / f"final_dryrun_seed{seed}_thresholds.json").write_text(
        json.dumps({"seed": seed, "split_hash": f"hash{seed}", "role": "val", "thresholds": thr}))


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(mpt, "RESULTS_DIR", str(tmp_path))
    write_seed(tmp_path, 1)
    write_seed(tmp_path, 2, flip=2)
    return tmp_path


def stats_for(results):
    seeds = [mpt.load_seed(s, "final_dryrun", CFG["methods"]) for s in CFG["seeds"]]
    return seeds, [mpt.seed_stats(sd, CFG) for sd in seeds]


def test_rates_are_computed_from_the_thresholds(results):
    seeds, stats = stats_for(results)
    a = stats[0][("pattern bank", 0.05)]
    assert a["tpr"] == pytest.approx(0.6)
    assert a["fpr"] == pytest.approx(0.1)
    assert a["prec"][0.01] == pytest.approx(precision(0.6, 0.1, 0.01, 10))
    b = stats[1][("LR + sentence classifier", 0.05)]
    assert b["tpr"] == pytest.approx(0.4)             # two of the six flagged poisoned documents are lost in seed 2


def test_source_and_attack_breakdown(results):
    seeds, stats = stats_for(results)
    a = stats[0][("pattern bank", 0.05)]
    assert set(a["source"]) == {"bipia", "pds"}        # pds_x is grouped with pds
    assert a["attack"] == {"direct": pytest.approx(0.6)}
    # poisoned bipia docs are the even i < 10 -> 0, 2, 4, 6, 8; flagged are i < 6 -> 0, 2, 4 -> 3 of 5
    assert a["source"]["bipia"][0] == pytest.approx(0.6)


def test_headline_cell_is_the_mean_over_seeds(results):
    seeds, stats = stats_for(results)
    text = mpt.t_headline(CFG, stats, "", "n.")
    row = [line for line in text.splitlines() if line.startswith("LR + sentence")][0]
    assert "0.50" in row                                # mean of 0.6 and 0.4 at the 5% budget
    assert "[DRY RUN" not in text


def test_dry_run_marker_is_in_every_caption(results):
    seeds, stats = stats_for(results)
    mark = "[DRY RUN on validation, not results] "
    for text in (mpt.t_headline(CFG, stats, mark, "n."), mpt.t_precision(CFG, stats, mark)):
        assert text.splitlines()[1].startswith("% DRY RUN")
        assert mark in text


def test_numbers_macros(results):
    seeds, stats = stats_for(results)
    text = mpt.numbers(CFG, stats, seeds, {"pattern": (1.5, 3.0), "sentence": (20.0, None)})
    assert r"\newcommand{\resTPRFivePat}{0.60}" in text
    assert r"\newcommand{\resTPRFiveLRsen}{0.50}" in text
    assert r"\newcommand{\resTPRminFiveLRsen}{0.40}" in text and r"\newcommand{\resTPRmaxFiveLRsen}{0.60}" in text
    assert r"\newcommand{\resHashB}{hash2}" in text
    assert r"\newcommand{\resLatMedPat}{1.5}" in text and r"\newcommand{\resLatPNinetyFivePat}{3.0}" in text
    assert "resLatPNinetyFiveSen" not in text           # no p95 given
    names = [line.split("{")[1].split("}")[0].lstrip("\\") for line in text.splitlines() if line.startswith("\\newcommand")]
    assert all(n.isalpha() for n in names), "macro names must be letters only"
    assert len(names) == len(set(names))


def test_missing_files_stop_with_the_command_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(mpt, "RESULTS_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        mpt.load_seed(7, "final_dryrun", CFG["methods"])
    assert "--dry-run" in str(e.value)


def test_a_method_missing_for_some_documents_is_refused(results):
    path = results / "final_dryrun_seed1_scores.csv"
    df = pd.read_csv(path)
    df = df.drop(df[(df["method"] == "pattern bank") & (df["id"] == "d3")].index)
    df.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        mpt.load_seed(1, "final_dryrun", CFG["methods"])


def test_latency_table(tmp_path):
    p = tmp_path / "lat.json"
    p.write_text(json.dumps({"hardware": {"cpu": "unit-test machine"}, "n": 200,
                             "detectors": {"pattern": {"median_ms": 0.5, "p95_ms": 0.9, "mean_ms": 0.6}}}))
    text = mpt.t_latency(str(p), "")
    assert "pattern & 0.5 & 0.9 & 0.6" in text and "unit-test machine" in text and "200 documents" in text


def test_tex_escaping():
    assert mpt.tex("a_b & 5%") == r"a\_b \& 5\%"
    assert mpt.ptex(0.0123) == r"1.2\%"
    assert mpt.ptex(0.0043) == r"0.43\%"
    assert np.isnan(mpt.mean([float("nan")]))


def test_cascade_table_averages_over_seeds_and_handles_designs_without_q():
    def row(design, q, budget, tpr, ms_clean, saving):
        r = dict.fromkeys(cascade_eval.COLUMNS, 0.0)
        r.update(design=design, q=q, budget=budget, tpr=tpr, fpr=0.04, ms_clean=ms_clean, ms_poisoned=ms_clean * 2, saving_clean=saving)
        return r

    frames = []
    for tpr in (0.4, 0.6):                                                  # two seeds
        frames.append(pd.DataFrame([row(cascade_eval.RUN_ALL, float("nan"), 0.05, tpr, 100.0, 0.0),
                                    row(cascade_eval.DESIGN_A, "", 0.05, tpr, 100.0, 0.0),
                                    row(cascade_eval.design_b(0.01), 0.01, 0.05, tpr, 30.0, 0.7),
                                    row(cascade_eval.design_b(0.01), 0.01, 0.01, 0.1, 30.0, 0.7)], columns=cascade_eval.COLUMNS))
    text = mpt.t_cascade(frames, "")
    lines = {line.split(" & ")[0]: line for line in text.splitlines() if " & " in line and not line.startswith("Design")}
    assert len(lines) == 3                                                  # run everything, A, B (the 1% budget rows are left out)
    assert lines[cascade_eval.RUN_ALL.replace("_", r"\_")].split(" & ")[1] == "0.50"       # mean of 0.4 and 0.6
    assert "70\\%" in lines["B: clear-early, q = 1\\%"]
    assert "nan" not in text