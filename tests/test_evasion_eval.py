"""
Tests for roadmap task 8.1: the evasion-rate evaluation (eval/evasion_eval.py), the shared figure style and figure F7.
Models and the pool are replaced by fakes; the figure is drawn from synthetic CSVs in a temporary folder.
"""

import math
import os
import sys
import types

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from data.evalsets import build_obfuscated as bo
from data.evalsets.obfuscated_A_sources import HOSTS_A, PAYLOADS_A
from eval import evasion_eval as ev
from eval import figstyle

ROWS, _ = bo.build_rows(HOSTS_A, PAYLOADS_A, "A")          # 180 rows: T0_none + 8 transformations, 20 each
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Fake:
    methods = ["pattern bank", "sentence classifier", "LR + sentence classifier"]
    thresholds = {m: {0.05: 0.5, 0.01: 0.9} for m in methods}

    def score(self, docs):
        pat = np.array([d["pat"] for d in docs])
        return {"pattern bank": pat, "sentence classifier": pat * 0.5, "LR + sentence classifier": pat * 0.9,
                "embedding": pat, "perplexity": pat, "naive average of the 3": pat, "LR on the 3 current scores": pat}


def _ext(pattern_of):
    return {"pattern": {r.id: pattern_of(r) for r in ROWS}, "embedding": {r.id: 0.0 for r in ROWS},
            "perplexity": {r.id: 0.0 for r in ROWS}}


# ----------------------------------------------------------------------------- statistics

def test_wilson_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for k, n in [(0, 20), (1, 20), (10, 20), (19, 20), (20, 20), (7, 160)]:
        lo, hi = ev.wilson(k, n)
        ref = stats.binomtest(k, n).proportion_ci(method="wilson")
        assert math.isclose(lo, ref.low, abs_tol=1e-6) and math.isclose(hi, ref.high, abs_tol=1e-6)
    assert all(math.isnan(v) for v in ev.wilson(0, 0))


def test_make_cell_evasion_is_one_minus_recall_with_mirrored_interval():
    c = ev.make_cell("pattern bank", "leet", 0.05, [True] * 15 + [False] * 5)
    assert (c["k"], c["n"]) == (15, 20)
    assert math.isclose(c["recall"], 0.75) and math.isclose(c["evasion"], 0.25)
    assert math.isclose(c["evasion_lo"], 1 - c["recall_hi"]) and math.isclose(c["evasion_hi"], 1 - c["recall_lo"])
    assert c["evasion_lo"] <= c["evasion"] <= c["evasion_hi"]


def test_transform_order_puts_the_control_first():
    order = ev.transform_order(ROWS)
    assert order[0] == "T0_none" and len(order) == 9 and order[1] == "zero_width"


# ----------------------------------------------------------------------------- cells

def test_pool_method_flags_use_train_thresholds_unchanged():
    ext = _ext(lambda r: 0.8 if r.attack_type in ("T0_none", "leet") else 0.1)
    flags = ev.pool_method_flags(ROWS, _Fake(), ext)
    assert set(flags) == set(ev.METHODS)
    assert flags["pattern bank"][0.05].sum() == 40 and flags["pattern bank"][0.01].sum() == 0   # 0.8 > 0.5, not > 0.9
    assert flags["sentence classifier"][0.05].sum() == 0                                         # 0.4 < 0.5


def test_build_cells_counts_per_transformation_and_method():
    ext = _ext(lambda r: 0.8 if r.attack_type in ("T0_none", "leet") else 0.1)
    flags = ev.pool_method_flags(ROWS, _Fake(), ext)
    cells = ev.build_cells(ROWS, flags)
    assert len(cells) == 3 * 2 * 9
    assert all(c["n"] == 20 for c in cells)
    leet = next(c for c in cells if (c["method"], c["transform"], c["budget"]) == ("pattern bank", "leet", 0.05))
    zw = next(c for c in cells if (c["method"], c["transform"], c["budget"]) == ("pattern bank", "zero_width", 0.05))
    assert leet["k"] == 20 and leet["evasion"] == 0.0 and zw["k"] == 0 and zw["evasion"] == 1.0
    assert ev.control_recall(cells, 0.05) == 1.0
    assert ev.control_recall(cells, 0.05, method="no such method") is None


# ----------------------------------------------------------------------------- report and CSV

def _cells(control_hits=17):
    flags = {m: {b: np.array([(r.attack_type == "T0_none" and int(r.id[6:8]) <= control_hits) or r.attack_type == "leet"
                              for r in ROWS]) for b in ev.BUDGETS} for m in ev.METHODS}
    return ev.build_cells(ROWS, flags)


def test_report_has_every_cell_with_n_the_control_check_and_the_caveats():
    text = ev.render_report(42, _cells(), ev.METHODS, [])
    assert "seed 42" in text and "T0_none" in text and "leet" in text
    assert text.count("(n=20)") == 3 * 2 * 9
    assert "Control check" in text and "17/20" not in text and "0.85" in text
    assert "Recall only" in text and "'before' measurement" in text and "Batch A" in text
    assert "WARNING" not in text
    warned = ev.render_report(42, _cells(), ev.METHODS, ["the pattern bank catches only 40%"])
    assert "**WARNING:** the pattern bank catches only 40%" in warned


def test_csv_has_the_documented_columns_and_all_rows(tmp_path):
    cells = _cells()
    base = ev.write_outputs(42, cells, ev.render_report(42, cells, ev.METHODS, []), out_dir=str(tmp_path))
    lines = open(base + ".csv", encoding="utf-8").read().splitlines()
    assert lines[0].split(",") == ev.CSV_COLUMNS
    assert len(lines) == 1 + len(cells)
    assert os.path.exists(base + ".md")


# ----------------------------------------------------------------------------- published classifier

def test_published_flags_use_train_clean_margins(monkeypatch):
    train = [dict(id=f"t{i}", y=0 if i < 100 else 1, text="x") for i in range(120)]
    fake = types.ModuleType("eval.baseline_eval")
    fake.cache_path = lambda model, w: f"train_{w}.csv"
    fake.read_cache = lambda path: {d["id"]: {"margin": float(i) / 10} for i, d in enumerate(train)}
    fake.score_missing = lambda docs, model, w, path, limit=None: (
        {d["id"]: {"margin": 100.0 if d["id"].endswith("0") else -100.0} for d in docs}, [])
    monkeypatch.setitem(sys.modules, "eval.baseline_eval", fake)
    monkeypatch.setattr(ev, "find_scores_file", lambda seed: "scores.csv")
    monkeypatch.setattr(ev, "load_docs", lambda seed, roles, path: {"train": train} if roles == ["train"] else {})
    out = ev.published_flags(42, ROWS, "some/model", 8)
    assert set(out) == {0.05, 0.01}
    assert out[0.05].sum() == sum(r.id.endswith("0") for r in ROWS)


def test_published_flags_stop_with_advice_when_train_scores_are_missing(monkeypatch):
    fake = types.ModuleType("eval.baseline_eval")
    fake.cache_path = lambda model, w: "train.csv"
    fake.read_cache = lambda path: {}
    monkeypatch.setitem(sys.modules, "eval.baseline_eval", fake)
    monkeypatch.setattr(ev, "find_scores_file", lambda seed: "scores.csv")
    monkeypatch.setattr(ev, "load_docs", lambda seed, roles, path: {"train": [dict(id="a", y=0, text="x")]})
    with pytest.raises(SystemExit, match="baseline_eval"):
        ev.published_flags(42, ROWS, "some/model", 8)


# ----------------------------------------------------------------------------- figure F7

def _write_seed_csvs(tmp_path, seeds=(42, 43, 44), with_published=True):
    methods = ev.METHODS + ([ev.PUB] if with_published else [])
    for s in seeds:
        rng = np.random.default_rng(s)
        flags = {m: {b: rng.random(len(ROWS)) < 0.6 for b in ev.BUDGETS} for m in methods}
        cells = ev.build_cells(ROWS, flags)
        ev.write_outputs(s, cells, ev.render_report(s, cells, methods, []), out_dir=str(tmp_path))


def test_seed_files_and_aggregate(tmp_path):
    _write_seed_csvs(tmp_path)
    files = ev.seed_files(str(tmp_path))
    assert list(files) == [42, 43, 44]
    agg = ev.aggregate({s: ev.read_cells(p) for s, p in files.items()})
    a = agg[(0.05, "pattern bank", "leet")]
    assert len(a["dots"]) == 3 and a["n"] == 20
    assert math.isclose(a["mean"], np.mean(a["dots"]))
    assert a["lo"] <= a["mean"] <= a["hi"]


def test_figure_is_drawn_as_pdf_and_png(tmp_path):
    _write_seed_csvs(tmp_path)
    files = ev.seed_files(str(tmp_path))
    pdf, png = ev.build_figure({s: ev.read_cells(p) for s, p in files.items()},
                               str(tmp_path / "figs" / "F7_evasion.pdf"), str(tmp_path / "prev" / "F7_evasion.png"))
    assert os.path.getsize(pdf) > 2000 and os.path.getsize(png) > 2000
    assert open(pdf, "rb").read(4) == b"%PDF"
    assert b"/Type3" not in open(pdf, "rb").read()          # fonttype 42: TrueType, no Type 3 fonts


def test_figure_without_the_published_classifier(tmp_path):
    _write_seed_csvs(tmp_path, seeds=(42,), with_published=False)
    files = ev.seed_files(str(tmp_path))
    pdf, _ = ev.build_figure({s: ev.read_cells(p) for s, p in files.items()},
                             str(tmp_path / "a.pdf"), str(tmp_path / "a.png"))
    assert os.path.getsize(pdf) > 2000


def test_caption_line_is_replaced_not_duplicated(tmp_path):
    p = str(tmp_path / "captions.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("**F1** first caption\n**F7** old text\n")
    ev.update_caption(p, "**F7** new text")
    ev.update_caption(p, "**F7** new text")
    assert open(p, encoding="utf-8").read().splitlines() == ["**F1** first caption", "**F7** new text"]


def test_style_rules():
    assert figstyle.SINGLE_W == 3.5 and figstyle.DOUBLE_W == 7.16
    assert set(figstyle.OKABE_ITO) == {"#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"}
    assert set(figstyle.METHOD_COLORS.values()) - {"#999999"} <= set(figstyle.OKABE_ITO)
    assert len(set(figstyle.METHOD_HATCHES.values())) == len(figstyle.METHOD_HATCHES)    # a second cue per method
    plt = figstyle.apply_style()
    assert plt.rcParams["pdf.fonttype"] == 42 and plt.rcParams["font.size"] == 8


def test_figure_mode_stops_without_csvs(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="evasion_eval"):
        ev.figure_main(str(tmp_path))


def test_only_train_and_validation_roles_are_requested():
    src = open(os.path.join(ROOT, "eval", "evasion_eval.py"), encoding="utf-8").read()
    assert 'load_docs(seed, ["train"]' in src
    assert "final_test" not in src
