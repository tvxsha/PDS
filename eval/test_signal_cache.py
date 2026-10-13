"""
Tests for eval/signal_cache.py (task 1.9). No models, no pool, no real split: a fake scorer, fake rows
and a temporary SIGNAL_DIR.

    python -m pytest eval/test_signal_cache.py
"""

import csv
import json
import os

import pytest

from eval import signal_cache as sc
from eval.protocol import ROLES, Row

HELD_OUT = ROLES[-1]          # the held-out role, written without quoting its name in this file


def make_rows(prefix, n, label=1):
    return [Row.make(id=f"{prefix}{i:03d}", text=f"document {prefix} {i}", label=label, source="ext_fake",
                     attack_type="x") for i in range(n)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temporary SIGNAL_DIR / EVALSET_DIR and a fake pool: roles -> rows."""
    pool = {"baseline": make_rows("b", 4, 0), "train": make_rows("t", 30), "val": make_rows("v", 10),
            HELD_OUT: make_rows("h", 5)}
    monkeypatch.setattr(sc, "SIGNAL_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(sc, "EVALSET_DIR", str(tmp_path / "evalsets"))
    monkeypatch.setattr(sc, "_split_hash", lambda seed: "abc123")
    monkeypatch.setattr(sc, "_rows_for_roles", lambda seed, roles: [r for role in roles for r in pool[role]])
    return tmp_path


class Counter:
    def __init__(self):
        self.calls = []

    def __call__(self, row):
        self.calls.append(row.id)
        return float(len(row.id))


def test_header_and_path(env):
    scorer = Counter()
    sc.score_missing("demo", 42, ["val"], scorer)
    path = sc.signal_path("demo", 42)
    assert os.path.basename(path) == "signal_demo_seed42_abc123.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["id", "score", "ms"]
    assert len(rows) == 1 + 10
    assert all(float(r[2]) >= 0 for r in rows[1:])


def test_resume_skips_cached_ids(env):
    scorer = Counter()
    out = sc.score_missing("demo", 42, ["val"], scorer)
    assert len(scorer.calls) == 10 and len(out) == 10
    scorer2 = Counter()
    sc.score_missing("demo", 42, ["val"], scorer2)
    assert scorer2.calls == []                                  # everything cached: scorer never called
    scorer3 = Counter()
    out3 = sc.score_missing("demo", 42, ["val", "train"], scorer3)
    assert len(scorer3.calls) == 30                              # only the new role is scored
    assert len(out3) == 40
    assert sc.load_signal("demo", 42)["v000"][0] == float(len("v000"))
    with open(sc.signal_path("demo", 42), encoding="utf-8") as f:
        assert sum(1 for line in f if line.startswith("id,")) == 1     # the header is written once


def test_held_out_role_needs_final(env):
    with pytest.raises(ValueError, match="test role requested"):
        sc.score_missing("demo", 42, [HELD_OUT], Counter())
    with pytest.raises(ValueError, match="test role requested"):
        sc.score_missing("demo", 42, ["val", HELD_OUT], Counter())
    with pytest.raises(ValueError, match="test role requested"):
        sc.get_embeddings(42, [HELD_OUT])                         # refused before any model is imported
    assert not os.path.exists(sc.signal_path("demo", 42))        # nothing was written
    out = sc.score_missing("demo", 42, [HELD_OUT], Counter(), final=True)
    assert len(out) == 5


def test_unknown_role_and_bad_names(env):
    with pytest.raises(ValueError):
        sc.score_missing("demo", 42, ["validation"], Counter())
    with pytest.raises(ValueError):
        sc.signal_path("../evil", 42)
    with pytest.raises(ValueError):
        sc.score_missing("demo", 42, ["val"], lambda row: float("nan"))


def test_half_written_last_line_is_dropped(env):
    sc.score_missing("demo", 42, ["val"], Counter())
    path = sc.signal_path("demo", 42)
    with open(path, "a", encoding="utf-8") as f:
        f.write("t000,12.5")                                      # a crash in the middle of a line (no newline)
    assert "t000" not in sc.load_signal("demo", 42)
    scorer = Counter()
    sc.score_missing("demo", 42, ["val", "train"], scorer)
    assert "t000" in scorer.calls                                 # re-scored, not trusted
    lines = open(path, encoding="utf-8").read().split("\n")
    assert all(len(line.split(",")) == 3 for line in lines if line)    # every line is whole again


def write_evalset(env, name, rows):
    os.makedirs(sc.EVALSET_DIR, exist_ok=True)
    with open(os.path.join(sc.EVALSET_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict()) + "\n")


def test_load_evalset_checks_source(env):
    write_evalset(env, "good", make_rows("e", 3))
    assert len(sc.load_evalset("good")) == 3
    bad = make_rows("e", 2)
    bad[0].source = "bipia"
    write_evalset(env, "bad", bad)
    with pytest.raises(ValueError, match="ext_"):
        sc.load_evalset("bad")
    with pytest.raises(FileNotFoundError):
        sc.load_evalset("missing")


def test_score_external_caches_and_rescores_changed_sets(env, monkeypatch):
    calls = {"base": 0, "extra": 0}

    def fake_base(seed, names):
        def mk(k):
            def fn(text):
                calls["base"] += 1
                return 0.1 * k + len(text) / 1000
            return fn
        return {n: mk(i) for i, n in enumerate(names, 1)}

    def extra_fn(text):
        calls["extra"] += 1
        return 0.5

    monkeypatch.setattr(sc, "_base_scorers", fake_base)
    write_evalset(env, "demo", make_rows("e", 4))
    out = sc.score_external("demo", 42, extra={"mine": extra_fn})
    assert set(out) == {"pattern", "embedding", "perplexity", "mine"}
    assert all(len(v) == 4 for v in out.values())
    assert calls == {"base": 12, "extra": 4}
    with open(sc.external_path("demo", 42), encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == ["id", "signal", "score", "ms"]
    sc.score_external("demo", 42, extra={"mine": extra_fn})        # everything cached
    assert calls == {"base": 12, "extra": 4}
    sc.score_external("demo", 42, extra={"mine": extra_fn, "other": extra_fn})   # one new signal only
    assert calls == {"base": 12, "extra": 8}
    write_evalset(env, "demo", make_rows("e", 5))                  # the set changed: everything again
    out = sc.score_external("demo", 42, extra={"mine": extra_fn})
    assert all(len(v) == 5 for v in out.values()) and calls["base"] == 27
    with pytest.raises(ValueError):
        sc.score_external("demo", 42, extra={"pattern": extra_fn})