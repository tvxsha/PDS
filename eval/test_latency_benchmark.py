"""Tests for eval.latency_benchmark (fake detectors, no models, no data). Run:  python -m pytest eval/test_latency_benchmark.py"""

import json
import time

import numpy as np
import pytest

from eval import cascade_eval, latency_benchmark as lb


def docs(n=10, base=300):
    return [{"id": f"d{i:02d}", "text": "x" * (base + 400 * i), "y": i % 2} for i in range(n)]


class FakeClock:
    """A clock that returns the given readings in order (two readings per timing: start and end)."""
    def __init__(self, readings):
        self.readings = list(readings)

    def __call__(self):
        return self.readings.pop(0)


def test_time_ms_uses_the_clock_around_the_call_only():
    calls = []
    clock = FakeClock([10.0, 10.0025])
    assert lb.time_ms(lambda t: calls.append(t), "doc", clock) == pytest.approx(2.5)
    assert calls == ["doc"]


def test_measure_takes_the_median_of_the_repeats_and_drops_one_stall():
    # one document, one detector, 3 repeats: 2 ms, 900 ms (a stall), 3 ms -> median 3 ms, max 900 ms
    clock = FakeClock([0.0, 0.002, 1.0, 1.9, 2.0, 2.003])
    rows = lb.measure({"d": lambda t: None}, docs(1), 3, clock)
    assert len(rows) == 1
    assert rows[0]["median_ms"] == pytest.approx(3.0)
    assert rows[0]["max_ms"] == pytest.approx(900.0)


def test_measure_gives_one_row_per_document_and_detector_and_calls_each_repeats_times():
    counts = {"a": 0, "b": 0}

    def make(name):
        def fn(text):
            counts[name] += 1
        return fn

    rows = lb.measure({"a": make("a"), "b": make("b")}, docs(4), 3)
    assert len(rows) == 8
    assert counts == {"a": 12, "b": 12}
    assert {r["detector"] for r in rows} == {"a", "b"}
    with pytest.raises(ValueError):
        lb.measure({"a": make("a")}, docs(1), 0)


def test_warm_up_runs_every_detector_on_every_text_without_timing():
    seen = []
    lb.warm_up({"a": lambda t: seen.append(("a", t)), "b": lambda t: seen.append(("b", t))}, ["t1", "t2"])
    assert seen == [("a", "t1"), ("b", "t1"), ("a", "t2"), ("b", "t2")]


def test_sampling_is_deterministic_and_independent_of_input_order():
    d = docs(50)
    a = lb.sample_docs(d, 10)
    b = lb.sample_docs(list(reversed(d)), 10)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert len({x["id"] for x in a}) == 10
    assert len(lb.sample_docs(d, 500)) == 50                   # asking for more than there is returns all


def test_buckets():
    assert lb.bucket_of(0) == "under 500 characters"
    assert lb.bucket_of(499) == "under 500 characters"
    assert lb.bucket_of(500) == "500 to 2,000 characters"
    assert lb.bucket_of(1999) == "500 to 2,000 characters"
    assert lb.bucket_of(2000) == "over 2,000 characters"
    assert lb.bucket_of(10 ** 6) == "over 2,000 characters"


def test_summary_statistics_by_hand():
    rows = [{"id": f"d{i}", "chars": 100 if i < 5 else 3000, "detector": "pattern", "median_ms": float(i + 1), "max_ms": 99.0}
            for i in range(10)]
    overall, by_length = lb.summarise(rows)
    s = overall["pattern"]
    assert s["n"] == 10 and s["median_ms"] == pytest.approx(5.5) and s["mean_ms"] == pytest.approx(5.5)
    assert s["p95_ms"] == pytest.approx(float(np.percentile(np.arange(1, 11), 95)))
    assert by_length["pattern"]["under 500 characters"]["median_ms"] == pytest.approx(3.0)
    assert by_length["pattern"]["over 2,000 characters"]["median_ms"] == pytest.approx(8.0)
    assert "500 to 2,000 characters" not in by_length["pattern"]        # empty buckets are left out


def test_detectors_are_listed_in_the_documented_order():
    rows = [{"id": "d", "chars": 10, "detector": n, "median_ms": 1.0, "max_ms": 1.0}
            for n in ["ours3", "zzz", "published classifier", "pattern"]]
    overall, _ = lb.summarise(rows)
    assert list(overall) == ["pattern", "published classifier", "ours3", "zzz"]


def test_json_is_what_the_cascade_and_the_paper_tables_read(tmp_path):
    detectors = {n: (lambda t: time.sleep(0.001)) for n in ["pattern", "embedding", "perplexity", "sentence classifier",
                                                           "published classifier", "ours3", "ours3 + sentence"]}
    sample = docs(6)
    rows = lb.measure(detectors, sample, 2)
    overall, by_length = lb.summarise(rows)
    hw = {"platform": "unit-test machine", "cpu": "fake", "logical_cores": 4, "threads_requested": 4, "power": "plugged in"}
    res = lb.result_dict(42, len(sample), 2, hw, overall, by_length)
    paths = lb.write_outputs(res, rows, overall, by_length, out_dir=str(tmp_path))
    data = json.load(open(paths["json"], encoding="utf-8"))
    assert data["n"] == 6 and set(data["detectors"]) == set(detectors)
    assert set(data["detectors"]["pattern"]) == {"median_ms", "p95_ms", "mean_ms"}
    ms, hardware, p95, keys = cascade_eval.read_latency(paths["json"])      # the reader of task 5.4
    assert set(ms) == {"pattern", "embedding", "perplexity", "sentence", "published"}
    assert all(v > 0 for v in ms.values()) and "unit-test machine" in hardware
    csv_lines = open(paths["csv"], encoding="utf-8").read().splitlines()
    assert csv_lines[0] == "id,chars,detector,median_ms,max_ms" and len(csv_lines) == 1 + 6 * 7
    assert "# Latency benchmark" in open(paths["md"], encoding="utf-8").read()


def test_a_real_stall_in_one_run_does_not_reach_the_summary():
    state = {"n": 0}

    def sometimes_slow(text):
        state["n"] += 1
        time.sleep(0.15 if state["n"] == 2 else 0.002)               # the second call stalls

    rows = lb.measure({"d": sometimes_slow}, docs(1), 3)
    assert rows[0]["median_ms"] < 50 < rows[0]["max_ms"]