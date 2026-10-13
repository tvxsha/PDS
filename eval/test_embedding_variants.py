"""Tests for the numeric core of eval/embedding_variants.py (numpy only, no models, no data files)."""

import numpy as np
import pytest

from eval.embedding_variants import MIN_DOMAIN, VARIANTS, Reference, select


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@pytest.fixture
def ref():
    """Two domains of 6 baseline documents each (around axis 0 and axis 1) plus a tiny domain of 2."""
    rng = np.random.default_rng(0)
    centres = {"A": np.eye(8)[0], "B": np.eye(8)[1], "C": np.eye(8)[2]}
    emb, doms = [], []
    for d, n in [("A", 6), ("B", 6), ("C", 2)]:
        for _ in range(n):
            emb.append(unit(centres[d] + 0.2 * rng.normal(size=8)))
            doms.append(d)
    return Reference(np.array(emb), doms)


def test_original_score_is_one_minus_max_cosine_over_two(ref):
    e = unit(np.eye(8)[0] + 0.1)
    expected = (1 - (ref.B @ e).max()) / 2
    assert ref.score("emb_max", e[None, :], ["A"])[0] == pytest.approx(expected)


def test_top_k_averages_the_k_highest_cosines(ref):
    e = unit(np.eye(8)[0] + 0.1)
    sims = np.sort(ref.B @ e)[::-1]
    assert ref.score("emb_top3", e[None, :], ["A"])[0] == pytest.approx((1 - sims[:3].mean()) / 2)
    assert ref.score("emb_top5", e[None, :], ["A"])[0] == pytest.approx((1 - sims[:5].mean()) / 2)


def test_domain_restriction_changes_the_score(ref):
    """A document that looks like domain B but is labelled domain A: normal for the original, odd inside domain A."""
    e = unit(np.eye(8)[1])
    own_pool = ref.score("emb_max", e[None, :], ["B"])[0]
    labelled_a = ref.score("emb_dom_max", e[None, :], ["A"])[0]
    labelled_b = ref.score("emb_dom_max", e[None, :], ["B"])[0]
    assert labelled_b == pytest.approx(own_pool)
    assert labelled_a > labelled_b + 0.2


def test_small_domain_falls_back_to_the_whole_baseline(ref):
    assert MIN_DOMAIN > 2
    assert ref.pool_key("C") is None and ref.pool_key("never seen") is None
    assert ref.pool_key("A") == "A"
    e = unit(np.eye(8)[2] + 0.1)
    for name in ("emb_dom_max", "emb_dom_z"):
        assert ref.score(name, e[None, :], ["C"])[0] == pytest.approx(ref.score(name, e[None, :], ["never seen"])[0])
    assert ref.score("emb_dom_max", e[None, :], ["C"])[0] == pytest.approx(ref.score("emb_max", e[None, :], ["C"])[0])


def test_z_score_is_higher_for_the_stranger(ref):
    near = unit(ref.B[0] + 0.01)
    far = unit(np.eye(8)[1])                    # domain B content shown to domain A
    z = ref.score("emb_dom_z", np.array([near, far]), ["A", "A"])
    assert z[1] > z[0]


def test_batch_scoring_equals_one_by_one(ref):
    rng = np.random.default_rng(1)
    E = unit(rng.normal(size=(10, 8)))
    doms = ["A", "B", "C", "A", "B", "C", "x", "A", "B", "C"]
    for name in VARIANTS + ("emb_max",):
        batch = ref.score(name, E, doms)
        single = np.array([ref.score(name, E[i:i + 1], [doms[i]])[0] for i in range(len(E))])
        assert np.allclose(batch, single, atol=1e-12)


def test_selection_keeps_the_original_unless_the_margin_is_met():
    def result(seed, auc):
        return {"seed": seed, "names": ["original"] + list(VARIANTS),
                "single": {n: {"auc_tr": a} for n, a in auc.items()}}
    small = {"original": 0.60, "emb_top3": 0.61, "emb_top5": 0.62, "emb_dom_max": 0.59, "emb_dom_z": 0.62}
    assert select([result(s, small) for s in (42, 43, 44)])["selected"] == "original"       # best gain 0.02 < 0.03
    big = dict(small, emb_dom_z=0.64)
    sel = select([result(s, big) for s in (42, 43, 44)])                                      # gain 0.04 >= 0.03
    assert sel["selected"] == "emb_dom_z" and sel["uses_variant"]