"""Tests for the pure parts of eval/fusion_v2.py (no models, no data files, no sklearn)."""

import json

import numpy as np
import pytest

from eval.fusion_v2 import (TIE, build_matrix, candidate_json, feature_list, read_selection, select_feature_set,
                            squash)


def aucs(**kw):
    return {k: [v, v, v] for k, v in kw.items()}


def test_read_selection_cases(tmp_path):
    p = tmp_path / "sel.json"
    assert read_selection(str(tmp_path / "does_not_exist.json")) == ("missing", None)
    p.write_text(json.dumps({"selected": "original", "uses_variant": False}))
    assert read_selection(str(p)) == ("original", None)
    p.write_text(json.dumps({"selected": "emb_abtt1", "uses_variant": True}))
    assert read_selection(str(p)) == ("variant", "emb_abtt1")
    p.write_text(json.dumps({"selected": "emb_abtt1", "uses_variant": False}))          # best variant named but not selected
    assert read_selection(str(p)) == ("original", None)
    p.write_text(json.dumps({"variant": "emb_center"}))                                 # another author's key name
    assert read_selection(str(p)) == ("variant", "emb_center")
    p.write_text(json.dumps({"selected": "Original stays"}))
    assert read_selection(str(p)) == ("original", None)
    p.write_text(json.dumps({"something": 1}))                                          # unknown layout: stop, do not guess
    with pytest.raises(SystemExit):
        read_selection(str(p))


def test_feature_sets_need_their_variants():
    assert feature_list("F0") == ["pattern", "embedding", "perplexity"]
    assert feature_list("F1") == ["pattern", "embedding", "perplexity", "sentence"]       # same column order as fit_everything
    assert feature_list("F4")[-1] == "published"
    assert feature_list("F2") is None and feature_list("F3") is None and feature_list("F5") is None
    assert feature_list("F2", v45="emb_dom_z") == ["pattern", "emb_dom_z", "perplexity", "sentence"]
    assert feature_list("F3", v44="emb_abtt1") == ["pattern", "emb_abtt1", "perplexity", "sentence"]
    assert feature_list("F5", v45="emb_dom_z") is None                                    # needs both
    assert feature_list("F5", v45="emb_dom_z", v44="emb_abtt1") == \
        ["pattern", "emb_dom_z", "emb_abtt1", "perplexity", "sentence"]


def test_matrix_columns_and_variant_rescaling():
    rng = np.random.default_rng(0)
    tr = {"pattern": rng.random(50), "embedding": 0.4 + 0.05 * rng.normal(size=50), "perplexity": rng.random(50),
          "sentence": rng.random(50), "published": rng.normal(size=50) * 8, "emb_dom_z": 3.0 + 2.0 * rng.normal(size=50)}
    ev = {k: v[:10] * 1.0 for k, v in tr.items()}
    Xtr, Xev = build_matrix(["pattern", "emb_dom_z", "perplexity", "sentence", "published"], tr, ev)
    assert Xtr.shape == (50, 5) and Xev.shape == (10, 5)
    # the variant takes the ORIGINAL embedding's train mean and spread; the eval rows use the TRAIN statistics
    assert Xtr[:, 1].mean() == pytest.approx(tr["embedding"].mean())
    assert Xtr[:, 1].std() == pytest.approx(tr["embedding"].std())
    assert np.allclose(Xev[:, 1], Xtr[:10, 1])
    # the published margin is squashed to [-1, 1]; the plain signals are untouched
    assert np.abs(Xtr[:, 4]).max() <= 1.0 and np.allclose(Xtr[:, 4], squash(tr["published"]))
    assert np.allclose(Xtr[:, 0], tr["pattern"]) and np.allclose(Xtr[:, 2], tr["perplexity"]) and np.allclose(Xtr[:, 3], tr["sentence"])


def test_the_clearly_best_set_wins():
    assert select_feature_set(aucs(F0=0.80, F1=0.93, F4=0.96))["selected"] == "F4"


def test_a_tie_prefers_fewer_features_and_cheaper_signals():
    sel = select_feature_set(aucs(F0=0.80, F1=0.930, F4=0.930 + TIE * 0.6))
    assert sel["best_by_auc"] == "F4" and sel["selected"] == "F1"                         # F4 gains less than 0.005: F1 stays
    sel = select_feature_set(aucs(F1=0.900, F4=0.900 + TIE + 0.001))
    assert sel["selected"] == "F4"                                                        # a real gain is taken
    sel = select_feature_set(aucs(F1=0.930, F2=0.931, F3=0.932, F5=0.933, F4=0.934))
    assert sel["selected"] == "F1"                                                        # everything within the tie: the incumbent
    sel = select_feature_set(aucs(F1=0.900, F2=0.940, F3=0.941, F4=0.944))
    assert sel["selected"] == "F2" and sel["tied_with_best"] == ["F2", "F3", "F4"]        # F2 before F3 before F4 in cost


def test_the_mean_over_seeds_decides_not_one_seed():
    sel = select_feature_set({"F1": [0.90, 0.90, 0.90], "F4": [0.99, 0.80, 0.80]})        # F4 wins one seed, loses the mean
    assert sel["selected"] == "F1"


def test_candidate_file_fields():
    sel = select_feature_set(aucs(F0=0.8, F1=0.93))
    c = candidate_json(sel, None, None, "missing")
    assert c["features"] == ["pattern", "embedding", "perplexity", "sentence"]
    assert c["embedding_variant"] == "original" and c["uses_published"] is False
    assert c["complete"] is False and c["missing_inputs"] == ["4.4"]                      # provisional until 4.4 exists
    assert candidate_json(sel, None, None, "original")["complete"] is True
    sel = select_feature_set(aucs(F1=0.9, F5=0.95, F4=0.94))
    c = candidate_json(sel, "emb_dom_z", "emb_abtt1", "variant")
    assert c["embedding_variants"] == ["emb_dom_z", "emb_abtt1"] and c["embedding_variant"] == "emb_dom_z + emb_abtt1"
    sel = select_feature_set(aucs(F1=0.9, F4=0.95))
    assert candidate_json(sel, None, None, "original")["uses_published"] is True