"""
Protocol evaluation: the held-out result the paper has been missing.

  fit on TRAIN  ->  choose thresholds on VAL  ->  report on TEST, once.

Everything that was previously tuned on the report set (calibration bounds,
fusion weights, thresholds) is now fitted on documents the test set never
touches. The embedding baseline is built ONLY from the split's "baseline"
documents, which are never scored or fitted on.

Needs: data/pooled/pooled.jsonl   (python -m data.pool_datasets)
       data/splits/split_seed42.json   (python -m eval.protocol --seed 42)

Usage:
    python -m eval.protocol_eval --seed 42            # score + fit; prints VALIDATION only
    python -m eval.protocol_eval --seed 42 --final    # ALSO the TEST report. Do this once,
                                                      # after code and thresholds are frozen.

Scoring is the slow part (GPT-2 + MiniLM per document), so raw scores are
cached in eval/results/pooled_scores_seed<N>_<splithash>.csv and the script
resumes if interrupted. Fitting and reporting take seconds.
"""

import argparse
import csv
import datetime
import json
import os
import subprocess
import time
from collections import defaultdict

import numpy as np

from eval.protocol import (POOLED_PATH, apply_split, load_split, parse_meta, read_jsonl,
                           validate_split)
from eval.stats import bootstrap_ci, compute_metrics, mcnemar_test

RESULTS_DIR = "eval/results"
SCORES = ["pattern_score", "embedding_score", "perplexity_score"]
FIELDS = ["id", "pattern_score", "embedding_score", "perplexity_score",
          "pattern_ms", "embedding_ms", "perplexity_ms"]


# --------------------------------------------------------------------------
# Scoring (slow, cached)
# --------------------------------------------------------------------------

def score_all(by_role, cache_path):
    cached = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = {r["id"]: r for r in csv.DictReader(f)}
    todo = [r for role in ("train", "val", "test") for r in by_role[role] if r.id not in cached]
    print(f"{len(cached)} documents already scored, {len(todo)} to score")
    if not todo:
        return cached

    from detectors import pattern_detector, embedding_detector, perplexity_detector
    print(f"Building embedding baseline from {len(by_role['baseline'])} baseline documents...")
    baseline = embedding_detector.BaselineCorpus([r.text for r in by_role["baseline"]])
    perplexity_detector._get_model_and_tokenizer()

    is_new = not os.path.exists(cache_path)
    start = time.time()
    with open(cache_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        for i, r in enumerate(todo, 1):
            p = pattern_detector.score_document(r.text)
            e = embedding_detector.score_document(r.text, baseline)
            q = perplexity_detector.score_document(r.text)
            row = {"id": r.id, "pattern_score": p.score, "embedding_score": e.score,
                   "perplexity_score": q.score, "pattern_ms": p.latency_ms,
                   "embedding_ms": e.latency_ms, "perplexity_ms": q.latency_ms}
            w.writerow(row)
            cached[r.id] = {k: str(v) for k, v in row.items()}
            if i % 25 == 0:
                f.flush()
                eta = (time.time() - start) / i * (len(todo) - i)
                print(f"  scored {i}/{len(todo)}  (~{eta / 60:.1f} min left)")
    return cached


def matrix(rows, cached):
    X = np.array([[float(cached[r.id][s]) for s in SCORES] for r in rows])
    y = np.array([r.label for r in rows])
    return X, y


# --------------------------------------------------------------------------
# Methods: every one is fitted on TRAIN, thresholded on VAL
# --------------------------------------------------------------------------

def best_threshold(y, s):
    """Threshold (predict poisoned if score >= t) maximising F1, ties -> lower FPR."""
    u = np.unique(s)
    cands = np.concatenate(([u[0] - 1e-9], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]))
    best_key, best_t = None, None
    for t in cands:
        m = compute_metrics(y, (s >= t).astype(int))
        key = (m["f1"], -m["fpr"])
        if best_key is None or key > best_key:
            best_key, best_t = key, t
    return float(best_t)


def best_threshold_at_fpr(y, s, max_fpr=0.05):
    """Lowest threshold whose FPR on these (validation) docs is <= max_fpr:
    the deployable operating point ("how many attacks do we catch if we may
    wrongly flag at most 5% of clean documents?")."""
    desc = np.sort(s[y == 0])[::-1]
    k = int(np.floor(max_fpr * len(desc)))
    if k >= len(desc):
        return float(desc[-1] - 1e-9)
    return float(np.nextafter(desc[k], np.inf))


def build_methods(Xtr, ytr):
    """name -> function mapping a raw-score matrix to one score per document."""
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)

    def cal(X):
        return np.clip((X - lo) / span, 0.0, 1.0)

    from fusion.learned import build_model
    lr_raw = build_model().fit(Xtr, ytr)
    lr_cal = build_model().fit(cal(Xtr), ytr)

    methods = {
        "pattern only": lambda X: X[:, 0],
        "embedding only": lambda X: X[:, 1],
        "perplexity only": lambda X: X[:, 2],
        "naive average (raw)": lambda X: X.mean(axis=1),
        "calibrated average": lambda X: cal(X).mean(axis=1),
        "learned LR (raw)": lambda X: lr_raw.predict_proba(X)[:, 1],
        "learned LR (calibrated)": lambda X: lr_cal.predict_proba(cal(X))[:, 1],
    }
    info = {"calibration_lo": lo.tolist(), "calibration_hi": hi.tolist(),
            "lr_raw_coef": lr_raw.coef_[0].tolist(), "lr_cal_coef": lr_cal.coef_[0].tolist()}
    return methods, info


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def git_state():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain",
                                              "detectors", "fusion", "eval"], text=True).strip())
        return sha, dirty
    except Exception:
        return "unknown", None


def val_report(methods, thresholds, thresholds_fpr, Xva, yva):
    print("\nVALIDATION (thresholds were chosen here, so these are still optimistic)\n")
    triv = compute_metrics(yva, np.ones_like(yva))
    print(f"reference: flag-everything gets F1 {triv['f1']:.2f} on this pool (precision {triv['precision']:.2f})\n")
    print(f"{'method':26s} {'thr':>7s}   P     R     F1    FPR  | recall @ FPR<=5%")
    for name, fn in methods.items():
        s = fn(Xva)
        m = compute_metrics(yva, (s >= thresholds[name]).astype(int))
        m5 = compute_metrics(yva, (s >= thresholds_fpr[name]).astype(int))
        print(f"{name:26s} {thresholds[name]:7.3f}  {m['precision']:.2f}  {m['recall']:.2f}  "
              f"{m['f1']:.2f}  {m['fpr']:.2f}  | {m5['recall']:.2f}")


def test_report(methods, thresholds, thresholds_fpr, info, test_rows, Xte, yte, seed, split_hash):
    preds = {n: (fn(Xte) >= thresholds[n]).astype(int) for n, fn in methods.items()}
    sha, dirty = git_state()
    L = [f"# Protocol evaluation: TEST split (seed {seed}, split {split_hash})", "",
         f"Code version: `{sha}`{' (UNCOMMITTED CHANGES in detectors/fusion/eval)' if dirty else ''}. "
         f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}.", "",
         f"Test set: {len(yte)} documents ({int((yte == 0).sum())} clean, {int((yte == 1).sum())} poisoned) "
         "from groups never seen in fitting, calibration or threshold choice.", "",
         "## Overall: thresholds chosen to maximise F1 on validation", "",
         "Read the first row first: on a pool with more poisoned than clean documents, flagging EVERYTHING "
         "already scores a respectable F1. A detector only means something if it beats that and keeps FPR low.", "",
         "| method | threshold | precision | recall | F1 | FPR | F1 95% CI | ROC-AUC |", "|---|---|---|---|---|---|---|---|"]
    triv = compute_metrics(yte, np.ones_like(yte))
    L.append(f"| flag everything (trivial) | - | {triv['precision']:.2f} | {triv['recall']:.2f} | "
             f"{triv['f1']:.2f} | {triv['fpr']:.2f} | - | 0.50 |")
    from sklearn.metrics import roc_auc_score
    for n, fn in methods.items():
        ci = bootstrap_ci(yte, preds[n], n_resamples=1000, seed=0)
        m = compute_metrics(yte, preds[n])
        L.append(f"| {n} | {thresholds[n]:.3f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | "
                 f"{m['fpr']:.2f} | {ci['f1']['low']:.2f}-{ci['f1']['high']:.2f} | "
                 f"{roc_auc_score(yte, fn(Xte)):.2f} |")

    preds_fpr = {n: (fn(Xte) >= thresholds_fpr[n]).astype(int) for n, fn in methods.items()}
    L += ["", "## Deployable operating point: threshold set so validation FPR <= 5%", "",
          "| method | recall (attacks caught) | test FPR | precision |", "|---|---|---|---|"]
    for n in methods:
        m = compute_metrics(yte, preds_fpr[n])
        L.append(f"| {n} | {m['recall']:.2f} | {m['fpr']:.2f} | {m['precision']:.2f} |")

    a, b = "calibrated average", "naive average (raw)"
    mc = mcnemar_test(yte, preds[a], preds[b])
    L += ["", f"McNemar, {a} vs {b}: b={mc['b']}, c={mc['c']}, p={mc['p_value']:.4f}", ""]

    sources = sorted({r.source for r in test_rows})
    key = ["calibrated average", "learned LR (calibrated)", "embedding only"]
    L += ["## Per source (threshold fixed globally; this is where transfer failures show)", "",
          "| source | n clean | n poisoned | " + " | ".join(f"{k}: recall / FPR" for k in key) + " |",
          "|---|---|---|" + "---|" * len(key)]
    for src in sources:
        idx = [i for i, r in enumerate(test_rows) if r.source == src]
        ys = yte[idx]
        cells = []
        for k in key:
            pr = preds[k][idx]
            rec = pr[ys == 1].mean() if (ys == 1).any() else float("nan")
            fpr = pr[ys == 0].mean() if (ys == 0).any() else float("nan")
            cells.append(f"{rec:.2f} / {fpr:.2f}")
        L.append(f"| {src} | {int((ys == 0).sum())} | {int((ys == 1).sum())} | " + " | ".join(cells) + " |")

    names = ["pattern only", "embedding only", "perplexity only", "naive average (raw)",
             "calibrated average", "learned LR (calibrated)"]
    L += ["", "## Failure map at scale: recall per attack type", "",
          "| attack type | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    by = defaultdict(list)
    for i, r in enumerate(test_rows):
        if r.label == 1:
            by[r.attack_type].append(i)
    for t, idx in sorted(by.items()):
        L.append(f"| {t} | {len(idx)} | " + " | ".join(f"{preds[n][idx].mean():.2f}" for n in names) + " |")

    pos = defaultdict(list)
    for i, r in enumerate(test_rows):
        if r.source == "bipia" and r.label == 1:
            pos[parse_meta(r.meta).get("position", "?")].append(i)
    if pos:
        L += ["", "## BIPIA: recall by attack position (truncation blind spot?)", "",
              "| position | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for p in ("start", "middle", "end"):
            if p in pos:
                L.append(f"| {p} | {len(pos[p])} | " + " | ".join(f"{preds[n][pos[p]].mean():.2f}" for n in names) + " |")

    L += ["", "## Learned-fusion coefficients (fitted on train)", "",
          "- raw: " + ", ".join(f"{k}={v:.3f}" for k, v in zip(SCORES, info["lr_raw_coef"])),
          "- calibrated: " + ", ".join(f"{k}={v:.3f}" for k, v in zip(SCORES, info["lr_cal_coef"])), ""]
    return "\n".join(L), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--final", action="store_true", help="also evaluate the TEST split")
    ap.add_argument("--force", action="store_true", help="re-run --final (counts as peeking; say so)")
    args = ap.parse_args()

    rows = read_jsonl(POOLED_PATH)
    split = load_split(args.seed)
    problems = validate_split(rows, split)
    if problems:
        raise SystemExit("Split is not valid for this pool (re-run eval.protocol): " + "; ".join(problems))
    by_role = apply_split(rows, split)
    print({k: len(v) for k, v in by_role.items()}, "| dropped:", len(split["dropped"]))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache = os.path.join(RESULTS_DIR, f"pooled_scores_seed{args.seed}_{split['hash']}.csv")
    cached = score_all(by_role, cache)

    Xtr, ytr = matrix(by_role["train"], cached)
    Xva, yva = matrix(by_role["val"], cached)
    methods, info = build_methods(Xtr, ytr)
    thresholds = {n: best_threshold(yva, fn(Xva)) for n, fn in methods.items()}
    thresholds_fpr = {n: best_threshold_at_fpr(yva, fn(Xva)) for n, fn in methods.items()}
    val_report(methods, thresholds, thresholds_fpr, Xva, yva)

    if not args.final:
        print("\n(Test split NOT evaluated. Add --final once everything is frozen.)")
        return

    lock = os.path.join(RESULTS_DIR, f"protocol_FINAL_seed{args.seed}.json")
    if os.path.exists(lock) and not args.force:
        raise SystemExit(f"The test split was already evaluated ({lock}). Running it again is "
                         "peeking; use --force only if you will say so in the paper.")
    test_rows = by_role["test"]
    Xte, yte = matrix(test_rows, cached)
    text, preds = test_report(methods, thresholds, thresholds_fpr, info, test_rows, Xte, yte, args.seed, split["hash"])
    md = os.path.join(RESULTS_DIR, f"protocol_report_seed{args.seed}.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write(text)
    with open(lock, "w", encoding="utf-8") as f:
        json.dump({"evaluated": datetime.datetime.now().isoformat(), "split_hash": split["hash"],
                   "thresholds": thresholds, "thresholds_fpr5": thresholds_fpr, "git": git_state()[0]}, f, indent=1)
    print("\n" + text)
    print(f"Saved {md}")


if __name__ == "__main__":
    main()