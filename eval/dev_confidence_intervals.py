"""
Task 0.9: confidence intervals for the development-set fusion table (Table I).

    python -m eval.dev_confidence_intervals

Reads eval/results/evaluation_raw.csv (77 documents: 60 poisoned, 17 clean), rebuilds every row of
the fusion table from the logged detector scores, and attaches a 95% bootstrap interval to
precision, recall and F1 (documents resampled with replacement, thresholds held fixed, via
eval.stats.bootstrap_ci).

Because the thresholds were tuned on these same 77 documents, the intervals alone understate the
uncertainty. So each tuned row also gets an "honest re-tuning" estimate: in each of 1000 resamples
the threshold is re-picked on the resample and F1 is measured on the documents the resample left
out. Its mean is a less optimistic F1 for "tune a threshold on a set like this, then use it".
(The learned rows reuse the 5-fold out-of-fold probabilities, so the model itself is never
scored on its own training documents; only the threshold is re-tuned.)

Writes eval/results/dev_confidence_intervals.md.
"""

import csv
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from eval.ablation import cascade, prf
from eval.stats import bootstrap_ci
from fusion.learned import FEATURE_NAMES, build_model

REPS = 1000


def load():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["true_label"]) for r in rows])
    S = np.array([[float(r[k]) for k in FEATURE_NAMES] for r in rows])
    return y, S


def minmax(S, lo, hi):
    return np.clip((S - lo) / (hi - lo), 0, 1)


def oof(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(build_model(), X, y, cv=skf, method="predict_proba")[:, 1]


def best_threshold(score, y, grid):
    best_f, best_t = -1.0, grid[0]
    for t in grid:
        f = prf(score >= t, y)[2]
        if f > best_f + 1e-12:
            best_f, best_t = f, t
    return best_t


def honest_f1(make_score, y, grid, reps=REPS, seed=0):
    """Re-tune the threshold on each bootstrap resample, score F1 on the left-out documents.
    make_score(train_idx) must return a score for ALL documents (any fitting uses train_idx only)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        if len(oob) < 5 or y[oob].sum() == 0 or (1 - y[oob]).sum() == 0 or y[idx].sum() == 0 or (1 - y[idx]).sum() == 0:
            continue
        score = make_score(idx)
        t = best_threshold(score[idx], y[idx], grid)
        out.append(prf(score[oob] >= t, y[oob])[2])
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)


def main():
    y, S = load()
    n = len(y)
    lo, hi = S.min(0), S.max(0)
    fine = np.linspace(0, 1, 1001)
    coarse = np.array([i / 20 for i in range(1, 20)])

    naive = S.mean(1)
    cal = minmax(S, lo, hi).mean(1)
    p_raw = oof(S, y)
    p_cal = oof(minmax(S, lo, hi), y)
    staged, _ = cascade(S, True)

    t_naive = best_threshold(naive, y, fine)
    t_cal = 0.20  # the value the paper reports (the 0.05-grid optimum; the 0.001 grid optimum is 0.194)
    t_raw = best_threshold(p_raw, y, coarse)
    t_lcal = best_threshold(p_cal, y, coarse)

    specs = [
        ("Flag every document", np.ones(n, bool), "-", None),
        ("Naive mean", naive >= 0.30, "0.30 (default)", None),
        ("Naive mean, re-tuned", naive >= t_naive, f"{t_naive:.3f} (tuned)", (lambda idx: naive, fine)),
        ("Learned, raw scores", p_raw >= t_raw, f"{t_raw:.2f} (tuned)", (lambda idx: p_raw, coarse)),
        ("Calibrated mean", cal >= t_cal, f"{t_cal:.2f} (tuned)",
         (lambda idx: minmax(S, S[idx].min(0), S[idx].max(0)).mean(1), coarse)),
        ("Learned, calibrated", p_cal >= t_lcal, f"{t_lcal:.2f} (tuned)", (lambda idx: p_cal, coarse)),
        ("Staged cascade", staged, "0.60/0.30/0.20", None),
    ]

    lines = ["# Development-set fusion table with 95% bootstrap intervals", "",
             f"{n} documents ({int(y.sum())} poisoned, {int((1 - y).sum())} clean). Documents are resampled with replacement "
             f"({REPS} resamples) and the thresholds are held fixed, so these intervals do NOT include the uncertainty from having "
             "tuned the thresholds on the same documents. Only 17 clean documents are present, so precision rests on very few "
             "false alarms and its interval is wide.", "",
             "| method | threshold | P (95% CI) | R (95% CI) | F1 (95% CI) | FP | FN | F1 with threshold re-tuned on resamples, scored on left-out documents |",
             "|---|---|---|---|---|---|---|---|"]
    for name, flag, thr, tune in specs:
        ci = bootstrap_ci(y, flag.astype(int), n_resamples=REPS, seed=0)
        p, r, f, tp, fp, fn = prf(flag, y)
        cells = [f"{ci[k]['point']:.2f} [{ci[k]['low']:.2f}, {ci[k]['high']:.2f}]" for k in ("precision", "recall", "f1")]
        if tune:
            make, grid = tune
            m, l, h, k = honest_f1(make, y, grid)
            honest = f"{m:.2f} [{l:.2f}, {h:.2f}]"
        else:
            honest = "n/a (no tuned threshold)" if name != "Staged cascade" else "n/a (thresholds fixed by hand earlier)"
        lines.append(f"| {name} | {thr} | {cells[0]} | {cells[1]} | {cells[2]} | {fp} | {fn} | {honest} |")

    # knife-edge check for the naive re-tuned row
    lines += ["", "## How fragile is the re-tuned naive threshold?", "",
              "| naive threshold | precision | recall | F1 | FP | FN |", "|---|---|---|---|---|---|"]
    for t in (0.095, 0.100, t_naive, 0.105, 0.110, 0.120):
        p, r, f, tp, fp, fn = prf(naive >= t, y)
        lines.append(f"| {t:.3f} | {p:.2f} | {r:.2f} | {f:.2f} | {fp} | {fn} |")
    lines += ["",
              f"The best naive threshold ({t_naive:.3f}) sits in a window less than a thousandth wide: moving it a few thousandths "
              "either way costs documents. That is the 'tuned on the evaluation set' problem in one table.", ""]
    report = "\n".join(lines)
    print(report)
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/dev_confidence_intervals.md", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print("Saved eval/results/dev_confidence_intervals.md")


if __name__ == "__main__":
    main()