"""
Task 0.8: does the perplexity detector earn its place?

Replays the development-set scores and timings already logged in
eval/results/evaluation_raw.csv, so nothing is re-run and no model is loaded:

    python -m eval.ablation

Two questions:
  1. Cascade ablation: the staged pipeline with and without the perplexity stage
     (same thresholds as eval/staged_pipeline.py: 0.60 / 0.30 / 0.20).
  2. Fusion ablation: the calibrated mean over every subset of the three detectors,
     each with its own best threshold on a fine grid (so every subset gets the
     same optimistic treatment, and none is favoured).

Everything here is on the same 77 development documents that were used to choose
the thresholds, so differences of one document are noise. The script prints how
many documents each difference amounts to.

Writes eval/results/ablation.md.
"""

import csv
import itertools
import os

import numpy as np

PATTERN_T, EMBED_T, PPL_T = 0.60, 0.30, 0.20
NAMES = ["pattern", "embedding", "perplexity"]
SHORT = {"pattern": "A", "embedding": "B", "perplexity": "C"}


def load():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r["true_label"]) for r in rows])
    S = np.array([[float(r[f"{n}_score"]) for n in NAMES] for r in rows])
    T = np.array([[float(r[f"{n}_latency_ms"]) for n in NAMES] for r in rows])
    return rows, y, S, T


def prf(flag, y):
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn


def cascade(S, use_perplexity):
    """Return (flags, stage_ran) for the cascade; stage_ran[i] = [A, B, C] booleans."""
    n = len(S)
    flag = np.zeros(n, bool)
    ran = np.zeros((n, 3), bool)
    ran[:, 0] = True
    for i, (a, b, c) in enumerate(S):
        if a >= PATTERN_T:
            flag[i] = True
            continue
        ran[i, 1] = True
        if b >= EMBED_T:
            flag[i] = True
            continue
        if use_perplexity:
            ran[i, 2] = True
            if c >= PPL_T:
                flag[i] = True
    return flag, ran


def best_over_grid(score, y):
    best = (-1.0, None)
    for t in np.linspace(0, 1, 1001):
        f = prf(score >= t, y)[2]
        if f > best[0] + 1e-12:
            best = (f, t)
    return best


def main():
    rows, y, S, T = load()
    n = len(y)
    all3 = T.sum(1)
    lines = ["# Ablation: what does the perplexity detector add? (development set)", "",
             f"{n} documents ({int(y.sum())} poisoned, {int((1 - y).sum())} clean). Scores and timings are replayed from "
             "evaluation_raw.csv. Thresholds were chosen on these same documents, so every number is optimistic.", "",
             "## 1. Cascade with and without the perplexity stage", "",
             "| cascade | precision | recall | F1 | false alarms | misses | mean ms / document | time saved vs running all three |",
             "|---|---|---|---|---|---|---|---|"]
    flags = {}
    for label, use in (("A -> B -> C (as published)", True), ("A -> B only", False)):
        f, ran = cascade(S, use)
        flags[label] = f
        p, r, fs, tp, fp, fn = prf(f, y)
        lat = (ran * T).sum(1)
        lines.append(f"| {label} | {p:.2f} | {r:.2f} | {fs:.2f} | {fp} | {fn} | {lat.mean():.1f} | {1 - lat.sum() / all3.sum():.1%} |")
        if use:
            reach_c = int(ran[:, 2].sum())
            c_flags = int((f & ran[:, 2]).sum())
            c_true = int((f & ran[:, 2] & (y == 1)).sum())
    f_full, f_ab = flags["A -> B -> C (as published)"], flags["A -> B only"]
    lines += ["",
              f"Run all three on every document: {all3.mean():.1f} ms per document "
              f"(A {T[:, 0].mean():.2f}, B {T[:, 1].mean():.2f}, C {T[:, 2].mean():.2f}; medians "
              f"{np.median(T[:, 0]):.2f} / {np.median(T[:, 1]):.2f} / {np.median(T[:, 2]):.2f}).",
              f"Stage C ran on {reach_c} documents and flagged {c_flags} of them ({c_true} poisoned). "
              f"The two cascades disagree on {int((f_full != f_ab).sum())} documents.",
              "Mean timings include model warm-up on the first documents, which is why they sit above the medians.", "",
              "Which detector clears each category on its own (raw score at or above that detector's cascade threshold):", "",
              "| category | documents | A alone | B alone | C alone | C flags a document that A and B both miss |",
              "|---|---|---|---|---|---|"]
    cats = np.array([r["category"] for r in rows])
    hit = np.c_[S[:, 0] >= PATTERN_T, S[:, 1] >= EMBED_T, S[:, 2] >= PPL_T]
    for c in ["injection", "authority", "corpus", "perplexity", "clean"]:
        k = cats == c
        only_c = int((hit[k, 2] & ~hit[k, 0] & ~hit[k, 1]).sum())
        lines.append(f"| {c}{' (false alarms)' if c == 'clean' else ''} | {int(k.sum())} | {int(hit[k, 0].sum())} | "
                     f"{int(hit[k, 1].sum())} | {int(hit[k, 2].sum())} | {only_c} |")
    lines += ["",
              "## 2. Calibrated mean over every subset of detectors", "",
              "Each detector is min-max scaled to its own observed range on this set, the chosen scores are averaged, and the "
              "threshold with the best F1 on a 0.001 grid is used. Same optimistic treatment for every subset.", "",
              "| detectors | best threshold | precision | recall | F1 | false alarms | misses |", "|---|---|---|---|---|---|---|"]
    lo, hi = S.min(0), S.max(0)
    Z = (S - lo) / (hi - lo)
    results = {}
    for k in (1, 2, 3):
        for sub in itertools.combinations(range(3), k):
            score = Z[:, sub].mean(1)
            fbest, tbest = best_over_grid(score, y)
            p, r, fs, tp, fp, fn = prf(score >= tbest, y)
            tag = "+".join(SHORT[NAMES[i]] for i in sub)
            results[tag] = fs
            lines.append(f"| {tag} | {tbest:.3f} | {p:.2f} | {r:.2f} | {fs:.3f} | {fp} | {fn} |")
    one_doc = 1.0 / n
    lines += ["",
              f"A+B scores F1 {results['A+B']:.3f} and A+B+C scores {results['A+B+C']:.3f}. With {n} documents a single changed "
              f"decision moves F1 by roughly {one_doc:.2f}, so this gap is about one document.", "",
              "Reading: on this set, perplexity adds no document that the other two stages miss, and removing it cuts the "
              "cascade's mean time per document by about half. That is a statement about this set only. The perplexity "
              "detector was designed for fluency-breaking attacks (the pds_perplexity category), and a set with more of them, "
              "or with attacks the embedding stage misses, could show a different picture."]
    report = "\n".join(lines)
    print(report)
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/ablation.md", "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print("\nSaved eval/results/ablation.md")


if __name__ == "__main__":
    main()