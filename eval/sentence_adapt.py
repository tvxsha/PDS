"""
Task 4.6c / 4.7: how do we handle a NEW corpus? (train + validation only)

    python -m eval.sentence_adapt                 # seed 42, 8 repetitions for part B
    python -m eval.sentence_adapt --reps 3        # faster

The leave-one-source-out check (eval.sentence_loso) showed that a threshold learned on other
sources does not transfer: it flagged 59% of clean BIPIA and nothing at all in PoisonedRAG.
Here the model is still trained on the OTHER sources, and then we give it the two things a real
deployer would have for their own corpus:

  Part A  clean-only calibration. A sample of m clean documents from the new corpus sets the
          threshold (split-conformal rule: expected false-alarm rate <= 5% by construction).
          No poisoned documents from the new corpus are used at all.
  Part B  A plus few-shot adaptation. In addition, c clean documents and about k labelled
          attack documents from the new corpus are added to the training data.

Calibration and adaptation documents are removed from the evaluation set together with every
other document of the same group (the same email, question or twin pair), so nothing is
evaluated on a document whose near-copy was used for setup. The test split is never loaded.
"""

import argparse
import math
import os

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from detectors.sentence_detector import SentenceInstructionDetector
from eval.sentence_eval import ALPHA, find_scores_file, load_docs
from eval.sentence_loso import lr_fit_predict, source_group

METHODS = ["LR on the 3 current scores", "sentence classifier", "LR + sentence classifier"]


def fit_score(train, test):
    """Fit everything on `train`, return {method: scores for `test`}."""
    ytr = np.array([d["y"] for d in train])
    gtr = np.array([d["grp"] for d in train])
    base_tr = np.c_[[d["pat"] for d in train], [d["emb"] for d in train], [d["ppl"] for d in train]]
    base_te = np.c_[[d["pat"] for d in test], [d["emb"] for d in test], [d["ppl"] for d in test]]
    model = SentenceInstructionDetector().fit([d["text"] for d in train], ytr)
    sen_te = np.array([model.score(d["text"]) for d in test])
    sen_oof = np.zeros(len(train))
    for a, b in GroupKFold(5).split(train, ytr, gtr):
        m = SentenceInstructionDetector().fit([train[i]["text"] for i in a], ytr[a])
        sen_oof[b] = [m.score(train[i]["text"]) for i in b]
    return {"LR on the 3 current scores": lr_fit_predict(base_tr, ytr, base_te),
            "sentence classifier": sen_te,
            "LR + sentence classifier": lr_fit_predict(np.c_[base_tr, sen_oof], ytr, np.c_[base_te, sen_te])}


def conformal_threshold(cal_scores, alpha=ALPHA):
    """Split-conformal rule: flag a document only if it scores above the ceil((m+1)(1-alpha))-th
    smallest clean calibration score. Needs m >= 19 for alpha = 5%, otherwise nothing can be flagged."""
    s = np.sort(cal_scores)
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return np.inf if k > len(s) else s[k - 1]


def rates(score, y, thr):
    flagged = score > thr
    return float(flagged[y == 1].mean()), float(flagged[y == 0].mean())  # TPR, FPR


def part_a(docs, held, rng, reps):
    tr = [d for d in docs if d["g"] != held]
    ho = [d for d in docs if d["g"] == held]
    y = np.array([d["y"] for d in ho])
    grp = np.array([d["grp"] for d in ho])
    scores = fit_score(tr, ho)
    clean_idx = np.where(y == 0)[0]
    rows = []
    for m in (20, 40, 60):
        if len(clean_idx) - m < 20:
            continue
        draws = [rng.choice(clean_idx, m, replace=False) for _ in range(reps)]
        for name in METHODS:
            tp, fp = [], []
            for cal in draws:
                keep = ~np.isin(grp, grp[cal])
                t, f = rates(scores[name][keep], y[keep], conformal_threshold(scores[name][cal]))
                tp.append(t)
                fp.append(f)
            rows.append((name, m, np.mean(tp), np.mean(fp), np.percentile(fp, 10), np.percentile(fp, 90)))
    return rows, int(y.sum()), int((1 - y).sum())


def part_b(docs, held, rng, reps, ks=(0, 5, 10, 20, 40), m=20, c=20):
    tr = [d for d in docs if d["g"] != held]
    ho = [d for d in docs if d["g"] == held]
    y = np.array([d["y"] for d in ho])
    grp = np.array([d["grp"] for d in ho])
    clean_idx = np.where(y == 0)[0]
    if len(clean_idx) < m + c + 20:
        return None
    acc = {(k, name): [] for k in ks for name in METHODS}
    for _ in range(reps):
        order = rng.permutation(clean_idx)
        cal, adapt_clean = order[:m], order[m:m + c]
        used = set(grp[cal]) | set(grp[adapt_clean])
        pois_groups = [g for g in rng.permutation(sorted(set(grp[y == 1]) - used))]
        for k in ks:
            chosen, n = [], 0
            for g in pois_groups:           # whole groups, so near-duplicates stay together
                if n >= k:
                    break
                members = np.where((grp == g) & (y == 1))[0]
                chosen += list(members)
                n += len(members)
            adapt = list(adapt_clean) + chosen
            keep = np.where(~np.isin(grp, grp[list(cal) + adapt]))[0]
            test_idx = list(keep) + list(cal)
            sc = fit_score(tr + [ho[i] for i in adapt], [ho[i] for i in test_idx])
            for name in METHODS:
                s_ev, s_cal = sc[name][:len(keep)], sc[name][len(keep):]
                t, f = rates(s_ev, y[keep], conformal_threshold(s_cal))
                acc[(k, name)].append((roc_auc_score(y[keep], s_ev), t, f))
    return {key: np.mean(v, axis=0) for key, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=8, help="repetitions for part B (part A uses 40x more, it is cheap)")
    ap.add_argument("--scores", default=None, help="path to the score cache CSV (found automatically if omitted)")
    args = ap.parse_args()
    scores_path = find_scores_file(args.seed, args.scores)
    print("Using score cache:", scores_path)

    docs = load_docs(args.seed, ["train", "val"], scores_path)  # the test role is never requested
    docs = docs["train"] + docs["val"]
    for d in docs:
        d["g"] = source_group(d["src"])
    print(f"{len(docs)} documents from train + validation. Test split not loaded.")
    rng = np.random.default_rng(args.seed)

    lines = ["# New-corpus handling: clean-only calibration and few-shot adaptation (train + validation only)", "",
             "The model is trained on the OTHER source groups. For the held-out corpus we then use only what a deployer "
             "would have. FPR/TPR below are measured on held-out documents that were NOT used for calibration or adaptation.", "",
             f"Target false-alarm rate: {ALPHA:.0%}. Compare with the fixed-threshold results in sentence_loso "
             "(BIPIA: 59% of clean flagged; PoisonedRAG: nothing flagged).", "",
             "## Part A: clean-only calibration (no poisoned document from the new corpus is used)", ""]
    for held in ["bipia", "poisonedrag", "pds"]:
        print("part A:", held, flush=True)
        rows, npos, nneg = part_a(docs, held, rng, reps=args.reps * 40)
        lines += [f"### {held} ({npos} poisoned / {nneg} clean)", "",
                  "| method | clean calibration docs | TPR | FPR (mean) | FPR (10th-90th pct) |", "|---|---|---|---|---|"]
        for name, m, t, f, lo, hi in rows:
            lines.append(f"| {name} | {m} | {t:.2f} | {f:.3f} | {lo:.3f} - {hi:.3f} |")
        lines.append("")

    lines += ["## Part B: calibration plus few-shot adaptation", "",
              "Training data = other sources + 20 clean documents of the new corpus + about k labelled attack documents of the "
              "new corpus (whole groups, so the actual count can be a little above k). A further, separate 20 clean documents set the threshold.",
              f"Means over {args.reps} random draws; AUC is threshold-free.", ""]
    for held in ["bipia", "poisonedrag", "pds"]:
        print("part B:", held, flush=True)
        res = part_b(docs, held, rng, args.reps)
        if res is None:
            lines += [f"### {held}", "", "Skipped: too few clean documents to hold out calibration, adaptation and evaluation sets.", ""]
            continue
        lines += [f"### {held}", "", "| attacks added (k) | method | AUC | TPR | FPR |", "|---|---|---|---|---|"]
        for (k, name), (auc, t, f) in res.items():
            lines.append(f"| {k} | {name} | {auc:.2f} | {t:.2f} | {f:.3f} |")
        lines.append("")
    lines += ["Caveats:", "",
              "- Only three source groups exist, and the clean calibration sets are small (20 to 60 documents), so every number is noisy.",
              "- Repetitions share the same underlying documents, so they do not give independent evidence; treat differences of a few points as noise.",
              "- The test split has not been used."]
    report = "\n".join(lines)
    print("\n" + report)
    path = f"eval/results/sentence_adapt_seed{args.seed}.md"
    os.makedirs("eval/results", exist_ok=True)
    open(path, "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()