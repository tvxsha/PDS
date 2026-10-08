"""
Tasks 1.11 / 1.12 / 1.5: evaluation at a fixed false-alarm budget (VALIDATION ONLY)

    python -m eval.fpr_eval              # seed 42
    python -m eval.fpr_eval --seed 43

Why: on the pooled data every method's best-F1 threshold collapses to "flag everything"
(F1 0.77 with precision 0.62), so F1 says nothing. Instead each method gets a threshold chosen
on the TRAIN split at a stated false-alarm budget (5% and 1% of clean documents), and that
threshold is applied unchanged to the validation split. The numbers are then what a
deployment would see: the realised recall and the realised false-alarm rate.

For learned methods the train scores are out-of-fold (GroupKFold by group), so a model is never
scored on documents it was fitted on. Intervals resample GROUPS (the same email, question or twin
pair), not single documents, because documents in a group are related.

Reads   data/pooled/pooled.jsonl, data/splits/split_seed{seed}.json, the score cache from
        protocol_eval (found by its header). Writes eval/results/fpr_eval_seed{seed}.md.
The test split is never loaded.
"""

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

from detectors.sentence_detector import SentenceInstructionDetector
from eval.sentence_eval import find_scores_file, load_docs

BUDGETS = (0.05, 0.01)
PREVALENCES = (0.01, 0.001)
REPS = 1000
SRC_GROUPS = ["bipia", "poisonedrag", "pds"]


def source_group(src):
    return "pds" if src.startswith("pds") else src


def lr_fit(Xtr, ytr):
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)


def oof_lr(X, y, groups):
    out = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        out[b] = lr_fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return out


def threshold_at_budget(clean_scores, alpha):
    """Smallest cut-off such that at most alpha of the clean scores are strictly above it."""
    return float(np.quantile(clean_scores, 1 - alpha))


def best_case_recall(score, y, alpha):
    """Recall at FPR <= alpha with the threshold picked ON the same data (optimistic reference)."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= alpha].max())


def precision(tpr, fpr, prevalence, n_clean):
    if fpr == 0:
        fpr = 3.0 / n_clean  # rule of three: zero false alarms seen in n documents means "up to about 3/n", not zero
    denom = tpr * prevalence + fpr * (1 - prevalence)
    return tpr * prevalence / denom if denom > 0 else float("nan")


def fit_everything(train, val):
    """Return (train_scores, val_scores): dicts method -> array, train scores out-of-fold."""
    ytr = np.array([d["y"] for d in train])
    gtr = np.array([d["grp"] for d in train])
    B = lambda ds: np.c_[[d["pat"] for d in ds], [d["emb"] for d in ds], [d["ppl"] for d in ds]]
    btr, bv = B(train), B(val)

    model = SentenceInstructionDetector().fit([d["text"] for d in train], ytr)
    sen_v = np.array([model.score(d["text"]) for d in val])
    sen_oof = np.zeros(len(train))
    for a, b in GroupKFold(5).split(train, ytr, gtr):
        m = SentenceInstructionDetector().fit([train[i]["text"] for i in a], ytr[a])
        sen_oof[b] = [m.score(train[i]["text"]) for i in b]

    xs_tr, xs_v = np.c_[btr, sen_oof], np.c_[bv, sen_v]
    tr = {"pattern bank": btr[:, 0], "embedding": btr[:, 1], "perplexity": btr[:, 2],
          "naive average of the 3": btr.mean(1), "sentence classifier": sen_oof,
          "LR on the 3 current scores": oof_lr(btr, ytr, gtr),
          "LR + sentence classifier": oof_lr(xs_tr, ytr, gtr)}
    va = {"pattern bank": bv[:, 0], "embedding": bv[:, 1], "perplexity": bv[:, 2],
          "naive average of the 3": bv.mean(1), "sentence classifier": sen_v,
          "LR on the 3 current scores": lr_fit(btr, ytr).predict_proba(bv)[:, 1],
          "LR + sentence classifier": lr_fit(xs_tr, ytr).predict_proba(xs_v)[:, 1]}
    return tr, va


def group_bootstrap(flags, y, grp, reps=REPS, seed=0):
    """flags: bool array. Resample groups; return (tpr_ci, fpr_ci)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(grp)
    members = {g: np.where(grp == g)[0] for g in uniq}
    tprs, fprs = [], []
    for _ in range(reps):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        yy, ff = y[idx], flags[idx]
        if yy.sum() and (1 - yy).sum():
            tprs.append(ff[yy == 1].mean())
            fprs.append(ff[yy == 0].mean())
    return np.percentile(tprs, [2.5, 97.5]), np.percentile(fprs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scores", default=None, help="path to the score cache CSV (found automatically if omitted)")
    args = ap.parse_args()
    seed = args.seed
    scores_path = find_scores_file(seed, args.scores)
    print("Using score cache:", scores_path)

    data = load_docs(seed, ["train", "val"], scores_path)  # the test role is never requested
    train, val = data["train"], data["val"]
    attack = {}
    with open("data/pooled/pooled.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            attack[r["id"]] = r["attack_type"]
    yv = np.array([d["y"] for d in val])
    gv = np.array([d["grp"] for d in val])
    srcg = np.array([source_group(d["src"]) for d in val])
    atk = np.array([attack[d["id"]] for d in val])
    n_pos, n_neg = int(yv.sum()), int((1 - yv).sum())
    print(f"train {len(train)} docs, validation {len(val)} docs ({n_pos} poisoned / {n_neg} clean). Test split not loaded.")
    print("fitting...", flush=True)
    tr_scores, va_scores = fit_everything(train, val)
    ytr = np.array([d["y"] for d in train])
    methods = list(va_scores)

    lines = ["# Evaluation at a fixed false-alarm budget (validation only)", "",
             f"Validation: {len(val)} documents ({n_pos} poisoned, {n_neg} clean). Each method's threshold is chosen on the TRAIN split "
             "(out-of-fold scores) so that at most the stated share of clean documents is flagged, then applied unchanged to validation. "
             "TPR is the share of poisoned documents flagged; FPR the share of clean documents flagged. Intervals (95%) resample groups. "
             f"With only {n_neg} clean validation documents, one false alarm is {1 / n_neg:.1%}, so FPR is coarse.", "",
             f"Flag-everything reference: TPR 1.00, FPR 1.00, precision {n_pos / len(val):.2f}, F1 {2 * n_pos / (n_pos + len(val)):.2f}. "
             "That is the bar the F1 numbers in earlier reports have to beat.", ""]

    flags_by = {}
    for alpha in BUDGETS:
        lines += [f"## False-alarm budget {alpha:.0%}", "",
                  "| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |",
                  "|---|---|---|---|---|---|"]
        for m in methods:
            thr = threshold_at_budget(tr_scores[m][ytr == 0], alpha)
            flags = va_scores[m] > thr
            flags_by[(alpha, m)] = flags
            tpr, fpr = flags[yv == 1].mean(), flags[yv == 0].mean()
            (tl, th), (fl, fh) = group_bootstrap(flags, yv, gv)
            prec = [precision(tpr, fpr, p, n_neg) for p in PREVALENCES]
            fa = f"{fpr * 10000:.0f}" if fpr > 0 else f"0 seen (up to ~{3.0 / n_neg * 10000:.0f})"
            lines.append(f"| {m} | {tpr:.2f} [{tl:.2f}, {th:.2f}] | {fpr:.3f} [{fl:.3f}, {fh:.3f}] | "
                         f"{best_case_recall(va_scores[m], yv, alpha):.2f} | {prec[0]:.1%} / {prec[1]:.2%} | {fa} |")
        lines.append("")
    lines += ["'best-case TPR' picks the threshold on validation itself (the optimistic number used in earlier reports). "
              "The first TPR column is the honest one: the threshold was fixed before validation was looked at. "
              "Compare methods at similar realised FPR: a method whose FPR overshoots the budget (for example 9% against 5%) "
              "gets extra recall for free.",
              "Precision uses the observed FPR; where no false alarm was seen it uses the rule-of-three upper bound "
              f"(3/{n_neg} = {3.0 / n_neg:.1%}) instead of zero, so zero-false-alarm methods are not flattered.", ""]

    alpha = 0.05
    lines += ["## By source (budget 5%): TPR / FPR", "",
              "| method | " + " | ".join(f"{g} ({int(((srcg == g) & (yv == 1)).sum())} pois / {int(((srcg == g) & (yv == 0)).sum())} clean)" for g in SRC_GROUPS) + " |",
              "|---|" + "---|" * len(SRC_GROUPS)]
    for m in methods:
        cells = []
        for g in SRC_GROUPS:
            k = srcg == g
            fl = flags_by[(alpha, m)][k]
            cells.append(f"{fl[yv[k] == 1].mean():.2f} / {fl[yv[k] == 0].mean():.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    types = sorted({t for t in atk[yv == 1]})
    show = ["pattern bank", "LR on the 3 current scores", "sentence classifier", "LR + sentence classifier"]
    lines += ["", "## By attack type (budget 5%): TPR among poisoned validation documents", "",
              "| attack type | n | " + " | ".join(show) + " |", "|---|---|" + "---|" * len(show)]
    for t in types:
        k = (atk == t) & (yv == 1)
        lines.append(f"| {t} | {int(k.sum())} | " + " | ".join(f"{flags_by[(alpha, m)][k].mean():.2f}" for m in show) + " |")

    lines += ["", "Caveats:", "",
              "- Validation numbers only; the test split has not been used. Thresholds come from train, so this is a fair preview, "
              "but any design choice made after looking at these tables makes them optimistic again.",
              "- Small cells (one attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. Read the intervals.",
              "- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so realised FPR on validation can sit a little off target.",
              "- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.",
              "- The calibrated average is not included here (its bounds come from the development set); it is in the protocol_eval table."]
    report = "\n".join(lines)
    print("\n" + report)
    path = f"eval/results/fpr_eval_seed{seed}.md"
    os.makedirs("eval/results", exist_ok=True)
    open(path, "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()