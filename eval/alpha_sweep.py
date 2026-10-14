"""
Task 5.3: false-alarm budget sweep (VALIDATION ONLY)

    python -m eval.alpha_sweep --seed 42
    python -m eval.alpha_sweep --all-seeds          # seeds 42, 43, 44

Two budgets (5%, 1%) are two points of a curve. This script traces the curve: for each method and each false-alarm
budget in 10%, 5%, 2%, 1%, 0.5%, 0.1% the threshold is the quantile of the out-of-fold TRAIN clean scores (never
anything from validation), applied unchanged to the validation documents. It reports the realised TPR and FPR with
group-bootstrap intervals and how many TRAIN clean documents lie above each threshold.

Methods: pattern bank, sentence classifier, LR on the 3 current scores, LR + sentence classifier, the fusion candidate of
task 4.10 (eval/results/fusion_v2_candidate.json) and the published classifier. They are built by the SAME code the final
test uses (eval/final_methods.py), so these curves and the held-out numbers describe the same models.

Validation has only about 170 clean documents, so one false alarm is about 0.6%. Budgets of 0.5% and 0.1% cannot be
verified: a cell with no false alarm only says the rate is below about 3/n (rule of three).

Reads   data/pooled/pooled.jsonl, data/splits/split_seed{seed}.json, the score cache from protocol_eval, the cached
        published-classifier scores, eval/results/fusion_v2_candidate.json
Writes  eval/results/alpha_sweep_seed{seed}.csv  (method,budget,thr,tpr,tpr_lo,tpr_hi,fpr,fpr_lo,fpr_hi,n_pos,n_neg)
        eval/results/alpha_sweep_seed{seed}.md
The held-out role is never requested. No torch import unless a published-classifier score is missing from the cache.
"""

import argparse
import csv
import os

import numpy as np

SEEDS = (42, 43, 44)
BUDGETS = (0.10, 0.05, 0.02, 0.01, 0.005, 0.001)
SWEEP_METHODS = ["pattern bank", "sentence classifier", "LR on the 3 current scores", "LR + sentence classifier",
                 "fusion candidate", "published classifier"]
CSV_COLUMNS = ["method", "budget", "thr", "tpr", "tpr_lo", "tpr_hi", "fpr", "fpr_lo", "fpr_hi", "n_pos", "n_neg"]
REPS = 1000


# --------------------------------------------------------------------------
# numbers (no models, no data files: these are what the tests exercise)
# --------------------------------------------------------------------------

def verifiable(alpha, n_neg):
    """A budget can be checked on validation only if it allows at least one false alarm among the clean documents."""
    return alpha * n_neg >= 1.0


def resolution_text(n_neg):
    return f"FPR resolution {100.0 / n_neg:.2f}%"


def pct(x):
    return f"{x * 100:g}%"


def sweep_one(train_clean, ev_scores, y, grp, alpha, reps=REPS):
    """One cell: threshold from the TRAIN clean scores only, applied to the evaluation scores."""
    from eval.fpr_eval import group_bootstrap, threshold_at_budget
    train_clean = np.asarray(train_clean, dtype=float)
    ev_scores, y = np.asarray(ev_scores, dtype=float), np.asarray(y)
    thr = threshold_at_budget(train_clean, alpha)
    flags = ev_scores > thr
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tpr, fpr = float(flags[y == 1].mean()), float(flags[y == 0].mean())
    (tl, th), (fl, fh) = group_bootstrap(flags, y, np.asarray(grp), reps=reps)
    if fpr == 0:
        fh = max(float(fh), 3.0 / n_neg)        # rule of three: no false alarm in n documents means "up to about 3/n"
    return {"thr": float(thr), "tpr": tpr, "tpr_lo": float(tl), "tpr_hi": float(th), "fpr": fpr,
            "fpr_lo": float(fl), "fpr_hi": float(fh), "n_pos": n_pos, "n_neg": n_neg,
            "train_above": int((train_clean > thr).sum()), "n_train_clean": int(len(train_clean))}


def write_csv(path, rows_by, budgets=BUDGETS):
    """rows_by[method][alpha] -> dict from sweep_one. One line per method and budget, columns as in the roadmap."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLUMNS)
        for method, cells in rows_by.items():
            for alpha in budgets:
                c = cells[alpha]
                w.writerow([method, repr(float(alpha)), repr(c["thr"]), repr(c["tpr"]), repr(c["tpr_lo"]), repr(c["tpr_hi"]),
                            repr(c["fpr"]), repr(c["fpr_lo"]), repr(c["fpr_hi"]), c["n_pos"], c["n_neg"]])


def sentence_for(method, cells, budgets=BUDGETS):
    """One sentence generated from the numbers: how the recall falls as the budget shrinks."""
    return f"{method}: recall " + ", ".join(f"{cells[a]['tpr']:.2f} at {pct(a)}" for a in budgets) + "."


def unverified_sentence(n_neg, budgets=BUDGETS):
    ok = [a for a in budgets if verifiable(a, n_neg)]
    bad = [a for a in budgets if not verifiable(a, n_neg)]
    if not bad:
        return f"Every budget above can be checked on validation ({resolution_text(n_neg)})."
    if not ok:
        return f"None of these budgets can be checked on validation ({resolution_text(n_neg)})."
    return (f"Only the budgets from {pct(max(ok))} down to {pct(min(ok))} can be checked on validation; the recall at "
            f"{', '.join(pct(a) for a in bad)} comes from a threshold whose false-alarm rate is not verified ({resolution_text(n_neg)}).")


# --------------------------------------------------------------------------
# one seed
# --------------------------------------------------------------------------

def analyse(seed):
    from eval import final_methods as registry
    from eval import fusion_v2
    from eval.protocol import load_split, split_hash
    from eval.sentence_eval import find_scores_file, load_docs

    cand = fusion_v2.load_candidate()                                    # stops with a clear message if 4.10 was not run
    scores_path = find_scores_file(seed)
    split = load_split(seed)
    data = load_docs(seed, ["train", "val"], scores_path)                # the held-out role is never requested
    train, val = data["train"], data["val"]
    for role, docs in (("train", train), ("val", val)):
        if len(docs) != len(split["roles"][role]):
            raise SystemExit(f"{role}: the score cache has {len(docs)} of the {len(split['roles'][role])} documents of this split.")
    ctx = registry.Context(seed=seed, eval_role="val", final=False, train=train, eval_docs=val, scores_path=scores_path,
                           split_hash=split_hash(split), config={})
    ytr, yva = ctx.y_train, ctx.y_eval
    gva = np.array([d["grp"] for d in val])
    rows_by = {}
    for name in SWEEP_METHODS:
        tr, es = registry.METHODS[name](ctx)
        tr, es = np.asarray(tr, dtype=float), np.asarray(es, dtype=float)
        if not (np.isfinite(tr).all() and np.isfinite(es).all()):
            raise SystemExit(f"method {name!r} returned a score that is not a finite number")
        rows_by[name] = {a: sweep_one(tr[ytr == 0], es, yva, gva, a) for a in BUDGETS}
        print(f"[seed {seed}] {name}: done", flush=True)
    return {"seed": seed, "rows_by": rows_by, "cand": cand, "n_val": len(val), "n_train": len(train),
            "n_pos": int((yva == 1).sum()), "n_neg": int((yva == 0).sum())}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def render(res):
    rows_by, n_neg, n_pos = res["rows_by"], res["n_neg"], res["n_pos"]
    cand = res["cand"]
    mark = lambda a: pct(a) + ("" if verifiable(a, n_neg) else "†")
    lines = [f"# False-alarm budget sweep, seed {res['seed']} (task 5.3, validation only)", "",
             f"Validation: {res['n_val']} documents ({n_pos} poisoned, {n_neg} clean). For every method and budget the threshold is the "
             f"quantile of the out-of-fold TRAIN clean scores ({res['n_train']} train documents), applied unchanged to validation; "
             "nothing here is tuned on validation. Intervals (95%) resample groups, not single documents.", "",
             f"**{resolution_text(n_neg)}.** One false alarm among the {n_neg} clean validation documents is {100.0 / n_neg:.2f}%. "
             f"Budgets marked † ({', '.join(pct(a) for a in BUDGETS if not verifiable(a, n_neg))}) allow less than one false alarm on "
             f"validation, so the realised FPR cannot confirm them: where no false alarm was seen the upper end of the FPR interval is the "
             f"rule-of-three bound 3/{n_neg} = {3.0 / n_neg:.1%}, not zero.", "",
             f"The fusion candidate is feature set {cand.get('feature_set', '?')} of task 4.10 ({', '.join(cand['features'])})"
             + ("" if cand.get("complete", True) else "; **the candidate is provisional** (" + ", ".join(cand.get('missing_inputs', []))
                + " not merged yet), so rerun this script after task 4.10 is rerun") + ".", "",
             "## Recall (TPR) as the allowed false-alarm rate shrinks", "",
             "| method | " + " | ".join(mark(a) for a in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for m, cells in rows_by.items():
        lines.append(f"| {m} | " + " | ".join(f"{cells[a]['tpr']:.2f}" for a in BUDGETS) + " |")
    lines += ["", "## Realised false-alarm rate (FPR) on validation", "",
              "| method | " + " | ".join(f"target {mark(a)}" for a in BUDGETS) + " |", "|---|" + "---|" * len(BUDGETS)]
    for m, cells in rows_by.items():
        lines.append(f"| {m} | " + " | ".join(f"{cells[a]['fpr']:.3f}" for a in BUDGETS) + " |")
    lines += ["", "A realised FPR above its target means the train threshold did not carry over to validation; such a method gets extra "
              "recall for free, so compare methods at similar realised FPR.", ""]
    for a in BUDGETS:
        lines += [f"## Budget {mark(a)}" + ("" if verifiable(a, n_neg) else " (not verifiable on validation)"), "",
                  "| method | threshold | train clean above it (expected) | TPR (95% CI) | FPR (95% CI) | n poisoned / clean |",
                  "|---|---|---|---|---|---|"]
        for m, cells in rows_by.items():
            c = cells[a]
            fpr_note = f"{c['fpr']:.3f} [{c['fpr_lo']:.3f}, {c['fpr_hi']:.3f}]" + (" (rule of 3)" if c["fpr"] == 0 else "")
            lines.append(f"| {m} | {c['thr']:.4f} | {c['train_above']} of {c['n_train_clean']} ({a * c['n_train_clean']:.1f}) | "
                         f"{c['tpr']:.2f} [{c['tpr_lo']:.2f}, {c['tpr_hi']:.2f}] | {fpr_note} | {c['n_pos']} / {c['n_neg']} |")
        lines.append("")
    lines += ["## In words (generated from the tables above)", ""]
    lines += [f"- {sentence_for(m, cells)}" for m, cells in rows_by.items()]
    lines += ["", unverified_sentence(n_neg)]
    lines += ["", "Caveats:", "",
              "- The pattern bank gives clean documents a score of (almost) zero, so its threshold does not move with the budget and its row is flat.",
              "- At the smallest budgets the TRAIN threshold itself rests on one or two clean documents (see the 'train clean above it' column), "
              "so it is noisy from seed to seed.",
              "- The fusion candidate is a logistic regression; when it is feature set F1 it equals 'LR + sentence classifier' and the two rows are identical.",
              "- The published classifier is not fitted on our data; the train split only fixes its cut-off.",
              "- Validation numbers only; the held-out split was not used. A choice made after reading these tables makes them optimistic."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-seeds", action="store_true", help="seeds 42, 43 and 44")
    args = ap.parse_args()
    os.makedirs("eval/results", exist_ok=True)
    for seed in (SEEDS if args.all_seeds else (args.seed,)):
        res = analyse(seed)
        csv_path, md_path = f"eval/results/alpha_sweep_seed{seed}.csv", f"eval/results/alpha_sweep_seed{seed}.md"
        write_csv(csv_path, res["rows_by"])
        text = render(res)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("\n" + text)
        print(f"Saved {csv_path}\nSaved {md_path}")


if __name__ == "__main__":
    main()