"""
Tasks 3.3 / 3.4 (first version): our detectors versus a published classifier, at a fixed
false-alarm budget (VALIDATION ONLY)

    python -m eval.baseline_eval --limit 20     # time 20 documents, print an estimate, stop (no report)
    python -m eval.baseline_eval                # seed 42, full run
    python -m eval.baseline_eval --seed 43
    python -m eval.baseline_eval --model meta-llama/Llama-Prompt-Guard-2-22M

What it does
  1. Scores every train and validation document with the published classifier (document score =
     most suspicious 512-token window) and caches the scores in
     eval/results/published_scores_<model>_w<windows>.csv, so a re-run, or another seed, only scores
     documents it has not seen. The cache is safe to share across seeds: the classifier is a fixed,
     pretrained model, so a document's score does not depend on any split.
  2. Rebuilds the methods of eval.fpr_eval (pattern bank, embedding, perplexity, LR on the three
     scores, sentence classifier, LR + sentence classifier) and adds the published classifier alone,
     LR + published, and LR + sentence + published.
  3. Same protocol as fpr_eval: thresholds from the TRAIN split at a 5% and a 1% false-alarm budget,
     applied unchanged to validation, with group-bootstrap intervals. Also reports the classifier
     "as shipped" (its own 0.5 probability cut-off, nothing tuned), who catches what (complementarity),
     and latency on the same documents.

Reads   data/pooled/pooled.jsonl, data/splits/split_seed{seed}.json, the score cache from protocol_eval.
Writes  eval/results/baseline_eval_seed{seed}_<model>.md
The test split is never loaded.
"""

import argparse
import csv
import json
import os
import platform
import time

import numpy as np

from detectors import published_classifier as pc
from eval.fpr_eval import (BUDGETS, PREVALENCES, SRC_GROUPS, best_case_recall, fit_everything,
                           group_bootstrap, lr_fit, oof_lr, precision, source_group, threshold_at_budget)
from eval.sentence_eval import find_scores_file, load_docs

PUB = "published classifier"
METHODS = ["pattern bank", "embedding", "perplexity", "LR on the 3 current scores", "sentence classifier",
           "LR + sentence classifier", PUB, "LR + published classifier", "LR + sentence + published"]
SHOW = ["LR on the 3 current scores", "LR + sentence classifier", PUB, "LR + published classifier",
        "LR + sentence + published"]


def cache_path(model_id, max_windows):
    return f"eval/results/published_scores_{model_id.replace('/', '__')}_w{max_windows}.csv"


def read_cache(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["id"]] = dict(margin=float(r["margin"]), windows=int(r["windows"]),
                                    truncated=int(r["truncated"]), ms=float(r["ms"]))
    return out


def score_missing(docs, model_id, max_windows, path, limit=None):
    """Score the documents that are not in the cache yet and append them to it. Returns (cache, ids scored now)."""
    cache = read_cache(path)
    todo = [d for d in docs if d["id"] not in cache]
    if limit is not None:
        todo = todo[:limit]
    done_now = []
    if todo:
        print(f"Scoring {len(todo)} documents with {model_id} ({len(cache)} already cached). "
              "You can stop with Ctrl+C and run again; finished documents are kept.", flush=True)
        pc.warm_up(model_id)
        new_file = not os.path.exists(path)
        os.makedirs("eval/results", exist_ok=True)
        t_start = time.perf_counter()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["id", "margin", "windows", "truncated", "ms"])
            for k, d in enumerate(todo, 1):
                t = time.perf_counter()
                margin, n_win, cut = pc.document_margin(d["text"], model_id, max_windows)
                ms = (time.perf_counter() - t) * 1000
                w.writerow([d["id"], f"{margin:.6f}", n_win, int(cut), f"{ms:.1f}"])
                cache[d["id"]] = dict(margin=margin, windows=n_win, truncated=int(cut), ms=ms)
                done_now.append(d["id"])
                if k % 25 == 0 or k == len(todo):
                    f.flush()
                    el = time.perf_counter() - t_start
                    print(f"  {k}/{len(todo)} scored, {el / 60:.1f} min elapsed, about "
                          f"{el / k * (len(todo) - k) / 60:.1f} min left", flush=True)
    return cache, done_now


def read_latency(scores_path):
    out = {}
    with open(scores_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["id"]] = (float(r["pattern_ms"]), float(r["embedding_ms"]), float(r["perplexity_ms"]))
    return out


def squash(margin):
    return np.clip(margin, -10, 10) / 10.0  # keeps the LR input on a scale like the other features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scores", default=None, help="path to the protocol_eval score cache (found automatically if omitted)")
    ap.add_argument("--model", default=pc.DEFAULT_MODEL)
    ap.add_argument("--max-windows", type=int, default=pc.MAX_WINDOWS)
    ap.add_argument("--limit", type=int, default=None, help="score at most N new documents, print timing, stop")
    args = ap.parse_args()
    seed = args.seed
    scores_path = find_scores_file(seed, args.scores)
    print("Using score cache:", scores_path)

    data = load_docs(seed, ["train", "val"], scores_path)  # the test role is never requested
    train, val = data["train"], data["val"]
    cache_file = cache_path(args.model, args.max_windows)
    cache, done_now = score_missing(train + val, args.model, args.max_windows, cache_file, args.limit)

    if args.limit is not None:
        if done_now:
            ms = np.array([cache[i]["ms"] for i in done_now])
            wins = np.array([cache[i]["windows"] for i in done_now])
            left = sum(d["id"] not in cache for d in train + val)
            print(f"\nScored {len(done_now)} documents: mean {ms.mean() / 1000:.1f} s per document, median {np.median(ms) / 1000:.1f} s, "
                  f"mean {wins.mean():.1f} windows per document.")
            print(f"{left} train/validation documents still to score for seed {seed}: about {left * ms.mean() / 60000:.0f} minutes.")
        print("No report written (--limit given).")
        return
    missing = [d["id"] for d in train + val if d["id"] not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} documents have no published-classifier score yet.")

    attack = {}
    with open("data/pooled/pooled.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            attack[r["id"]] = r["attack_type"]
    ytr = np.array([d["y"] for d in train])
    gtr = np.array([d["grp"] for d in train])
    yv = np.array([d["y"] for d in val])
    gv = np.array([d["grp"] for d in val])
    srcg = np.array([source_group(d["src"]) for d in val])
    atk = np.array([attack[d["id"]] for d in val])
    n_pos, n_neg = int(yv.sum()), int((1 - yv).sum())
    print(f"train {len(train)} docs, validation {len(val)} docs ({n_pos} poisoned / {n_neg} clean). Test split not loaded.")
    print("fitting our own methods (this part takes a few minutes)...", flush=True)
    tr, va = fit_everything(train, val)

    pub_tr = np.array([cache[d["id"]]["margin"] for d in train])
    pub_va = np.array([cache[d["id"]]["margin"] for d in val])
    tr[PUB], va[PUB] = pub_tr, pub_va
    base_tr = np.column_stack([tr["pattern bank"], tr["embedding"], tr["perplexity"]])
    base_va = np.column_stack([va["pattern bank"], va["embedding"], va["perplexity"]])
    variants = {"LR + published classifier": ([base_tr, squash(pub_tr)], [base_va, squash(pub_va)]),
                "LR + sentence + published": ([base_tr, tr["sentence classifier"], squash(pub_tr)],
                                              [base_va, va["sentence classifier"], squash(pub_va)])}
    for name, (cols_tr, cols_va) in variants.items():
        xtr, xva = np.column_stack(cols_tr), np.column_stack(cols_va)
        tr[name] = oof_lr(xtr, ytr, gtr)
        va[name] = lr_fit(xtr, ytr).predict_proba(xva)[:, 1]

    lines = ["# Our detectors versus a published classifier, at a fixed false-alarm budget (validation only)", "",
             f"Published classifier: `{args.model}`, windows of {pc.MAX_LEN} tokens (at most {args.max_windows} per document), "
             "document score = most suspicious window, measured as a logit margin so that probabilities saturating at 0 or 1 "
             "do not hide the ranking.", "",
             f"Validation: {len(val)} documents ({n_pos} poisoned, {n_neg} clean). Each method's threshold is chosen on the TRAIN split "
             "(out-of-fold scores for learned methods) so that at most the stated share of clean documents is flagged, then applied "
             "unchanged to validation. The published classifier is not fitted on our data, so for it the train split only fixes the "
             f"cut-off. Intervals (95%) resample groups. With only {n_neg} clean validation documents, one false alarm is {1 / n_neg:.1%}.", ""]

    flags_by = {}
    for alpha in BUDGETS:
        lines += [f"## False-alarm budget {alpha:.0%}", "",
                  "| method | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | false alarms per 10,000 clean |",
                  "|---|---|---|---|---|---|"]
        for m in METHODS:
            thr = threshold_at_budget(tr[m][ytr == 0], alpha)
            flags = va[m] > thr
            flags_by[(alpha, m)] = flags
            tpr, fpr = flags[yv == 1].mean(), flags[yv == 0].mean()
            (tl, th), (fl, fh) = group_bootstrap(flags, yv, gv)
            prec = [precision(tpr, fpr, p, n_neg) for p in PREVALENCES]
            fa = f"{fpr * 10000:.0f}" if fpr > 0 else f"0 seen (up to ~{3.0 / n_neg * 10000:.0f})"
            lines.append(f"| {m} | {tpr:.2f} [{tl:.2f}, {th:.2f}] | {fpr:.3f} [{fl:.3f}, {fh:.3f}] | "
                         f"{best_case_recall(va[m], yv, alpha):.2f} | {prec[0]:.1%} / {prec[1]:.2%} | {fa} |")
        lines.append("")

    shipped = pub_va > 0  # the model's own default: injection probability above 0.5, nothing tuned
    tpr, fpr = shipped[yv == 1].mean(), shipped[yv == 0].mean()
    (tl, th), (fl, fh) = group_bootstrap(shipped, yv, gv)
    lines += ["## The published classifier as shipped (its own 0.5 cut-off, nothing tuned on our data)", "",
              f"TPR {tpr:.2f} [{tl:.2f}, {th:.2f}], FPR {fpr:.3f} [{fl:.3f}, {fh:.3f}]. This is the cleanest number in the report: "
              "no threshold of ours is involved.", ""]

    alpha = 0.05
    lines += ["## By source (budget 5%): TPR / FPR", "",
              "| method | " + " | ".join(f"{g} ({int(((srcg == g) & (yv == 1)).sum())} pois / {int(((srcg == g) & (yv == 0)).sum())} clean)"
                                         for g in SRC_GROUPS) + " |", "|---|" + "---|" * len(SRC_GROUPS)]
    for m in SHOW:
        cells = []
        for g in SRC_GROUPS:
            k = srcg == g
            fl_ = flags_by[(alpha, m)][k]
            cells.append(f"{fl_[yv[k] == 1].mean():.2f} / {fl_[yv[k] == 0].mean():.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## By attack type (budget 5%): TPR among poisoned validation documents", "",
              "| attack type | n | " + " | ".join(SHOW) + " |", "|---|---|" + "---|" * len(SHOW)]
    for t in sorted({t for t in atk[yv == 1]}):
        k = (atk == t) & (yv == 1)
        lines.append(f"| {t} | {int(k.sum())} | " + " | ".join(f"{flags_by[(alpha, m)][k].mean():.2f}" for m in SHOW) + " |")

    ours, theirs = flags_by[(alpha, "LR + sentence classifier")], flags_by[(alpha, PUB)]
    pos, neg = yv == 1, yv == 0
    lines += ["", "## Who catches what (budget 5%): our best method (LR + sentence classifier) against the published classifier", "",
              f"Poisoned validation documents ({n_pos}): caught by both {int((ours & theirs & pos).sum())}, only by the published classifier "
              f"{int((theirs & ~ours & pos).sum())}, only by ours {int((ours & ~theirs & pos).sum())}, by neither {int((~ours & ~theirs & pos).sum())}.",
              f"Clean validation documents ({n_neg}): falsely flagged by both {int((ours & theirs & neg).sum())}, only by the published classifier "
              f"{int((theirs & ~ours & neg).sum())}, only by ours {int((ours & ~theirs & neg).sum())}.",
              f"Flag a document if EITHER flags it: TPR {(ours | theirs)[pos].mean():.2f}, FPR {(ours | theirs)[neg].mean():.3f} "
              "(the false-alarm rate adds up, which is why the learned combination is the fairer comparison).", ""]

    lat = read_latency(scores_path)
    ours_ms = np.array([lat[d["id"]] for d in val if d["id"] in lat])
    pub_ms = np.array([cache[d["id"]]["ms"] for d in val])
    pub_win = np.array([cache[d["id"]]["windows"] for d in val])
    cut = int(sum(cache[d["id"]]["truncated"] for d in train + val))

    def row(name, x):
        return f"| {name} | {x.mean():.1f} | {np.median(x):.1f} | {np.percentile(x, 95):.1f} |"

    lines += ["## Latency per validation document (ms)", "",
              "| detector | mean | median | 95th percentile |", "|---|---|---|---|",
              row("pattern bank", ours_ms[:, 0]), row("embedding", ours_ms[:, 1]), row("perplexity", ours_ms[:, 2]),
              row("all three of ours", ours_ms.sum(1)), row(f"published classifier ({args.model})", pub_ms), "",
              f"The published classifier used a mean of {pub_win.mean():.1f} windows per document; {cut} of {len(train) + len(val)} train and "
              f"validation documents were longer than {args.max_windows} windows and were cut off. Our detectors' times come from the "
              f"protocol_eval run on the same documents, measured at a different time, so treat the comparison as indicative. "
              f"Machine for the published classifier: {platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, CPU only.", ""]

    lines += ["Caveats:", "",
              "- Validation numbers only; the test split has not been used. Any design choice made after looking at these tables makes them optimistic again.",
              "- Contamination is not checked (task 3.7). We do not know how much of BIPIA, PoisonedRAG or similar public text the classifier saw during "
              "its own training, so a high score on a public source must not be read as generalisation.",
              "- The classifier was built for prompt-injection text. A low score on PoisonedRAG says it is the wrong tool for knowledge-corruption "
              "attacks, not that it is a bad classifier.",
              "- Documents longer than the window cap are scored on their first windows only.",
              "- 'LR + published' rows are fitted on train and use out-of-fold train scores for the threshold, exactly like the other learned rows.",
              "- Small cells (a handful of documents per attack type, one false alarm = a whole percentage point) are noisy. Read the intervals."]
    report = "\n".join(lines)
    print("\n" + report)
    path = f"eval/results/baseline_eval_seed{seed}_{args.model.replace('/', '__')}.md"
    os.makedirs("eval/results", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()