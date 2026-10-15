"""
deepset/prompt-injections as an external source (roadmap task 2.10). TRAIN and VALIDATION only.

    python -m data.evalsets.build_deepset            # once: writes data/evalsets/deepset.jsonl
    python -m eval.deepset_eval --seed 42            # also 43 and 44
    python -m eval.deepset_eval --seed 42 --skip-published      # without the DeBERTa classifier (no 700 MB download)

deepset items are short chat prompts, many of them German; they are NOT retrieved documents. This
script reports them as an external check of the instruction signals, and nothing more. Nothing is
translated or filtered silently: every table is given for all rows and for an "English-looking"
subset, and the share of rows outside that subset is printed.

(a) TRANSFER IN. The models fitted on the TRAIN role of the pool (eval.apply_models.fit_models),
    with their TRAIN-derived 5% and 1% thresholds, are applied unchanged to deepset. Reported:
    TPR on the injection prompts and FPR on the legitimate prompts, for every method.
(b) TRANSFER OUT. A sentence classifier is fitted on deepset only. Its 5% / 1% thresholds come from
    5-fold out-of-fold scores of the legitimate deepset prompts (so it never sees the documents it
    is scored on). It is then applied to the seed's VALIDATION documents: AUC, TPR and FPR.
(c) PUBLISHED CLASSIFIER. protectai/deberta-v3-base-prompt-injection-v2 scores deepset; its cut-off
    comes from the seed's TRAIN clean margins (the cache that eval.baseline_eval wrote), exactly as
    in eval.baseline_eval, plus the classifier "as shipped" (margin > 0, nothing tuned).

Reads   data/evalsets/deepset.jsonl, data/pooled/pooled.jsonl, data/splits/split_seed{seed}.json,
        the pooled score cache (found by its header), the published-classifier train cache.
Writes  eval/results/deepset_eval_seed{seed}.md and .csv
        (columns direction,method,budget,tpr,fpr,n_pos,n_neg,subset), plus the caches
        external_deepset_seed{seed}_<hash>.csv and published_scores_deepset_<model>_w<windows>.csv.
The held-out role is never requested.
"""

import argparse
import csv
import math
import os
import re

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from detectors.sentence_detector import SentenceInstructionDetector
from eval.apply_models import fit_models
from eval.fpr_eval import BUDGETS, SRC_GROUPS, source_group, threshold_at_budget
from eval.sentence_eval import find_scores_file, load_docs
from eval.signal_cache import external_path, load_evalset, score_external

SET = "deepset"
CSV_COLUMNS = ["direction", "method", "budget", "tpr", "fpr", "n_pos", "n_neg", "subset"]
ENGLISH_WORDS = ("the", "and", "is", "to", "of", "you", "a", "in")
ENGLISH_MIN = 3
SENTENCE_ON_DEEPSET = "sentence classifier (fit on deepset)"
PUB = "published classifier"
PUB_SHIPPED = "published classifier (as shipped, margin > 0)"


def english_looking(text):
    """At least ENGLISH_MIN distinct words of ENGLISH_WORDS appear in the text as whole words."""
    words = set(re.findall(r"[A-Za-z]+", text.lower()))
    return sum(w in words for w in ENGLISH_WORDS) >= ENGLISH_MIN


def wilson(k, n, z=1.96):
    """95% Wilson interval for k successes in n trials; (nan, nan) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cell(flags, y):
    """Counts behind one TPR / FPR pair: flags (bool array) against labels y (1 = injection)."""
    flags, y = np.asarray(flags, dtype=bool), np.asarray(y)
    return dict(tp=int(flags[y == 1].sum()), n_pos=int((y == 1).sum()),
                fp=int(flags[y == 0].sum()), n_neg=int((y == 0).sum()))


def rate(k, n):
    return k / n if n else float("nan")


def record(direction, method, budget, subset, flags, y):
    return dict(direction=direction, method=method, budget=budget, subset=subset, **cell(flags, y))


def csv_row(rec):
    return [rec["direction"], rec["method"], rec["budget"], f"{rate(rec['tp'], rec['n_pos']):.4f}",
            f"{rate(rec['fp'], rec['n_neg']):.4f}", rec["n_pos"], rec["n_neg"], rec["subset"]]


def fmt_rate(k, n):
    if n == 0:
        return "n/a (n=0)"
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}] (n={n})"


def subsets_of(rows):
    return {"all": np.ones(len(rows), dtype=bool),
            "english_like": np.array([english_looking(r.text) for r in rows], dtype=bool)}


# --------------------------------------------------------------------------- (a) transfer in

def transfer_in(rows, fm, ext):
    """Pool-trained models and TRAIN thresholds applied to deepset. Returns (records, scores per method)."""
    docs = [dict(id=r.id, text=r.text, pat=ext["pattern"][r.id], emb=ext["embedding"][r.id],
                 ppl=ext["perplexity"][r.id]) for r in rows]
    y = np.array([r.label for r in rows])
    scores = fm.score(docs)
    recs = []
    for budget in BUDGETS:
        for m in fm.methods:
            flags = scores[m] > fm.thresholds[m][budget]
            for name, mask in subsets_of(rows).items():
                recs.append(record("transfer_in", m, budget, name, flags[mask], y[mask]))
    return recs, scores


# --------------------------------------------------------------------------- (c) published classifier

def published_margins(seed, rows, model_id, max_windows):
    """(margins on deepset, {budget: cut-off from TRAIN clean margins}, cache path of the deepset margins)."""
    from eval import baseline_eval as be

    train = load_docs(seed, ["train"], find_scores_file(seed))["train"]
    train_cache_path = be.cache_path(model_id, max_windows)
    train_cache = be.read_cache(train_cache_path)
    missing = [d["id"] for d in train if d["id"] not in train_cache]
    if missing:
        raise SystemExit(f"{len(missing)} train documents have no published-classifier score in {train_cache_path}. "
                         "Run  python -m eval.baseline_eval  first (or use --skip-published).")
    print("Using published-classifier train cache:", train_cache_path)
    clean = np.array([train_cache[d["id"]]["margin"] for d in train if d["y"] == 0])
    cutoffs = {a: threshold_at_budget(clean, a) for a in BUDGETS}
    path = f"eval/results/published_scores_{SET}_{model_id.replace('/', '__')}_w{max_windows}.csv"
    cache, _ = be.score_missing([dict(id=r.id, text=r.text) for r in rows], model_id, max_windows, path)
    print("Using published-classifier deepset cache:", path)
    return np.array([cache[r.id]["margin"] for r in rows]), cutoffs, path


def published_records(rows, margins, cutoffs):
    y = np.array([r.label for r in rows])
    recs = []
    for name, mask in subsets_of(rows).items():
        for budget in BUDGETS:
            recs.append(record("transfer_in", PUB, budget, name, (margins > cutoffs[budget])[mask], y[mask]))
        recs.append(record("transfer_in", PUB_SHIPPED, "shipped", name, (margins > 0)[mask], y[mask]))
    return recs


# --------------------------------------------------------------------------- (b) transfer out

def transfer_out(seed, rows, val_docs):
    """Sentence classifier fitted on deepset only, applied to the validation documents of the pool.
    Returns (records, info) where info holds the AUC values."""
    texts = [r.text for r in rows]
    y = np.array([r.label for r in rows])
    oof = np.zeros(len(rows))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, y):
        m = SentenceInstructionDetector().fit([texts[i] for i in a], y[a])
        oof[b] = [m.score(texts[i]) for i in b]
    cutoffs = {a: threshold_at_budget(oof[y == 0], a) for a in BUDGETS}
    final = SentenceInstructionDetector().fit(texts, y)

    yv = np.array([d["y"] for d in val_docs])
    sv = np.array([final.score(d["text"]) for d in val_docs])
    srcg = np.array([source_group(d["src"]) for d in val_docs])
    masks = {"val_all": np.ones(len(val_docs), dtype=bool)}
    masks.update({f"val_{g}": srcg == g for g in SRC_GROUPS})

    recs, auc = [], {}
    for name, mask in masks.items():
        auc[name] = roc_auc_score(yv[mask], sv[mask]) if len(set(yv[mask])) == 2 else float("nan")
        for budget in BUDGETS:
            recs.append(record("transfer_out", SENTENCE_ON_DEEPSET, budget, name, (sv > cutoffs[budget])[mask], yv[mask]))
    info = dict(auc=auc, oof_auc=roc_auc_score(y, oof), cutoffs=cutoffs)
    return recs, info


# --------------------------------------------------------------------------- report

def _table(recs, direction, budget, subsets, methods):
    lines = ["| method | " + " | ".join(f"{s}: TPR (95% Wilson) / FPR" for s in subsets) + " |",
             "|---|" + "---|" * len(subsets)]
    for m in methods:
        cells = []
        for s in subsets:
            r = next((x for x in recs if x["direction"] == direction and x["method"] == m
                      and x["budget"] == budget and x["subset"] == s), None)
            cells.append("-" if r is None else f"{fmt_rate(r['tp'], r['n_pos'])} / {fmt_rate(r['fp'], r['n_neg'])}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines


def render_report(seed, rows, recs_in, recs_out, info, published_used):
    n_pos = sum(r.label for r in rows)
    n_eng = int(subsets_of(rows)["english_like"].sum())
    lines = [f"# deepset/prompt-injections as an external source, seed {seed} (train and validation only)", "",
             f"deepset: {len(rows)} prompts ({n_pos} injection, {len(rows) - n_pos} legitimate). "
             f"Not English-looking: {len(rows) - n_eng} of {len(rows)} ({(len(rows) - n_eng) / len(rows):.1%}); the rule is "
             f"'at least {ENGLISH_MIN} of {{{', '.join(ENGLISH_WORDS)}}} appear as whole words'. Nothing was translated or removed; "
             "every table below gives all rows and the English-looking subset side by side. These are short chat prompts, "
             "not retrieved documents: read the numbers as an external check of the instruction signals, not as RAG results.", ""]
    methods_in = [m for m in dict.fromkeys(r["method"] for r in recs_in if r["method"] not in (PUB, PUB_SHIPPED))]
    for budget in BUDGETS:
        lines += [f"## (a) Transfer in, false-alarm budget {budget:.0%}", "",
                  "Models fitted on the TRAIN role of the pool; the threshold of each method comes from TRAIN clean scores "
                  "(out-of-fold for learned methods) and is applied unchanged to deepset.", ""]
        lines += _table(recs_in, "transfer_in", budget, ["all", "english_like"],
                        methods_in + ([PUB] if published_used else []))
        lines.append("")
    if published_used:
        lines += ["## (c) The published classifier as shipped (its own cut-off, nothing tuned)", ""]
        lines += _table(recs_in, "transfer_in", "shipped", ["all", "english_like"], [PUB_SHIPPED])
        lines.append("")
    lines += ["## (b) Transfer out: sentence classifier fitted on deepset only, applied to the pool's validation documents", "",
              f"5% / 1% cut-offs from 5-fold out-of-fold scores of the legitimate deepset prompts "
              f"({info['cutoffs'][0.05]:.3f} / {info['cutoffs'][0.01]:.3f}). In-corpus reference: out-of-fold AUC on deepset "
              f"{info['oof_auc']:.3f}.", "",
              "| validation subset | AUC |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in info["auc"].items()]
    lines.append("")
    for budget in BUDGETS:
        lines += [f"### Budget {budget:.0%}", ""]
        lines += _table(recs_out, "transfer_out", budget, list(info["auc"]), [SENTENCE_ON_DEEPSET])
        lines.append("")
    lines += ["Caveats:", "",
              "- Validation numbers only; the test split has not been used. Thresholds in (a) and (c) come from the pool's TRAIN role, in (b) from deepset's own out-of-fold scores.",
              "- deepset prompts are short and partly German; an instruction signal tuned on English documents can fail here for language reasons alone. The English-looking subset is a rough rule, not a language detector.",
              "- Each deepset prompt is its own group, so the Wilson intervals treat prompts as independent; they are rough, and the cells are small (n is printed in every cell).",
              "- The numbers describe how the signals behave on a different kind of text. They say nothing about retrieved documents, and none of it was tuned after looking at deepset.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-published", action="store_true", help="do not run the DeBERTa classifier")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()
    seed = args.seed

    scores_path = find_scores_file(seed)
    print("Using score cache:", scores_path)
    rows = load_evalset(SET)
    print(f"deepset: {len(rows)} prompts. The held-out role is not requested.")
    ext = score_external(SET, seed)
    print("Using external cache:", external_path(SET, seed))

    fm = fit_models(seed, scores_path)
    recs_in, _ = transfer_in(rows, fm, ext)
    published_used = not args.skip_published
    if published_used:
        from detectors import published_classifier as pc

        model = args.model or pc.DEFAULT_MODEL
        windows = args.max_windows or pc.MAX_WINDOWS
        margins, cutoffs, _ = published_margins(seed, rows, model, windows)
        recs_in += published_records(rows, margins, cutoffs)

    val_docs = load_docs(seed, ["val"], scores_path)["val"]
    recs_out, info = transfer_out(seed, rows, val_docs)

    report = render_report(seed, rows, recs_in, recs_out, info, published_used)
    print("\n" + report)
    os.makedirs("eval/results", exist_ok=True)
    base = f"eval/results/deepset_eval_seed{seed}"
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLUMNS)
        for rec in recs_in + recs_out:
            w.writerow(csv_row(rec))
    print(f"\nSaved {base}.md and {base}.csv")


if __name__ == "__main__":
    main()
