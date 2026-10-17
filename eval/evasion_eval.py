"""
Evasion rates of the obfuscations (roadmap task 8.1): the "before" measurement for the normalise-then-match patch (8.4).

    python -m eval.evasion_eval --seed 42            # also 43 and 44; writes the report and the CSV
    python -m eval.evasion_eval --seed 42 --skip-published       # without the DeBERTa classifier (no 700 MB download)
    python -m eval.evasion_eval --figure             # F7 from the CSVs of the seeds that have been run

The models fitted on the TRAIN role of the pool (eval.apply_models.fit_models) and their TRAIN-derived 5% and 1%
thresholds are applied unchanged to data/evalsets/obfuscated_A.jsonl (task 2.5). Every document of that set carries the
same injected sentence, so a document that is not flagged is an evasion. For every transformation and method:
recall = share of documents flagged, evasion rate = 1 - recall, with 95% Wilson intervals; n is printed in every cell
(20 documents per transformation). The row "T0_none" (untransformed controls) must show high recall for the pattern
bank, otherwise the experiment says nothing: the script warns when it does not.

Methods: pattern bank, sentence classifier, LR + sentence classifier and, unless --skip-published, the published
classifier (protectai/deberta-v3-base-prompt-injection-v2; its cut-offs come from the seed's TRAIN clean margins, as in
eval.baseline_eval).

Reads   data/evalsets/obfuscated_A.jsonl, data/pooled/pooled.jsonl, data/splits/split_seed{seed}.json, the pooled score
        cache (found by its header), the published-classifier train cache.
Writes  eval/results/evasion_eval_seed{seed}.md and .csv (columns method,transform,budget,k,n,recall,recall_lo,recall_hi,
        evasion,evasion_lo,evasion_hi), the cache external_obfuscated_A_seed{seed}_<hash>.csv and
        published_scores_obfuscated_A_<model>_w<windows>.csv; with --figure also paper/figures/F7_evasion.pdf,
        eval/results/figures/F7_evasion.png and the F7 line of paper/figures/captions.md.
Only the TRAIN and VALIDATION roles are ever read. This is the Batch A (design) set: recall only, never a false-alarm rate.
"""

import argparse
import csv
import glob
import math
import os
import re

import numpy as np

from eval.apply_models import fit_models
from eval.fpr_eval import BUDGETS, threshold_at_budget
from eval.sentence_eval import find_scores_file, load_docs
from eval.signal_cache import external_path, load_evalset, score_external

SET = "obfuscated_A"
CONTROL = "T0_none"
METHODS = ["pattern bank", "sentence classifier", "LR + sentence classifier"]
PUB = "published classifier"
CSV_COLUMNS = ["method", "transform", "budget", "k", "n", "recall", "recall_lo", "recall_hi",
               "evasion", "evasion_lo", "evasion_hi"]
MIN_CONTROL_RECALL = 0.7
RESULTS_DIR = "eval/results"
FIG_PDF = "paper/figures/F7_evasion.pdf"
FIG_PNG = "eval/results/figures/F7_evasion.png"
CAPTIONS = "paper/figures/captions.md"
CAPTION_F7 = ("**F7** Evasion rate (share of documents with the injected sentence that are NOT flagged) per text "
              "transformation, at the 5% and 1% false-alarm budgets; bars are the mean over three seeds, dots the "
              "single seeds, error bars the mean 95% Wilson interval. `none` is the untransformed control. Batch A "
              "(design batch, 20 documents per transformation): recall only, not a false-alarm rate.")


# --------------------------------------------------------------------------- statistics

def wilson(k, n, z=1.959963984540054):
    """95% Wilson interval (z = the exact 97.5% normal quantile, as scipy uses) for k successes in n trials;
    (nan, nan) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def make_cell(method, transform, budget, flags):
    """One table cell from the boolean flags of the documents of one transformation."""
    flags = np.asarray(flags, dtype=bool)
    n, k = int(len(flags)), int(flags.sum())
    lo, hi = wilson(k, n)
    return dict(method=method, transform=transform, budget=budget, k=k, n=n,
                recall=k / n if n else float("nan"), recall_lo=lo, recall_hi=hi,
                evasion=1 - k / n if n else float("nan"), evasion_lo=1 - hi, evasion_hi=1 - lo)


def transform_order(rows):
    """T0_none first, then the transformations in the order of the evaluation set."""
    seen = list(dict.fromkeys(r.attack_type for r in rows))
    return ([CONTROL] if CONTROL in seen else []) + [t for t in seen if t != CONTROL]


def build_cells(rows, flags_by_method, budgets=BUDGETS):
    """flags_by_method: {method: {budget: bool array over rows}} -> list of cells (one per method, budget, transform)."""
    cells = []
    types = np.array([r.attack_type for r in rows])
    for method, per_budget in flags_by_method.items():
        for budget in budgets:
            for t in transform_order(rows):
                cells.append(make_cell(method, t, budget, per_budget[budget][types == t]))
    return cells


def control_recall(cells, budget, method="pattern bank"):
    c = next((x for x in cells if x["method"] == method and x["transform"] == CONTROL and x["budget"] == budget), None)
    return None if c is None else c["recall"]


# --------------------------------------------------------------------------- scoring

def pool_method_flags(rows, fm, ext):
    """Flags of the pool-trained methods: {method: {budget: bool array}}, TRAIN-derived thresholds applied unchanged."""
    docs = [dict(id=r.id, text=r.text, pat=ext["pattern"][r.id], emb=ext["embedding"][r.id],
                 ppl=ext["perplexity"][r.id]) for r in rows]
    scores = fm.score(docs)
    return {m: {b: scores[m] > fm.thresholds[m][b] for b in BUDGETS} for m in METHODS}


def published_flags(seed, rows, model_id, max_windows):
    """{budget: bool array} for the published classifier, cut-offs from the seed's TRAIN clean margins."""
    from eval import baseline_eval as be

    train = load_docs(seed, ["train"], find_scores_file(seed))["train"]
    train_path = be.cache_path(model_id, max_windows)
    train_cache = be.read_cache(train_path)
    missing = [d["id"] for d in train if d["id"] not in train_cache]
    if missing:
        raise SystemExit(f"{len(missing)} train documents have no published-classifier score in {train_path}. "
                         "Run  python -m eval.baseline_eval  first (or use --skip-published).")
    print("Using published-classifier train cache:", train_path)
    clean = np.array([train_cache[d["id"]]["margin"] for d in train if d["y"] == 0])
    cutoffs = {b: threshold_at_budget(clean, b) for b in BUDGETS}
    path = f"{RESULTS_DIR}/published_scores_{SET}_{model_id.replace('/', '__')}_w{max_windows}.csv"
    cache, _ = be.score_missing([dict(id=r.id, text=r.text) for r in rows], model_id, max_windows, path)
    print("Using published-classifier cache for", SET + ":", path)
    margins = np.array([cache[r.id]["margin"] for r in rows])
    return {b: margins > cutoffs[b] for b in BUDGETS}


# --------------------------------------------------------------------------- report

def fmt_cell(c):
    return f"{c['evasion']:.2f} [{c['evasion_lo']:.2f}, {c['evasion_hi']:.2f}] (n={c['n']})"


def render_report(seed, cells, methods, warnings):
    types = list(dict.fromkeys(c["transform"] for c in cells))
    lines = [f"# Evasion rates of the obfuscations, seed {seed} (train and validation only)", "",
             "Evaluation set: data/evalsets/obfuscated_A.jsonl (Batch A, the design batch). Every document carries the "
             "same injected sentence; a transformation is applied to that sentence only. The models and their 5% / 1% "
             "false-alarm thresholds were fitted on the TRAIN role of the pool and are applied unchanged. "
             "Evasion rate = 1 - recall = share of documents that are NOT flagged; cells show the rate with its 95% "
             "Wilson interval and n. `T0_none` is the untransformed control: the pattern bank must catch most of "
             "those, otherwise the experiment says nothing.", ""]
    for w in warnings:
        lines += [f"**WARNING:** {w}", ""]
    for budget in BUDGETS:
        lines += [f"## False-alarm budget {budget:.0%}", "",
                  "| transformation | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
        for t in types:
            row = []
            for m in methods:
                c = next(x for x in cells if x["method"] == m and x["transform"] == t and x["budget"] == budget)
                row.append(fmt_cell(c))
            lines.append(f"| {t} | " + " | ".join(row) + " |")
        lines.append("")
    pb = [control_recall(cells, b) for b in BUDGETS]
    lines += ["Control check (pattern bank, recall on the untransformed documents): "
              + ", ".join(f"{b:.0%} budget {r:.2f}" for b, r in zip(BUDGETS, pb) if r is not None) + ".", "",
              "Caveats:", "",
              "- Recall only. These sets contain only poisoned documents; no false-alarm rate is measured here "
              "(it comes from the pool, and the cost of any defence is measured in task 8.4).",
              "- Batch A is the design batch (Rule 5): it may be looked at while defences are developed. This table is the "
              "'before' measurement for the normalise-then-match patch; the blind Batch B is used once at the freeze.",
              "- 20 documents per transformation: the intervals are wide. The 20 documents of a transformation "
              "are different hosts but share the same 20 payloads, and the same 20 payloads are used for every "
              "transformation, so rows are not independent samples.",
              "- Evasion is measured against the cut-offs of the TRAIN role; it describes these eight simple text "
              "tricks only and says nothing about adaptive attackers.", ""]
    return "\n".join(lines)


def write_outputs(seed, cells, report, out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"evasion_eval_seed{seed}")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(CSV_COLUMNS)
        for c in cells:
            w.writerow([c["method"], c["transform"], c["budget"], c["k"], c["n"],
                        f"{c['recall']:.4f}", f"{c['recall_lo']:.4f}", f"{c['recall_hi']:.4f}",
                        f"{c['evasion']:.4f}", f"{c['evasion_lo']:.4f}", f"{c['evasion_hi']:.4f}"])
    return base


# --------------------------------------------------------------------------- figure F7

def read_cells(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(dict(method=r["method"], transform=r["transform"], budget=float(r["budget"]), n=int(r["n"]),
                            evasion=float(r["evasion"]), evasion_lo=float(r["evasion_lo"]),
                            evasion_hi=float(r["evasion_hi"])))
    return out


def seed_files(results_dir=RESULTS_DIR):
    found = {}
    for p in glob.glob(os.path.join(results_dir, "evasion_eval_seed*.csv")):
        m = re.search(r"evasion_eval_seed(\d+)\.csv$", p)
        if m:
            found[int(m.group(1))] = p
    return dict(sorted(found.items()))


def aggregate(per_seed):
    """per_seed: {seed: cells} -> {(budget, method, transform): dict(mean, lo, hi, dots, n)} (mean over seeds)."""
    keys = {}
    for seed, cells in per_seed.items():
        for c in cells:
            keys.setdefault((round(c["budget"], 4), c["method"], c["transform"]), []).append((seed, c))
    out = {}
    for key, items in keys.items():
        ev = [c["evasion"] for _, c in items]
        out[key] = dict(mean=float(np.mean(ev)), lo=float(np.mean([c["evasion_lo"] for _, c in items])),
                        hi=float(np.mean([c["evasion_hi"] for _, c in items])), dots=ev, n=items[0][1]["n"])
    return out


def label_of(transform):
    return "none" if transform == CONTROL else transform.replace("_", " ")


def build_figure(per_seed, pdf_path=FIG_PDF, png_path=FIG_PNG):
    """Grouped bars: evasion rate per transformation, one bar per method, one panel per budget; seed dots over the bars."""
    from eval.figstyle import DOUBLE_W, METHOD_COLORS, METHOD_HATCHES, apply_style

    plt = apply_style()
    agg = aggregate(per_seed)
    methods = [m for m in METHODS + [PUB] if any(k[1] == m for k in agg)]
    types = list(dict.fromkeys(k[2] for k in agg))
    budgets = sorted({k[0] for k in agg}, reverse=True)
    fig, axes = plt.subplots(len(budgets), 1, figsize=(DOUBLE_W, 1.9 * len(budgets) + 0.5), sharex=True, squeeze=False)
    width = 0.8 / len(methods)
    for ax, budget in zip(axes[:, 0], budgets):
        for j, m in enumerate(methods):
            xs, means, lo_err, hi_err, dots = [], [], [], [], []
            for i, t in enumerate(types):
                a = agg.get((budget, m, t))
                if a is None:
                    continue
                x = i - 0.4 + width * (j + 0.5)
                xs.append(x)
                means.append(a["mean"])
                lo_err.append(max(a["mean"] - a["lo"], 0))
                hi_err.append(max(a["hi"] - a["mean"], 0))
                dots.append((x, a["dots"]))
            ax.bar(xs, means, width * 0.92, color=METHOD_COLORS[m], hatch=METHOD_HATCHES[m], edgecolor="black",
                   linewidth=0.4, label=m, yerr=[lo_err, hi_err], error_kw=dict(elinewidth=0.6, capsize=1.2, ecolor="black"))
            for x, ds in dots:
                ax.plot([x] * len(ds), ds, linestyle="none", marker="o", markersize=2.2, markerfacecolor="white",
                        markeredgecolor="black", markeredgewidth=0.5, zorder=4)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel(f"Evasion rate ({budget:.0%} budget)")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[-1, 0].set_xticks(range(len(types)))
    axes[-1, 0].set_xticklabels([label_of(t) for t in types], rotation=25, ha="right")
    axes[-1, 0].set_xlabel("Text transformation applied to the injected sentence")
    axes[0, 0].legend(ncol=len(methods), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.4,
                      columnspacing=1.0)
    fig.tight_layout()
    for p in (pdf_path, png_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p)
    plt.close(fig)
    return pdf_path, png_path


def update_caption(path=CAPTIONS, line=CAPTION_F7):
    """Write or replace the F7 line of paper/figures/captions.md; every other line is left untouched."""
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    kept = [ln for ln in lines if not ln.startswith("**F7**")]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(kept + [line]) + "\n")


def figure_main(results_dir=RESULTS_DIR):
    files = seed_files(results_dir)
    if not files:
        raise SystemExit("no eval/results/evasion_eval_seed<N>.csv found: run  python -m eval.evasion_eval --seed N  first")
    print("Figure F7 from seeds:", ", ".join(map(str, files)))
    if len(files) < 3:
        print("WARNING: fewer than three seeds; the final figure needs seeds 42, 43 and 44.")
    pdf, png = build_figure({s: read_cells(p) for s, p in files.items()})
    update_caption()
    print(f"Saved {pdf}, {png} and the F7 line of {CAPTIONS}")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--figure", action="store_true", help="draw F7 from the CSVs written by earlier runs")
    ap.add_argument("--skip-published", action="store_true", help="do not run the DeBERTa classifier")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()
    if args.figure:
        return figure_main()
    if args.seed is None:
        ap.error("give --seed N (or --figure)")
    seed = args.seed

    scores_path = find_scores_file(seed)
    print("Using score cache:", scores_path)
    rows = load_evalset(SET)
    print(f"{SET}: {len(rows)} documents. Only the train and validation roles are used.")
    ext = score_external(SET, seed)
    print("Using external cache:", external_path(SET, seed))

    fm = fit_models(seed, scores_path)
    flags = pool_method_flags(rows, fm, ext)
    methods = list(METHODS)
    if not args.skip_published:
        from detectors import published_classifier as pc

        model = args.model or pc.DEFAULT_MODEL
        windows = args.max_windows or pc.MAX_WINDOWS
        flags[PUB] = published_flags(seed, rows, model, windows)
        methods.append(PUB)

    cells = build_cells(rows, flags)
    warnings = []
    for b in BUDGETS:
        r = control_recall(cells, b)
        if r is not None and r < MIN_CONTROL_RECALL:
            warnings.append(f"the pattern bank catches only {r:.0%} of the untransformed controls at the {b:.0%} "
                            f"budget (need at least {MIN_CONTROL_RECALL:.0%}); read the table with care, the "
                            "experiment is weak.")
    report = render_report(seed, cells, methods, warnings)
    print("\n" + report)
    base = write_outputs(seed, cells, report)
    print(f"\nSaved {base}.md and {base}.csv")


if __name__ == "__main__":
    main()
