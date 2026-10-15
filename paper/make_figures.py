"""
Paper figures from the final run (graphs only; the tables come from eval/make_paper_tables.py).

    python paper/make_figures.py            # dry run: reads the VALIDATION files, every figure carries a DRY RUN stamp
    python paper/make_figures.py --final    # reads the held-out files (final_test_*), no stamp

Run it from the repository root. It reads only files that already exist (scores, thresholds, cascade CSVs,
eval/results/latency_benchmark.json) and writes PDFs into paper/figures/. It never opens a document and never
fits anything, so it can be repeated as often as needed.

Figures (the names are what paper/sections/*.tex includes):
  fig_roc.pdf        ROC curves (mean over the three seeds) with the 1% and 5% operating points
  fig_source.pdf     recall and false-alarm rate by source at the 5% budget
  fig_attack.pdf     recall by attack type (heat map) at the 5% budget
  fig_precision.pdf  precision against the share of poisoned documents (Bayes), 1% and 5% budgets
  fig_scores.pdf     score distributions of LR + sentence classifier, clean vs poisoned, with both thresholds
  fig_latency.pdf    median and 95th-percentile latency per signal, and median latency by document length
  fig_cascade.pdf    expected cost against recall for the cascade designs (needs the cascade extra)

Seeds overlap heavily, so error bars are the range over the three seeds, not confidence intervals.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from sklearn.metrics import roc_curve               # noqa: E402

from eval import final_test                         # noqa: E402
from eval.fpr_eval import precision                 # noqa: E402
from eval.make_paper_tables import load_seed        # noqa: E402

RESULTS_DIR = "eval/results"
LATENCY = os.path.join(RESULTS_DIR, "latency_benchmark.json")
OUT_DIR = "paper/figures"

COL1, COL2 = 3.45, 7.1                              # IEEE column and page width, inches

# Okabe-Ito colours (safe for colour-vision deficiency); lines also differ in style. The last field (hatch) is no longer used.
STYLE = {
    "pattern bank":               ("Pattern bank",     "#000000", "-",  "o", ""),
    "embedding":                  ("Embedding",        "#E69F00", "--", "s", "//"),
    "perplexity":                 ("Perplexity",       "#56B4E9", "-.", "^", "\\\\"),
    "naive average of the 3":     ("Naive average",    "#6A3D9A", ":",  "v", ".."),
    "sentence classifier":        ("Sentence clf.",    "#009E73", "-",  "D", "xx"),
    "LR on the 3 current scores": ("LR (A-C)",         "#882255", "--", "P", "++"),
    "LR + sentence classifier":   ("LR + sentence",    "#D55E00", "-",  "*", ""),
    "published classifier":       ("Published clf.",   "#0072B2", "-.", "X", "oo"),
}
INK, GRID = "#222222", "#DDDDDD"


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def finish(fig, name, out, dry):
    if dry:
        fig.text(0.5, 0.5, "DRY RUN\nvalidation, not results", ha="center", va="center", rotation=25,
                 fontsize=22, color="#C00000", alpha=0.18, weight="bold")
    path = os.path.join(out, name)
    fig.savefig(path)
    plt.close(fig)
    print("  " + path)


def label(m):
    return STYLE[m][0]


def tpr_fpr(sd, m, b):
    fl = sd["scores"][m] > sd["thr"][m][str(b)]
    y = sd["y"]
    return float(fl[y == 1].mean()), float(fl[y == 0].mean())


# ---------------------------------------------------------------------------------------------------------------------

def fig_roc(seeds, cfg, out, dry):
    methods = [m for m in cfg["methods"] if m != "naive average of the 3"]
    grid = np.linspace(0, 0.20, 401)
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    for m in methods:
        curves = []
        for sd in seeds:
            fpr, tpr, _ = roc_curve(sd["y"], sd["scores"][m], drop_intermediate=False)
            curves.append(np.interp(grid, fpr, tpr))
        _, color, ls, mk, _ = STYLE[m]
        ax.plot(grid, np.mean(curves, axis=0), color=color, ls=ls, label=label(m))
        for b, face in ((0.05, color), (0.01, "white")):
            pts = np.array([tpr_fpr(sd, m, b) for sd in seeds]).mean(axis=0)
            ax.plot(pts[1], pts[0], mk, color=color, mfc=face, mec=color, ms=4.2, mew=0.9, zorder=5)
    for b in cfg["budgets"]:
        ax.axvline(b, color=INK, lw=0.6, ls=":")
        ax.text(b + 0.002, 0.985, f"{int(b * 100)}% budget", rotation=90, fontsize=6, va="top", ha="left")
    ax.set_xlim(0, 0.20)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("False-alarm rate (clean documents flagged)")
    ax.set_ylabel("Recall (poisoned documents flagged)")
    h, lab = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", color=INK, mfc=INK, ls="", ms=4), Line2D([], [], marker="o", color=INK, mfc="white", ls="", ms=4)]
    lab += ["operating point, 5% budget", "operating point, 1% budget"]
    ax.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=2.2, labelspacing=0.25,
              columnspacing=1.0)
    finish(fig, "fig_roc.pdf", out, dry)


def fig_source(seeds, cfg, out, dry):
    methods = list(cfg["methods"])
    groups = sorted(set(seeds[0]["srcg"]))
    names = {"bipia": "BIPIA", "poisonedrag": "PoisonedRAG", "pds": "in-house"}
    fig, axes = plt.subplots(2, 1, figsize=(COL2, 3.6), sharex=True, gridspec_kw={"hspace": 0.12})
    w = 0.8 / len(methods)
    for ax, which, ylab in ((axes[0], 0, "Recall"), (axes[1], 1, "False-alarm rate")):
        for j, m in enumerate(methods):
            vals = []
            for g in groups:
                per_seed = []
                for sd in seeds:
                    fl = sd["scores"][m] > sd["thr"][m]["0.05"]
                    mask = (sd["srcg"] == g) & (sd["y"] == (1 - which))
                    per_seed.append(fl[mask].mean())
                vals.append(per_seed)
            mean = np.array([np.mean(v) for v in vals])
            err = np.clip(np.array([[np.mean(v) - np.min(v) for v in vals], [np.max(v) - np.mean(v) for v in vals]]), 0, None)
            x = np.arange(len(groups)) + (j - (len(methods) - 1) / 2) * w
            _, color, _, _, _ = STYLE[m]
            ax.bar(x, mean, w * 0.92, color=color, edgecolor="white", linewidth=0.4, label=label(m))
            ax.errorbar(x, mean, yerr=err, fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.2)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.0 if which == 0 else None)
    axes[1].axhline(0.05, color=INK, lw=0.7, ls="--")
    axes[1].text(len(groups) - 0.55, 0.052, "5% budget", fontsize=6, ha="right", va="bottom")
    ns = []
    for g in groups:
        pos = int(((seeds[0]["srcg"] == g) & (seeds[0]["y"] == 1)).sum())
        neg = int(((seeds[0]["srcg"] == g) & (seeds[0]["y"] == 0)).sum())
        ns.append(f"{names.get(g, g)}\n({pos} poisoned / {neg} clean, seed {seeds[0]['seed']})")
    axes[1].set_xticks(np.arange(len(groups)))
    axes[1].set_xticklabels(ns)
    axes[0].legend(ncol=8, loc="lower center", bbox_to_anchor=(0.5, 1.02), columnspacing=0.9, handlelength=1.3, handletextpad=0.4)
    finish(fig, "fig_source.pdf", out, dry)


def fig_attack(seeds, cfg, out, dry):
    methods = list(cfg["methods"])
    atks = sorted(set(seeds[0]["atk"][seeds[0]["y"] == 1]))
    mat = np.zeros((len(methods), len(atks)))
    for i, m in enumerate(methods):
        for k, a in enumerate(atks):
            vals = []
            for sd in seeds:
                fl = sd["scores"][m] > sd["thr"][m]["0.05"]
                mask = (sd["atk"] == a) & (sd["y"] == 1)
                if mask.any():
                    vals.append(fl[mask].mean())
            mat[i, k] = np.mean(vals) if vals else np.nan
    ns = []
    for a in atks:
        c = [int(((sd["atk"] == a) & (sd["y"] == 1)).sum()) for sd in seeds]
        ns.append(str(c[0]) if len(set(c)) == 1 else f"{min(c)}-{max(c)}")
    pretty = {"bipia_code": "BIPIA\ncode", "bipia_text": "BIPIA\ntext", "pds_adversarial": "in-house\nadversarial",
              "pds_authority": "in-house\nauthority", "pds_corpus": "in-house\ncorpus", "pds_injection": "in-house\ninjection",
              "pds_perplexity": "in-house\nperplexity", "poisonedrag_nq": "PoisonedRAG\nNQ"}
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(atks)))
    ax.set_xticklabels([f"{pretty.get(a, a.replace('_', ' '))}\n(n = {n})" for a, n in zip(atks, ns)], fontsize=6.5)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([label(m) for m in methods])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(methods)):
        for k in range(len(atks)):
            v = mat[i, k]
            ax.text(k, i, "-" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Recall at the 5% budget", fontsize=7)
    cb.outline.set_visible(False)
    finish(fig, "fig_attack.pdf", out, dry)


def fig_precision(seeds, cfg, out, dry):
    prev = np.logspace(-3, -1, 120)
    n_neg = float(np.mean([(sd["y"] == 0).sum() for sd in seeds]))
    picks = [("LR + sentence classifier", "0.05", "-"), ("LR + sentence classifier", "0.01", "--"),
             ("sentence classifier", "0.05", "-"), ("pattern bank", "0.05", "-"), ("published classifier", "0.05", "-.")]
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    for m, b, ls in picks:
        t, f = np.array([tpr_fpr(sd, m, float(b)) for sd in seeds]).mean(axis=0)
        _, color, _, _, _ = STYLE[m]
        ax.plot(prev, [precision(t, f, p, n_neg) for p in prev], color=color, ls=ls, label=f"{label(m)} ({int(float(b) * 100)}%)")
    for p in cfg["prevalences"]:
        ax.axvline(p, color=INK, lw=0.6, ls=":")
        ax.text(p * 1.06, 0.97, f"{p * 100:g}%", fontsize=6, va="top")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e-1)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["0.1%", "1%", "10%"])
    ax.set_xlabel("Share of poisoned documents in the retrieved stream")
    ax.set_ylabel("Precision (share of alerts that are real)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), labelspacing=0.25, handlelength=2.2, title="method (false-alarm budget)",
              title_fontsize=6.5, alignment="left")
    finish(fig, "fig_precision.pdf", out, dry)


def fig_scores(seeds, cfg, out, dry):
    sd = seeds[0]
    m = "LR + sentence classifier"
    s, y = sd["scores"][m], sd["y"]
    bins = np.linspace(0, 1, 26)
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    ax.hist(s[y == 0], bins=bins, color="#0072B2", alpha=0.55, label=f"clean (n = {int((y == 0).sum())})")
    ax.hist(s[y == 1], bins=bins, color="#D55E00", alpha=0.55, label=f"poisoned (n = {int(y.sum())})")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 400)
    for b, ls in ((0.05, "--"), (0.01, ":")):
        t = sd["thr"][m][str(b)]
        ax.axvline(t, color=INK, lw=0.8, ls=ls)
        ax.text(t - 0.01, 330, f"{int(b * 100)}% budget", rotation=90, fontsize=6, va="top", ha="right")
    ax.set_xlabel("Fused score (LR + sentence classifier)")
    ax.set_ylabel(f"Documents (seed {sd['seed']}, log scale)")
    ax.legend(loc="upper left")
    finish(fig, "fig_scores.pdf", out, dry)


def fig_latency(out, dry):
    if not os.path.exists(LATENCY):
        print(f"  skipped fig_latency.pdf: {LATENCY} does not exist")
        return
    with open(LATENCY, encoding="utf-8") as f:
        lat = json.load(f)
    det, bl = lat["detectors"], lat.get("by_length", {})
    pretty = {"pattern": "Pattern", "sentence classifier": "Sentence clf.", "embedding": "Embedding", "perplexity": "Perplexity",
              "published classifier": "Published clf.", "ours3": "A-C together", "ours3 + sentence": "A-D together"}
    order = [k for k in sorted(det, key=lambda k: det[k]["median_ms"])]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL2, 2.4), sharey=True, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    y = np.arange(len(order))
    med = np.array([det[k]["median_ms"] for k in order])
    p95 = np.array([det[k]["p95_ms"] for k in order])
    a.barh(y, med, 0.55, color="#0072B2", label="median")
    a.plot(p95, y, "D", color="#D55E00", ms=3.8, label="95th percentile")
    a.set_yticks(y)
    a.set_yticklabels([pretty.get(k, k) for k in order])
    a.set_xscale("log")
    a.set_xlim(0.04, 1e4)
    a.invert_yaxis()
    a.set_xlabel("Milliseconds per document (log scale)")
    a.legend(loc="upper right")
    a.set_title("(a) per signal", loc="left", fontsize=7.5)
    if bl:
        buckets = ["under 500 characters", "500 to 2,000 characters", "over 2,000 characters"]
        keys = [k for k in order if k in bl]
        w = 0.8 / len(buckets)
        shades = ["#BDD7E7", "#6BAED6", "#08519C"]
        for j, bk in enumerate(buckets):
            vals = [bl[k][bk]["median_ms"] for k in keys]
            b.barh(np.arange(len(keys)) + (j - 1) * w, vals, w * 0.92, color=shades[j], label=bk.replace("under", "<").replace("over", ">"))
        b.set_xscale("log")
        b.set_xlim(0.04, 1e4)
        b.set_xlabel("Median milliseconds (log scale)")
        b.legend(loc="upper right", title="document length", title_fontsize=6.5)
        b.set_title("(b) by document length", loc="left", fontsize=7.5)
    finish(fig, "fig_latency.pdf", out, dry)


def fig_cascade(cfg, prefix, out, dry):
    paths = [os.path.join(RESULTS_DIR, f"{prefix}_seed{s}_cascade.csv") for s in cfg["seeds"]]
    if not all(os.path.exists(p) for p in paths):
        print("  skipped fig_cascade.pdf: the cascade files are missing")
        return
    fr = pd.concat([pd.read_csv(p) for p in paths])
    fr["qkey"] = fr["q"].fillna(-1)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7), sharey=True, gridspec_kw={"wspace": 0.08})
    look = {-1.0: None, 0.01: ("D", "#009E73", "B: clear-early, q = 1%"), 0.02: ("v", "#0072B2", "B: clear-early, q = 2%"),
            0.05: ("^", "#CC79A7", "B: clear-early, q = 5%")}
    for ax, b in zip(axes, cfg["budgets"]):
        sub = fr[np.isclose(fr["budget"], b)]
        g = sub.groupby(["design", "qkey"], sort=False)[["ms_clean", "tpr"]].mean().reset_index()
        for _, r in g.iterrows():
            if r["design"] == "run everything + LR":
                mk, col, lab = "o", "#000000", "run everything + LR"
            elif r["design"].startswith("A"):
                mk, col, lab = "s", "#E69F00", "A: OR cascade"
            else:
                mk, col, lab = look[round(r["qkey"], 2)]
            ax.plot(r["ms_clean"], r["tpr"], mk, color=col, ms=6, mec="white", mew=0.6, label=lab)
        ax.set_xlabel("Expected milliseconds per clean document")
        ax.set_title(f"({'ab'[list(cfg['budgets']).index(b)]}) {int(b * 100)}% false-alarm budget", loc="left", fontsize=7.5)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 0.8)
    axes[0].set_ylabel("Recall")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=5, columnspacing=1.0, handletextpad=0.3)
    finish(fig, "fig_cascade.pdf", out, dry)


def main():
    ap = argparse.ArgumentParser(description="Paper figures from the final run (see the top of this file).")
    ap.add_argument("--final", action="store_true", help="read the held-out files (final_test_*); default is the dry run")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    cfg, _ = final_test.load_config()
    role = final_test.HELD_OUT if args.final else "val"
    prefix = f"final_{final_test.MODES[role]}"
    dry = not args.final
    seeds = [load_seed(s, prefix, cfg["methods"]) for s in cfg["seeds"]]

    setup()
    os.makedirs(args.out, exist_ok=True)
    print("Wrote:")
    fig_roc(seeds, cfg, args.out, dry)
    fig_source(seeds, cfg, args.out, dry)
    fig_attack(seeds, cfg, args.out, dry)
    fig_precision(seeds, cfg, args.out, dry)
    fig_scores(seeds, cfg, args.out, dry)
    fig_latency(args.out, dry)
    fig_cascade(cfg, prefix, args.out, dry)
    if dry:
        print("\nDRY RUN: validation numbers, every figure is stamped. Use --final after the three test runs.")


if __name__ == "__main__":
    main()