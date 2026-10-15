"""
Shared figure style for the paper (rules of roadmap task 1.6, first used by figure F7 of task 8.1).

    from eval.figstyle import apply_style, METHOD_COLORS, METHOD_HATCHES, SINGLE_W, DOUBLE_W

Rules: single-column width 3.5 in or double-column 7.16 in, font size 8, no title inside the figure (captions are
written in LaTeX), `pdf.fonttype 42`, Okabe-Ito colours, the SAME colour per method in every figure, hatches as a
second cue so the figures are readable in greyscale, axis labels with units, seed values drawn as dots over the bars.
"""

SINGLE_W = 3.5
DOUBLE_W = 7.16

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# one colour (and one hatch) per method, identical in every figure
METHOD_COLORS = {
    "pattern bank": "#0072B2",
    "embedding": "#56B4E9",
    "perplexity": "#CC79A7",
    "naive average of the 3": "#F0E442",
    "sentence classifier": "#E69F00",
    "LR on the 3 current scores": "#000000",
    "LR + sentence classifier": "#009E73",
    "published classifier": "#D55E00",
    "cascade": "#999999",
}
METHOD_HATCHES = {
    "pattern bank": "",
    "embedding": "///",
    "perplexity": "\\\\\\",
    "naive average of the 3": "...",
    "sentence classifier": "xxx",
    "LR on the 3 current scores": "---",
    "LR + sentence classifier": "|||",
    "published classifier": "ooo",
    "cascade": "+++",
}


def apply_style():
    """Set the matplotlib defaults of the paper figures (non-interactive backend, 8 pt, TrueType fonts in PDF)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "hatch.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
        "savefig.dpi": 200,
    })
    return plt
