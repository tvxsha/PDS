"""
Reproduces the "boilerplate anchoring" calibration-fragility finding as a
clean, standalone experiment on real data already in the repo (no model
re-runs needed -- uses the cached scores in eval/results/*.csv).

THE PROBLEM
-----------
calibrated_fusion.py (and adversarial_test.py, which freezes these same
bounds) min-max normalizes each detector's raw scores using the single
lowest and single highest score observed in the original 77-doc eval set.
That means ONE document sets each boundary. If that one document happens
to be boilerplate-ish/README-like text that just barely resembles an
attack (or vice versa), the entire calibration scale for every other
document -- including every future document scored against these frozen
bounds -- is anchored to that one point.

pattern_score and perplexity_score are already naturally bounded to
[0, 1] with many documents tied at each extreme, so min-max does nothing
there and isn't fragile. embedding_score is the detector actually at
risk: its raw range is a narrow, continuous band (~0.18-0.43) set by
individual documents, not ties -- exactly the single-anchor scenario.

THE EXPERIMENT
---------------
For embedding_score:
  1. Compute the real min-max bounds (as calibrated_fusion.py does).
  2. Remove the single document that sets the low anchor, recompute the
     bound on the remaining 76 docs -- this is "how much would calibration
     have been different if that one clean doc had been phrased slightly
     differently." Same for the high anchor.
  3. Do the identical leave-one-out test using a 5th/95th percentile
     (quantile) bound instead of raw min/max -- quantiles should barely
     move, since dropping 1 of 77 docs changes a percentile rank by ~1.3%.
  4. Quantify the practical effect: how much does the WHOLE DATASET's
     average calibrated embedding score shift when the anchor doc is
     removed, under min-max vs quantile -- this is the actual fragility,
     not just a bounds number.
  5. Show the concrete real-world instance: adv_010.txt (the one
     adversarial doc that evaded detection in eval/results/adversarial_results.csv,
     a credential-exposure attack framed as neutral README/license text)
     recalibrated under both the original and anchor-removed min-max
     bounds, vs. under quantile bounds.

Scope note: "quantile calibration" here means percentile-based min-max
(5th/95th instead of 0th/100th) as a drop-in, more outlier-robust
replacement for the SAME normalize-then-average fusion scheme already in
calibrated_fusion.py. It is not the conformal-prediction thresholding in
fusion/conformal.py (task 5.1) -- that's a different mechanism (choosing
a decision threshold with a formal FPR guarantee) and is Tvisha's/
Karishma's follow-on work (5.1/5.3). Worth flagging in review in case
"quantile" was meant to point at that instead.

Usage: python -m eval.calibration_fragility
Output: eval/results/calibration_fragility.png
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

EVAL_CSV = "eval/results/evaluation_raw.csv"
ADV_CSV = "eval/results/adversarial_results.csv"
OUT_PNG = "eval/results/calibration_fragility.png"

SCORE_KEYS = ["pattern_score", "embedding_score", "perplexity_score"]


def load_scores():
    with open(EVAL_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in SCORE_KEYS}


def minmax_bounds(values):
    return float(np.min(values)), float(np.max(values))


def quantile_bounds(values, low_q=5, high_q=95):
    return float(np.percentile(values, low_q)), float(np.percentile(values, high_q))


def normalize(values, lo, hi):
    span = hi - lo if hi > lo else 1.0
    return np.clip((values - lo) / span, 0.0, 1.0)


def leave_one_out_shift(values, bound_fn, which):
    """
    Removes the document that sets the `which` ("low" or "high") bound,
    recomputes the bound on the remaining docs, and returns
    (original_bound, bound_without_anchor, avg_renormalized_score_shift).

    The last number is the real-world consequence: how much the WHOLE
    dataset's average normalized score moves just because that one
    document is in or out of the calibration set.
    """
    lo, hi = bound_fn(values)
    anchor_idx = int(np.argmin(values)) if which == "low" else int(np.argmax(values))
    remaining = np.delete(values, anchor_idx)

    lo2, hi2 = bound_fn(remaining)
    bound_before = lo if which == "low" else hi
    bound_after = lo2 if which == "low" else hi2

    norm_with_anchor = normalize(values, lo, hi)
    norm_without_anchor = normalize(values, lo2, hi2)
    avg_shift = float(np.mean(np.abs(norm_with_anchor - norm_without_anchor)))

    return bound_before, bound_after, avg_shift


def main():
    scores = load_scores()

    print("=" * 70)
    print("CALIBRATION FRAGILITY: min-max vs quantile, leave-one-anchor-out")
    print("=" * 70)

    results = {}  # {score_key: {"minmax": avg_shift, "quantile": avg_shift}}
    for key in SCORE_KEYS:
        values = scores[key]
        print(f"\n--- {key} ---")
        results[key] = {}
        for method_name, bound_fn in [("minmax", minmax_bounds), ("quantile(5/95)", quantile_bounds)]:
            shifts = []
            for which in ["low", "high"]:
                before, after, avg_shift = leave_one_out_shift(values, bound_fn, which)
                shifts.append(avg_shift)
                print(f"  {method_name:16s} {which:4s} bound: {before:.4f} -> {after:.4f} "
                      f"(removing 1 doc)  | dataset-avg normalized-score shift: {avg_shift:.4f}")
            results[key][method_name] = float(np.mean(shifts))

    # Concrete real instance: adv_010, the evaded credential-exposure attack,
    # recalibrated on embedding_score under each bound scheme.
    print("\n" + "=" * 70)
    print("CONCRETE CASE: adv_010.txt (evaded credential-exposure attack)")
    print("=" * 70)
    with open(ADV_CSV, encoding="utf-8") as f:
        adv_rows = {r["filename"]: r for r in csv.DictReader(f)}
    adv010_embedding = float(adv_rows["adv_010.txt"]["embedding_score"])
    print(f"adv_010.txt raw embedding_score: {adv010_embedding:.4f}")

    emb = scores["embedding_score"]
    lo, hi = minmax_bounds(emb)
    _, hi_no_anchor, _ = leave_one_out_shift(emb, minmax_bounds, "high")
    qlo, qhi = quantile_bounds(emb)

    for label, lo_b, hi_b in [
        ("min-max (current, fixed bounds)", lo, hi),
        ("min-max (if the high-anchor doc were absent)", lo, hi_no_anchor),
        ("quantile 5/95 (current)", qlo, qhi),
    ]:
        cal = float(normalize(np.array([adv010_embedding]), lo_b, hi_b)[0])
        print(f"  {label:48s} -> calibrated embedding contribution: {cal:.4f}")

    # Plot: dataset-average normalized-score shift when the anchor doc is
    # removed, grouped by detector, min-max vs quantile.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(SCORE_KEYS))
    width = 0.35
    minmax_vals = [results[k]["minmax"] for k in SCORE_KEYS]
    quantile_vals = [results[k]["quantile(5/95)"] for k in SCORE_KEYS]

    ax.bar(x - width / 2, minmax_vals, width, label="min-max (current)", color="#c0392b")
    ax.bar(x + width / 2, quantile_vals, width, label="quantile (5th/95th pct)", color="#2980b9")
    ax.set_xticks(x)
    ax.set_xticklabels(SCORE_KEYS)
    ax.set_ylabel("Avg. dataset-wide normalized-score shift\nwhen the single anchor doc is removed")
    ax.set_title("Calibration fragility: min-max vs quantile bounds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nPlot saved to {OUT_PNG}")


if __name__ == "__main__":
    main()
