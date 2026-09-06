"""
Fixes the scale-mismatch problem in naive fixed fusion: pattern and
perplexity output scores that regularly hit 1.0, but embedding's raw
scores never exceed ~0.43 on this dataset. Averaging these directly
systematically under-weights embedding regardless of how good its signal
actually is.

Fix: min-max normalize each detector's scores to its OWN observed range
before averaging, so all three contribute on a comparable 0-1 scale.

Usage: python -m eval.calibrated_fusion
"""

import csv

CATEGORIES = ["injection", "authority", "corpus", "perplexity"]
SCORE_KEYS = ["pattern_score", "embedding_score", "perplexity_score"]


def minmax_normalize(rows, score_key):
    values = [float(r[score_key]) for r in rows]
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    return {id(r): (float(r[score_key]) - lo) / span for r in rows}, lo, hi


def main():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # normalize each detector independently
    normalized = {}
    print("=== Observed ranges used for calibration ===")
    for key in SCORE_KEYS:
        norm_map, lo, hi = minmax_normalize(rows, key)
        normalized[key] = norm_map
        print(f"{key:18s} observed range: [{lo:.3f}, {hi:.3f}]")

    # attach calibrated combined score to each row
    for r in rows:
        calibrated_scores = [normalized[key][id(r)] for key in SCORE_KEYS]
        r["calibrated_fusion_score"] = sum(calibrated_scores) / len(calibrated_scores)

    print("\n" + "=" * 60)
    print("CALIBRATED FIXED FUSION -- threshold sweep")
    print("=" * 60)
    print(f"{'threshold':>10s} {'precision':>10s} {'recall':>8s} {'f1':>8s}")
    best_f1, best_t = -1, None
    for t in [i / 20 for i in range(1, 20)]:
        tp = sum(1 for r in rows if r["true_label"] == "1" and r["calibrated_fusion_score"] >= t)
        fp = sum(1 for r in rows if r["true_label"] == "0" and r["calibrated_fusion_score"] >= t)
        fn = sum(1 for r in rows if r["true_label"] == "1" and r["calibrated_fusion_score"] < t)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        marker = ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
            marker = "  <-- best so far"
        print(f"{t:>10.2f} {precision:>10.2f} {recall:>8.2f} {f1:>8.2f}{marker}")

    print(f"\nBEST calibrated fusion: threshold={best_t:.2f}  f1={best_f1:.2f}")

    print("\n" + "=" * 60)
    print(f"FAILURE MAP at best calibrated threshold ({best_t:.2f})")
    print("=" * 60)
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        rate = sum(1 for r in cat_rows if r["calibrated_fusion_score"] >= best_t) / len(cat_rows)
        print(f"{cat:12s} {rate:>10.0%}")


if __name__ == "__main__":
    main()