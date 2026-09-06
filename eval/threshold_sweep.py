"""
Sweeps candidate thresholds for the embedding detector to find the one that
actually maximizes F1 -- rather than assuming 0.5 or guessing a single
number. Also reports how good the BEST possible threshold is, which tells
you whether this detector is worth keeping at all or is fundamentally weak
on this dataset (useful either way for your report).

Usage: python -m eval.threshold_sweep
"""

import csv


def compute_metrics(rows, score_key, threshold):
    tp = sum(1 for r in rows if r["true_label"] == "1" and float(r[score_key]) >= threshold)
    fp = sum(1 for r in rows if r["true_label"] == "0" and float(r[score_key]) >= threshold)
    fn = sum(1 for r in rows if r["true_label"] == "1" and float(r[score_key]) < threshold)
    tn = sum(1 for r in rows if r["true_label"] == "0" and float(r[score_key]) < threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1, tp, fp, fn, tn


def main():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for score_key in ["pattern_score", "embedding_score", "perplexity_score"]:
        print(f"\n=== Threshold sweep: {score_key} ===")
        print(f"{'threshold':>10s} {'precision':>10s} {'recall':>8s} {'f1':>8s}")

        best_f1 = -1
        best_threshold = None
        for t in [i / 20 for i in range(1, 20)]:  # 0.05 to 0.95 in steps of 0.05
            precision, recall, f1, *_ = compute_metrics(rows, score_key, t)
            marker = ""
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
                marker = "  <-- best so far"
            print(f"{t:>10.2f} {precision:>10.2f} {recall:>8.2f} {f1:>8.2f}{marker}")

        print(f"BEST for {score_key}: threshold={best_threshold:.2f}  f1={best_f1:.2f}")


if __name__ == "__main__":
    main()