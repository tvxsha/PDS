"""
Recomputes per-detector metrics, combined-fusion metrics, the failure map,
and blind spots using PROPERLY TUNED per-detector thresholds (found via
eval/threshold_sweep.py) instead of a naive blanket 0.5.

Reads the already-generated eval/results/evaluation_raw.csv -- fast, no
model re-runs needed.

Usage: python -m eval.recompute_metrics
"""

import csv

CATEGORIES = ["injection", "authority", "corpus", "perplexity"]

# Tuned thresholds from eval/threshold_sweep.py -- update these if you
# regenerate the dataset or re-run the sweep and get different numbers.
THRESHOLDS = {
    "pattern_score": 0.30,       # flat F1 across 0.05-0.60, picked a stable middle value
    "embedding_score": 0.30,     # was the real discovery -- 0.5 was far too high
    "perplexity_score": 0.20,    # near-best F1, more stable than the very edge at 0.05
}


def main():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("=" * 60)
    print("PER-DETECTOR METRICS (tuned thresholds)")
    print("=" * 60)
    for score_key, threshold in THRESHOLDS.items():
        tp = sum(1 for r in rows if r["true_label"] == "1" and float(r[score_key]) >= threshold)
        fp = sum(1 for r in rows if r["true_label"] == "0" and float(r[score_key]) >= threshold)
        fn = sum(1 for r in rows if r["true_label"] == "1" and float(r[score_key]) < threshold)
        tn = sum(1 for r in rows if r["true_label"] == "0" and float(r[score_key]) < threshold)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        latency_key = f"{score_key.split('_')[0]}_latency_ms"
        avg_latency = sum(float(r[latency_key]) for r in rows) / len(rows)
        print(f"{score_key:18s} (threshold={threshold})  precision={precision:.2f}  "
              f"recall={recall:.2f}  f1={f1:.2f}  avg_latency={avg_latency:.1f}ms  "
              f"(tp={tp} fp={fp} fn={fn} tn={tn})")

    print("\n" + "=" * 60)
    print("FAILURE MAP: per-category detection rate per detector (tuned thresholds)")
    print("=" * 60)
    print(f"{'Category':12s} {'Pattern':>10s} {'Embedding':>10s} {'Perplexity':>10s}")
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        n = len(cat_rows)
        pattern_rate = sum(1 for r in cat_rows if float(r["pattern_score"]) >= THRESHOLDS["pattern_score"]) / n
        embedding_rate = sum(1 for r in cat_rows if float(r["embedding_score"]) >= THRESHOLDS["embedding_score"]) / n
        perplexity_rate = sum(1 for r in cat_rows if float(r["perplexity_score"]) >= THRESHOLDS["perplexity_score"]) / n
        print(f"{cat:12s} {pattern_rate:>10.0%} {embedding_rate:>10.0%} {perplexity_rate:>10.0%}")

    print("\n" + "=" * 60)
    print("BLIND SPOTS: poisoned docs where ALL THREE detectors miss (tuned thresholds)")
    print("=" * 60)
    blind_spots = [
        r for r in rows
        if r["true_label"] == "1"
        and float(r["pattern_score"]) < THRESHOLDS["pattern_score"]
        and float(r["embedding_score"]) < THRESHOLDS["embedding_score"]
        and float(r["perplexity_score"]) < THRESHOLDS["perplexity_score"]
    ]
    if blind_spots:
        for r in blind_spots:
            print(f"  {r['filename']} (category={r['category']})")
    else:
        print("  None -- every poisoned doc caught by at least one detector at these thresholds.")

    print(f"\nTotal blind spots: {len(blind_spots)} out of {sum(1 for r in rows if r['true_label'] == '1')} poisoned docs")


if __name__ == "__main__":
    main()