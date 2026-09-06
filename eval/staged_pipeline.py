"""
Simulates a staged/tiered detection pipeline: run cheap detectors first,
only pay for the expensive one (perplexity) when the cheap ones are
inconclusive. Reuses the real per-detector scores AND latencies already
logged in eval/results/evaluation_raw.csv -- no need to re-run detectors,
this just replays the decision cascade against real timings.

Stage order rationale (informed by earlier findings):
  1. Pattern -- essentially free, and precision=1.00 when it fires, so a
     strong pattern match is trusted immediately with no further checks.
  2. Embedding -- cheap (~21ms) AND, per the learned-fusion feature
     importance, your strongest overall signal. Runs second so most
     documents get resolved before paying perplexity's cost.
  3. Perplexity -- slowest (~100ms+), only run when stages 1-2 are both
     inconclusive.

Thresholds below match the tuned values from earlier analysis
(eval/recompute_metrics.py, eval/calibrated_fusion.py) -- update here if
those change.

Usage: python -m eval.staged_pipeline
"""

import csv

PATTERN_REJECT_THRESHOLD = 0.6   # strong pattern match -> reject immediately
EMBEDDING_THRESHOLD = 0.30       # tuned threshold from earlier diagnosis
PERPLEXITY_THRESHOLD = 0.20      # tuned threshold from earlier diagnosis

CATEGORIES = ["injection", "authority", "corpus", "perplexity"]


def staged_verdict(pattern_score, embedding_score, perplexity_score):
    """
    Returns (verdict, stages_run, total_latency_contribution_keys)
    Simulates the cascade -- does NOT re-run detectors, just decides which
    stages "would have" executed based on already-computed scores.
    """
    stages_run = ["pattern"]
    if pattern_score >= PATTERN_REJECT_THRESHOLD:
        return "Reject", stages_run

    stages_run.append("embedding")
    if embedding_score >= EMBEDDING_THRESHOLD:
        return "Flag/Reject", stages_run

    stages_run.append("perplexity")
    if perplexity_score >= PERPLEXITY_THRESHOLD:
        return "Flag/Reject", stages_run

    return "Allow", stages_run


def main():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total_staged_latency = 0.0
    total_always_all_latency = 0.0
    stage_usage_counts = {"pattern": 0, "embedding": 0, "perplexity": 0}

    results = []
    for r in rows:
        pattern_score = float(r["pattern_score"])
        embedding_score = float(r["embedding_score"])
        perplexity_score = float(r["perplexity_score"])

        verdict, stages_run = staged_verdict(pattern_score, embedding_score, perplexity_score)

        staged_latency = sum(float(r[f"{stage}_latency_ms"]) for stage in stages_run)
        always_all_latency = (
            float(r["pattern_latency_ms"])
            + float(r["embedding_latency_ms"])
            + float(r["perplexity_latency_ms"])
        )

        total_staged_latency += staged_latency
        total_always_all_latency += always_all_latency
        for stage in stages_run:
            stage_usage_counts[stage] += 1

        results.append({
            "filename": r["filename"],
            "category": r["category"],
            "true_label": r["true_label"],
            "verdict": verdict,
            "stages_run": "+".join(stages_run),
        })

    n = len(rows)
    print(f"Total documents: {n}")
    print("\n=== Stage usage (how often each stage actually ran) ===")
    for stage, count in stage_usage_counts.items():
        print(f"{stage:12s} ran on {count}/{n} docs ({count/n:.0%})")

    print(f"\n=== Latency comparison ===")
    print(f"Always run all 3:  total={total_always_all_latency:.1f}ms  avg/doc={total_always_all_latency/n:.2f}ms")
    print(f"Staged pipeline:   total={total_staged_latency:.1f}ms  avg/doc={total_staged_latency/n:.2f}ms")
    savings_pct = (1 - total_staged_latency / total_always_all_latency) * 100
    print(f"Time saved: {savings_pct:.1f}%")

    print(f"\n=== Accuracy of staged pipeline (treating Flag/Reject as detection) ===")
    tp = sum(1 for r in results if r["true_label"] == "1" and r["verdict"] != "Allow")
    fp = sum(1 for r in results if r["true_label"] == "0" and r["verdict"] != "Allow")
    fn = sum(1 for r in results if r["true_label"] == "1" and r["verdict"] == "Allow")
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"precision={precision:.2f}  recall={recall:.2f}  f1={f1:.2f}  (tp={tp} fp={fp} fn={fn})")
    print("(compare to calibrated fixed fusion f1=0.97 -- staged pipeline uses a simpler")
    print(" cascade rule rather than a weighted combination, so some difference is expected)")

    print(f"\n=== Failure map (staged pipeline) ===")
    for cat in CATEGORIES:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        rate = sum(1 for r in cat_results if r["verdict"] != "Allow") / len(cat_results)
        print(f"{cat:12s} {rate:>10.0%}")


if __name__ == "__main__":
    main()