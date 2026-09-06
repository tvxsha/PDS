"""
Full evaluation run: scores every document in the dataset with all 3
detectors + fixed fusion, then computes:
  - per-detector precision/recall/F1
  - combined-system precision/recall/F1
  - per-attack-category detection rate per detector (the failure map)
  - latency stats per detector
  - a raw results CSV you can dig into further or hand to a stats tool

Usage: python -m eval.run_evaluation
(needs internet on first run, for model downloads -- same as the detector
smoke tests)
"""

import csv
import glob
import os
import random
from collections import defaultdict

from detectors import pattern_detector, embedding_detector, perplexity_detector
from fusion.fixed import combine_scores

random.seed(42)

CATEGORIES = ["injection", "authority", "corpus", "perplexity"]


def load_docs(path_glob):
    docs = []
    for fpath in sorted(glob.glob(path_glob)):
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
        docs.append((os.path.basename(fpath), text))
    return docs


def get_category(filename: str, is_poisoned: bool) -> str:
    if not is_poisoned:
        return "clean"
    for cat in CATEGORIES:
        if filename.startswith(cat):
            return cat
    return "unknown"


def main():
    print("Loading dataset...")
    clean_docs = load_docs("data/clean/*.txt")
    poisoned_docs = load_docs("data/poisoned/*.txt")
    print(f"  {len(clean_docs)} clean, {len(poisoned_docs)} poisoned")

    # Split clean docs: baseline (for embedding detector reference) vs held-out test set.
    # This avoids the embedding detector trivially matching a doc to itself.
    random.shuffle(clean_docs)
    split_point = int(len(clean_docs) * 0.7)
    baseline_clean = clean_docs[:split_point]
    test_clean = clean_docs[split_point:]
    print(f"  Using {len(baseline_clean)} clean docs as embedding baseline, "
          f"{len(test_clean)} held out for testing")

    print("Building embedding baseline corpus (this downloads the model on first run)...")
    baseline_texts = [text for _, text in baseline_clean]
    baseline_corpus = embedding_detector.BaselineCorpus(baseline_texts)

    print("Warming up perplexity model...")
    perplexity_detector._get_model_and_tokenizer()

    # Build the full test set: held-out clean docs (label=0) + all poisoned docs (label=1)
    test_set = [(fname, text, 0, "clean") for fname, text in test_clean]
    test_set += [
        (fname, text, 1, get_category(fname, is_poisoned=True))
        for fname, text in poisoned_docs
    ]

    print(f"Scoring {len(test_set)} documents with all 3 detectors...")
    rows = []
    for i, (fname, text, true_label, category) in enumerate(test_set):
        pattern_result = pattern_detector.score_document(text)
        embedding_result = embedding_detector.score_document(text, baseline_corpus)
        perplexity_result = perplexity_detector.score_document(text)

        fusion_result = combine_scores([pattern_result, embedding_result, perplexity_result])

        rows.append({
            "filename": fname,
            "category": category,
            "true_label": true_label,  # 1 = poisoned, 0 = clean
            "pattern_score": pattern_result.score,
            "pattern_latency_ms": pattern_result.latency_ms,
            "embedding_score": embedding_result.score,
            "embedding_latency_ms": embedding_result.latency_ms,
            "perplexity_score": perplexity_result.score,
            "perplexity_latency_ms": perplexity_result.latency_ms,
            "fixed_fusion_score": fusion_result.combined_score,
            "fixed_fusion_verdict": fusion_result.verdict,
        })

        if (i + 1) % 20 == 0:
            print(f"  scored {i + 1}/{len(test_set)}...")

    # Save raw results
    os.makedirs("eval/results", exist_ok=True)
    out_path = "eval/results/evaluation_raw.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nRaw results saved to {out_path}")

    # ---- Compute metrics ----
    print("\n" + "=" * 60)
    print("PER-DETECTOR METRICS (threshold = 0.5 on individual score)")
    print("=" * 60)
    for detector_key, score_key in [
        ("Pattern", "pattern_score"),
        ("Embedding", "embedding_score"),
        ("Perplexity", "perplexity_score"),
    ]:
        tp = sum(1 for r in rows if r["true_label"] == 1 and r[score_key] >= 0.5)
        fp = sum(1 for r in rows if r["true_label"] == 0 and r[score_key] >= 0.5)
        fn = sum(1 for r in rows if r["true_label"] == 1 and r[score_key] < 0.5)
        tn = sum(1 for r in rows if r["true_label"] == 0 and r[score_key] < 0.5)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        avg_latency = sum(r[f"{score_key.split('_')[0]}_latency_ms"] for r in rows) / len(rows)
        print(f"{detector_key:12s}  precision={precision:.2f}  recall={recall:.2f}  "
              f"f1={f1:.2f}  avg_latency={avg_latency:.1f}ms  (tp={tp} fp={fp} fn={fn} tn={tn})")

    print("\n" + "=" * 60)
    print("COMBINED (FIXED FUSION) METRICS")
    print("=" * 60)
    tp = sum(1 for r in rows if r["true_label"] == 1 and r["fixed_fusion_verdict"] != "Allow")
    fp = sum(1 for r in rows if r["true_label"] == 0 and r["fixed_fusion_verdict"] != "Allow")
    fn = sum(1 for r in rows if r["true_label"] == 1 and r["fixed_fusion_verdict"] == "Allow")
    tn = sum(1 for r in rows if r["true_label"] == 0 and r["fixed_fusion_verdict"] == "Allow")
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print(f"(counting 'Flag for review' or 'Reject' as a detection)")
    print(f"precision={precision:.2f}  recall={recall:.2f}  f1={f1:.2f}  (tp={tp} fp={fp} fn={fn} tn={tn})")

    print("\n" + "=" * 60)
    print("FAILURE MAP: per-category detection rate per detector (threshold 0.5)")
    print("=" * 60)
    print(f"{'Category':12s} {'Pattern':>10s} {'Embedding':>10s} {'Perplexity':>10s} {'Fixed Fusion':>13s}")
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        n = len(cat_rows)
        pattern_rate = sum(1 for r in cat_rows if r["pattern_score"] >= 0.5) / n
        embedding_rate = sum(1 for r in cat_rows if r["embedding_score"] >= 0.5) / n
        perplexity_rate = sum(1 for r in cat_rows if r["perplexity_score"] >= 0.5) / n
        fusion_rate = sum(1 for r in cat_rows if r["fixed_fusion_verdict"] != "Allow") / n
        print(f"{cat:12s} {pattern_rate:>10.0%} {embedding_rate:>10.0%} {perplexity_rate:>10.0%} {fusion_rate:>13.0%}")

    # Blind spot check: attacks that NO detector caught individually
    print("\n" + "=" * 60)
    print("BLIND SPOTS: poisoned docs where ALL THREE detectors scored < 0.5")
    print("=" * 60)
    blind_spots = [
        r for r in rows
        if r["true_label"] == 1
        and r["pattern_score"] < 0.5
        and r["embedding_score"] < 0.5
        and r["perplexity_score"] < 0.5
    ]
    if blind_spots:
        for r in blind_spots:
            print(f"  {r['filename']} (category={r['category']})")
    else:
        print("  None -- every poisoned doc was caught by at least one detector.")

    print(f"\nDone. Full results in {out_path} for further analysis.")


if __name__ == "__main__":
    main()
