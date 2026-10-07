"""
The segmented detector was tested against the whole-document detector's
threshold (0.30) and looked worse on false positives -- but that's not a
fair comparison. Max-aggregation across segments naturally produces higher
scores than whole-document averaging, even on clean text, purely because
taking a maximum over several samples tends to be larger than their mean.
Comparing it to a threshold tuned for a different scoring method isn't
meaningful.

This script runs the segmented detector across the FULL labeled evaluation
set (same 17 clean + 60 poisoned split as run_evaluation.py) and sweeps
thresholds to find the segmented detector's OWN best operating point, so
it can be fairly compared against the whole-document detector's own best
F1 (0.92 at threshold 0.30, from eval/threshold_sweep.py).

Usage: python -m eval.tune_segmented_threshold
"""

import csv
import glob
import os
import random

from detectors import segmented_embedding_detector, embedding_detector

random.seed(42)

CATEGORIES = ["injection", "authority", "corpus", "perplexity"]


def rebuild_baseline_and_test_sets():
    clean_paths = sorted(glob.glob("data/clean/*.txt"))
    clean_docs = []
    for fpath in clean_paths:
        with open(fpath, encoding="utf-8") as f:
            clean_docs.append((os.path.basename(fpath), f.read()))
    random.shuffle(clean_docs)
    split_point = int(len(clean_docs) * 0.7)
    baseline_clean = clean_docs[:split_point]
    test_clean = clean_docs[split_point:]
    baseline_texts = [text for _, text in baseline_clean]
    baseline_corpus = embedding_detector.BaselineCorpus(baseline_texts)
    return baseline_corpus, test_clean


def get_category(filename, is_poisoned):
    if not is_poisoned:
        return "clean"
    for cat in CATEGORIES:
        if filename.startswith(cat):
            return cat
    return "unknown"


def main():
    print("Rebuilding baseline and test set...")
    baseline_corpus, test_clean = rebuild_baseline_and_test_sets()

    poisoned_docs = []
    for fpath in sorted(glob.glob("data/poisoned/*.txt")):
        with open(fpath, encoding="utf-8") as f:
            poisoned_docs.append((os.path.basename(fpath), f.read()))

    test_set = [(fname, text, 0, "clean") for fname, text in test_clean]
    test_set += [(fname, text, 1, get_category(fname, True)) for fname, text in poisoned_docs]

    print(f"Scoring {len(test_set)} documents with the SEGMENTED detector...")
    rows = []
    for fname, text, true_label, category in test_set:
        result = segmented_embedding_detector.score_document(text, baseline_corpus)
        rows.append({"filename": fname, "category": category, "true_label": true_label,
                     "segmented_score": result.score})

    print("\n" + "=" * 60)
    print("SEGMENTED DETECTOR -- own threshold sweep")
    print("=" * 60)
    print(f"{'threshold':>10s} {'precision':>10s} {'recall':>8s} {'f1':>8s}")
    best_f1, best_t = -1, None
    for t in [i / 20 for i in range(1, 20)]:
        tp = sum(1 for r in rows if r["true_label"] == 1 and r["segmented_score"] >= t)
        fp = sum(1 for r in rows if r["true_label"] == 0 and r["segmented_score"] >= t)
        fn = sum(1 for r in rows if r["true_label"] == 1 and r["segmented_score"] < t)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        marker = ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
            marker = "  <-- best so far"
        print(f"{t:>10.2f} {precision:>10.2f} {recall:>8.2f} {f1:>8.2f}{marker}")

    print(f"\nSEGMENTED detector best: threshold={best_t:.2f}  f1={best_f1:.2f}")
    print("Compare to WHOLE-DOCUMENT detector's own best: f1=0.92 at threshold=0.30")

    print("\n" + "=" * 60)
    print(f"Failure map at segmented detector's own best threshold ({best_t:.2f})")
    print("=" * 60)
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        rate = sum(1 for r in cat_rows if r["segmented_score"] >= best_t) / len(cat_rows)
        print(f"{cat:12s} {rate:>10.0%}")


if __name__ == "__main__":
    main()