"""
Tunes the threshold for the blended (max + mean) segmented aggregation and
compares its own best F1 against both prior results:
  - whole-document detector's own best: F1=0.92 (threshold=0.30)
  - pure-max segmented detector's own best: F1=0.90 (threshold=0.30)

Usage: python -m eval.tune_blended_segmented
"""

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

    print(f"Scoring {len(test_set)} documents with the BLENDED detector (max_weight=0.5)...")
    rows = []
    for fname, text, true_label, category in test_set:
        result = segmented_embedding_detector.score_document_blended(text, baseline_corpus)
        rows.append({"filename": fname, "category": category, "true_label": true_label,
                     "blended_score": result.score})

    print("\n" + "=" * 60)
    print("BLENDED DETECTOR (0.5*max + 0.5*mean) -- threshold sweep")
    print("=" * 60)
    print(f"{'threshold':>10s} {'precision':>10s} {'recall':>8s} {'f1':>8s}")
    best_f1, best_t = -1, None
    for t in [i / 20 for i in range(1, 20)]:
        tp = sum(1 for r in rows if r["true_label"] == 1 and r["blended_score"] >= t)
        fp = sum(1 for r in rows if r["true_label"] == 0 and r["blended_score"] >= t)
        fn = sum(1 for r in rows if r["true_label"] == 1 and r["blended_score"] < t)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        marker = ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
            marker = "  <-- best so far"
        print(f"{t:>10.2f} {precision:>10.2f} {recall:>8.2f} {f1:>8.2f}{marker}")

    print(f"\nBLENDED detector best: threshold={best_t:.2f}  f1={best_f1:.2f}")
    print("Compare to: whole-document f1=0.92, pure-max segmented f1=0.90")

    print("\n" + "=" * 60)
    print(f"Failure map at blended detector's best threshold ({best_t:.2f})")
    print("=" * 60)
    for cat in CATEGORIES:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        rate = sum(1 for r in cat_rows if r["blended_score"] >= best_t) / len(cat_rows)
        print(f"{cat:12s} {rate:>10.0%}")

    # also check the specific adversarial case that motivated this whole line of work
    print("\n" + "=" * 60)
    print("Checking adv_010.txt specifically (the boilerplate-anchoring case)")
    print("=" * 60)
    adv_010_path = "data/adversarial/adv_010.txt"
    if os.path.exists(adv_010_path):
        with open(adv_010_path, encoding="utf-8") as f:
            text = f.read()
        result = segmented_embedding_detector.score_document_blended(text, baseline_corpus)
        caught = "CAUGHT" if result.score >= best_t else "still evaded"
        print(f"adv_010.txt blended score: {result.score:.3f}  ({caught} at threshold {best_t:.2f})")


if __name__ == "__main__":
    main()