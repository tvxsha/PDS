"""
Tests the new segmented embedding detector against TWO things:
  1. All 10 adversarial docs -- does it now catch adv_010 (the one that
     evaded the whole-document detector)?
  2. The full main dataset (clean + poisoned) -- does segment-level scoring
     introduce NEW false positives on clean docs? This matters because
     scoring the "worst" segment is inherently more sensitive than
     averaging -- a single unusual sentence in an otherwise normal document
     could now trigger a false alarm that whole-document averaging would
     have smoothed over. This is the real cost/benefit test, not just
     "does it catch the one attack we already know about."

Usage: python -m eval.test_segmented_detector
"""

import glob
import os
import random

from detectors import segmented_embedding_detector, embedding_detector

random.seed(42)  # matches run_evaluation.py's split


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


def main():
    print("Rebuilding baseline (same split as main evaluation)...")
    baseline_corpus, test_clean = rebuild_baseline_and_test_sets()

    # --- Part 1: adversarial docs ---
    print("\n" + "=" * 60)
    print("PART 1: Adversarial docs -- whole-document vs segmented")
    print("=" * 60)
    adv_files = sorted(glob.glob("data/adversarial/*.txt"))
    for fpath in adv_files:
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
        whole_doc_result = embedding_detector.score_document(text, baseline_corpus)
        segmented_result = segmented_embedding_detector.score_document(text, baseline_corpus)
        improved = "  <-- IMPROVED" if segmented_result.score > whole_doc_result.score + 0.05 else ""
        print(f"{os.path.basename(fpath)}: whole_doc={whole_doc_result.score:.3f}  "
              f"segmented={segmented_result.score:.3f}{improved}")

    # --- Part 2: false positive check on held-out clean docs ---
    print("\n" + "=" * 60)
    print("PART 2: Held-out CLEAN docs -- checking for NEW false positives")
    print("=" * 60)
    EMBEDDING_THRESHOLD = 0.30  # same tuned threshold as the whole-document detector
    whole_doc_fp = 0
    segmented_fp = 0
    for fname, text in test_clean:
        whole_doc_result = embedding_detector.score_document(text, baseline_corpus)
        segmented_result = segmented_embedding_detector.score_document(text, baseline_corpus)

        whole_flagged = whole_doc_result.score >= EMBEDDING_THRESHOLD
        segmented_flagged = segmented_result.score >= EMBEDDING_THRESHOLD

        if whole_flagged:
            whole_doc_fp += 1
        if segmented_flagged:
            segmented_fp += 1

        if segmented_flagged and not whole_flagged:
            print(f"  NEW false positive from segmentation: {fname} "
                  f"(whole_doc={whole_doc_result.score:.3f}, segmented={segmented_result.score:.3f})")

    print(f"\nWhole-document false positives: {whole_doc_fp}/{len(test_clean)}")
    print(f"Segmented false positives: {segmented_fp}/{len(test_clean)}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("If Part 1 shows adv_010 improved and Part 2 shows no/few new false")
    print("positives, segment-level scoring is a genuine improvement, not just")
    print("a trade of one failure mode for another.")


if __name__ == "__main__":
    main()