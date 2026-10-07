"""
Tests a veto mechanism: flag a document if EITHER
  (a) the whole-document embedding score crosses its own tuned threshold (0.30), OR
  (b) any single segment's raw anomaly score crosses a separate, higher
      "veto" threshold, regardless of the document-level average.

This differs from pure max-aggregation (which replaces the whole-document
score entirely) and from blending (which averages max and mean into one
number). Here, the whole-document detector still does most of the work,
and the veto only fires for segments so extreme that they're worth
flagging on their own -- if that bar can be set high enough to avoid
new false positives.

IMPORTANT CHECK BUILT IN: this script explicitly reports where adv_010's
worst segment score (0.319) falls relative to clean documents' worst
segment scores. If clean documents commonly produce higher single-segment
scores than adv_010 does, no veto threshold can catch adv_010 without
also flagging those clean documents -- which would mean the veto approach
is fundamentally unable to fix this specific case, not just untuned.

Usage: python -m eval.veto_mechanism_test
"""

import glob
import os
import random

from detectors import segmented_embedding_detector, embedding_detector

random.seed(42)

CATEGORIES = ["injection", "authority", "corpus", "perplexity"]
WHOLE_DOC_THRESHOLD = 0.30  # whole-document detector's own tuned threshold


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

    print(f"Scoring {len(test_set)} documents (whole-document score + max segment score)...")
    rows = []
    for fname, text, true_label, category in test_set:
        whole_doc_result = embedding_detector.score_document(text, baseline_corpus)
        segmented_result = segmented_embedding_detector.score_document(text, baseline_corpus)
        rows.append({
            "filename": fname, "category": category, "true_label": true_label,
            "whole_doc_score": whole_doc_result.score,
            "max_segment_score": segmented_result.score,  # pure max, already computed
        })

    # --- The critical sanity check, before any threshold sweeping ---
    print("\n" + "=" * 60)
    print("SANITY CHECK: where does adv_010 fall vs clean docs' max-segment scores?")
    print("=" * 60)
    clean_max_scores = sorted([r["max_segment_score"] for r in rows if r["category"] == "clean"], reverse=True)
    print(f"Clean docs' max-segment scores (highest 5): {[f'{s:.3f}' for s in clean_max_scores[:5]]}")

    adv_010_path = "data/adversarial/adv_010.txt"
    adv_010_max_score = None
    if os.path.exists(adv_010_path):
        with open(adv_010_path, encoding="utf-8") as f:
            text = f.read()
        result = segmented_embedding_detector.score_document(text, baseline_corpus)
        adv_010_max_score = result.score
        print(f"adv_010.txt max-segment score: {adv_010_max_score:.3f}")
        n_clean_above = sum(1 for s in clean_max_scores if s >= adv_010_max_score)
        print(f"{n_clean_above}/{len(clean_max_scores)} clean documents score AT LEAST as high as adv_010 "
              f"on this metric.")
        if n_clean_above > 0:
            print("This means NO veto threshold can catch adv_010 without also flagging "
                  "at least that many clean documents -- the veto approach cannot cleanly "
                  "separate this case, regardless of tuning.")

    # --- Still run the sweep, to see the actual best achievable tradeoff ---
    print("\n" + "=" * 60)
    print("VETO RULE -- threshold sweep (whole-doc >= 0.30 OR max-segment >= veto_threshold)")
    print("=" * 60)
    print(f"{'veto_thresh':>12s} {'precision':>10s} {'recall':>8s} {'f1':>8s}")
    best_f1, best_t = -1, None
    for t in [i / 20 for i in range(1, 20)]:
        tp = fp = fn = 0
        for r in rows:
            flagged = (r["whole_doc_score"] >= WHOLE_DOC_THRESHOLD) or (r["max_segment_score"] >= t)
            if r["true_label"] == 1 and flagged:
                tp += 1
            elif r["true_label"] == 0 and flagged:
                fp += 1
            elif r["true_label"] == 1 and not flagged:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        marker = ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
            marker = "  <-- best so far"
        print(f"{t:>12.2f} {precision:>10.2f} {recall:>8.2f} {f1:>8.2f}{marker}")

    print(f"\nVETO rule best: veto_threshold={best_t:.2f}  f1={best_f1:.2f}")
    print("Compare to: whole-document f1=0.92, pure-max segmented f1=0.90, blended f1=0.90")

    if adv_010_max_score is not None:
        caught = adv_010_max_score >= best_t
        print(f"\nAt this best veto threshold, adv_010.txt would be: "
              f"{'CAUGHT' if caught else 'still evaded'}")


if __name__ == "__main__":
    main()