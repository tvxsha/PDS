"""
Adversarial red-teaming: tests whether hand-crafted attacks, specifically
designed to mimic real README genre/vocabulary (to evade the embedding
detector), avoid all pattern-detector trigger phrases, and stay fluent
(to evade perplexity), can slip past the calibrated fusion pipeline.

IMPORTANT methodological point: this reuses the calibration bounds
(min/max per detector) computed from the ORIGINAL evaluation dataset,
not recomputed including these new adversarial docs. Recomputing bounds
using the very docs you're testing would be circular -- it would let an
extreme adversarial score simply stretch the range and look "normal" by
definition. A real deployed system calibrates once on known data, then
applies that fixed transform to new incoming documents -- this mirrors
that.

Also reconstructs the EXACT SAME embedding baseline corpus split used in
eval/run_evaluation.py (same random seed, same 70% split) so the
embedding detector is being tested under the same conditions as the main
evaluation, not a different/easier baseline.

Usage: python -m eval.adversarial_test
"""

import csv
import glob
import os
import random

from detectors import pattern_detector, embedding_detector, perplexity_detector

random.seed(42)  # MUST match eval/run_evaluation.py's seed exactly


def rebuild_baseline_corpus():
    """Reconstructs the identical baseline split used in run_evaluation.py."""
    clean_paths = sorted(glob.glob("data/clean/*.txt"))
    clean_docs = []
    for fpath in clean_paths:
        with open(fpath, encoding="utf-8") as f:
            clean_docs.append((os.path.basename(fpath), f.read()))
    random.shuffle(clean_docs)
    split_point = int(len(clean_docs) * 0.7)
    baseline_clean = clean_docs[:split_point]
    baseline_texts = [text for _, text in baseline_clean]
    return embedding_detector.BaselineCorpus(baseline_texts)


def load_calibration_bounds():
    """Reads min/max per detector from the original evaluation results --
    these bounds are FIXED and reused, not recalculated from new data."""
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bounds = {}
    for key in ["pattern_score", "embedding_score", "perplexity_score"]:
        values = [float(r[key]) for r in rows]
        bounds[key] = (min(values), max(values))
    return bounds


def calibrate(raw_score, lo, hi):
    """Applies the FIXED calibration transform. Clips to [0,1] since a new
    document can, in principle, score outside the original observed range --
    when that happens it's itself worth noting, not just silently clipped
    away."""
    span = hi - lo if hi > lo else 1.0
    normalized = (raw_score - lo) / span
    clipped = max(0.0, min(1.0, normalized))
    out_of_range = normalized != clipped
    return clipped, out_of_range


def main():
    print("Rebuilding embedding baseline (same split as main evaluation)...")
    baseline_corpus = rebuild_baseline_corpus()

    print("Loading fixed calibration bounds from original evaluation...")
    bounds = load_calibration_bounds()
    for key, (lo, hi) in bounds.items():
        print(f"  {key}: [{lo:.3f}, {hi:.3f}]")

    print("Warming up perplexity model...")
    perplexity_detector._get_model_and_tokenizer()

    adv_files = sorted(glob.glob("data/adversarial/*.txt"))
    if not adv_files:
        print("\nNo files found in data/adversarial/ -- create them first.")
        return

    print(f"\nScoring {len(adv_files)} adversarial documents...\n")

    CALIBRATED_THRESHOLD = 0.15  # matches the primary threshold chosen earlier
    evaded_count = 0

    for fpath in adv_files:
        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        pattern_result = pattern_detector.score_document(text)
        embedding_result = embedding_detector.score_document(text, baseline_corpus)
        perplexity_result = perplexity_detector.score_document(text)

        cal_pattern, oor_p = calibrate(pattern_result.score, *bounds["pattern_score"])
        cal_embedding, oor_e = calibrate(embedding_result.score, *bounds["embedding_score"])
        cal_perplexity, oor_pp = calibrate(perplexity_result.score, *bounds["perplexity_score"])

        combined = (cal_pattern + cal_embedding + cal_perplexity) / 3
        verdict = "CAUGHT" if combined >= CALIBRATED_THRESHOLD else "EVADED"
        if verdict == "EVADED":
            evaded_count += 1

        oor_note = ""
        if oor_e:
            oor_note = "  [embedding score OUTSIDE original calibration range -- notable]"

        print(f"--- {os.path.basename(fpath)}: {verdict}{oor_note} ---")
        print(f"  raw: pattern={pattern_result.score:.2f}  embedding={embedding_result.score:.3f}  "
              f"perplexity={perplexity_result.score:.2f}")
        print(f"  calibrated combined: {combined:.3f} (threshold={CALIBRATED_THRESHOLD})")
        print(f"  content: {text[:100]}...")
        print()

    print("=" * 60)
    print(f"EVASION RATE: {evaded_count}/{len(adv_files)} adversarial docs evaded detection "
          f"({evaded_count/len(adv_files):.0%})")
    print("=" * 60)


if __name__ == "__main__":
    main()