"""
Sanity check: confirms the perplexity detector actually flags the generated
garbled docs and leaves the clean docs alone, BEFORE you commit to using
this dataset for real evaluation. Run this on your machine (needs internet
for the GPT-2 download on first run).

Usage: python -m eval.validate_perplexity_category
"""

import glob
from detectors.perplexity_detector import score_document, _get_model_and_tokenizer


def main():
    print("Warming up model...")
    _get_model_and_tokenizer()

    clean_files = sorted(glob.glob("data/clean/*.txt"))[:10]  # sample, not all 55
    garbled_files = sorted(glob.glob("data/poisoned/perplexity_*.txt"))

    print(f"\n--- Clean docs (expect mostly LOW scores) ---")
    clean_scores = []
    for f in clean_files:
        text = open(f, encoding="utf-8").read()
        result = score_document(text)
        clean_scores.append(result.score)
        print(f"{f}: score={result.score:.2f}  ({result.triggered_reason})")

    print(f"\n--- Garbled docs (expect mostly HIGH scores) ---")
    garbled_scores = []
    for f in garbled_files:
        text = open(f, encoding="utf-8").read()
        result = score_document(text)
        garbled_scores.append(result.score)
        print(f"{f}: score={result.score:.2f}  ({result.triggered_reason})")

    avg_clean = sum(clean_scores) / len(clean_scores)
    avg_garbled = sum(garbled_scores) / len(garbled_scores)
    print(f"\nAverage clean score: {avg_clean:.2f}")
    print(f"Average garbled score: {avg_garbled:.2f}")

    if avg_garbled > avg_clean + 0.3:
        print("\nLooks good -- clear separation between clean and garbled scores.")
    else:
        print("\nWarning: scores don't separate clearly. May need to adjust thresholds")
        print("in perplexity_detector.py, or the corruption method needs to be stronger.")


if __name__ == "__main__":
    main()
