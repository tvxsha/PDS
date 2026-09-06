"""
Finds the specific documents the calibrated fusion still misses at its
best threshold, so you can actually read them and understand WHY they're
hard -- much stronger for your report than just citing a percentage.

Usage: python -m eval.find_remaining_misses
"""

import csv

SCORE_KEYS = ["pattern_score", "embedding_score", "perplexity_score"]
BEST_THRESHOLD = 0.15  # from eval/calibrated_fusion.py's sweep -- update if you regenerate data


def minmax_normalize(rows, score_key):
    values = [float(r[score_key]) for r in rows]
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    return {id(r): (float(r[score_key]) - lo) / span for r in rows}


def main():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    normalized = {key: minmax_normalize(rows, key) for key in SCORE_KEYS}
    for r in rows:
        r["calibrated_fusion_score"] = sum(normalized[key][id(r)] for key in SCORE_KEYS) / len(SCORE_KEYS)

    misses = [
        r for r in rows
        if r["true_label"] == "1" and r["calibrated_fusion_score"] < BEST_THRESHOLD
    ]

    print(f"Found {len(misses)} missed poisoned docs at threshold {BEST_THRESHOLD}:\n")
    for r in misses:
        print(f"--- {r['filename']} (category={r['category']}) ---")
        print(f"  raw scores: pattern={r['pattern_score']}  embedding={r['embedding_score']}  "
              f"perplexity={r['perplexity_score']}")
        print(f"  calibrated combined: {r['calibrated_fusion_score']:.3f}")
        # print the actual document content so you can read what it says
        try:
            with open(f"data/poisoned/{r['filename']}", encoding="utf-8") as doc_file:
                print(f"  content: {doc_file.read()}")
        except FileNotFoundError:
            print("  (could not locate original file to print content)")
        print()


if __name__ == "__main__":
    main()