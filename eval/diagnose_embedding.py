"""
Diagnostic: checks whether the embedding detector's 0% recall is a
thresholding problem (scores are there but too low to cross 0.5) or a
genuine lack of signal (scores don't separate clean from poisoned at all).

Reads the already-generated eval/results/evaluation_raw.csv -- no need to
re-run the full evaluation.

Usage: python -m eval.diagnose_embedding
"""

import csv


def main():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    clean_scores = [float(r["embedding_score"]) for r in rows if r["true_label"] == "0"]
    poisoned_scores = [float(r["embedding_score"]) for r in rows if r["true_label"] == "1"]

    def stats(scores, label):
        if not scores:
            print(f"{label}: no data")
            return
        print(f"{label}: min={min(scores):.3f}  max={max(scores):.3f}  "
              f"mean={sum(scores)/len(scores):.3f}  n={len(scores)}")

    print("=== Embedding score distribution ===")
    stats(clean_scores, "Clean docs   ")
    stats(poisoned_scores, "Poisoned docs")

    print("\n=== Per-category breakdown (poisoned only) ===")
    for cat in ["injection", "authority", "corpus", "perplexity"]:
        cat_scores = [float(r["embedding_score"]) for r in rows if r["category"] == cat]
        stats(cat_scores, f"{cat:12s}")

    # Check: is there ANY separation, even below the 0.5 threshold?
    if clean_scores and poisoned_scores:
        clean_mean = sum(clean_scores) / len(clean_scores)
        poison_mean = sum(poisoned_scores) / len(poisoned_scores)
        print(f"\nClean mean: {clean_mean:.3f}  |  Poisoned mean: {poison_mean:.3f}")
        gap = poison_mean - clean_mean
        if gap > 0.05:
            print(f"There IS separation (gap={gap:.3f}), just compressed below the 0.5 threshold.")
            print("This is a THRESHOLDING problem -- fixable by lowering the threshold or rescaling.")
            suggested_threshold = (clean_mean + poison_mean) / 2
            print(f"Try a threshold around {suggested_threshold:.3f} instead of 0.5.")
        elif gap > 0.01:
            print(f"There's a small gap ({gap:.3f}) but it's weak -- may need a different scaling approach,")
            print("not just a threshold change.")
        else:
            print(f"Essentially NO separation (gap={gap:.3f}). This looks like a genuine limitation:")
            print("on this dataset, embedding similarity alone doesn't distinguish clean from poisoned docs.")
            print("This is a legitimate, reportable finding -- not a bug to fix.")


if __name__ == "__main__":
    main()