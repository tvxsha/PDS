"""
Run this as you build the dataset to check progress against the targets
in data/README.md. No dependencies -- just counts files.

Usage (from the pds/ folder): python -m eval.dataset_check
"""

import os
from collections import Counter

CLEAN_DIR = "data/clean"
POISONED_DIR = "data/poisoned"
TARGET_CLEAN = 50
TARGET_PER_CATEGORY = 12
CATEGORIES = ["injection", "authority", "corpus", "perplexity"]


def count_files(directory):
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".txt")]


def main():
    clean_files = count_files(CLEAN_DIR)
    poisoned_files = count_files(POISONED_DIR)

    print(f"Clean docs: {len(clean_files)} / {TARGET_CLEAN} target")

    category_counts = Counter()
    unrecognized = []
    for f in poisoned_files:
        matched = False
        for cat in CATEGORIES:
            if f.startswith(cat):
                category_counts[cat] += 1
                matched = True
                break
        if not matched:
            unrecognized.append(f)

    print("\nPoisoned docs by category:")
    for cat in CATEGORIES:
        n = category_counts[cat]
        flag = "  <-- needs more" if n < TARGET_PER_CATEGORY else ""
        print(f"  {cat}: {n} / {TARGET_PER_CATEGORY} target{flag}")

    if unrecognized:
        print(f"\nWarning: {len(unrecognized)} file(s) don't match a known category prefix:")
        for f in unrecognized:
            print(f"  {f}")

    total_poisoned = sum(category_counts.values())
    print(f"\nTotal poisoned: {total_poisoned}, total clean: {len(clean_files)}")


if __name__ == "__main__":
    main()
