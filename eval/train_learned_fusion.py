"""
Trains and evaluates the learned fusion model using 5-fold stratified
cross-validation, rather than a single train/test split.

Why cross-validation instead of a split: with only 77 test documents
(17 clean + 60 poisoned across 4 categories), a single 70/30 split would
leave only ~4-5 examples per category in the test set -- too few to trust.
Cross-validation gets an out-of-fold prediction for every document (never
predicted by a model that was trained on it), so we can compute real
metrics and a real failure map across the FULL dataset while still
avoiding train/test leakage. This is standard, defensible practice for
small datasets -- worth stating explicitly in your report's methodology.

Usage: python -m eval.train_learned_fusion
"""

import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fusion.learned import build_model, get_feature_importance, FEATURE_NAMES

CATEGORIES = ["injection", "authority", "corpus", "perplexity"]


def main():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X = np.array([[float(r[feat]) for feat in FEATURE_NAMES] for r in rows])
    y = np.array([int(r["true_label"]) for r in rows])

    print(f"Dataset: {len(y)} docs ({sum(y)} poisoned, {len(y) - sum(y)} clean)")

    # 5-fold stratified CV: each fold keeps the same poisoned/clean ratio
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = build_model()

    # out-of-fold predicted probabilities -- each doc scored by a model
    # that never saw it during training
    oof_probs = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

    for r, prob in zip(rows, oof_probs):
        r["learned_fusion_score"] = prob

    # Now fit on the FULL dataset for interpretability (feature importance) --
    # this final model isn't used for evaluation metrics above, only to
    # inspect which signal it learned to weight most heavily
    full_model = build_model()
    full_model.fit(X, y)
    importances = get_feature_importance(full_model)

    print("\n" + "=" * 60)
    print("LEARNED FEATURE IMPORTANCE (coefficients from full-dataset fit)")
    print("=" * 60)
    for name, coef in sorted(importances.items(), key=lambda x: -abs(x[1])):
        print(f"{name:18s} coefficient={coef:+.3f}")
    print("(larger magnitude = more influence; positive = higher score -> more likely poisoned)")

    print("\n" + "=" * 60)
    print("LEARNED FUSION -- threshold sweep (on out-of-fold predictions)")
    print("=" * 60)
    print(f"{'threshold':>10s} {'precision':>10s} {'recall':>8s} {'f1':>8s}")
    best_f1, best_t = -1, None
    for t in [i / 20 for i in range(1, 20)]:
        preds = (oof_probs >= t).astype(int)
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        marker = ""
        if f1 > best_f1:
            best_f1, best_t = f1, t
            marker = "  <-- best so far"
        print(f"{t:>10.2f} {precision:>10.2f} {recall:>8.2f} {f1:>8.2f}{marker}")

    print(f"\nBEST learned fusion: threshold={best_t:.2f}  f1={best_f1:.2f}")
    print("(compare this to calibrated fixed fusion's f1=0.97 at threshold=0.15)")

    print("\n" + "=" * 60)
    print(f"FAILURE MAP at best learned-fusion threshold ({best_t:.2f})")
    print("=" * 60)
    for cat in CATEGORIES:
        cat_indices = [i for i, r in enumerate(rows) if r["category"] == cat]
        if not cat_indices:
            continue
        cat_probs = oof_probs[cat_indices]
        rate = (cat_probs >= best_t).mean()
        print(f"{cat:12s} {rate:>10.0%}")

    # save for further inspection if needed
    with open("eval/results/learned_fusion_scores.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "category", "true_label", "learned_fusion_score"])
        for r in rows:
            writer.writerow([r["filename"], r["category"], r["true_label"], r["learned_fusion_score"]])
    print("\nOut-of-fold scores saved to eval/results/learned_fusion_scores.csv")


if __name__ == "__main__":
    main()