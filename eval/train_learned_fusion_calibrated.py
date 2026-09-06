"""
Same as train_learned_fusion.py, but trains on MIN-MAX NORMALIZED scores
(same calibration as fusion/fixed.py's calibrated version) instead of raw
scores. Tests the hypothesis that the plain learned-fusion model
underperformed because it wasn't given properly scaled inputs -- i.e.
that manual calibration wasn't just a nice-to-have, it addressed something
the model couldn't fully compensate for on its own via regularized
coefficients alone.

Usage: python -m eval.train_learned_fusion_calibrated
"""

import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from fusion.learned import build_model, get_feature_importance, FEATURE_NAMES

CATEGORIES = ["injection", "authority", "corpus", "perplexity"]


def minmax_normalize_column(values):
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    return [(v - lo) / span for v in values]


def main():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # build calibrated (min-max normalized per column) feature matrix
    raw_columns = {feat: [float(r[feat]) for r in rows] for feat in FEATURE_NAMES}
    calibrated_columns = {feat: minmax_normalize_column(vals) for feat, vals in raw_columns.items()}

    X = np.array([[calibrated_columns[feat][i] for feat in FEATURE_NAMES] for i in range(len(rows))])
    y = np.array([int(r["true_label"]) for r in rows])

    print(f"Dataset: {len(y)} docs ({sum(y)} poisoned, {len(y) - sum(y)} clean)")
    print("(features are min-max normalized per detector before training)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = build_model()
    oof_probs = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]

    full_model = build_model()
    full_model.fit(X, y)
    importances = get_feature_importance(full_model)

    print("\n" + "=" * 60)
    print("LEARNED FEATURE IMPORTANCE ON CALIBRATED INPUTS")
    print("=" * 60)
    for name, coef in sorted(importances.items(), key=lambda x: -abs(x[1])):
        print(f"{name:18s} coefficient={coef:+.3f}")

    print("\n" + "=" * 60)
    print("LEARNED FUSION ON CALIBRATED INPUTS -- threshold sweep")
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

    print(f"\nBEST learned fusion (calibrated inputs): threshold={best_t:.2f}  f1={best_f1:.2f}")
    print("Compare to: naive fixed=0.70, calibrated fixed=0.97, learned on raw=0.88")

    print("\n" + "=" * 60)
    print(f"FAILURE MAP at best threshold ({best_t:.2f})")
    print("=" * 60)
    for cat in CATEGORIES:
        cat_indices = [i for i, r in enumerate(rows) if r["category"] == cat]
        if not cat_indices:
            continue
        cat_probs = oof_probs[cat_indices]
        rate = (cat_probs >= best_t).mean()
        print(f"{cat:12s} {rate:>10.0%}")


if __name__ == "__main__":
    main()