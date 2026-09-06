"""
Learned fusion: a logistic regression trained on the 3 detectors' RAW
scores (not pre-calibrated) to predict poisoned vs. clean.

Deliberately trained on raw scores rather than the min-max calibrated
ones used in fusion/fixed.py's calibrated version -- the point of this
module is to test whether a learned model can discover appropriate
per-detector weighting automatically (compensating for embedding's
compressed 0.18-0.43 range on its own), without needing us to manually
calibrate first. If it matches or beats calibrated fixed fusion, that's
a genuinely interesting finding: manual calibration wasn't necessary,
a learned model finds it on its own. If it falls short, that's equally
interesting: manual calibration captured something a simple linear
model couldn't.
"""

from sklearn.linear_model import LogisticRegression

FEATURE_NAMES = ["pattern_score", "embedding_score", "perplexity_score"]


def build_model() -> LogisticRegression:
    """A fresh, untrained model with a fixed random state for reproducibility."""
    return LogisticRegression(random_state=42)


def get_feature_importance(model: LogisticRegression) -> dict:
    """
    Returns each feature's learned coefficient. Larger magnitude = more
    influence on the final decision. Sign matters too: positive means
    higher detector score pushes toward "poisoned".
    """
    coefs = model.coef_[0]
    return {name: float(coef) for name, coef in zip(FEATURE_NAMES, coefs)}