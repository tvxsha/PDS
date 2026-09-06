"""
Fixed-weight fusion: combines the 3 detector scores into a single verdict
using a simple weighted average. This is your baseline fusion method --
compare it against the learned fusion (fusion/learned.py, built later)
to see whether learned weighting actually beats hand-picked weights.
"""

from dataclasses import dataclass

# Starting weights -- equal by default. Once you have real evaluation
# results, you can hand-tune these (e.g. if perplexity turns out least
# reliable, lower its weight) and compare against the learned version.
DEFAULT_WEIGHTS = {
    "instruction_pattern": 1.0,
    "embedding_anomaly": 1.0,
    "perplexity_fluency": 1.0,
}

# Verdict thresholds on the final combined score (0-1 scale)
FLAG_THRESHOLD = 0.3
REJECT_THRESHOLD = 0.6


@dataclass
class FusionResult:
    combined_score: float
    verdict: str  # "Allow" / "Flag for review" / "Reject"
    component_scores: dict


def combine_scores(detector_results: list, weights: dict = None) -> FusionResult:
    """
    detector_results: list of DetectorResult objects (from detectors/base.py),
    one per detector, for a single document.
    """
    weights = weights or DEFAULT_WEIGHTS
    component_scores = {r.detector_name: r.score for r in detector_results}

    total_weight = sum(weights.get(name, 0) for name in component_scores)
    if total_weight == 0:
        combined = 0.0
    else:
        combined = sum(
            component_scores[name] * weights.get(name, 0)
            for name in component_scores
        ) / total_weight

    if combined >= REJECT_THRESHOLD:
        verdict = "Reject"
    elif combined >= FLAG_THRESHOLD:
        verdict = "Flag for review"
    else:
        verdict = "Allow"

    return FusionResult(combined_score=combined, verdict=verdict, component_scores=component_scores)
