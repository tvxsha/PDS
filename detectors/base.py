"""
Shared interface for all PDS detectors.

Every detector module should expose a function that returns a DetectorResult
so the fusion layer and latency analysis can treat all three the same way.
"""

import time
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class DetectorResult:
    detector_name: str
    score: float          # 0.0 (clean) to 1.0 (poisoned) -- keep this scale consistent across all 3 detectors
    triggered_reason: str  # plain-English reason, feeds the explainability layer later
    latency_ms: float = 0.0
    raw_details: dict = field(default_factory=dict)  # anything extra you want to log for debugging


def timed(detector_name):
    """
    Decorator: wraps a detector's core scoring function, measures latency,
    and packages the return value into a DetectorResult automatically.

    Your scoring function should just return (score, reason, raw_details_dict).
    This decorator adds the name + latency so you don't have to repeat timing
    code in every module.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            score, reason, raw_details = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            return DetectorResult(
                detector_name=detector_name,
                score=score,
                triggered_reason=reason,
                latency_ms=elapsed_ms,
                raw_details=raw_details or {},
            )
        return wrapper
    return decorator
