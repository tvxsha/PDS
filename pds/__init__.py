"""
PDS: Poison Detection System.

The installable entry point to the project's three detectors (instruction
pattern, embedding anomaly, perplexity fluency) and fixed-weight fusion,
so other code -- a demo app, a RAG pipeline, a notebook -- can do:

    import pds
    result = pds.scan("some document text")
    print(result.verdict, result.reasons, result.latency_ms)

instead of wiring up the detectors and fusion layer by hand each time.

This wraps the EXISTING detectors/ and fusion/ modules (unchanged) -- it
doesn't reimplement any scoring logic, just gives it one clean call.

Run from the repo root (same convention as everything in eval/): the
embedding detector's default baseline corpus is built from data/clean/*.txt
relative to the current working directory, same as eval/run_evaluation.py
and eval/adversarial_test.py already assume. Pass your own `baseline` to
scan() if you're calling this from somewhere else.
"""

import glob
from dataclasses import dataclass, field

from detectors import pattern_detector, embedding_detector, perplexity_detector
from detectors.base import DetectorResult
from fusion.fixed import combine_scores, FusionResult

__version__ = "0.1.0"
__all__ = ["scan", "ScanResult", "build_baseline_from_dir"]

_default_baseline = None  # lazy-built once, reused across calls -- see embedding_detector.BaselineCorpus


@dataclass
class ScanResult:
    """
    What pds.scan() returns. The three fields the task asked for --
    verdict, reasons, latency_ms -- are plain attributes (not a 3-tuple:
    this dataclass also carries the combined score and the raw per-detector
    results, which the demo app and any caller debugging a verdict will want).

    verdict: "Allow" / "Flag for review" / "Reject"  (from fusion/fixed.py)
    reasons: one human-readable reason per detector, in detector order
    latency_ms: total wall-clock time across all 3 detectors for this scan
    combined_score: the fused 0-1 score fusion/fixed.py computed the verdict from
    detector_results: the raw DetectorResult objects, one per detector, if you
        need per-detector scores/latencies rather than just the summary
    """

    verdict: str
    reasons: list = field(default_factory=list)
    latency_ms: float = 0.0
    combined_score: float = 0.0
    detector_results: list = field(default_factory=list)


def build_baseline_from_dir(path: str = "data/clean/*.txt") -> embedding_detector.BaselineCorpus:
    """Builds a BaselineCorpus from every .txt file matching `path` (glob pattern)."""
    paths = sorted(glob.glob(path))
    if not paths:
        raise FileNotFoundError(
            f"No clean baseline documents found at '{path}'. Run pds.scan() from the "
            f"repo root, or pass your own baseline= (see embedding_detector.BaselineCorpus)."
        )
    texts = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            texts.append(f.read())
    return embedding_detector.BaselineCorpus(texts)


def _get_default_baseline():
    global _default_baseline
    if _default_baseline is None:
        _default_baseline = build_baseline_from_dir()
    return _default_baseline


def scan(text: str, *, baseline=None, weights: dict = None) -> ScanResult:
    """
    Runs all 3 detectors on `text` and fuses them with fixed-weight fusion.

    baseline: an embedding_detector.BaselineCorpus to compare against. If
        omitted, lazily builds one from data/clean/*.txt the first time
        scan() is called (and reuses it for every call after that).
    weights: passed through to fusion.fixed.combine_scores -- see
        fusion/fixed.py's DEFAULT_WEIGHTS if you want to override them.
    """
    baseline = baseline or _get_default_baseline()

    results: list[DetectorResult] = [
        pattern_detector.score_document(text),
        embedding_detector.score_document(text, baseline),
        perplexity_detector.score_document(text),
    ]

    fusion_result: FusionResult = combine_scores(results, weights=weights)

    total_latency = sum(r.latency_ms for r in results)
    reasons = [r.triggered_reason for r in results]

    return ScanResult(
        verdict=fusion_result.verdict,
        reasons=reasons,
        latency_ms=total_latency,
        combined_score=fusion_result.combined_score,
        detector_results=results,
    )
