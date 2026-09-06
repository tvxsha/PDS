from detectors.base import DetectorResult


def _short_reason(result: DetectorResult) -> str:
    """
    Turn one detector's full triggered_reason into a short phrase suitable
    for stitching into a single verdict sentence.
    """
    if result.detector_name == "instruction_pattern":
        # pattern_detector.py's reason already starts with "flagged: " when
        # triggered -- strip that since we add our own "Flagged:" prefix.
        reason = result.triggered_reason
        if reason.startswith("flagged: "):
            reason = reason[len("flagged: "):]
        return reason

    if result.detector_name == "embedding_anomaly":
        level = "high" if result.score > 0.6 else "moderate"
        return f"{level} embedding anomaly ({result.score:.2f})"

    if result.detector_name == "perplexity_fluency":
        if "garbled" in result.triggered_reason:
            return f"unusually garbled phrasing ({result.score:.2f})"
        if "smooth" in result.triggered_reason:
            return f"unusually smooth phrasing ({result.score:.2f})"
        return f"unusual fluency ({result.score:.2f})"

    # fallback for any future detector -- keep it working, not pretty
    return f"{result.detector_name} anomaly ({result.score:.2f})"


def format_verdict(
    pattern_result: DetectorResult,
    embedding_result: DetectorResult,
    perplexity_result: DetectorResult,
    threshold: float = 0.5,
) -> str:
    """
    Build one plain-English verdict string from the 3 detector results.

    Only a detector whose score is strictly above `threshold` contributes
    its reason to the sentence -- detectors at or below threshold are
    treated as "didn't flag anything" and are left out entirely.

    Example:
        format_verdict(pattern_result, embedding_result, perplexity_result)
        -> "Flagged: high embedding anomaly (0.38) + contains override phrase"
    """
    results = [pattern_result, embedding_result, perplexity_result]
    triggered = [r for r in results if r.score > threshold]

    if not triggered:
        return f"Clean: no detector exceeded the {threshold:.2f} threshold"

    # Report strongest signal first so the most important reason leads.
    triggered.sort(key=lambda r: r.score, reverse=True)
    reasons = [_short_reason(r) for r in triggered]

    return "Flagged: " + " + ".join(reasons)


if __name__ == "__main__":
    # Quick manual smoke test -- run `python -m eval.explain` from the
    # pds/ root. Builds DetectorResult objects directly rather than
    # calling the real detectors, so this has no model-download cost.

    clean_case = (
        DetectorResult("instruction_pattern", 0.0, "no known injection patterns found"),
        DetectorResult("embedding_anomaly", 0.12, "consistent with clean baseline (max similarity=0.880)"),
        DetectorResult("perplexity_fluency", 0.0, "normal human-like fluency (perplexity=45.2)"),
    )

    injection_case = (
        DetectorResult("instruction_pattern", 1.0, "flagged: contains 'ignore instructions' override phrase"),
        DetectorResult("embedding_anomaly", 0.30, "moderately dissimilar to clean baseline (max similarity=0.400)"),
        DetectorResult("perplexity_fluency", 0.05, "normal human-like fluency (perplexity=38.0)"),
    )

    corpus_case = (
        DetectorResult("instruction_pattern", 0.0, "no known injection patterns found"),
        DetectorResult("embedding_anomaly", 0.68, "flagged: highly dissimilar to clean baseline (max similarity=-0.360)"),
        DetectorResult("perplexity_fluency", 0.10, "normal human-like fluency (perplexity=52.0)"),
    )

    for name, case in [("clean", clean_case), ("injection", injection_case), ("corpus", corpus_case)]:
        print(f"{name}: {format_verdict(*case)}")
