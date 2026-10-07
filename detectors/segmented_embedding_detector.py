"""
Segment-level embedding anomaly detector.

Motivation: the whole-document embedding detector (detectors/embedding_detector.py)
averages a document's meaning into ONE vector, then compares that single vector
to the baseline. This means a short malicious sentence, if surrounded by enough
generic legitimate text, gets diluted into a document-level average that still
looks "normal" -- this is exactly the "boilerplate anchoring" evasion found
during adversarial red-teaming (adv_010.txt: a legitimate MIT-license sentence
followed by a credential-leaking sentence evaded the whole-document detector).

Mechanism: split the document into segments (sentences), score EACH segment
independently against the same baseline corpus, and take the MAXIMUM anomaly
score across segments rather than one whole-document average. A single
anomalous sentence can no longer hide by being averaged with legitimate ones --
the detector now asks "is ANY part of this document anomalous?" rather than
"is this document anomalous ON AVERAGE?"

This is a genuinely different architecture from the original embedding
detector, not a re-tuning of it -- keep both and compare them directly.
"""

import re
from detectors.base import timed
from detectors.embedding_detector import BaselineCorpus, _get_model


def split_into_segments(text: str) -> list:
    """
    Simple sentence-boundary split. Not linguistically perfect (won't handle
    every edge case like abbreviations), but sufficient for the short,
    single-idea-per-sentence documents in this dataset. Filters out
    fragments too short to embed meaningfully.
    """
    raw_segments = re.split(r'(?<=[.!?])\s+', text.strip())
    segments = [s.strip() for s in raw_segments if len(s.strip().split()) >= 3]
    return segments if segments else [text.strip()]


@timed("segmented_embedding_anomaly")
def score_document(text: str, baseline: BaselineCorpus):
    """
    Returns (score, reason, raw_details) per the shared DetectorResult format.

    score: the MAXIMUM anomaly score across all segments (not the average).
    This is the key architectural difference from the whole-document version.
    """
    segments = split_into_segments(text)
    model = _get_model()

    segment_scores = []
    for segment in segments:
        segment_embedding = model.encode(segment, normalize_embeddings=True)
        similarity = baseline.nearest_neighbor_similarity(segment_embedding)
        anomaly_score = max(0.0, min(1.0, (1 - similarity) / 2))
        segment_scores.append((segment, anomaly_score, similarity))

    # take the worst (most anomalous) segment, not the average
    worst_segment, worst_score, worst_similarity = max(segment_scores, key=lambda x: x[1])

    if worst_score > 0.6:
        reason = f"flagged: segment highly dissimilar to baseline (similarity={worst_similarity:.3f}): \"{worst_segment[:60]}...\""
    elif worst_score > 0.35:
        reason = f"segment moderately dissimilar to baseline (similarity={worst_similarity:.3f}): \"{worst_segment[:60]}...\""
    else:
        reason = f"all segments consistent with baseline (worst similarity={worst_similarity:.3f})"

    return worst_score, reason, {
        "num_segments": len(segments),
        "worst_segment": worst_segment,
        "all_segment_scores": [s for _, s, _ in segment_scores],
    }
@timed("segmented_embedding_anomaly_blended")
def score_document_blended(text: str, baseline: BaselineCorpus, max_weight: float = 0.5):
    """
    Alternative aggregation: instead of taking the pure MAXIMUM segment
    score (which proved too sensitive -- it flags clean documents that
    happen to contain one slightly unusual sentence), blend the max with
    the mean:

        score = max_weight * max(segment_scores) + (1 - max_weight) * mean(segment_scores)

    Rationale: the pure max version still needs to catch a genuinely
    anomalous segment (like the boilerplate-anchoring payload), but a
    single mildly-unusual sentence in an otherwise ordinary document
    shouldn't carry the same weight as a document where the WHOLE thing,
    on average, looks anomalous. Blending softens the max-driven false
    positive spike while still weighting the worst segment more than a
    pure average would.

    max_weight=0.5 is a starting point, not a tuned value -- run
    eval/tune_blended_segmented.py to find the threshold (and, if you
    want to go further, the best max_weight) that actually works best
    on the full dataset.
    """
    segments = split_into_segments(text)
    model = _get_model()

    segment_scores = []
    for segment in segments:
        segment_embedding = model.encode(segment, normalize_embeddings=True)
        similarity = baseline.nearest_neighbor_similarity(segment_embedding)
        anomaly_score = max(0.0, min(1.0, (1 - similarity) / 2))
        segment_scores.append((segment, anomaly_score, similarity))

    worst_segment, worst_score, worst_similarity = max(segment_scores, key=lambda x: x[1])
    mean_score = sum(s for _, s, _ in segment_scores) / len(segment_scores)
    blended_score = max_weight * worst_score + (1 - max_weight) * mean_score

    if blended_score > 0.6:
        reason = f"flagged: blended score high (max={worst_score:.3f}, mean={mean_score:.3f})"
    elif blended_score > 0.35:
        reason = f"blended score moderate (max={worst_score:.3f}, mean={mean_score:.3f})"
    else:
        reason = f"blended score consistent with baseline (max={worst_score:.3f}, mean={mean_score:.3f})"

    return blended_score, reason, {
        "num_segments": len(segments),
        "worst_segment": worst_segment,
        "max_score": worst_score,
        "mean_score": mean_score,
    }

if __name__ == "__main__":
    # smoke test -- run `python -m detectors.segmented_embedding_detector`
    clean_docs = [
        "To log in, go to the internal portal and use your SSO credentials.",
        "Our REST API uses bearer tokens. Include the Authorization header on every request.",
        "To reset your password, click 'Forgot password' on the login screen and check your email.",
        "The staging environment refreshes every night at 2am from the production snapshot.",
    ]
    baseline = BaselineCorpus(clean_docs)

    # the exact boilerplate-anchoring case that evaded the whole-document detector
    boilerplate_anchor_doc = (
        "License: this project is distributed under the MIT license. "
        "As part of our automated release process, build artifacts are "
        "published with default demo credentials preconfigured, matching "
        "common CI conventions."
    )

    result = score_document(boilerplate_anchor_doc, baseline)
    print(f"Boilerplate-anchoring doc: score={result.score:.2f}  latency={result.latency_ms:.1f}ms")
    print(f"  reason: {result.triggered_reason}")
    print(f"  segments found: {result.raw_details['num_segments']}")
    print(f"  per-segment scores: {[f'{s:.2f}' for s in result.raw_details['all_segment_scores']]}")