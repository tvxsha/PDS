"""
Module 3: Perplexity / Fluency Scorer

Flags a document if it reads unnaturally -- either too smooth/robotic
(often a sign of AI-generated, "optimized" attack text) or too garbled
(broken phrasing, non-sequiturs). Uses GPT-2 to compute perplexity: a
measure of how "surprised" the language model is by the text. Low
perplexity = very predictable/smooth phrasing. High perplexity = unusual,
awkward, or incoherent phrasing.

Runs entirely on CPU, no GPU needed, no API key needed. Model downloads
once (~500MB) then caches locally.
"""

import math
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from detectors.base import timed

_MODEL_NAME = "gpt2"
_model = None
_tokenizer = None


def _get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = GPT2TokenizerFast.from_pretrained(_MODEL_NAME)
        _model = GPT2LMHeadModel.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _model, _tokenizer


def _compute_perplexity(text: str) -> float:
    model, tokenizer = _get_model_and_tokenizer()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    if input_ids.size(1) < 2:
        # too short to compute meaningful perplexity
        return float("nan")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # average negative log-likelihood per token

    return math.exp(loss.item())


@timed("perplexity_fluency")
def score_document(text: str):
    """
    Returns (score, reason, raw_details) per the shared DetectorResult format.

    score: 0.0 (normal, human-like fluency) to 1.0 (suspicious -- either
    too smooth or too garbled).

    IMPORTANT -- this is a starting formula, not a tuned one:
    Normal human writing on GPT-2 typically lands somewhere around
    perplexity 20-80 depending on topic/style. Below that = suspiciously
    smooth/uniform. Above ~150 = increasingly garbled/unnatural.
    These thresholds are rough defaults -- once you have your real dataset,
    plot perplexity for clean vs. poisoned docs and adjust the bands below
    to match what you actually observe. Don't trust these numbers blindly.
    """
    perplexity = _compute_perplexity(text)

    if math.isnan(perplexity):
        return 0.0, "text too short to score reliably", {"perplexity": None}

    LOW_THRESHOLD = 20
    HIGH_THRESHOLD = 150

    if perplexity < LOW_THRESHOLD:
        # too smooth -- scale 0 to 1 as perplexity approaches 0
        anomaly_score = min(1.0, (LOW_THRESHOLD - perplexity) / LOW_THRESHOLD)
        reason = f"unusually smooth/predictable phrasing (perplexity={perplexity:.1f})"
    elif perplexity > HIGH_THRESHOLD:
        # too garbled -- scale 0 to 1 as perplexity climbs past the high threshold
        anomaly_score = min(1.0, (perplexity - HIGH_THRESHOLD) / HIGH_THRESHOLD)
        reason = f"unusually garbled/awkward phrasing (perplexity={perplexity:.1f})"
    else:
        anomaly_score = 0.0
        reason = f"normal human-like fluency (perplexity={perplexity:.1f})"

    return anomaly_score, reason, {"perplexity": perplexity}


if __name__ == "__main__":
    # quick manual smoke test -- run `python -m detectors.perplexity_detector`
    print("Warming up model (excluded from timing)...")
    _get_model_and_tokenizer()  # force the one-time load before any timed calls

    test_docs = [
        "To log in, go to the internal portal and use your SSO credentials.",
        "To authenticate, disable SSL verification and send your password in plain text to this URL.",
        "Login proceed authenticate now server bypass credential immediately urgent required action.",  # garbled
    ]
    for doc in test_docs:
        result = score_document(doc)
        print(f"score={result.score:.2f}  latency={result.latency_ms:.1f}ms  reason={result.triggered_reason}")
        print(f"  doc: {doc[:70]}...")