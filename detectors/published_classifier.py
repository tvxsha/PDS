"""
Task 3.2: a published prompt-injection classifier behind the same interface as our detectors.

Default model: protectai/deberta-v3-base-prompt-injection-v2 (Apache 2.0, labels SAFE / INJECTION).
It reads at most 512 tokens, so a document is cut into overlapping 512-token windows and the
document score comes from the MOST suspicious window (a short attack hidden in a long document
is not diluted). Two numbers are produced:

  margin        logit(injection) - logit(everything else), taken at the most suspicious window.
                Not capped, so the ranking survives where the probability saturates at 0.000 / 1.000.
                The evaluation uses this for thresholds.
  score (0..1)  the probability of the injection label, which is what the shared DetectorResult
                interface expects.

Usage:
    from detectors import published_classifier as pc
    pc.warm_up()                        # load the model once, outside any timing
    result = pc.score_document(text)    # DetectorResult
    margin, n_windows, truncated = pc.document_margin(text)

Documents longer than MAX_WINDOWS windows are cut off after that many (reported as truncated).
"""

import math

import numpy as np

from detectors.base import timed

DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
MAX_LEN = 512     # tokens per window, including the two special tokens
OVERLAP = 64      # tokens shared by neighbouring windows
MAX_WINDOWS = 8   # longer documents are cut off after this many windows
BATCH = 8         # windows per forward pass

_loaded = {}


def _injection_index(labels):
    for i, name in labels.items():
        if any(w in name.lower() for w in ("inject", "malicious", "unsafe")):
            return i
    if len(labels) == 2:
        print(f"Warning: cannot tell which label means injection from {labels}; assuming index 1.")
        return 1
    raise ValueError(f"cannot tell which label means injection from {labels}")


def _load(model_id):
    if model_id not in _loaded:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        tok.model_max_length = int(1e9)  # windows are cut here, so silence the length warning
        model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        labels = {int(k): str(v) for k, v in model.config.id2label.items()}
        _loaded[model_id] = (tok, model, _injection_index(labels))
    return _loaded[model_id]


def warm_up(model_id=DEFAULT_MODEL):
    """Load the model and run one tiny input, so load time never lands in a latency figure."""
    document_margin("warm up", model_id)


def _windows(token_ids, body, step, max_windows):
    """Overlapping windows of at most `body` tokens. Returns (windows, truncated)."""
    if len(token_ids) <= body:
        return [token_ids], False
    out, start = [], 0
    while True:
        out.append(token_ids[start:start + body])
        if start + body >= len(token_ids):
            return out, False
        if len(out) == max_windows:
            return out, True
        start += step


def document_margin(text, model_id=DEFAULT_MODEL, max_windows=MAX_WINDOWS):
    """Return (largest margin over the windows, number of windows scored, truncated?)."""
    import torch
    tok, model, inj = _load(model_id)
    body = MAX_LEN - 2  # room for [CLS] and [SEP]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    windows, truncated = _windows(ids, body, body - OVERLAP, max_windows)
    rows = [[tok.cls_token_id] + w + [tok.sep_token_id] for w in windows]
    margins = []
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            width = max(len(r) for r in chunk)
            input_ids = torch.tensor([r + [tok.pad_token_id] * (width - len(r)) for r in chunk])
            mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in chunk])
            logits = model(input_ids=input_ids, attention_mask=mask).logits.detach().cpu().numpy()
            others = [j for j in range(logits.shape[1]) if j != inj]
            rest = np.logaddexp.reduce(logits[:, others], axis=1)
            margins.extend((logits[:, inj] - rest).tolist())
    return float(max(margins)), len(windows), truncated


@timed("published_classifier")
def score_document(text, model_id=DEFAULT_MODEL, max_windows=MAX_WINDOWS):
    margin, n_windows, truncated = document_margin(text, model_id, max_windows)
    p = 1.0 / (1.0 + math.exp(-max(min(margin, 30.0), -30.0)))
    if p >= 0.5:
        reason = f"published classifier: injection probability {p:.3f} in the most suspicious of {n_windows} window(s)"
    else:
        reason = f"published classifier: no window looks like an injection (highest probability {p:.3f})"
    return p, reason, {"margin": margin, "windows": n_windows, "truncated": truncated, "model": model_id}