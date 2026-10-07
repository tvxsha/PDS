"""
Sentence-level instruction classifier (roadmap task 4.6).

Scores every sentence for "does this try to give the reader/model an instruction?"
and uses the highest-scoring sentence as the document score, so a short attack hidden
inside a long benign document is not diluted.

Training needs only DOCUMENT labels. Every sentence of a poisoned document is labelled
poisoned and every sentence of a clean document clean. Benign sentences occur in both
classes (for example the same email with and without an inserted instruction), so their
weights cancel and the model learns the word patterns that occur only in attacks.

Usage:
    model = SentenceInstructionDetector().fit(train_texts, train_labels)
    score = model.score(text)                  # 0..1
    probs, sentences = model.score_units(text) # per-sentence scores (for explanations)
"""

import random
import re

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from detectors.base import timed

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_YOU = re.compile(r"\b(you|your)\b")
_MODAL = re.compile(r"\b(please|ensure|make sure|must|should|kindly|do not|don't)\b")
_OUTPUT = re.compile(r"\b(response|answer|reply|output|implementation)\b")


def split_units(text, cap=300):
    """Split a document into sentence-like units (lines first, then sentences)."""
    units = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            units += [s.strip() for s in _SENTENCE_END.split(line) if s.strip()]
    return (units or [text])[:cap]


def _hand_features(sentence):
    low = sentence.lower()
    return [
        sentence.strip().endswith("?"),
        bool(_YOU.search(low)),
        bool(_MODAL.search(low)),
        bool(_OUTPUT.search(low)),
        "```" in sentence,
        len(low.split()) < 40,
    ]


class SentenceInstructionDetector:
    def __init__(self, max_units_per_doc=60, c=1.0, seed=3):
        self.max_units_per_doc = max_units_per_doc
        self.c = c
        self.seed = seed
        self.vec = None
        self.clf = None

    def _matrix(self, units):
        hand = csr_matrix(np.array([_hand_features(s) for s in units], dtype=float))
        return hstack([self.vec.transform(units), hand])

    def fit(self, texts, labels):
        rng = random.Random(self.seed)
        units, y = [], []
        for text, label in zip(texts, labels):
            u = split_units(text)
            if len(u) > self.max_units_per_doc:
                u = rng.sample(u, self.max_units_per_doc)
            units += u
            y += [int(label)] * len(u)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit(units)
        self.clf = LogisticRegression(C=self.c, max_iter=3000, class_weight="balanced")
        self.clf.fit(self._matrix(units), y)
        return self

    def score_units(self, text):
        units = split_units(text)
        probs = self.clf.predict_proba(self._matrix(units))[:, 1]
        return probs, units

    def score(self, text):
        probs, _ = self.score_units(text)
        return float(probs.max())


# ---- optional: same interface as the other detectors (DetectorResult) -------------------
_MODEL = None


def set_model(model):
    global _MODEL
    _MODEL = model


@timed("sentence_instruction")
def score_document(text):
    if _MODEL is None:
        raise RuntimeError("call set_model(fitted SentenceInstructionDetector) first")
    probs, units = _MODEL.score_units(text)
    i = int(np.argmax(probs))
    reason = f"most instruction-like sentence ({probs[i]:.2f}): {units[i][:120]!r}"
    return float(probs[i]), reason, {"n_units": len(units)}