# Prior Art Search — PDS (RAG Poisoning Detection)

Task 11.2. Searched Google Patents (via web search + Justia mirrors, since
patents.google.com blocks automated fetching), USPTO full-text search, and
arXiv, on: **"RAG poisoning detection"** and **"cascaded detector
calibration."** This is a working list for the team's own orientation and
for the Oct 7 review — **not a substitute for the formal IPR/patent-cell
review in task 11.1**, which must happen before any public posting.

## Closest prior art found

### 1. US 12,107,885 B1 — "Prompt Injection Classifier Using Intermediate Results"
**Assignee:** HiddenLayer, Inc. Filed Apr 2024, granted Oct 2024.
[patents.justia.com/patent/12107885](https://patents.justia.com/patent/12107885)

Covers a **staged/cascade pipeline**: a lightweight local classifier screens
every prompt, and suspicious cases escalate to a more expensive central
ensemble. Conceptually the closest match to PDS's staged pipeline
(`eval/staged_pipeline.py`) — cheap-first, escalate-when-uncertain.

**What differs:** HiddenLayer's detection signal is the **LLM's own
internal intermediate-layer activations** (a white-box, model-internal
probe) and is framed around protecting the *model* from a malicious
*prompt*, independent of the model's own weights. PDS instead scores
**document content itself**, pre-retrieval, using three independent,
model-external signals (regex pattern matching, embedding distance to a
trusted corpus, and GPT-2 perplexity) — it does not require access to the
production LLM's internals at all, and targets RAG corpus/retrieval
poisoning specifically, not general prompt injection at the chat-input
level. The claims also do not require score calibration/normalization
across multiple heterogeneous detectors, which is PDS's main empirical
finding (Section on calibrated fusion, F1 0.70 → 0.97).

### 2. US 12,021,885 B2 — "Aggregating Results from Multiple Anomaly Detection Engines"
**Assignee:** IBM. Filed Dec 2020, granted Jun 2024.
[patents.justia.com/patent/12021885](https://patents.justia.com/patent/12021885)

Combines scores from multiple anomaly detectors via a **weighted linear
combination**, with weights derived from a resource-relationship graph
(IT infrastructure monitoring — networks, storage, applications).

**What differs:** General-purpose IT/infrastructure domain, not text or
RAG. No score calibration/normalization step — weights are relationship-
derived, not score-scale-derived. No staged/cascaded latency
consideration. PDS's calibration step (min-max, and the quantile
alternative in `eval/calibration_fragility.py`) solves a different problem:
making *heterogeneous raw score scales* (e.g. embedding detector capped
near 0.43, pattern/perplexity spanning the full 0-1 range) comparable
*before* any weighting is applied — this patent's method would inherit
that scale-mismatch problem unmodified if applied to PDS's detectors.

### 3. US 2024/0330772 A1 — "Calibrated Model Intervention with Conformal Threshold"
**Status:** Pending application, filed Mar 2024, published Oct 2024.
[patents.justia.com/patent/20240330772](https://patents.justia.com/patent/20240330772)

Uses **conformal prediction** to threshold a classifier's output into a
calibrated membership set, auto-acting when exactly one class qualifies
and escalating to manual review otherwise. Domain-agnostic (loan approval,
access control, risk analysis examples given).

**What differs:** This is **single-model** confidence calibration, not
multi-detector score fusion — there is one classifier being calibrated,
not three independent signals being combined. It is the closest prior art
specifically to Phase 5's planned conformal-calibration work
(`fusion/conformal.py`), so that phase should explicitly position itself
against this application once it's further along: PDS's planned
contribution there is applying conformal/split-conformal thresholding to a
**fused, multi-signal cascade score** with a formal FPR guarantee on the
*combined* decision, not on a single model's raw output.

## Searches that returned no close match
- `"RAG poisoning detection"` alone mostly surfaced academic papers (e.g.
  RevPRAG, arXiv:2411.18948; "When Context Bites," arXiv:2608.06947) rather
  than patents — RAG-specific poisoning *detection* (as opposed to RAG
  poisoning *attacks*, which are well-studied) appears to be a thinner
  patent landscape than prompt injection generally.
- No patent found combining: (a) document-content-level detection
  (pre-retrieval), (b) three independent, named signal types matching
  PDS's (pattern/embedding/perplexity), and (c) explicit cross-detector
  score calibration before fusion, in the same claim.

## Where PDS's actual novelty case is strongest
Based on this search, **the calibration-and-fusion mechanism** (not any
single detector, all three of which are individually well-established in
the literature per `related_work.md`) is the most defensible angle,
specifically:
1. Calibrating heterogeneous detector score *scales* before fusion
   (addresses #2's gap).
2. The quantile-calibration fragility analysis itself
   (`eval/calibration_fragility.py`) — no prior art found quantifying
   *how much* a single calibration-set document can shift a fused
   detector's output, which is a novel empirical measurement regardless of
   which calibration method is ultimately chosen.
3. Phase 5's planned conformal thresholding **applied to a fused,
   multi-detector cascade score** (vs. #3's single-model case).

None of this is a substitute for a real freedom-to-operate search by the
IPR/patent cell (task 11.1) — this is a team-level sanity check to inform
*what to emphasize* when that meeting happens, not a legal clearance.
