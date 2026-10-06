
## 1. Problem
RAG systems retrieve documents from a corpus and feed them to an LLM. If an
attacker can get a malicious document into that corpus (or injected into the
retrieval results), they can hijack the LLM's behavior — prompt injection,
instruction override, credential exfiltration — without ever touching the
model itself. This is "RAG poisoning."

## 2. Gap
Existing defenses are mostly single-signal: a keyword/pattern filter, OR a
perplexity check, OR an embedding-similarity check — each with known,
different blind spots. No open comparison exists of how these signals fail
*differently*, or whether combining them is actually worth the latency cost.

## 3. Method — PDS (Poison Detection System)
Three independent, cheap-to-expensive detectors, fused into one verdict:
- **Pattern** (instant): regex match against known override/injection phrases
- **Embedding anomaly**: cosine distance from a trusted clean-document baseline
- **Perplexity**: GPT-2-based fluency scoring (catches unnaturally
  smooth/robotic OR garbled text)

Combined via fusion — we tested 4 variants: naive fixed average, learned
(logistic regression), **calibrated fixed** (min-max normalized before
averaging), and a staged/tiered cascade.

## 4. Results — fusion comparison
| Method | F1 | Notes |
|---|---|---|
| Naive fixed fusion | 0.70 | Hurt by raw score-scale mismatch across detectors |
| Learned fusion (raw) | 0.88 | Partial fix via regularized logistic regression |
| **Calibrated fixed fusion** | **0.97** | Min-max normalize each detector before averaging |
| Learned fusion (calibrated) | 0.98 | Best accuracy; confirms embedding as strongest signal |
| Staged/tiered pipeline | 0.97 | Same F1 as calibrated, **72.5% less latency** |

**Headline:** proper calibration closes almost the entire gap to a learned
model — and a staged cascade gets the same accuracy for a fraction of the
compute cost. [PENDING: updated numbers after 0.2's credential-regex fix]

## 5. Failure map / cross-domain generalization
[PENDING — Karishma's 0.3: per-category scores, detector triggers, F1 on the
12 cross-domain (HR/support-policy) documents]

## 6. Adversarial red-teaming finding
**9/10 (90%) of hand-crafted evasion attempts were caught.** All 10 avoided
every pattern-trigger phrase and used fluent, natural README-style language
specifically to evade embedding and perplexity too.

**The one evasion — "boilerplate anchoring":** `adv_010.txt` opened with
maximally generic, ubiquitous legitimate text (standard license boilerplate),
which diluted the whole-document embedding average enough that a malicious
clause later in the same document slipped under threshold (combined score
0.106, threshold 0.15). [PENDING: does 0.2's new credential regex now catch
this one via pattern detection instead?]

**Follow-on finding — calibration itself is fragile:** min-max
normalization lets a single clean document set each detector's entire scale.
We tested this directly on the embedding detector: removing just the ONE
document that sets the calibration boundary shifts the *whole dataset's*
normalized scores by ~2.1% on average — quantile-based (5th/95th percentile)
bounds are ~1.8x more stable under the same test. Caveat: more stable
calibration isn't automatically better *detection* — the decision threshold
would need separate re-tuning if we switched methods.

## 7. Limitations
- Dataset is small (77 docs) and partly self-authored — Phase 2 is scaling
  this to 300+ from public sources (BIPIA, PoisonedRAG, deepset, etc.)
- No comparison yet against published baselines (DeBERTa classifier, Meta
  Prompt Guard) — Phase 3
- Three earlier fusion variants we tried — segmented, blend, veto — did NOT
  beat calibrated fusion (negative result, reported honestly, not hidden)
- Calibration bounds are fixed from a one-time training set; staleness as
  the document distribution shifts over time is untested

## 8. Next steps
- Phase 1: formal train/val/test protocol with bootstrap CIs (no more
  tuning-on-test) — unblocks everything downstream
- Phase 2: dataset to 300+ docs from public sources
- Phase 3: baseline comparison
- Phase 5: conformal calibration with a formal false-positive guarantee
- Target: IEEE-indexed venue submission after IP/patent-cell review


