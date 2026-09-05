# PDS Roadmap

Working title: *Where Poison Detectors Fail: A Comparative Failure and Latency
Analysis of RAG Defense Signals*

This roadmap tracks every task needed to go from current state to a finished
paper + demo. Tasks are grouped by phase, with dependencies noted so you know
what blocks what — since you're mostly doing this solo, the dependency order
matters more than the week labels.

---

## Phase 0 — Foundation (do first, everything else depends on this)

- [x] Project structure scaffolded (`detectors/`, `data/`, `eval/`, `fusion/`)
- [x] Shared detector interface (`detectors/base.py` — `DetectorResult` + `@timed` decorator)
- [x] Module 1: Instruction pattern detector — built and tested
- [ ] Git repo pushed to GitHub
- [ ] **Dataset construction started** — this blocks almost everything below, start immediately
  - [ ] 50-60 clean docs (domain: coding/API reference material where possible)
  - [ ] 12-15 docs per poisoned category × 4 categories (48-60 total):
    - `injection_###.txt` — direct instruction-override attacks
    - `authority_###.txt` — fake system/policy spoofing
    - `corpus_###.txt` — subtle corpus poisoning, no obvious override language
    - `perplexity_###.txt` — garbled/awkward generated text
  - [ ] Run `eval/dataset_check.py` periodically to track progress against targets
  - [ ] Note source of each doc (real vs. synthetic) in case you need to justify it later

---

## Phase 1 — Core Detectors (Module 2 and 3)

*Depends on: nothing blocking, can build in parallel with dataset construction*

- [ ] **Module 2: Embedding anomaly detector**
  - [ ] Pick embedding library (e.g. `sentence-transformers`, free/local, no GPU needed)
  - [ ] Compute embeddings for the clean baseline corpus
  - [ ] For a new doc: measure distance (cosine similarity) to nearest clean-corpus neighbors
  - [ ] Score 0-1 scaled from that distance, same interface as `pattern_detector.py`
  - [ ] Test against sample clean + poisoned docs
- [ ] **Module 3: Perplexity / fluency scorer**
  - [ ] Load a local small LM (GPT-2 via `transformers`, runs on CPU)
  - [ ] Compute perplexity for a doc's text
  - [ ] Score 0-1 scaled from perplexity (very low = too smooth/robotic, very high = garbled — decide if you want a U-shaped score or split into two signals)
  - [ ] Test against sample clean + poisoned docs
- [ ] Confirm both modules follow `@timed(...)` + `score_document()` shape from `base.py`, so fusion plugs in with zero rework

---

## Phase 2 — Fusion

*Depends on: all 3 detectors working, at least a partial dataset*

- [ ] Fixed weighted-average fusion (`fusion/fixed.py`) — combine the 3 raw scores into one verdict
- [ ] Learned fusion (`fusion/learned.py`) — logistic regression or small decision tree trained on [3 scores] → poisoned/clean label
  - [ ] Split dataset into train/test before training this (don't evaluate on the same docs you trained on)
  - [ ] Extract feature importances/coefficients — this is your "which signal matters for which attack type" finding
- [ ] Staged/tiered pipeline (`fusion/staged.py`) — run pattern detector first (cheapest), only run perplexity (slowest) if it passes; measure time saved vs. always running all three

---

## Phase 3 — Evaluation

*Depends on: full dataset built, fusion methods working*

- [ ] Per-detector precision/recall/F1
- [ ] Combined-system (fixed) precision/recall/F1
- [ ] Combined-system (learned) precision/recall/F1
- [ ] Per-attack-type detection rate per detector — the failure/blind-spot map (your core novelty story)
- [ ] Confusion matrix across all attack categories
- [ ] Latency (ms) per detector + staged-pipeline time savings
- [ ] Naive baseline comparison (e.g. pattern-matching alone vs. full ensemble) — shows the ensemble's actual marginal value

---

## Phase 4 — Extended Novelty Work

*Depends on: Phase 3 baseline results existing*

- [ ] **Cross-domain generalization test**
  - [ ] Build/collect a small (10-15 doc) test set from a different domain than your primary corpus
  - [ ] Re-run evaluation on this set, compare accuracy drop vs. primary domain
- [ ] **Adversarial red-teaming** (highest-novelty item, prioritize if time allows)
  - [ ] Hand-craft poison specifically designed to evade the *combined* system (paraphrase injection phrasing, smooth text to dodge perplexity, embed near-legit topics to dodge anomaly detection)
  - [ ] Run these through the full pipeline, report evasion rate
  - [ ] Use findings to describe the system's actual blind spots as an attacker would exploit them
- [ ] **Explainability formatting**
  - [ ] Combine each detector's `triggered_reason` into a single plain-English verdict string
  - [ ] Near-zero extra work — mostly presentation of data you already have

---

## Phase 5 — Writeup & Presentation

*Depends on: Phase 3 (required) and as much of Phase 4 as you complete*

- [ ] Failure analysis writeup (the core paper narrative)
- [ ] Report sections: problem framing, related work honesty statement, method, evaluation, failure analysis, adversarial findings, limitations (small sample caveat if relevant), conclusion
- [ ] Final presentation deck update (based on existing `PDS_Pitch_Deck.pptx`)
- [ ] Demo prep (if required) — likely just running the pipeline live against a couple of sample docs

---

## Notes

- **Dataset size matters for credibility.** 12-15 docs per category minimum, not 5 — small samples won't hold up if your teacher questions statistical meaningfulness.
- **Build detectors to output raw scores, not just pass/fail** — needed for learned fusion and any threshold-tuning later.
- **Log timing from day one** on every detector call — retrofitting latency measurement later means re-running everything.
- **If time runs short**, the priority order for cutting scope is: Phase 4 adversarial work is impressive but optional; Phase 3 evaluation is not optional; a smaller-than-planned dataset (still ≥10/category) is acceptable with an honest limitations note, but skipping the failure-map entirely guts the whole novelty angle.