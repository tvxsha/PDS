# PDS Roadmap

Working title: *Where Poison Detectors Fail: A Comparative Failure and Latency
Analysis of RAG Defense Signals*

**Team:** Tvisha, Karishma, Sree Vali

This roadmap reflects where the project actually stands right now, not the
original starting plan. Tasks are split so that everything on the critical
path — the stuff that determines whether the core system works and whether
the paper's central claims hold up — stays with Tvisha. Karishma and Sree
Vali get self-contained, clearly-specified tasks that produce real, gradeable
project content, but nothing that blocks progress if it comes back late or
needs rework.

---

## Current status (as of this point in the project)

- [x] Project structure, shared detector interface (`detectors/base.py`)
- [x] Module 1: instruction pattern detector — built, tested
- [x] Module 2: embedding anomaly detector — built, tested
- [x] Module 3: perplexity/fluency detector — built, tested
- [x] Dataset v1: 55 clean docs (real READMEs from 10 open-source projects) + 60 poisoned docs (15 per category: injection, authority, corpus, perplexity)
- [x] Fixed-weight fusion built and evaluated
- [x] **Major finding**: naive equal-weight fusion (F1=0.70) badly underperforms because embedding detector's raw score range (0.18-0.43) is compressed compared to pattern/perplexity's (0-1) — averaging dilutes its signal
- [x] **Calibrated fusion** (min-max normalize each detector before averaging) fixes this: F1=0.95, recall=0.97, precision=0.94
- [x] Failure map by category at calibrated thresholds: authority/corpus/perplexity all ~100%, injection ~87%
- [ ] Currently isolating the specific injection docs the calibrated fusion still misses, to understand the residual failure mode

**This means the project is already past the point the original roadmap assumed** — core detection + a real, defensible fusion improvement story already exist. What's left is: learned fusion (to add interpretability, not necessarily beat 0.95), the extended novelty work (cross-domain, adversarial red-teaming), and the actual writeup.

---

## Tvisha's tasks

- [ ] Finish diagnosing the remaining injection misses (in progress)
- [ ] Learned fusion (`fusion/learned.py`) — logistic regression/decision tree on the 3 (calibrated) scores, trained/tested on a proper split
- [ ] Extract and interpret feature importances from the learned model — this is the "which signal matters for which attack type" finding
- [ ] Staged/tiered pipeline + latency-savings measurement
- [ ] Adversarial red-teaming: hand-crafting poison designed to evade the calibrated fusion, measuring evasion rate
- [ ] All threshold tuning and calibration decisions
- [ ] Final consolidated metrics table and failure-map figure for the paper
- [ ] The core "failure analysis" narrative section of the report (the actual argument, not just describing what was built)
- [ ] All git merges into `main` — review anything teammates produce before merging

---

## Karishma's tasks

Self-contained, doesn't touch the codebase's internals, produces real report/deck content.

### Task K1: Cross-domain test set (10-15 docs)
Build a small second test set from a domain *different* from the primary
corpus (which is open-source software READMEs). Suggested: general internal
policy/wiki-style writing, or customer-support style documentation — anything
that isn't API/library documentation.
- Deliverable: 10-15 `.txt` files, same format as `data/clean/*.txt`
- **Hand these to Tvisha as plain `.txt` files** (don't touch the eval scripts) — Tvisha will run them through the existing pipeline
- Roughly 1-2 hours of work

### Task K2: Related work / literature honesty section
Find and summarize (in your own words, 2-3 sentences each) 4-6 real papers
or sources from 2023-2025 on: RAG data poisoning, prompt injection detection,
or embedding-based anomaly detection for text. The assignment brief already
references some — start there, then search for 2-3 more independently.
- Deliverable: a `related_work.md` file with source, one-line summary, and how it connects to our 3 detectors
- This becomes a real section of the final report — genuinely important, not busywork

### Task K3: Presentation deck
Update `PDS_Pitch_Deck.pptx` into the final presentation, once Tvisha shares
final numbers (don't build slides with placeholder/guessed numbers — wait
for real ones to avoid redoing slides).
- Include: problem framing, the 3-detector approach, the calibration finding,
  final metrics table, failure map, adversarial findings if ready
- Deliverable: updated `.pptx`

**Checkpoint:** K1 should be done early (it's needed for Tvisha's cross-domain
evaluation). K2 can happen in parallel any time. K3 has to wait until near
the end.

---

## Sree Vali's tasks

Also self-contained; focused on documentation, dataset QA, and one small
coding task with a very tight spec.

### Task S1: Dataset provenance audit
Review `data/_provenance/*.tsv` against the actual files in `data/clean/`
and `data/poisoned/`. Confirm every file is accounted for, sources are
correctly labeled, and write a short paragraph for the report's methodology
section explaining in plain language how the dataset was constructed
(real READMEs for clean, template-generated + hand-crafted for poisoned).
- Deliverable: a short `dataset_methodology.md` write-up
- Also flag anything that looks wrong or mislabeled back to Tvisha

### Task S2: Explainability formatting (small, well-specified coding task)
Write a single function that takes the 3 raw detector scores + reasons
(already available as `DetectorResult` objects) and produces one
plain-English verdict string, e.g.:
`"Flagged: high embedding anomaly (0.38) + contains override phrase"`
- Exact spec: function `format_verdict(pattern_result, embedding_result, perplexity_result) -> str`
- Only include a detector's reason if its score is above a passed-in threshold
- **Do this in a new file** `eval/explain.py`, don't modify existing detector files
- Send the file to Tvisha to review before it's merged

### Task S3: Report sections — problem framing & limitations
Draft (in prose, for the report, not code): the "Problem, Explained Simply"
section (can adapt from the original pitch PDF) and a "Limitations" section
covering: dataset size caveats, the corpus-poisoning/topic-mismatch caveat
Tvisha will describe, and the fact that attack categories were generated
rather than sourced from real incidents.
- Deliverable: `report_sections_draft.md`

**Checkpoint:** S1 can start immediately. S2 needs `detectors/base.py`
(already exists) — can start immediately. S3 should wait until Tvisha shares
the calibration/topic-mismatch finding so it's described accurately.

---

## How handoffs should work

To keep things organized as a team:
- Karishma and Sree Vali's work happens in their own files (dataset files,
  markdown write-ups, the one small `explain.py` function) — they shouldn't
  need to touch `detectors/`, `fusion/`, or `eval/run_evaluation.py`, since
  those are still actively changing as Tvisha works through the roadmap
- Send deliverables to Tvisha to review and merge in, rather than pushing
  directly to `main`, so nothing conflicts with work in progress
- If working in git branches: `git checkout -b karishma/cross-domain-set`
  and `git checkout -b sreevali/dataset-audit` — Tvisha reviews and merges

---

## Remaining timeline (rough order)

1. Finish injection-miss diagnosis
2. Learned fusion + feature importance
3. Staged/tiered pipeline + latency writeup
4. Cross-domain evaluation (once Karishma delivers K1)
5. Adversarial red-teaming
6. Final metrics consolidation + failure analysis writeup
7. Merge in teammates' sections, assemble final report + deck

---

## Notes

- The calibration finding (naive fusion underweights embedding due to
  score-scale mismatch) is arguably the single strongest result so far —
  make sure it's front and center in the report, not buried as a footnote.
- Don't let K3 (slides) start before final numbers exist — redoing slides
  from guessed numbers wastes time neither teammate can spare.
- If Karishma or Sree Vali's deliverables come back incomplete or low-effort,
  the fallback is: S1/S3 can be absorbed into Tvisha's writeup with minimal
  extra work (they're writing tasks, recoverable). K1 (cross-domain set) is
  the one task worth following up on directly if it stalls, since it's the
  only teammate task that blocks a piece of the core evaluation.