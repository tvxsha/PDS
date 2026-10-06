# PDS Roadmap — Task Assignments

Working title: *Where Poison Detectors Fail: A Comparative Failure and Latency Analysis of RAG Defense Signals*

Team: **Tvisha** (T), **Karishma** (K), **Sree Vali** (S)

## How to read this file
- 🟢 **READY** = can be started right now, no dependencies
- 🔴 **BLOCKED** = waiting on the task IDs in the "Depends on" column
- ✅ **DONE** = finished and merged to `main`
- ⏱ = needed for the Oct 7 review
- Update the Status column in your own PR when you finish a task, so blocked tasks turn green for the next person.
- Rules: no tuning on the test split, ever. Test is run once on a frozen git tag. Every number in the paper must trace to a script in `eval/`.

## Already done
- ✅ Pattern, embedding (MiniLM) and perplexity (GPT-2) detectors
- ✅ Dataset v1: 55 clean, 60 poisoned, 10 adversarial, 12 cross-domain (this is now the **development set**)
- ✅ Fixed, calibrated, learned and staged fusion
- ✅ Failure map, blind-spot analysis, adversarial testing (boilerplate-anchoring finding)
- ✅ Segmented / blend / veto experiments (negative results)
- ✅ Summary report, README, related work (6 papers)

## Critical path (what unblocks the most people)
`1.1` (Tvisha) + `1.2` (Sree) -> `1.4` (Tvisha) unlocks Phases 3, 4, 5, 8.
`2.1`, `2.2`, `2.3`, `2.4`, `2.5` -> `2.6` unlocks the baselines, label audit, datasheet and final results.
Do these first.

---

## Phase 0 — Clean-up and Oct 7 review ⏱

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 0.1 | ⏱ Run `git status` / `git branch`; commit and push any local files not on `main` (segmented, blend, veto scripts, credential patterns); move experiments to `eval/experiments/` | Tvisha | 🟢 READY | — | `eval/experiments/*` on `main` | `git status` is clean and `main` has every local script |
| 0.2 | ⏱ Add the two credential regexes to `detectors/pattern_detector.py`; re-run `run_evaluation`, `calibrated_fusion`, `adversarial_test`, `generate_summary_report` | Tvisha | 🟢 READY | — | Updated `eval/results/*`, `summary_report.md` | New numbers committed; adv_010 result known |
| 0.3 | ⏱ Write and run cross-domain evaluation on Karishma's 12 docs, report per-doc scores, detector triggers and F1 | Karishma | 🟢 READY | — | `eval/cross_domain_eval.py`, `eval/results/cross_domain.csv` | Script runs; results table pasted in PR |
| 0.4 | ⏱ Add `requirements.txt` (pinned), fix repo URL case to `tvxsha/PDS`, add a "Reproduce results" section to the README | Karishma | 🟢 READY | — | `requirements.txt`, README | A fresh venv + `pip install -r requirements.txt` runs `eval.run_evaluation` |
| 0.5 | ⏱ Rebase `sreevali/dataset-audit` and `sreevali/explain-verdict` onto current `main` (avoid stale deletions), open PRs, get a review, merge | Sree Vali | 🟢 READY | — | `eval/explain.py`, `dataset_methodology.md` on `main` | Both PRs merged, no files deleted by accident |
| 0.6 | ⏱ Prepare review slides and demo script: problem -> gap -> method -> results -> failure map -> adversarial finding -> limitations -> next steps; write `demo_commands.md`; rehearse once | Sree Vali | 🟢 READY | — (final numbers fill in after 0.2, 0.3) | `slides/`, `demo_commands.md` | One full run-through with timings under the review slot |
| 0.7 | ⏱ Update `paper.tex` with final numbers, negative-results section, credential-pattern finding, cross-domain results; add `paper/` to repo | Tvisha | 🔴 BLOCKED | 0.2, 0.3 | `paper/paper.tex`, `paper/paper.pdf` | PDF compiles; every table matches the CSVs |

## Phase 1 — Evaluation protocol (fixes "tuned on the test set")

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 1.1 | Define common doc schema (`id, text, label, source, attack_type, domain, length`); write stratified 60/20/20 train/val/test split by (label, attack_type, source); save split IDs | Tvisha | 🟢 READY | — | `eval/protocol.py`, `data/splits/split_seed{N}.json` | Split is deterministic per seed; no doc in two splits |
| 1.2 | Stats helpers: bootstrap 95% CI (1000 resamples) for precision/recall/F1/FPR, McNemar's test, multi-seed aggregation (mean ± std) | Sree Vali | 🟢 READY | — | `eval/stats.py` + small unit test | Functions take `y_true, y_pred`; test passes on synthetic data |
| 1.3 | Latency benchmark: warm-up, >=100 runs per detector, median and p95, hardware printout | Karishma | 🟢 READY | — | `eval/latency_benchmark.py`, `eval/results/latency.csv` | Prints a table; first-call model-load overhead is excluded |
| 1.4 | Retrofit every eval script (fusion, learned, calibrated, staged) to the protocol: fit on train, tune on val, report on test with CIs | Tvisha | 🔴 BLOCKED | 1.1, 1.2 | Updated `eval/*.py` | No script touches test labels before the final run |
| 1.5 | `make_results.py`: one command regenerates all tables and figures | Sree Vali | 🔴 BLOCKED | 1.4 | `eval/make_results.py`, `eval/results/tables/` | A clean clone produces every paper table |
| 1.6 | ROC and precision-recall curves with AUC + CIs; reliability (calibration) plots for fused scores | Karishma | 🔴 BLOCKED | 1.4 | `eval/plot_curves.py`, `paper/figs/` | Vector PDFs saved |

## Phase 2 — Dataset at scale (fixes "77 self-written docs")
Target: >=300 clean, >=300 poisoned, >=50 per attack type. Check each source's licence and availability first.

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 2.1 | `peek_datasets.py` + loaders for BIPIA and PoisonedRAG into the common schema | Karishma | 🟢 READY | — | `eval/peek_datasets.py`, `data/loaders/bipia.py`, `data/loaders/poisonedrag.py` | Each loader returns schema-conformant rows with source/licence |
| 2.2 | Loaders for deepset/prompt-injections, Open-Prompt-Injection, HackAPrompt (separate "direct injection" slice) | Sree Vali | 🟢 READY | — | `data/loaders/deepset.py`, `opi.py`, `hackaprompt.py` | Same schema; counts printed per source |
| 2.3 | Sample clean passages from the same corpora (NQ/MS-MARCO/Wikipedia/READMEs/Stack Overflow); match length and domain distributions to poisoned docs | Tvisha | 🔴 BLOCKED | 1.1 | `data/loaders/clean_sampler.py` | Length histograms of clean vs poisoned overlap |
| 2.4 | Expand adversarial set to 100+: LLM-generated attacks, paraphrases of known attacks, new categories; log generator and prompt per doc | Karishma | 🟢 READY | — | `data/adversarial/`, provenance TSV | >=100 docs with provenance |
| 2.5 | Detector-aware and obfuscation attacks (boilerplate anchoring, homoglyphs, zero-width chars, base64, split-sentence, fluent phrasing) | Tvisha | 🟢 READY | — | `data/adversarial_aware/`, provenance TSV | >=40 docs, each tagged with the evasion technique |
| 2.6 | Pool all sources into one dataset; dedupe; assign splits; write provenance and counts report | Tvisha | 🔴 BLOCKED | 1.1, 2.1, 2.2, 2.3, 2.4, 2.5 | `data/pooled/`, `data/pooled_stats.md` | Counts meet targets; no duplicate docs across splits |
| 2.7 | Label audit: independently double-label ~100 random docs; compute Cohen's kappa; resolve disagreements | Sree Vali | 🔴 BLOCKED | 2.6 | `eval/label_audit.py`, `data/audit/` | Kappa reported; fixes applied to labels |
| 2.8 | Dataset card / datasheet (collection, labelling, licences, limitations, intended use) | Sree Vali | 🔴 BLOCKED | 2.6 | `data/DATASHEET.md` | Covers every source |

## Phase 3 — Baselines (fixes "no comparison with existing detectors")

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 3.1 | Choose >=3 published baselines (a DeBERTa prompt-injection classifier, Meta Prompt Guard, a perplexity-filter baseline); request gated access; verify they load and run on CPU | Sree Vali | 🟢 READY | — | `docs/baselines.md` | Each baseline scores a sample doc on your machine |
| 3.2 | Wrap each baseline in the `DetectorResult` interface (score, reason, latency) | Karishma | 🔴 BLOCKED | 3.1 | `detectors/baselines/*.py` | Same interface as our detectors; unit tested |
| 3.3 | Run baselines under the protocol: choose thresholds on **val only**, report F1, per-category recall, FPR, median/p95 latency with CIs on test | Tvisha | 🔴 BLOCKED | 1.4, 2.6, 3.2 | `eval/run_baselines.py`, `eval/results/baselines.csv` | One table: ours vs baselines |
| 3.4 | Complementarity analysis: which docs each detector uniquely catches; does adding a baseline to the cascade help? | Sree Vali | 🔴 BLOCKED | 3.3 | `eval/complementarity.py` + figure | Overlap (Venn/UpSet) figure plus a written paragraph |

## Phase 4 — Ablations and error analysis

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 4.1 | Ablation: each detector alone, each pair, all three; for each fusion variant | Karishma | 🔴 BLOCKED | 1.4 | `eval/ablation.py`, table | Table with CIs |
| 4.2 | Swap in a stronger small LM for perplexity and a stronger embedding model; report effect size | Sree Vali | 🔴 BLOCKED | 1.4 | `eval/model_swap.py`, table | Before/after table |
| 4.3 | Manual error analysis: 20 false negatives and 20 false positives, categorised by cause | Tvisha | 🔴 BLOCKED | 1.4, 2.6 | `eval/results/error_analysis.md` | Causes grouped into a failure taxonomy |

## Phase 5 — Novel contribution: calibration with a false-positive guarantee

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 5.1 | Implement quantile and split-conformal thresholding (clean FPR <= alpha); first test on the dev set | Tvisha | 🟢 READY | — | `fusion/conformal.py` | Achieved FPR tracks target alpha on dev data |
| 5.2 | Reproduce min-max calibration fragility (boilerplate anchoring) as a clean experiment: show how bounds shift under adversarial boilerplate | Sree Vali | 🟢 READY | — | `eval/calibration_fragility.py` | Plot of score shift, min-max vs quantile |
| 5.3 | Sweep alpha in {0.01, 0.05, 0.10}; plot achieved FPR vs target and recall at each alpha, on the pooled data | Karishma | 🔴 BLOCKED | 5.1 | `eval/conformal_sweep.py`, figure | Figure and table with CIs |
| 5.4 | Cost-aware cascade: choose stage order and early-exit thresholds to minimise latency under a recall floor; Pareto plot | Karishma | 🔴 BLOCKED | 1.4 | `fusion/cost_aware.py`, figure | Pareto curve vs fixed cascade |

## Phase 6 — Working system and demo

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 6.1 | Package PDS as an installable library: `pyproject.toml`, `pds.scan(doc) -> verdict, reasons, latency` | Sree Vali | 🟢 READY | — | `pds/__init__.py`, `pyproject.toml` | `pip install -e .` then `import pds` works |
| 6.2 | Streamlit/Gradio demo: paste text, see each detector's score, cascade path, verdict, latency | Karishma | 🟢 READY | — | `demo/app.py` | Runs locally with one command |
| 6.3 | Unit tests for detectors, fusion and protocol; GitHub Actions CI | Karishma | 🟢 READY | — | `tests/`, `.github/workflows/ci.yml` | CI green on `main` |
| 6.4 | LangChain / LlamaIndex retriever middleware that filters retrieved chunks | Tvisha | 🔴 BLOCKED | 6.1 | `pds/integrations/` | Works in a small example RAG app |
| 6.5 | 2-3 minute demo video and architecture diagram | Sree Vali | 🔴 BLOCKED | 6.2 | `docs/demo.mp4`, `paper/figs/architecture.pdf` | Video and diagram exist |

## Phase 7 — End-to-end impact

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 7.1 | Build a small RAG pipeline (retriever + LLM) over a clean corpus; inject poisoned docs | Tvisha | 🟢 READY | — | `rag/pipeline.py` | Pipeline answers queries; poisoning is controllable |
| 7.2 | Harness to measure attack success rate, clean-query accuracy and latency with and without PDS | Karishma | 🔴 BLOCKED | 7.1 | `rag/measure_asr.py` | Table: ASR before/after, accuracy cost |
| 7.3 | Sweeps: poisoning rate (1, 5, 10 docs), corpus size, 2+ retrievers, 2+ LLMs | Sree Vali | 🔴 BLOCKED | 7.2, 6.1 | `rag/sweeps.py`, figures | Sweep figures with CIs |

## Phase 8 — Robustness

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 8.1 | Evaluate all detectors on obfuscation and detector-aware attacks; report evasion rate per technique | Tvisha | 🔴 BLOCKED | 1.4, 2.5 | `eval/robustness_obfuscation.py` | Evasion-rate table |
| 8.2 | Paraphrase robustness: rewrite attacks at several strengths; recall drop per detector | Sree Vali | 🔴 BLOCKED | 1.4, 2.4 | `eval/robustness_paraphrase.py` | Recall-vs-strength plot |
| 8.3 | Multi-document (instruction split across two chunks) and position sensitivity (start / middle / end of long docs) | Karishma | 🔴 BLOCKED | 1.4 | `eval/robustness_position.py` | Table by position |

## Phase 9 — Released benchmark

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 9.1 | Attack taxonomy document with one example per class (injection, authority spoofing, fluent corpus poisoning, perplexity artefact, credential exposure, obfuscation) | Karishma | 🟢 READY | — | `docs/taxonomy.md` | Each class defined with an example |
| 9.2 | Leaderboard script: add a detector, get the standard metrics table | Tvisha | 🔴 BLOCKED | 3.3 | `eval/leaderboard.py` | New detector scored in one command |

## Phase 10 — Paper

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 10.1 | Related work expanded to 20-30 papers (RAG poisoning, prompt-injection detection, perplexity defences, conformal calibration) | Karishma | 🟢 READY | — | `related_work.md`, `paper/refs.bib` | >=20 verified references with citations |
| 10.2 | Threat model, formal problem statement (cascade as a decision problem), and ethics / responsible-release section | Sree Vali | 🟢 READY | — | `paper/sections/threat_model.tex`, `ethics.tex` | Attacker capabilities and scope stated precisely |
| 10.3 | Limitations and negative-results sections (segmentation, blend, veto) | Sree Vali | 🟢 READY | — | `paper/sections/limitations.tex` | Every limitation names how a later phase addresses it |
| 10.4 | Dataset section: sources, sizes, labelling, kappa, splits | Karishma | 🔴 BLOCKED | 2.6, 2.7 | `paper/sections/dataset.tex` | Numbers match `pooled_stats.md` |
| 10.5 | Method and results sections: main table, ablations, baselines, failure map, latency, calibration | Tvisha | 🔴 BLOCKED | 3.3, 4.1, 5.3 | `paper/sections/results.tex` | Every number traces to a script |
| 10.6 | Abstract, introduction (3 contributions), headline figure, final integration, compile and format check | Tvisha | 🔴 BLOCKED | 10.1, 10.2, 10.3, 10.4, 10.5 | `paper/paper.tex`, `paper/paper.pdf` | PDF compiles within venue page limit |
| 10.7 | Everyone: read the whole paper once and send comments before submission | All | 🔴 BLOCKED | 10.6 | PR comments | All comments resolved |

## Phase 11 — Submission and IP

| ID | Task | Owner | Status | Depends on | Output / file | Done when |
|---|---|---|---|---|---|---|
| 11.1 | Meet VIT's IPR / patent cell **before** any public posting; ask what disclosure is safe | Tvisha | 🟢 READY | — | notes in `docs/ip.md` | Written answer on what can be posted and when |
| 11.2 | Prior-art search (Google Patents, Lens.org) on "RAG poisoning detection" and "cascaded detector calibration"; 1-page invention disclosure | Sree Vali | 🟢 READY | — | `docs/prior_art.md` | List of closest prior art and what differs |
| 11.3 | Shortlist 3-5 venues (IEEE-indexed with real peer review): deadlines, page limits, double-blind rules, template | Karishma | 🟢 READY | — | `docs/venues.md` | Table with deadlines and links; one recommended target |

---

## Per-person checklist

### Tvisha (18 tasks)
**Ready now:** 0.1, 0.2, 1.1, 2.5, 5.1, 7.1, 11.1
**Blocked:** 0.7 (needs 0.2, 0.3) · 1.4 (needs 1.1, 1.2) · 2.3 (needs 1.1) · 2.6 (needs 1.1, 2.1-2.5) · 3.3 (needs 1.4, 2.6, 3.2) · 4.3 (needs 1.4, 2.6) · 6.4 (needs 6.1) · 8.1 (needs 1.4, 2.5) · 9.2 (needs 3.3) · 10.5 (needs 3.3, 4.1, 5.3) · 10.6 (needs 10.1-10.5)

### Karishma (18 tasks)
**Ready now:** 0.3, 0.4, 1.3, 2.1, 2.4, 6.2, 6.3, 9.1, 10.1, 11.3
**Blocked:** 1.6 (needs 1.4) · 3.2 (needs 3.1) · 4.1 (needs 1.4) · 5.3 (needs 5.1) · 5.4 (needs 1.4) · 7.2 (needs 7.1) · 8.3 (needs 1.4) · 10.4 (needs 2.6, 2.7)

### Sree Vali (18 tasks)
**Ready now:** 0.5, 0.6, 1.2, 2.2, 3.1, 5.2, 6.1, 10.2, 10.3, 11.2
**Blocked:** 1.5 (needs 1.4) · 2.7 (needs 2.6) · 2.8 (needs 2.6) · 3.4 (needs 3.3) · 4.2 (needs 1.4) · 6.5 (needs 6.2) · 7.3 (needs 7.2, 6.1) · 8.2 (needs 1.4, 2.4)

## Dependency order
Phase 0 and the READY tasks run in parallel now -> Phase 1 (1.4 is the bottleneck) -> Phases 2, 3, 4, 5 -> Phases 7, 8 -> Phase 10 -> Phase 11 (submission).