# PDS Roadmap (single-owner version, updated Oct 7, 2026)

Working title: *Where Poison Detectors Fail: A Comparative Failure and Latency Analysis of RAG Defense Signals*

Owner of all remaining work: **Tvisha**. Work already merged by **Sree Vali** (stats helpers, calibration-fragility script, `explain`, `pds` package, review slides and demo script, threat model, ethics, limitations, prior-art notes, baselines notes, dataset loaders) and **Karishma** (the 12 cross-domain benign documents) stays credited in git history and the author list. This file only changes who does the *remaining* work.

## How to read this file
- ✅ **DONE** = finished and on `main` (or, for paper items, in the current PDF)
- 🟡 **BUILT** = code or text exists but still needs a run, a commit/push, or a paper update
- 🟢 **READY** = can start now
- 🔴 **BLOCKED** = waiting on the IDs in "Depends on"
- ⏱ = worth doing before or at the Oct 7 review
- **R1-R10** = the ten reviewer suggestions (table below); every one maps to at least one task
- Est. = rough hands-on time for you, not counting compute

## Rules that do not change
1. **No tuning on the test split, ever.** Thresholds, calibration bounds and the embedding baseline are chosen on train/validation only. Test is run once (`eval.protocol_eval --final`) on a frozen git tag.
2. Every number in the paper traces to a script in `eval/` and a file in `eval/results/`.
3. All dev-set numbers (the 77 documents) are labelled *development-set, optimistic* wherever they appear.
4. Anything written after seeing a failure (the credential regexes, the "your answer/response" rule) is reported as a **patch, not independent evidence**. Attack sets used to evaluate a mitigation are written and frozen **before** the mitigation is implemented.
5. Negative results stay in the paper.

## Where we are
| Area | State |
|---|---|
| Detectors, fusion, staged cascade | ✅ three detectors + naive, calibrated, learned, staged fusion |
| Dev-set paper | 🟡 10 pages, 5 figures, 15 equations, 25 references; the updated `paper.tex` and `figures/` are not in the repo yet (H4) |
| Evaluation protocol (group-aware 25% baseline / 60-20-20 split, validation thresholds, `--final` once, bootstrap CIs, McNemar, flag-everything row) | 🟡 code written, **not yet run** |
| Pooled dataset | 🟡 1,068 docs (438 clean / 630 poisoned): PDS dev 137, BIPIA 540, PoisonedRAG 400; deepset skipped (`datasets` library conflict) |
| Zero-shot transfer of the frozen dev pipeline | 🟡 BIPIA done (all detectors at chance, AUC 0.46-0.54); PoisonedRAG still to run (E1) |
| "your answer/response" pattern rule | 🟡 written, **not yet added** (waits for the saved v1 run, E2) |
| Datasheet | 🟡 draft with `[CONFIRM]` items |

## The ten reviewer suggestions and where they live
| # | Suggestion | Tasks | Status |
|---|---|---|---|
| R1 | Held-out test set; thresholds on validation; test touched once | 1.1, 1.4, 1.7 | 🟡 code written, not run |
| R2 | Class imbalance: many more clean docs; precision at realistic prevalence | 1.5, 2.6, 2.9 | 🟡 pool has 438 clean; prevalence-adjusted precision not written |
| R3 | Confidence intervals | 1.2, 1.4, 0.9 | ✅ helpers; 🟡 not yet in any table |
| R4 | One published detector as a baseline | 3.1, 3.2, 3.3 | 🟢 not started |
| R5 | Boilerplate subtraction before embedding | 4.4 | 🟢 not started |
| R6 | Multi-domain baseline, nearest-domain scoring | 4.5 | 🟢 not started |
| R7 | Is perplexity worth its cost? Pattern+embedding ablation | 0.8, 4.1 | 🟡 dev-set ablation computed (stage C flags **0** extra docs); not yet in paper or on held-out data |
| R8 | End-to-end attack success rate | 7.1, 7.2, 7.3 | 🟢 not started (needs an LLM) |
| R9 | Threshold consistency (0.15 vs 0.20) | 0.7 | 🟢 not fixed |
| R10 | Obfuscated attacks (base64, homoglyph, zero-width) | 2.5, 8.1, 8.4 | 🟢 not started |

---

## Today (Oct 7): order of operations ⏱
You run the commands; I draft the text and code. Each step is small and ends with something saved.

| Step | What | Who | Est. |
|---|---|---|---|
| 1 | E1: run `external_transfer` on the **original** code (pool now has PoisonedRAG); save as `external_transfer_v1.md/.csv` | you | 5 min |
| 2 | 0.7 + 0.8 + 0.9: fix threshold consistency in the paper, add the perplexity ablation and dev-set bootstrap CIs | me (needs your yes) | 20 min |
| 3 | E2: add the "your answer/response" rule, rerun, save `external_transfer_v2.md/.csv` | you | 10 min |
| 4 | 1.1/1.4: `python -m eval.protocol --seed 42`, then `python -m eval.protocol_eval --seed 42` (validation only; **no** `--final`) | you | 15-30 min |
| 5 | H1-H4: commit and push everything from your machine on a branch | you | 10 min |

Do **not** run `--final` today unless every threshold and design choice is frozen (see 1.7).

---

## Phase 0 - Paper hygiene and review ⏱

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 0.1 | Commit local experiment scripts (segmented/blend/veto) to `eval/experiments/` | 🟡 | - | - | `eval/experiments/*` on `main` | `git status` clean | 10 min |
| 0.2 | Credential-exposure regexes in `detectors/pattern_detector.py` | ✅ | - | - | merged in PR #6 | - | - |
| 0.3 | Cross-domain evaluation on the 12 benign docs | ✅ | - | - | `eval/cross_domain_eval.py`, `cross_domain.csv` | - | - |
| 0.4 | Pinned `requirements.txt` + "Reproduce results" in the README | 🟢 | - | - | `requirements.txt`, README | Fresh venv + `pip install -r` runs `eval.run_evaluation` | 30 min |
| 0.5 | `explain` and dataset-audit PRs | ✅ | - | - | on `main` | - | - |
| 0.6 | Review slides and demo script | ✅ | - | - | `slides_outline.md`, `demo_commands.md` | Rehearse once, with the new figures | 30 min |
| 0.7 ⏱ | **Threshold consistency.** `adversarial_test.py` and `cross_domain_eval.py` use 0.15; the paper's tables use 0.20. Pick one rule: report **both** thresholds side by side, or move the scripts to 0.20. Correct the text (see note below) | 🟢 | R9 | - | edited scripts + paper | Every count in the paper states its threshold | 20 min |
| 0.8 ⏱ | **Perplexity ablation (dev set).** Report: stage C flags 0 additional docs in the staged cascade; A→B alone gives the same F1 (0.97) and 4 false positives at ~56 ms vs 105 ms; A+B mean 0.976 vs 0.983 at best threshold. Add as a short subsection + table row | 🟡 | R7 | - | `paper.tex`, `eval/ablation.py` | Paper states what C adds and what it costs | 25 min |
| 0.9 ⏱ | **Bootstrap CIs on the dev-set tables** (Table I, the per-detector F1s). Expect wide intervals (17 clean docs); say so | 🟢 | R3 | - | `paper.tex` | Every F1 in Table I has a 95% CI | 25 min |
| 0.10 | Remove internal references from the paper ("Phase 2", "task 11.1", file paths) and confirm the author email | 🟢 | - | - | `sections/*.tex` | Paper reads as a stand-alone paper | 20 min |

**Note for 0.7 (verified on the existing result files):**
- Cross-domain benign docs flagged: **12/12 at 0.15, 11/12 at 0.20**. In-domain clean docs flagged: **4/17 at 0.15, 1/17 at 0.20**. The paper currently quotes 12/12 next to 1/17, which mixes thresholds. The conclusion (other-domain documents are flagged far more often) holds at either threshold.
- `adv_007` scores 0.194 before the credential patch, so at 0.20 the first red-team round would be **8/10**, not 9/10 (final 10/10 is unchanged). `adv_010` is 0.106 before the patch.

## Phase 1 - Evaluation protocol (R1, R2, R3)

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 1.1 | Schema + group-aware split (baseline / train / val / test), saved with a hash | 🟡 | R1 | - | `eval/protocol.py`, `data/splits/split_seed42.json` | Split deterministic; no group in two roles | 10 min to run |
| 1.2 | Bootstrap CIs, McNemar, multi-seed aggregation | ✅ | R3 | - | `eval/stats.py`, `test_stats.py` | - | - |
| 1.3 | Latency benchmark: warm-up, >=100 runs per detector, median and p95, hardware printout | 🟢 | - | - | `eval/latency_benchmark.py` | Table; first-call load excluded | 40 min |
| 1.4 | Validation run: fit/tune on train+val, report val table with CIs | 🟡 | R1, R3 | 1.1 | `eval/protocol_eval.py` | No script touches test labels | 30 min |
| 1.5 | **Prevalence-adjusted precision.** Report precision at prevalence 1%, 0.1%, 0.01% from the measured TPR/FPR: `P = TPR·π / (TPR·π + FPR·(1−π))`, plus the false alarms per 10,000 docs | 🟢 | R2 | 1.4 | `eval/protocol_eval.py`, table | Table in report and paper | 45 min |
| 1.6 | ROC and PR curves with AUC + CIs; reliability plots for fused scores | 🔴 | R3 | 1.4 | `eval/plot_curves.py`, `paper/figures/` | Vector PDFs | 1 h |
| 1.7 | **Freeze and run test once.** Checklist: thresholds from validation only; every design choice frozen; clean git tree; tag the commit; `--final` once | 🔴 | R1 | 1.4, 2.6, 3.3, 4.1 | `protocol_report_seed42.md`, `protocol_FINAL_seed42.json` | One test run, git SHA recorded | 20 min |
| 1.8 | Retrofit older scripts (fusion, learned, calibrated, staged) onto the protocol; rewrite `make_results.py` so one command regenerates every table | 🔴 | R1 | 1.4 | `eval/make_results.py` | Clean clone reproduces every paper table | 3 h |

## Phase 2 - Dataset at scale (R2, R10)
Target: ≥300 poisoned and ≥1,000 clean documents, ≥50 per attack type, so one clean doc moves FPR by ≤0.1%.

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 2.1 | Loaders: BIPIA and PoisonedRAG | 🟡 | R2 | - | `data/loaders/bipia.py`, `poisonedrag.py` | Built and run; commit pending | - |
| 2.2 | Loaders: deepset, Open-Prompt-Injection, HackAPrompt | ✅ | - | - | `data/loaders/*` | deepset skipped locally (see 2.10) | - |
| 2.3 | Length-matched clean sampler | 🟡 | R2 | 1.1 | `data/loaders/clean_sampler.py` | Length histograms overlap | - |
| 2.4 | Expand the adversarial set to ≥100 (generated, paraphrased, new categories); log generator and prompt per doc | 🟢 | R10 | - | `data/adversarial/`, provenance TSV | ≥100 docs with provenance | 3 h |
| 2.5 | **Obfuscation generator.** Apply base64, homoglyph (Cyrillic look-alikes), zero-width characters inside trigger phrases, leetspeak, spacing and sentence-splitting to a fixed seed set of ≥20 injection/authority docs → ≥100 variants, each tagged with its technique | 🟢 | R10 | - | `data/generate_obfuscated.py`, `data/adversarial_aware/` | ≥100 docs; every one tagged; pattern-detector result available immediately (no models needed) | 1.5 h |
| 2.6 | Pool all sources, dedupe, assign splits, stats report | 🟡 | R2 | 2.1-2.5 | `data/pooled/`, `pooled_stats.md` | Counts meet targets; no duplicate across splits | rerun after 2.4/2.5/2.9 |
| 2.7 | Label audit: double-label ~100 docs, Cohen's kappa | 🔴 | - | 2.6 | `eval/label_audit.py`, `data/audit/` | Kappa reported | 2 h |
| 2.8 | Datasheet (resolve `[CONFIRM]` items: corpus-doc authorship, adversarial authorship, BEIR/NQ and deepset licences) | 🟡 | - | 2.6 | `data/DATASHEET.md` | No `[CONFIRM]` left | 45 min |
| 2.9 | **More clean documents** from NQ/Wikipedia/READMEs/policy text, length-matched, to ≥1,000 clean | 🟢 | R2 | 2.3 | `pooled.jsonl` | ≥1,000 clean, ≥3 domains | 1 h |
| 2.10 | Fix the `datasets`/`huggingface-hub` conflict (pin `huggingface-hub==1.30.0`) so deepset loads, or drop deepset and say so | 🟢 | - | - | `requirements.txt` | Pool builds with or without deepset, documented | 20 min |

## Phase 3 - Published baseline (R4)
One baseline is enough; two is better.

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 3.1 | Verify an open DeBERTa prompt-injection classifier loads and runs on CPU on your machine (check the exact Hugging Face model id before relying on it); request Meta Prompt Guard access as an optional second | 🟡 | R4 | - | `docs/baselines.md` | Each baseline scores a sample doc | 45 min |
| 3.2 | Wrap in the `DetectorResult` interface (score, reason, latency) | 🟢 | R4 | 3.1 | `detectors/baselines/*.py` | Same interface as ours; unit test | 45 min |
| 3.3 | Run under the protocol: threshold on **validation only**; F1, per-category recall, FPR, latency with CIs; one table "ours vs published" | 🔴 | R4 | 1.4, 2.6, 3.2 | `eval/run_baselines.py`, `baselines.csv` | One comparison table | 1 h |
| 3.4 | Complementarity: which docs each detector uniquely catches; does adding the baseline to the cascade help? | 🔴 | R4 | 3.3 | `eval/complementarity.py` | Overlap figure + paragraph | 1 h |

## Phase 4 - Ablations and mitigations (R5, R6, R7)

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 4.1 | Ablation: each detector alone, each pair, all three, for each fusion variant, on validation, then test once. Dev-set result already in hand (0.8) | 🔴 | R7 | 1.4 | `eval/ablation.py`, table with CIs | Table answers "does perplexity earn 283 ms?" | 1 h |
| 4.2 | Stronger LM for perplexity / stronger embedding model; effect size | 🔴 | - | 1.4 | `eval/model_swap.py` | Before/after table | 2 h |
| 4.3 | Manual error analysis: 20 FN + 20 FP, categorised | 🔴 | - | 1.4, 2.6 | `error_analysis.md` | Failure taxonomy | 1.5 h |
| 4.4 | **Boilerplate subtraction.** Before embedding, drop sentences whose nearest baseline *sentence* has cosine ≥ θ; embed the rest. Choose θ on validation, **not** on `adv_010`. First write and freeze a fresh set of ≥20 boilerplate-anchoring attacks, then implement. Compare against the segment-level negative result | 🟢 | R5 | 1.4 | `detectors/embedding_variants.py`, `eval/boilerplate_subtraction.py` | Failure mode + tested mitigation, with the false-positive cost reported | 3 h |
| 4.5 | **Multi-domain baseline.** Build the embedding baseline from ≥3 domains (READMEs, policy/HR text, one more); score each document against its nearest domain. Hold out different benign docs for testing than were used to build the baseline (write more policy docs, or split the 12). Success = other-domain FPR drops while poison recall holds | 🟢 | R6 | 2.9 | `eval/multidomain_baseline.py` | FPR on held-out HR/support docs: README-only vs multi-domain | 3 h |

## Phase 5 - Calibration with a false-positive guarantee

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 5.1 | Quantile and split-conformal thresholding (clean FPR ≤ α); first test on dev | 🟢 | - | - | `fusion/conformal.py` | Achieved FPR tracks α | 2 h |
| 5.2 | Calibration-fragility experiment (min-max vs quantile) | ✅ | - | - | `eval/calibration_fragility.py` | - | - |
| 5.3 | Sweep α in {0.01, 0.05, 0.10}; achieved FPR vs target and recall, on the pooled data | 🔴 | - | 5.1, 2.6 | `eval/conformal_sweep.py` | Figure + table with CIs | 1.5 h |
| 5.4 | Cost-aware cascade: stage order and exit thresholds under a recall floor; Pareto plot | 🔴 | R7 | 1.4, 4.1 | `fusion/cost_aware.py` | Pareto vs fixed cascade (include the A→B-only cascade) | 2 h |

## Phase 6 - Working system and demo (after the core)

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 6.1 | Installable library: `pds.scan(doc) -> verdict, reasons, latency` | 🟡 | - | - | `pds/`, `pyproject.toml` | `pip install -e .` then `import pds` works | 20 min to verify |
| 6.2 | Streamlit/Gradio demo | 🟢 | - | - | `demo/app.py` | Runs with one command | 2 h |
| 6.3 | Unit tests + GitHub Actions CI | 🟢 | - | - | `tests/`, `ci.yml` | CI green | 2 h |
| 6.4 | LangChain / LlamaIndex retriever middleware | 🔴 | - | 6.1 | `pds/integrations/` | Works in a small RAG app | 2 h |
| 6.5 | Demo video + architecture diagram (diagram ✅: `figures/architecture.tex`) | 🟡 | - | 6.2 | `docs/demo.mp4` | Video exists | 1.5 h |

## Phase 7 - End-to-end impact (R8) 🧪 stretch
Detection metrics are a proxy; this is the number a practitioner cares about. It needs an LLM, so decide the budget first (Q3 below).

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 7.0 | Smallest honest version: ~30 queries over a small clean corpus, poison 1 doc per query, measure attack success with and without PDS using one LLM. Check what BIPIA's own evaluation code offers before writing a harness | 🟢 | R8 | - | `rag/measure_asr.py` | Table: ASR before/after, clean-query accuracy cost | 4 h |
| 7.1 | Full pipeline (retriever + LLM) with controllable poisoning | 🔴 | R8 | 7.0 | `rag/pipeline.py` | Pipeline answers queries | 3 h |
| 7.2 | Sweeps: poisoning rate (1, 5, 10 docs), corpus size, ≥2 retrievers, ≥2 LLMs | 🔴 | R8 | 7.1 | `rag/sweeps.py` | Sweep figures with CIs | 4 h |

## Phase 8 - Robustness (R10)

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 8.1 | Evaluate all detectors on the obfuscation set; evasion rate per technique (the pattern detector will likely collapse; saying so is fine) | 🔴 | R10 | 2.5, 1.4 | `eval/robustness_obfuscation.py` | Evasion-rate table | 1 h |
| 8.2 | Paraphrase robustness at several strengths | 🔴 | - | 1.4, 2.4 | `eval/robustness_paraphrase.py` | Recall-vs-strength plot | 2 h |
| 8.3 | Multi-document attacks and position sensitivity (start/middle/end of long docs; check the 256-word-piece truncation hypothesis for the embedding model) | 🔴 | - | 1.4 | `eval/robustness_position.py` | Table by position | 2 h |
| 8.4 | **Normalise-then-match defence:** Unicode NFKC, strip zero-width characters, map common homoglyphs, decode base64-looking spans, then run the regexes. Report recall recovered and any new false positives | 🔴 | R10 | 8.1 | `detectors/normalize.py` | Before/after evasion table | 2 h |

## Phase 9 - Released benchmark (keep small)

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 9.1 | Attack taxonomy with one example per class (injection, authority spoofing, fluent corpus poisoning, perplexity artefact, credential exposure, obfuscation, boilerplate anchoring) | 🟢 | - | - | `docs/taxonomy.md` | Each class defined with an example | 1 h |
| 9.2 | Leaderboard script: add a detector, get the standard table | 🔴 | R4 | 3.3 | `eval/leaderboard.py` | One command | 1.5 h |

## Phase 10 - Paper

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 10.1 | Related work, ≥20 verified references | ✅ | - | - | 25 references in `paper.tex` | Spot-check page numbers before submission | 30 min |
| 10.2 | Threat model, formal problem statement, ethics | ✅ | - | - | `sections/threat_model.tex`, `ethics.tex` | Symbol fix applied (latency cost is now c_i) | - |
| 10.3 | Limitations and negative results | ✅ | - | - | `sections/limitations.tex` | Strip internal phase references (0.10) | - |
| 10.4 | **External-validity section:** zero-shot BIPIA/PoisonedRAG results (v1), then the v2 rule as a labelled patch | 🔴 | R2 | E1, E2 | `paper.tex` | v1 and v2 side by side; patch wording kept | 1 h |
| 10.5 | **Held-out results section:** validation-chosen thresholds, test once, CIs, prevalence-adjusted precision, flag-everything row, baseline comparison | 🔴 | R1-R4 | 1.7, 3.3 | `paper.tex` | Every number traces to a script | 2 h |
| 10.6 | Mitigation sections: boilerplate subtraction, multi-domain baseline, normalise-then-match, perplexity ablation | 🔴 | R5-R7, R10 | 4.1, 4.4, 4.5, 8.4 | `paper.tex` | Each states the cost as well as the gain | 2 h |
| 10.7 | Rewrite abstract/introduction/conclusion around the held-out numbers (abstract stays free of numbers if required) | 🔴 | - | 10.5, 10.6 | `paper.tex` | PDF compiles within the venue page limit | 1.5 h |
| 10.8 | Final read-through by Karishma and Sree before submission | 🔴 | - | 10.7 | comments | All resolved | - |

## Phase 11 - Submission and IP

| ID | Task | Status | From | Depends on | Output / file | Done when | Est. |
|---|---|---|---|---|---|---|---|
| 11.1 | Meet VIT's IPR / patent cell **before** any public posting | 🟢 | - | - | `docs/ip.md` | Written answer on what can be posted and when | 1 h |
| 11.2 | Prior-art search and one-page disclosure | ✅ | - | - | `docs/prior_art.md` | Honest reading: not likely patentable | - |
| 11.3 | Shortlist 3-5 IEEE-indexed venues: deadlines, page limits, template | 🟢 | - | - | `docs/venues.md` | One recommended target | 1 h |

## Housekeeping (small, do whenever you touch the repo)

| ID | Task | Est. |
|---|---|---|
| H1 | On your machine: `git checkout -b <branch>`, `git add` the Phase 2 files (`eval/protocol.py`, `eval/protocol_eval.py`, `eval/external_transfer.py`, `data/loaders/*`, `data/pool_datasets.py`, `data/DATASHEET.md`), commit, `git push -u origin <branch>`, open a PR | 10 min |
| H2 | Make sure `.gitignore` has `data/external/` and `data/pooled/pooled.jsonl` | 2 min |
| H3 | Confirm `detectors/perplexity_detector.py` has the 1024-token truncation fix | 2 min |
| H4 | Copy the updated paper into the repo: `paper/paper.tex`, `paper/sections/threat_model.tex`, `paper/figures/*` (the figure script reads `eval/results/*.csv`) | 10 min |
| H5 | The perplexity/pattern score cache `eval/results/pooled_scores_seed*.csv` does **not** invalidate when detector code changes; delete it after editing any detector | 1 min |

## External transfer (E-tasks, do in this order)

| ID | Task | Status | Est. |
|---|---|---|---|
| E1 | `python -m eval.external_transfer --sources bipia,poisonedrag`; then `copy eval\results\external_transfer.md eval\results\external_transfer_v1.md` (and `.csv`) **before** touching any detector | 🟢 ⏱ | 5 min |
| E2 | Add the "your answer/response" pattern rule to `detectors/pattern_detector.py`, rerun, save as `external_transfer_v2.md/.csv` | 🔴 (after E1) | 10 min |
| E3 | Write down the honest reading: v1 = zero-shot transfer of the frozen pipeline; v2 = the rule was written after reading BIPIA, so it is a patch | 🔴 (after E2) | 15 min |

## Critical path for one person
`E1 → 0.7/0.8/0.9 → E2 → 1.1/1.4 → (2.9, 2.5, 2.4 in parallel) → 2.6 rerun → 3.1-3.3 → 4.1, 4.4, 4.5 → 8.1/8.4 → 1.7 (test once) → 10.4-10.7 → 11.x`
Stretch items (7.x, 6.x, 5.4) go after the paper's held-out numbers exist.

## Open questions only you can answer
1. **What time is the review today?** It decides whether step 4 (protocol validation run) fits.
2. DATASHEET `[CONFIRM]` items (2.8): who wrote the corpus and adversarial docs, and the licences.
3. Is there an LLM API key or budget for 7.0? If not, 7.x stays future work, or runs a small local model.
4. Is the author email on the paper correct, and what order should the authors be in?
5. Which venue (11.3), and when is its deadline?