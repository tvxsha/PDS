# PDS Roadmap — team version (rev 4, 13 Oct 2026)

Working title: *Where Poison Detectors Fail: A Comparative Failure and Latency Analysis of RAG Defense Signals*

Team: **Tvisha** (lead; owns the pool, the splits, the one-time test run and the paper's results), **Karishma**, **SreeValli**.
Repository: `tvxsha/PDS`, branch `main` (everything below assumes PR #11 is merged).
Deadline: ______ (Tvisha fills in; count backwards from it using the waves in Section 11).

> This file is written for people **and for their Claude sessions**. Every task card says what to read first, what to create, which commands to run, and when the task is done. A Claude that follows a card literally should produce code that fits the project. Section 12 has the prompt to start a task with.

## Contents
1. How to use this file
2. The project in one page (primer)
3. Repository map and interfaces
4. Rules (not negotiable)
5. Setup (do this first)
6. Team split at a glance
7. Already done, by person
8. To do: Tvisha
9. To do: Karishma
10. To do: SreeValli
11. Order of work: waves, dependencies, freeze
12. Templates: Claude start prompt, PR description, result report
13. Glossary and metric definitions
14. Known pitfalls (read once)

---

## 1. How to use this file

1. Find your name in Sections 8 to 10. Do your tasks in wave order (Section 11). One task = one branch = one pull request.
2. Before you start a task, give your Claude **Sections 2 to 5 and 14, plus the task card** (use the prompt in Section 12). Tell it to read the files under "Read first" before writing code.
3. Tick your own checkbox in the list at the top of your section in the same PR that finishes the task (`- [ ]` becomes `- [x]`).
4. If a card is unclear or a file it mentions does not exist, stop and ask Tvisha. Do not guess file names, function names or numbers.
5. Cross-review: Karishma reviews SreeValli's PRs and SreeValli reviews Karishma's (10 minutes each, check the "Done when" list). Tvisha does the final review and the merge.

---

## 2. The project in one page (primer)

**Problem.** A RAG pipeline retrieves documents and puts them in an LLM's prompt. An attacker can (a) hide instructions inside a document (*indirect prompt injection*, benchmark **BIPIA**) or (b) plant passages that make the system give a chosen wrong answer (*knowledge/corpus poisoning*, benchmark **PoisonedRAG**). PDS screens each document **before** it reaches the LLM.

**What we built.**

| Signal | File | What it computes | Typical cost (CPU, median) |
|---|---|---|---|
| Pattern bank | `detectors/pattern_detector.py` | max weight of matched regexes (`SUSPICIOUS_PATTERNS`), 0 to 1 | about 0.1 ms |
| Embedding anomaly | `detectors/embedding_detector.py` | MiniLM (`all-MiniLM-L6-v2`); `(1 - max cosine similarity to a baseline corpus) / 2` | 35 to 59 ms |
| Perplexity | `detectors/perplexity_detector.py` | GPT-2 perplexity on the first 1,024 tokens, banded to 0 to 1 | 190 to 270 ms (mean about 800 to 1,000 ms because of long documents) |
| Sentence instruction classifier | `detectors/sentence_detector.py` | TF-IDF (1 to 2 grams) + hand features + logistic regression per sentence; document score = highest sentence probability | not yet benchmarked (task 1.3) |
| Published baseline | `detectors/published_classifier.py` | `protectai/deberta-v3-base-prompt-injection-v2`, 512-token windows, document score = most suspicious window (logit margin) | about 250 ms |
| Fusion | `fusion/fixed.py`, `fusion/learned.py`, `fusion/` + `eval/fpr_eval.py` | naive average, calibrated average, logistic regression on the scores | negligible |

**Thesis of the paper.** Cheap defence signals look excellent on small hand-made data and fail on realistic data. We show *where* each fails, at what latency, under an honest protocol (held-out splits, fixed false-alarm budgets, several seeds, no tuning on test).

**Evaluation idea in one paragraph.** Every method gets a threshold chosen on the TRAIN split so that at most 5% (or 1%) of CLEAN training documents are flagged (out-of-fold scores for learned methods). That threshold is applied unchanged to the VALIDATION split, and we report recall (TPR) and the realised false-alarm rate (FPR) with group-bootstrap confidence intervals, precision at 1% and 0.1% attack prevalence, and results per source. The TEST split is used once at the very end.

**Where we are (validation split, seeds 42 / 43 / 44, recall at the 5% false-alarm budget, threshold fixed on train).**

| Method | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| Pattern bank | 0.24 | 0.27 | 0.26 |
| Embedding | 0.08 | 0.13 | 0.04 |
| Perplexity | 0.08 | 0.04 | 0.05 |
| Naive average of the 3 | 0.22 | 0.26 | 0.25 |
| LR on the 3 detectors | 0.33 | 0.38 | 0.32 |
| Sentence classifier | 0.64 | 0.66 | 0.61 |
| LR + sentence classifier | 0.64 | 0.72 | 0.69 |
| Published classifier (cut-off from our train split) | 0.10 | 0.12 | 0.11 |

At the 1% budget, LR + sentence reaches 0.36 / 0.38 / 0.32 recall with 0.6% / 0.6% / 0% false alarms. At 1% attack prevalence about 60% to 85% of its alerts would still be false.

**Findings that shape the work.**
1. The old 77-document development set gave F1 about 0.97. On the pooled held-out data "flag everything" already has F1 0.70, and no single cheap detector beats it at a useful false-alarm rate.
2. The pattern bank catches nothing on PoisonedRAG (recall 0.00); embedding and perplexity are near chance alone.
3. The sentence classifier is strong when trained on the same sources (0.61 to 0.66) but does not transfer to a source it has not seen (leave-one-source-out AUC about 0.58 to 0.60 on BIPIA, about 0.70 on PoisonedRAG; with a threshold carried over it flags 76% to 80% of clean BIPIA documents). Most of its in-corpus gain is on PoisonedRAG, and we have not yet shown it is not just learning the attack generator's style (task N1).
4. The published classifier is a prompt-injection model; on our documents it catches about 10% to 14% of attacks at 11% to 13% false alarms, and adds nothing to our best fusion.
5. Latency: the three cheap detectors together have a median of about 220 to 330 ms (perplexity dominates); the published classifier costs about the same at the median.

**Pool and splits (FROZEN).** `python -m data.pool_datasets` builds `data/pooled/pooled.jsonl`: **2,291 documents, 1,143 clean and 1,148 poisoned**.

| Source | clean | poisoned | What it is |
|---|---|---|---|
| `bipia` | 578 | 578 | BIPIA email / table / code host documents, each once clean and once with one inserted attack (text or code) |
| `poisonedrag` | 498 | 500 | 100 PoisonedRAG targets x 5 adversarial passages; clean = length-matched BEIR NQ passages |
| `pds_dev` | 55 | 60 | our hand-built README-style set (4 attack categories x 15) |
| `pds_adv` | 0 | 10 | first adversarial red-team set |
| `pds_xdomain` | 12 | 0 | benign HR / support / policy documents |

Split per seed (identical sizes for seeds 42, 43, 44): **baseline 285** (clean only, used only to build the embedding detector's reference corpus), **train 1,116**, **val 373** (201 poisoned / 172 clean), **test 373**; 144 poisoned BIPIA twins of baseline groups are dropped. Documents are split by **group** (a BIPIA host and its poisoned copy, or the 5 passages of one PoisonedRAG target, never straddle roles).
Split hashes: seed 42 = `6bdb8ff98e`, seed 43 = `93a61f0778`, seed 44 = `4d1126b320`.

---

## 3. Repository map and interfaces

```
PDS/
├── ROADMAP.md                    this file
├── README.md                     overview (its results table is stale: task 6.1)
├── pyproject.toml                installable package: pds, detectors, fusion
├── detectors/
│   ├── base.py                   DetectorResult + @timed(name) decorator (shared interface)
│   ├── pattern_detector.py       SUSPICIOUS_PATTERNS; score_document(text)
│   ├── embedding_detector.py     BaselineCorpus(list_of_texts); score_document(text, baseline)
│   ├── perplexity_detector.py    score_document(text)
│   ├── sentence_detector.py      SentenceInstructionDetector().fit(texts, labels); .score(text); .score_units(text)
│   ├── published_classifier.py   score_document(text); document_margin(text) -> (margin, n_windows, truncated)
│   └── segmented_embedding_detector.py   experiment, not in the main pipeline
├── fusion/                       fixed.py (weights, thresholds), learned.py (LR on the 3 scores)
├── pds/__init__.py               pds.scan(text) -> ScanResult (3 detectors + fixed fusion)
├── data/
│   ├── clean/ poisoned/ adversarial/ cross_domain/    hand-built sets (55 / 60 / 10 / 12 files)
│   ├── _provenance/*.tsv         where each hand-built document came from
│   ├── loaders/                  bipia.py, poisonedrag.py, clean_sampler.py, pds_local.py, deepset.py, hackaprompt.py, opi.py
│   ├── pool_datasets.py          builds the pool (defaults = the frozen pool)
│   ├── pooled/pooled.jsonl       GIT-IGNORED; rebuild locally (Section 5)
│   ├── pooled_stats.md           counts, lengths, shortcut check (committed)
│   ├── splits/split_seed{42,43,44}.json     roles baseline/train/val/test (FROZEN, committed)
│   ├── evalsets/                 NEW: separate evaluation sets in Row schema (*.jsonl) + their builders; never part of the pool
│   └── external/                 git-ignored downloads (do not put our own sets here)
├── eval/
│   ├── protocol.py               Row schema, make_split, validate_split, load_split, apply_split, read_jsonl
│   ├── protocol_eval.py          scores the pool with the 3 detectors -> score cache; validation report; --final = test (once)
│   ├── fpr_eval.py               fixed-false-alarm-budget evaluation (the main results tables)
│   ├── sentence_eval.py          helpers find_scores_file, load_docs, recall_at_fpr (+ an older report)
│   ├── sentence_loso.py          leave-one-source-out experiment
│   ├── sentence_adapt.py         clean-only calibration and few-shot adaptation experiment
│   ├── baseline_eval.py          published classifier vs ours, complementarity, latency table
│   ├── ablation.py, dev_confidence_intervals.py    analyses of the old 77-document dev set
│   ├── stats.py, test_stats.py   compute_metrics, bootstrap_ci, mcnemar_test, aggregate_seeds
│   ├── (older dev-set scripts)   run_evaluation.py, calibrated_fusion.py, staged_pipeline.py, adversarial_test.py, cross_domain_eval.py, external_transfer.py, explain.py, ...
│   └── results/                  every report (.md) and score cache (.csv), committed
├── docs/                         baselines.md, prior_art.md (+ new notes from the tasks below)
├── tests/                        NEW (task 6.3); older tests live next to their code (eval/test_stats.py)
├── demo/                         NEW (task 6.2): Streamlit demo
├── requirements.txt, requirements-ci.txt, requirements-demo.txt     NEW (tasks 0.4, 6.2)
├── paper/                        OLD paper source (task H4 replaces it); later paper/figures/ and paper/tables/ (generated)
└── related_work.md, dataset_methodology.md, slides_outline.md, demo_commands.md
```

**Interfaces you will use.**

```python
# eval/protocol.py
from eval.protocol import Row, read_jsonl, load_split, apply_split, split_hash, parse_meta, ROLES, POOLED_PATH
# Row fields: id, text, label (0 clean / 1 poisoned), source, attack_type, domain, length, group, meta ("k=v|k=v")
# ROLES = ("baseline", "train", "val", "test")

# eval/sentence_eval.py
from eval.sentence_eval import find_scores_file, load_docs
docs = load_docs(seed, ["train", "val"], find_scores_file(seed))
# -> {"train": [ {id, text, y, src, grp, pat, emb, ppl}, ... ], "val": [...]}
# NOTE: no domain / attack_type / meta in these dicts; join to the pool rows by id when you need them.

# eval/fpr_eval.py
from eval.fpr_eval import (fit_everything, threshold_at_budget, best_case_recall, group_bootstrap,
                           precision, lr_fit, oof_lr, source_group, BUDGETS, PREVALENCES)
train_scores, val_scores = fit_everything(train, val)   # dict: method name -> np.array; train scores are out-of-fold
t = threshold_at_budget(train_scores["LR + sentence classifier"][ytr == 0], 0.05)   # threshold from TRAIN clean scores
# precision(tpr, fpr, prevalence, n_clean); source_group maps pds_dev / pds_adv / pds_xdomain to "pds"

# detectors/base.py
from detectors.base import DetectorResult, timed
# DetectorResult(detector_name, score 0..1, triggered_reason, latency_ms, raw_details)
# @timed("name") wraps a function that returns (score, reason, raw_details_dict)

# detectors/embedding_detector.py
baseline = BaselineCorpus([texts of the BASELINE-role documents])
res = score_document(text, baseline)       # DetectorResult

# detectors/sentence_detector.py
m = SentenceInstructionDetector().fit(train_texts, train_labels)
m.score(text)                              # max sentence probability
probs, units = m.score_units(text)         # per-sentence probabilities and sentences
```

**Interfaces created by tasks 1.9 and 1.7a** (they exist on `main` after those tasks are merged; read the card for the details).

```python
# eval/signal_cache.py  (task 1.9)
signal_path(name, seed); load_signal(name, seed)                  # eval/results/signal_<name>_seed<N>_<splithash>.csv: id,score,ms
score_missing(name, seed, roles, scorer, final=False)             # scorer(row)->float; "test" in roles needs final=True (only final_test.py does that)
get_embeddings(seed, roles)                                       # (ids, matrix) MiniLM embeddings, cached as eval/results/emb_minilm_*.npz (git-ignored)
load_evalset(set_name); score_external(set_name, seed, extra=None)  # data/evalsets/<set_name>.jsonl scored with the seed's detectors; cache external_<set>_seed<N>_<hash>.csv

# eval/apply_models.py  (task 1.9)
fm = fit_models(seed)       # fm.thresholds[method][0.05], fm.sentence_model, fm.score(docs) -> {method: np.ndarray}; same method names as fit_everything

# eval/final_test.py  (task 1.7a)  python -m eval.final_test --seed N --dry-run | --final
# outputs: eval/results/final_<dryrun|test>_seed<N>_scores.csv (id,label,source,attack_type,group,method,score) and ..._thresholds.json
```

**Evaluation sets** (`data/evalsets/<name>.jsonl`, one `Row` per line, `source` starts with `ext_`) are separate from the pool: adversarial A / B, obfuscated A / B, deepset, paraphrases. They contain no clean documents unless said so, so on them only recall is reported; false alarms come from the pool.

**Files that exist on disk but must not be hand-edited:** `data/pooled_stats.md`, anything in `eval/results/` (regenerate with the script), `data/splits/*.json`.

**Score caches (committed in `eval/results/`).**
- `pooled_scores_seed<N>_<splithash>.csv`: columns `id, pattern_score, embedding_score, perplexity_score, pattern_ms, embedding_ms, perplexity_ms`. The embedding scores depend on the seed's baseline documents, so the cache is per seed. **It also contains rows for the TEST ids (they were scored once so the final run is fast). Never join or evaluate those rows.**
- `published_scores_protectai__deberta-v3-base-prompt-injection-v2_w8.csv`: columns `id, margin, windows, truncated, ms`; shared across seeds, keyed by document id.
- The `*_ms` columns contain occasional huge values from when a laptop went to sleep mid-run. Use medians and 95th percentiles, never means, and use the dedicated benchmark (task 1.3) for latency claims.

---

## 4. Rules (not negotiable)

1. **The TEST split is never used before the freeze.** No `--final`, no peeking, no "just looking" at test rows or test scores. Any script you write must request roles explicitly (`["train", "val"]`, or `baseline` where needed). Only Tvisha runs the test split, once per seed (tasks 1.7a and 1.7).
2. **The pool and the splits are frozen.** Do not change `data/pool_datasets.py` settings, `data/loaders/*` behaviour for existing sources, or `data/splits/*`. New data (more adversarial documents, obfuscated variants, deepset, paraphrases) are **separate evaluation sets** in `data/evalsets/` with their own eval script (not `data/external/`, which is git-ignored). They are never added to the pool. Only Tvisha may change the pool or splits, and then all three seeds must be re-run.
3. **Thresholds come from TRAIN, never from validation or test.** Operating points are the 5% and 1% false-alarm budgets set from the train clean-score quantile (out-of-fold scores for learned methods). "Best-case" numbers (threshold picked on validation itself) may be shown only if labelled optimistic.
4. **Always run seeds 42, 43 and 44** and show all three (or mean and range). Never report only the best seed.
5. **Patches are labelled.** Anything written after looking at failures (a new regex, a normaliser, a tuned parameter) is reported as a *patch, not independent evidence* and is evaluated on data written or generated afterwards by someone who has not tuned against it.
6. **Negative results stay in the paper.** "It did not help" with numbers is a result.
7. **Every number traces to a script and a file** in `eval/results/`. Reports are produced by scripts, never edited by hand, and end with a caveats section ("Validation numbers only; the test split has not been used.").
8. **Small cells are flagged.** Always print n. With few clean documents, "zero false alarms" only means "below 3/n" (rule of three). Group bootstrap for intervals when documents share groups.
9. **Scripts run from the repo root** as `python -m eval.<name> --seed N`, are deterministic (fixed seeds), print which cache files they used, and write their report to `eval/results/<name>_seed<N>.md`.
10. **No new heavy dependencies without asking Tvisha.** Never `pip install datasets` into the project venv (it pulls `huggingface-hub` 2.x and breaks `transformers`). Keep `huggingface-hub` below 2.0.
11. **Do not commit** the pool file, `venv/`, `__pycache__/`, model weights, API keys, personal data, or other people's datasets beyond what is already in the repo. Run `git status` and read it before every commit.
12. **Git workflow.** Branch from an up-to-date `main`: `git checkout main`, `git pull`, `git checkout -b <yourname>/<task-id>-<short-name>` (example `karishma/2.4-adversarial-set`). One task per PR. Do not force-push to `main`. If you touch a protected file (below), say so in the PR.
    **Protected files (change only after agreeing with Tvisha):** `eval/protocol.py`, `eval/protocol_eval.py`, `eval/fpr_eval.py`, `eval/sentence_eval.py`, `data/pool_datasets.py`, `data/splits/*`, `ROADMAP.md` (except your own checkboxes).
13. **No AI attribution in git.** Commit messages and PR descriptions must not contain `Co-Authored-By` lines, "Generated with ..." footers, or session links. Write plain commit messages: one imperative sentence, optionally a short body.
14. **Be honest in text.** Say what a result shows and what it does not. Do not call an in-corpus number "generalisation". Do not fix a number by re-running until it looks better.
15. **Attack text stays harmless.** Adversarial and injected documents use obviously fake targets (`example.com`, fake keys like `sk-TEST-0000`). No working malware, no real credentials, no personal data.

---

## 5. Setup (do this first; Karishma and SreeValli: tasks 0.A and 0.B)

Requirements: Git, VS Code, Python 3.11 to 3.14 (Tvisha uses 3.14), about 3 GB free disk, internet for the first run.

**1. Clone and enter the folder**
```
git clone https://github.com/tvxsha/PDS.git
cd PDS
```

**2. Create and activate the virtual environment**

Windows (cmd):
```
python -m venv venv
venv\Scripts\activate
```
macOS / Linux:
```
python3 -m venv venv
source venv/bin/activate
```

**3. Install packages.** When `requirements.txt` exists (task 0.4), use `pip install -r requirements.txt`. Until then:
```
python -m pip install --upgrade pip
pip install numpy scipy scikit-learn matplotlib pandas pyarrow pytest torch "transformers==5.16.1" "sentence-transformers==6.0.1" "huggingface-hub>=1.5.0,<2.0"
pip check
```
`pip check` must print `No broken requirements found.` (The quotes matter in Windows cmd.) If a pinned version will not install on your Python version, tell Tvisha. Do not loosen pins silently.

**4. Build the pool** (downloads BIPIA, PoisonedRAG and the BEIR NQ corpus; several minutes):
```
python -m data.pool_datasets
```
It should end with `Pooled 2291 documents (1143 clean / 1148 poisoned)`. If the BEIR download fails, ask Tvisha for `pooled.jsonl` and put it at `data/pooled/pooled.jsonl` (create the folder).

**5. Check that your pool is identical to everyone else's**
```
python -m eval.protocol --seed 42
python -m eval.protocol --seed 43
python -m eval.protocol --seed 44
```
The lines `split hash:` must read `6bdb8ff98e`, `93a61f0778`, `4d1126b320`. Then run `git status`: nothing may be modified. If `data/splits/*` or `data/pooled_stats.md` show changes, your pool differs. Run `git checkout -- data/splits data/pooled_stats.md` and tell Tvisha.

**6. Smoke test** (uses the committed score caches; takes a few minutes; downloads no models):
```
python -m eval.fpr_eval --seed 42
```
At the 5% budget you should see pattern bank TPR 0.24, sentence classifier 0.64, LR + sentence classifier 0.64 (within 0.01). Then `git status` should still show nothing modified (the report is regenerated identically). Models (MiniLM about 90 MB, GPT-2 about 500 MB, DeBERTa about 700 MB) download only when a task actually runs a detector.

**Harmless messages:** "You are sending unauthenticated requests to the HF Hub", the `torch.jit.script ... not supported in Python 3.14` FutureWarning, and "`loss_type=None` was set in the config".

---

## 6. Team split at a glance

Core hours are hands-on time (not counting compute). Stretch tasks are dropped first if time runs short. Tvisha additionally reviews and merges every PR (about 3 hours, not counted below).

| Person | Core tasks | Core hours | Stretch tasks | Stretch hours | Main areas |
|---|---|---|---|---|---|
| Tvisha | 0.4, 1.9, N1, 4.5, 4.10, 5.3, 5.4, 1.7a, 1.7, 11.1, 10.9, H4, 10.5, 10.6, 10.7 | 24 | 4.2, 7.2, 7.3 | 6 | frozen test run, fusion, cascade, per-domain baseline, diagnosis of the sentence classifier, paper results |
| Karishma | 0.A, 2.4, 2.7, 3.7, 4.3, 4.4, 9.1, N2, N3, 11.3, 6.1, 10.2, 10.3, 10.4, 10.8, 10.12 | 26.5 | 2.8, 8.2, 6.5, 9.3, 6.6 | 7.5 | new data and annotation (adversarial set, label audit, datasheet), boilerplate subtraction, error analysis, block/review policy, paper related work / data / limitations |
| SreeValli | 0.B, 1.3, 2.10, 2.5, 8.3, 5.1, 8.1, 8.4, 6.2, 6.3, N4, 1.6, 10.1, 10.10, 10.11, 10.13 | 26.5 | 4.1, 4.9, 6.4, 7.0 | 9.5 | latency benchmark, figures, conformal thresholds, obfuscation and robustness, deepset, demo, tests and CI, paper tables / system / reproducibility |


---

## 7. Already done, by person

This list was written from the git history. If something you did is missing or wrong, fix this section in your next PR. Everything below is merged on `main` (PRs #1 to #11).

### Tvisha: already done
- Project scaffold and the shared detector interface (`detectors/base.py`: `DetectorResult`, `@timed`).
- The three detectors: instruction pattern bank, embedding anomaly (MiniLM + `BaselineCorpus`), perplexity (GPT-2, banded).
- Hand-built datasets and generators with provenance (`data/clean`, `data/poisoned`, `data/_provenance/*.tsv`, `data/generate_*.py`).
- Fusion: fixed weights, calibrated, learned (logistic regression) and a staged cascade on the old 77-document dev set.
- Regex fixes (possessive-pronoun patterns, credential patch) and the first adversarial red-team (10 documents, `data/adversarial/adv_001` to `adv_010`).
- Segmented-embedding experiments and the summary-report generator.
- External transfer to BIPIA and PoisonedRAG (`eval/external_transfer.py`, v1 and v2 reports).
- The evaluation protocol: `Row` schema, group-aware splits, `protocol_eval.py`, `fpr_eval.py` (fixed false-alarm budgets, group bootstrap, precision at low prevalence).
- Pool v2: 2,291 documents from five sources, BIPIA host de-duplication, shortcut checks (type-only AUC 0.5), splits for seeds 42 to 44 with hashes.
- The sentence instruction classifier with leave-one-source-out and clean-only adaptation experiments.
- The published-baseline wrapper and evaluation (`detectors/published_classifier.py`, `eval/baseline_eval.py`).
- Ablation and confidence-interval scripts for the old dev set.
- Reviewed and merged PRs #1 to #11; keeps an 11-page paper draft outside the repo (task H4 brings it in).

### Karishma: already done
- Cross-domain benign set: 12 benign HR / support / policy documents (`data/cross_domain/crossdomain_001` to `012`) used to check false alarms on instruction-like but harmless text; they are now the `pds_xdomain` source in the pool.
- `related_work.md` (PR #1).

### SreeValli: already done
- `dataset_methodology.md` and the dataset provenance audit (PR #2).
- `eval/explain.py` with `format_verdict()` (PR #3; the per-detector-threshold fix was committed by Tvisha).
- `eval/stats.py` (`compute_metrics`, `bootstrap_ci`, `mcnemar_test`, `aggregate_seeds`) with `eval/test_stats.py`.
- `eval/calibration_fragility.py` and its figure.
- The installable `pds` package (`pds.scan`) and `pyproject.toml`.
- `slides_outline.md` and `demo_commands.md`.
- `docs/baselines.md` and `docs/prior_art.md`.
- Paper sections `paper/sections/ethics.tex`, `limitations.tex`, `threat_model.tex`.
- Dataset loaders `data/loaders/_schema.py`, `deepset.py`, `hackaprompt.py`, `opi.py`.

---

## 8. To do: Tvisha

**Core tasks (tick your own box in the PR that finishes the task):**

- [x] **0.4** requirements.txt and .gitignore check (0.5 h, wave 0)
- [x] **1.9** Shared plumbing: score-cache lookup, signal cache, model re-fit helper (3 h, wave 0)
- [x] **N1** Diagnose the sentence classifier (is it learning instructions or generator style?) (2.5 h, wave 1)
- [x] **4.5** Domain-aware embedding variants (2.5 h, wave 1)
- [x] **1.7a** The one-shot final test script (with dry run) (2 h, wave 1)
- [ ] **4.10** Re-fit fusion with the new signals (1.5 h, wave 2)
- [ ] **5.3** False-alarm budget sweep (1.5 h, wave 2)
- [ ] **5.4** Cost-aware cascade (3 h, wave 2)
- [ ] **1.7** Freeze and run the test split once per seed (0.5 h, wave 3)
- [ ] **10.9** Authorship and acknowledgements (0.25 h, wave 3)
- [x] **H4** Bring the updated paper source into the repo (0.25 h, wave 3)
- [ ] **10.5** Paper: held-out (test) results section (2 h, wave 3)
- [ ] **10.6** Paper: mitigations and what they cost (2 h, wave 3)
- [ ] **10.7** Paper: abstract, introduction, conclusion (1.5 h, wave 3)

**Stretch (only if core is done and time is left):**

- [ ] **4.2** Stronger perplexity and embedding models (2 h)
- [ ] **7.2** Sensitivity sweeps (3 h)
- [ ] **7.3** Sweeps write-up (1 h)

Core total: 24 h in 15 tasks.

Tvisha also reviews and merges every PR (about 3 hours), answers questions, and is the only person who runs the final test split.

#### 0.4 · requirements.txt and .gitignore check
**Owner:** Tvisha · **Est:** 0.5 h · **Wave:** 0 · **Tier:** core · **Depends on:** — · **Blocks:** 6.2, 6.3, 10.10

**Why.** Teammates currently install from a pip line in this file. One pinned file avoids "works on my machine" and is needed for CI and the reproducibility appendix.
**Read first.** `pyproject.toml`, `.gitignore`, Section 5 of this file.
**Do.**
1. In your working venv run `pip freeze > freeze_full.txt` (do not commit it). Create `requirements.txt` with only the direct dependencies: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `pandas`, `pyarrow`, `pytest`, `torch`, `transformers==5.16.1`, `sentence-transformers==6.0.1`, `huggingface-hub>=1.5.0,<2.0`. Pin each other package to the version that is installed on your machine (`pip show <name>`), written as `name==version`.
2. Create `requirements-ci.txt` with the packages that need no model downloads: `numpy`, `scipy`, `scikit-learn`, `pandas`, `matplotlib`, `pytest` (pinned the same way). CI (task 6.3) installs this file.
3. Open `.gitignore` and make sure it contains: `venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `data/pooled/pooled.jsonl`, `data/external/`, `eval/results/*.npz`, `*.log`, `freeze_full.txt`. It must NOT ignore `data/evalsets/`, `eval/results/*.csv`, `eval/results/*.md` or `eval/results/*.json`.
**Create / edit.** `requirements.txt`, `requirements-ci.txt`, `.gitignore`.
**Output.** The two requirements files.
**Done when.** In a brand-new venv `pip install -r requirements.txt` followed by `pip check` prints `No broken requirements found.`; `git status` shows only the three files above.
**Watch out.** Quote the `huggingface-hub` line if you test it by hand in cmd. Never add `datasets` (Rule 10).

#### 1.9 · Shared plumbing: score-cache lookup, signal cache, model re-fit helper
**Owner:** Tvisha · **Est:** 3 h · **Wave:** 0 · **Tier:** core · **Depends on:** — · **Blocks:** N1, 4.5, 1.7a, 4.4, 2.10, 2.5, 8.3, 6.3, 4.10, N2, N3, 5.1, 8.1, 4.2, 7.2, 8.2, 4.1, 4.9, 7.0

**Why.** Almost every other task needs (a) the right score cache for a seed, (b) a place to store new per-document signals without touching the protected caches, (c) a way to apply the train-fitted models and thresholds to documents that are not in the pool (adversarial, obfuscated, deepset). Building this once, with a guard against the test split, keeps the code of all three people consistent.
**Read first.** `eval/sentence_eval.py` (`find_scores_file`, `load_docs`), `eval/fpr_eval.py` (`fit_everything`, `threshold_at_budget`), `eval/protocol.py` (`load_split`, `split_hash`, `apply_split`, `Row`), `eval/baseline_eval.py` (`read_cache`, `score_missing` show the resume-able CSV pattern to copy), Section 3 of this file.
**Do.**
1. **Part A: fix `find_scores_file`** in `eval/sentence_eval.py` (protected file: say so in the PR). New logic: load `data/splits/split_seed{seed}.json`, compute `split_hash(split)`, return `eval/results/pooled_scores_seed{seed}_{hash}.csv` if it exists; otherwise search CSVs by the header as now but keep only paths whose basename contains BOTH `seed{seed}_` and the hash; if none, `raise SystemExit` with a message telling the user to run `python -m eval.protocol_eval --seed N`. Never choose by modification time. Keep the `override` argument.
2. **Part B: create `eval/signal_cache.py`** with exactly this interface (other cards rely on these names):
```python
SIGNAL_DIR = "eval/results"
def signal_path(name, seed) -> str            # eval/results/signal_<name>_seed<seed>_<splithash>.csv
def load_signal(name, seed) -> dict           # {id: (score, ms)}; {} if the file does not exist
def score_missing(name, seed, roles, scorer, final=False) -> dict
    # scorer(row) -> float, row is an eval.protocol.Row. Scores rows of the requested roles that are not cached yet,
    # appends "id,score,ms" lines (flush every 25 rows, so a crash or sleep loses little), times each call with
    # time.perf_counter(), and returns {id: (score, ms)} for everything cached.
    # raises ValueError("test role requested") if "test" in roles and final is False.
def get_embeddings(seed, roles) -> tuple      # (ids, float32 matrix) of MiniLM embeddings, cached per id in eval/results/emb_minilm_seed<seed>_<splithash>.npz (extended when new ids are requested); same role guard
def evalset_path(set_name) -> str             # data/evalsets/<set_name>.jsonl  (Row schema, one JSON object per line)
def load_evalset(set_name) -> list            # list of Row
def external_path(set_name, seed) -> str      # eval/results/external_<set_name>_seed<seed>_<splithash>.csv   (long format: id,signal,score,ms)
def score_external(set_name, seed, extra=None) -> dict
    # scores every document of the evaluation set with the three base detectors (embedding baseline = THIS seed's "baseline"-role documents)
    # plus any extra {signal_name: fn(text)->float}; caches in external_path; returns {signal: {id: score}}.
```
3. **Part C: create `eval/apply_models.py`:**
```python
class FittedModels:
    seed: int
    sentence_model: object  # the SentenceInstructionDetector fitted on all of TRAIN
    thresholds: dict      # {method: {0.05: t, 0.01: t}}, thresholds from TRAIN clean scores (out-of-fold for learned methods)
    val_scores: dict      # {method: np.ndarray} for the seed's validation documents (used by --check)
    def score(self, docs) -> dict   # docs: list of dict(id, text, pat, emb, ppl) -> {method: np.ndarray}
def fit_models(seed) -> FittedModels
```
   `fit_models` repeats the fitting done inside `fit_everything` (sentence detector on all of train; LR on 3 scores; LR on 3 scores + out-of-fold sentence scores) but keeps the fitted objects, and uses the same method names as `fit_everything` ("pattern bank", "embedding", "perplexity", "naive average of the 3", "sentence classifier", "LR on the 3 current scores", "LR + sentence classifier"). Add `python -m eval.apply_models --seed 42 --check`, which prints `OK` only if the validation scores from `fit_models(42)` equal those from `fit_everything` to 1e-9 (this catches drift if anyone edits `fpr_eval.py`).
4. **Tests** in `eval/test_signal_cache.py` (pytest, no models): a fake scorer and a temporary `SIGNAL_DIR` (monkeypatch) prove that (i) resuming skips cached ids, (ii) `roles=["test"]` raises unless `final=True`, (iii) the CSV header is `id,score,ms`.
**Create / edit.** `eval/sentence_eval.py` (protected, small), new `eval/signal_cache.py`, `eval/apply_models.py`, `eval/test_signal_cache.py`; add `data/evalsets/__init__.py` (empty) and a one-line `data/evalsets/README.md` ("separate evaluation sets in Row schema; never part of the pool").
**Output.** The three modules, the test file.
**Done when.** `python -m pytest eval/test_signal_cache.py` passes; `python -m eval.apply_models --seed 42 --check` (and 43, 44) prints OK; `python -m eval.fpr_eval --seed 42` still prints pattern 0.24 / sentence 0.64 / LR+sentence 0.64; after deleting the three `pooled_scores_*` files from a scratch copy, `find_scores_file(42)` fails with the clear message instead of picking a wrong file.
**Watch out.** The test role may be passed only with `final=True`, and only `eval/final_test.py` does that. Do not import torch at module import time in `signal_cache.py` or `apply_models.py` (CI has no torch); import detectors inside the functions that need them.

#### N1 · Diagnose the sentence classifier (is it learning instructions or generator style?)
**Owner:** Tvisha · **Est:** 2.5 h · **Wave:** 1 · **Tier:** core · **Depends on:** 1.9 · **Blocks:** 10.5, 10.8

**Why.** The sentence classifier is our best signal (recall about 0.64 at 5% false alarms) but most of its in-corpus gain is on PoisonedRAG, where the pattern bank scores 0.00, and it does not transfer to a held-out source (leave-one-source-out AUC about 0.58 to 0.71). PoisonedRAG poisoned passages are machine-written, the clean ones are Wikipedia passages. We must know whether the classifier detects instructions or just "machine-written vs Wikipedia". The answer decides what the paper may claim.
**Read first.** `detectors/sentence_detector.py` (`SentenceInstructionDetector`: `.vec` TF-IDF, `.clf` logistic regression, 6 hand features appended after the TF-IDF columns in this order: ends with "?", has you/your, has modal word, has output word, has code fence, shorter than 40 words), `eval/fpr_eval.py` (`fit_everything`, `source_group`), `eval/sentence_loso.py`, `eval/results/sentence_loso_seed42.md`.
**Do.** Write `eval/diagnose_sentence.py` (`--seed N`, train + validation only, deterministic). The report sections:
1. **Features:** the 25 most positive and 25 most negative coefficients of `clf`, with names from `vec.get_feature_names_out()` followed by the 6 hand-feature names.
2. **Where the recall is:** val TPR / FPR at the train-derived 5% threshold for "sentence classifier", per source group and per `attack_type` (join to the pool rows by id; `load_docs` has no attack_type).
3. **Control 1, length:** AUC on val, per source group, of (a) character length, (b) number of sentences (`split_units`).
4. **Control 2, style features:** logistic regression (class_weight balanced) on 6 document-level style features: log characters, mean words per sentence, share of capitalised words, punctuation per 100 characters, digit share, type-token ratio. Fit on train, report val AUC and TPR at the 5% threshold taken from out-of-fold train scores (`oof_lr`), per source group.
5. **Control 3, bag of style:** document-level TF-IDF on character 3 to 5-grams (`analyzer="char_wb"`, `max_features=50000`) + logistic regression, same reporting (threshold from out-of-fold train scores).
6. **Top sentences:** for val, the 15 highest-scoring sentences (with probability, source, label) among poisoned PoisonedRAG documents, among clean documents (these are the false alarms), and among poisoned BIPIA documents. Truncate each sentence to 200 characters.
7. A caveats section (what the controls can and cannot show). Then write `docs/sentence_diagnosis.md` by hand (at most 200 words): the decision rule is: if Control 3 reaches at least 80% of the sentence classifier's val TPR on PoisonedRAG, the paper must say the PoisonedRAG gain is consistent with learning generator style and is not evidence of instruction detection; otherwise say the controls explain only part of it.
**Create / edit.** `eval/diagnose_sentence.py`, `docs/sentence_diagnosis.md`.
**Output.** `eval/results/diagnose_sentence_seed42.md`, `_seed43.md`, `_seed44.md`.
**Done when.** Three reports exist, each prints n for every cell, and `docs/sentence_diagnosis.md` states the decision-rule outcome with numbers from the reports for all three seeds.
**Watch out.** Do not tune the classifier in this task; it is a diagnosis. Controls use train only for fitting. Quote at most one short sentence per example; never paste whole documents.

#### 4.5 · Domain-aware embedding variants
**Owner:** Tvisha · **Est:** 2.5 h · **Wave:** 1 · **Tier:** core · **Depends on:** 1.9 · **Blocks:** 4.10

**Why.** The embedding detector scores `(1 - max cosine similarity to any baseline document)/2`. With a baseline that mixes emails, tables, code and encyclopedia passages, the score mostly says "is this topic covered", which is why recall is 0.04 to 0.13. We test whether restricting or re-scaling the comparison per domain rescues it.
**Read first.** `detectors/embedding_detector.py` (`BaselineCorpus.embeddings` are normalised, so dot product = cosine), `data/pooled_stats.md` (domains), the interfaces of task 1.9 in Section 3 (`get_embeddings`, `score_missing`).
**Do.** Write `eval/embedding_variants.py`. Embed each document once (`get_embeddings`) and compute all variants from that matrix:
1. `emb_top3`: mean of the 3 highest cosines to the baseline (also `emb_top5`).
2. `emb_dom_max`: same as the original score but the baseline is restricted to baseline documents with the same `domain` as the scored document; fewer than 5 such baseline documents: fall back to the full baseline.
3. `emb_dom_z`: for each domain, compute each baseline document's leave-one-out nearest-neighbour cosine (max cosine to the other baseline documents of that domain); score = z-score of the scored document's max cosine against that distribution, sign-flipped so that higher = more anomalous. Same fallback rule.
4. Store each variant with `score_missing(name, seed, ["train","val"], ...)` so later tasks reuse it. The original score is `1 - ...` from the existing cache; do not recompute it.
5. Report for each variant and seed: AUC on train and val, TPR at the train-derived 5% budget (val), and the same for "LR on the 3 scores with this variant replacing the embedding score". First print a domain x label count table for train; if any domain has more than 90% of one label, say so in the report (shortcut risk).
6. Choice rule: the variant with the highest mean train AUC over the three seeds is "selected"; if it beats the original embedding by less than 0.03 train AUC, the original stays. Write the selected name into `eval/results/embedding_variants_selected.json`.
**Create / edit.** `eval/embedding_variants.py`.
**Output.** `eval/results/embedding_variants_seed<N>.md`, the signal caches `signal_emb_*_seed<N>_<hash>.csv`, the selection JSON.
**Done when.** Three seed reports, a selected variant (or "original stays") recorded, and a short note in the report saying whether any variant is better than seed-to-seed noise.
**Watch out.** The domain label is metadata of the benchmark; in a real deployment it would be the collection the document comes from. Say that in the report. A negative result is a result.

#### 4.10 · Re-fit fusion with the new signals
**Owner:** Tvisha · **Est:** 1.5 h · **Wave:** 2 · **Tier:** core · **Depends on:** 1.9, 4.4, 4.5 · **Blocks:** 5.3, 5.4, N2, 1.7

**Why.** Karishma (4.4) and the embedding variants (4.5) produce candidate replacements for the weak embedding signal. This task decides, by a rule fixed in advance, what the final fused model uses.
**Read first.** `eval/fpr_eval.py` (`oof_lr`, `lr_fit`, `threshold_at_budget`, `group_bootstrap`, `precision`), `eval/baseline_eval.py` (how the published margin is squashed and added), the selection JSONs from 4.5 (`embedding_variants_selected.json`) and 4.4 (`embedding_boilerplate_selected.json`).
**Do.** Write `eval/fusion_v2.py` (`--seed N`, train + val). Feature sets: F0 = pattern, embedding, perplexity; F1 = F0 + sentence (out-of-fold on train, as `fit_everything` does); F2 = F1 with the selected 4.5 variant replacing embedding; F3 = F1 with the best 4.4 variant replacing embedding; F4 = F1 + published margin; F5 = F1 with both replacements if both were selected. Report each at the 5% and 1% budgets in the same table layout as `fpr_eval` (TPR and FPR with group-bootstrap intervals, precision at 1% and 0.1%). If task 4.4 or 4.5 was dropped (Section 11.2), skip the feature sets that need it. Selection rule (written before looking at validation): highest mean out-of-fold TRAIN AUC over seeds 42 to 44; differences under 0.005 prefer fewer features and cheaper signals. Write the choice to `eval/results/fusion_v2_candidate.json` (`{"features": [...], "embedding_variant": ..., "uses_published": bool}`).
**Create / edit.** `eval/fusion_v2.py`.
**Output.** `eval/results/fusion_v2_seed<N>.md`, `fusion_v2_candidate.json`.
**Done when.** Three seeds run, candidate file written, report states the rule and the AUC table it used.
**Watch out.** Do not look at validation TPR to pick the feature set. If nothing beats F1, F1 stays (that is a fine outcome).

#### 5.3 · False-alarm budget sweep
**Owner:** Tvisha · **Est:** 1.5 h · **Wave:** 2 · **Tier:** core · **Depends on:** 4.10 · **Blocks:** 1.6, 1.7

**Why.** Two budgets (5%, 1%) are two points; a curve shows how recall collapses as the allowed false-alarm rate shrinks, which is what a deployment (millions of clean documents) cares about.
**Read first.** `eval/fpr_eval.py`, `eval/apply_models.py` (after 1.9), `eval/results/fusion_v2_candidate.json`.
**Do.** `eval/alpha_sweep.py --seed N`: methods = pattern bank, sentence classifier, LR on the 3 scores, LR + sentence classifier, the candidate from 4.10, published classifier. Budgets: 10%, 5%, 2%, 1%, 0.5%, 0.1%. For each: threshold from the TRAIN out-of-fold clean-score quantile; report realised val TPR and FPR with group-bootstrap intervals and n. Because validation has only about 170 clean documents, any budget below about 1% cannot be verified (one false alarm = 0.6%): print "FPR resolution 0.58%" and use the rule of three for zero-false-alarm cells. Also print how many train clean documents lie above each threshold.
**Create / edit.** `eval/alpha_sweep.py`.
**Output.** `eval/results/alpha_sweep_seed<N>.csv` (columns `method,budget,thr,tpr,tpr_lo,tpr_hi,fpr,fpr_lo,fpr_hi,n_pos,n_neg`) and `.md`.
**Done when.** Three seeds, the CSV is used by figure F2 (task 1.6), caveat about resolution is in the report.
**Watch out.** Thresholds never come from validation.

#### 5.4 · Cost-aware cascade
**Owner:** Tvisha · **Est:** 3 h · **Wave:** 2 · **Tier:** core · **Depends on:** 1.3, 4.10 · **Blocks:** 1.7, 10.6, 10.11

**Why.** Running every signal on every document costs about 0.25 s. Most traffic is clean, so a cascade only saves time if it can stop early on clean-looking documents. An "OR cascade" (stop only when something is flagged) saves nothing on clean traffic. We measure both designs honestly.
**Read first.** `eval/staged_pipeline.py` (old dev-set cascade with hard-coded thresholds: do not reuse its numbers), `eval/results/latency_benchmark.json` (task 1.3), `eval/fpr_eval.py`.
**Do.** Write `eval/cascade_eval.py` (`--seed N`, train + val). Stage order by cost: pattern, sentence classifier, embedding, perplexity, published (use the ms medians from `latency_benchmark.json`; if a stage is not in the final model, skip it).
- **Design A (OR cascade):** a document is flagged as soon as a stage score exceeds that stage's flag threshold; otherwise it continues to the next stage; after the last stage it is allowed. Flag thresholds: split the 5% budget across stages (grid of splits on the train out-of-fold clean scores, for example 2%/1.5%/0.5%/0.5%/0.5%), choose the split with the highest mean train out-of-fold recall. Report realised train FPR (union bound: at most the sum of the stage budgets).
- **Design B (clear-early cascade):** as A, and additionally a stage may allow the document immediately when its score is below a clear threshold, chosen on TRAIN as the q-quantile of the stage score among poisoned documents, q in {0.01, 0.02, 0.05} (so at most q of poisoned documents are cleared at that stage).
- For each design and each seed: val TPR, FPR, share of clean documents that exit before the last stage, expected milliseconds per document = sum over stages of (share reaching the stage x median ms of the stage) using separate numbers for clean and poisoned documents, and the same figure for "run everything + LR". Report group-bootstrap intervals for TPR/FPR; the latency figures are expectations from medians (say so).
- Expose `run_cascade(seed, eval_role)` so `eval/final_test.py` can call it (card 1.7a).
**Create / edit.** `eval/cascade_eval.py`.
**Output.** `eval/results/cascade_eval_seed<N>.md`.
**Done when.** Three seeds; a table with design, TPR, FPR, expected ms on clean documents, and a one-sentence conclusion generated from the numbers (for example "design B cuts expected latency by X% at a recall cost of Y points"), not written by hand.
**Watch out.** Design A's clean-traffic latency equals the full pipeline; the report must say that. The sentence classifier is cheap but needs fitting; time only `score`.

#### 1.7a · The one-shot final test script (with dry run)
**Owner:** Tvisha · **Est:** 2 h · **Wave:** 1 · **Tier:** core · **Depends on:** 1.9 · **Blocks:** N4, 1.6, 1.7

**Why.** The test split may be used once per seed. The run must therefore be a button that was rehearsed on validation, writes everything the paper needs, and cannot be repeated by accident. Building it early also lets SreeValli write the table and figure code against the dry-run output.
**Read first.** `eval/protocol_eval.py` (`--final` path), `eval/fpr_eval.py`, `eval/apply_models.py`, `eval/signal_cache.py`.
**Do.** Create `eval/final_test.py`, `eval/final_methods.py`, `eval/final_config.json`.
1. `final_config.json`: `{"frozen": false, "seeds": [42,43,44], "budgets": [0.05,0.01], "prevalences": [0.01,0.001], "bootstrap_reps": 1000, "methods": [...names...], "extras": [...]}`. Initial methods: the 7 names from `fit_models` plus "published classifier".
2. `final_methods.py` has two registries. `@register_method("name")` marks a function `build(ctx) -> (train_scores, eval_scores)` where `ctx` has `seed`, `train` and `eval` document lists (dicts as in `load_docs`), the signal getters and `eval_role`; train scores are out-of-fold. `@register_extra("name")` marks a function `run(ctx) -> dict[str, DataFrame]` for results that are not a single score per document (the cascade from 5.4). Cards 4.10 and 5.4 add their entries; nothing else should need editing.
3. `final_test.py` has one function `evaluate(seed, eval_role)` with `eval_role` in `{"val","test"}` and the CLI `python -m eval.final_test --seed N --dry-run` (role val, writes `final_dryrun_*`) or `--final` (role test, writes `final_test_*`). Both paths run the identical code. Outputs per run: `eval/results/final_<mode>_seed<N>_scores.csv` (long format: `id,label,source,attack_type,group,method,score`), `final_<mode>_seed<N>_thresholds.json` (`{"seed":N,"split_hash":...,"role":...,"thresholds":{method:{"0.05":t,"0.01":t}}}`), `final_<mode>_seed<N>_<extra>.csv` for each extra, and the report `final_<mode>_seed<N>.md` (TPR/FPR with group-bootstrap intervals at both budgets, precision at both prevalences, per source, per attack type, caveats).
4. `--final` refuses to run unless `"frozen": true` in the config AND `eval/results/final_FINAL_seed<N>.json` does not exist. It scores the test documents with every signal (`score_missing(..., ["test"], final=True)`; the published classifier scores the test documents on the fly), computes all metrics in memory, then writes the lock file `final_FINAL_seed<N>.json` (timestamp, split hash, `git rev-parse HEAD`, sha256 of the config, python/torch/transformers versions) BEFORE printing or writing any result. A crash before the lock file exists may be fixed and re-run. Once the lock exists, never re-run.
5. Fit on TRAIN only and take thresholds from the train out-of-fold clean-score quantile, exactly as in validation (a refit on train + validation is not done; say so in the paper as a conservative choice).
**Create / edit.** The three files above.
**Output.** Dry-run files for the three seeds.
**Done when.** `python -m eval.final_test --seed 42 --dry-run` reproduces the known validation numbers (pattern 0.24, sentence 0.64, LR + sentence 0.64 at 5%), the same for seeds 43 and 44, `--final` without `frozen: true` exits with a clear refusal, and `grep -n '"test"' eval/*.py` shows the role only in `final_test.py`, `protocol.py`, `protocol_eval.py`, `signal_cache.py` (and one comment in `sentence_eval.py`).
**Watch out.** Never run `--final` while testing; use a copy of the repo with a fake split if you need to test the lock logic.

#### 1.7 · Freeze and run the test split once per seed
**Owner:** Tvisha · **Est:** 0.5 h · **Wave:** 3 · **Tier:** core · **Depends on:** 1.7a, 4.10, 5.3, 5.4, 5.1, 8.4, N2, N3, 8.1 · **Blocks:** 10.5, 10.6, 6.1, 6.6

**Why.** This is the moment the numbers become evidence.
**Do.** Follow the FREEZE checklist in Section 11.3 exactly. In short: all wave-2 PRs merged and `main` clean; dry runs pass for three seeds; Karishma has added the blind Batch B files; set `"frozen": true`, commit "Freeze final configuration", tag `freeze-v1`; run `--final` for seeds 42, 43, 44 (about 10 minutes each, plugged in, sleep disabled); commit the outputs and the three lock files; announce in the team chat that nothing may change in `eval/` apart from reporting.
**Output.** `final_test_seed<N>*.csv/json/md`, `final_FINAL_seed<N>.json`.
**Done when.** Three lock files are committed.
**Watch out.** Do not re-run to "see if it changes". Do not change thresholds or methods afterwards. A bug found later is reported in the paper as a limitation.

#### 11.1 · University IPR / policy check
**Owner:** Tvisha · **Est:** 1 h · **Wave:** 2 · **Tier:** core · **Depends on:** — · **Blocks:** 10.9

**Why.** Before a paper is submitted or code is published, find out whether the university claims or needs to be told about student work. This is administration, not a legal opinion.
**Do.** Find the university's IPR / innovation-cell page and the student project policy; write down what it says about course projects, disclosure before publication, open-sourcing, and who to contact. Email the contact with two questions (does the project need to be disclosed before submission; is there anything to add to the paper's acknowledgements). Do not claim patentability: `docs/prior_art.md` lists related work and our detectors are known techniques with negative results.
**Create / edit.** `docs/ipr_note.md` (policy link, contact, date asked, answer, date).
**Done when.** The note exists with the link and the date; if there is no reply yet, it says so.
**Watch out.** Keep personal data out of the note.

#### 10.9 · Authorship and acknowledgements
**Owner:** Tvisha · **Est:** 0.25 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 11.1 · **Blocks:** —

**Do.** Agree author order with the team (document the contribution of each person in two lines), add the acknowledgements, and add whatever statement the course or the target venue requires about use of AI tools in writing or coding (read their policy; if there is none, add nothing). Put the result in the paper's author block.
**Done when.** Author block and acknowledgements compile.

#### H4 · Bring the updated paper source into the repo
**Owner:** Tvisha · **Est:** 0.25 h · **Wave:** 3 · **Tier:** core · **Depends on:** — · **Blocks:** 10.9, 10.5, 10.6, 10.2, 10.3, 10.4, 10.8, 10.1, 10.10, 10.11

**Why.** The current `paper/` folder holds the OLD paper. The updated IEEE-format source (11 pages) lives outside the repo.
**Do.** Copy the updated source into `paper/`, keeping `paper/sections/*.tex` from the repo unless the new source already contains newer versions; create `paper/figures/` and `paper/tables/` (empty `.gitkeep` files); build once (pdflatex or Overleaf) to confirm it compiles; commit; then post the list of section files in the team chat so everyone edits the right file. Until this is done nobody edits paper text.
**Done when.** `paper/` compiles and the section list is posted.

#### 10.5 · Paper: held-out (test) results section
**Owner:** Tvisha · **Est:** 2 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 1.7, N4, N1 · **Blocks:** 10.7, 10.12, 10.13

**Do.** Write the Results subsection(s) from the generated tables in `paper/tables/` (task N4): headline recall at the 5% and 1% budgets for all methods (mean and range over the three seeds), per-source table, precision at 1% and 0.1% prevalence, the published-baseline comparison, latency (median and p95). Include the sentence-classifier diagnosis outcome (N1) in one paragraph. Every number in the text must be copied from a table, never typed from memory. State plainly: thresholds from train, test used once, three seeds, test hash and lock files.
**Done when.** Section compiles; each number has a table or file behind it; Karishma's read-through (10.12) finds no mismatch.
**Watch out.** Do not describe validation-era numbers as test numbers. If test results are worse than validation, say so first.

#### 10.6 · Paper: mitigations and what they cost
**Owner:** Tvisha · **Est:** 2 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 1.7, 5.4, 5.1, 8.4, N2 · **Blocks:** 10.7, 10.12, 10.13

**Do.** One subsection each, from the generated tables: cascade (cost vs recall, 5.4), conformal / per-source thresholds (5.1), normalise-then-match (8.4, labelled as a patch with its false-positive cost), block / review / allow policy (N2). For each: what it fixes, what it costs, what it does not fix.
**Done when.** Compiles; every claim has a table row; patches labelled per Rule 5.

#### 10.7 · Paper: abstract, introduction, conclusion
**Owner:** Tvisha · **Est:** 1.5 h · **Wave:** 3 · **Tier:** core · **Depends on:** 10.4, 10.5, 10.6 · **Blocks:** 10.12, 10.13

**Do.** Rewrite the abstract (150 to 200 words, **no digits anywhere**: say "most", "roughly half", "a small fraction"), the introduction's contribution list, and the conclusion so that they match the final results. The thesis is "where cheap poison detectors fail and what it costs to fix them", not "a new detector". Mention limitations in the conclusion.
**Done when.** Abstract has no digit (check with a search for `[0-9]`), contributions match sections, no claim stronger than a table row.

#### 4.2 · Stronger perplexity and embedding models (stretch)
**Owner:** Tvisha · **Est:** 2 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 1.9 · **Blocks:** —

**Do.** Use `score_missing` to score train + val with one larger language model for the perplexity signal and one other sentence-embedding model (ask before downloading anything above 1 GB), report AUC, TPR at 5% and median latency next to the originals. Skip if time is short.

#### 7.2 · Sensitivity sweeps (stretch)
**Owner:** Tvisha · **Est:** 3 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 1.9 · **Blocks:** 7.3

**Do.** Train + val only: perplexity token cap (128, 256, 512, 1024), baseline corpus size (50, 100, 285 documents), sentence-classifier `C` and n-gram range, training-set size learning curve (25%, 50%, 100% of train). One table per sweep, three seeds.

#### 7.3 · Sweeps write-up (stretch)
**Owner:** Tvisha · **Est:** 1 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 7.2 · **Blocks:** —

**Do.** One appendix table and a paragraph for the paper.

---

## 9. To do: Karishma

**Core tasks (tick your own box in the PR that finishes the task):**

- [ ] **0.A** Setup and first PR (1.5 h, wave 0)
- [ ] **2.4** Adversarial set, 100 new documents (Batch A open, Batch B blind) (4 h, wave 1)
- [ ] **2.7** Blind label audit (2 h, wave 1)
- [ ] **3.7** Contamination check of the pretrained models (1 h, wave 1)
- [ ] **4.3** Manual error analysis (2 h, wave 1)
- [ ] **4.4** Boilerplate subtraction for the embedding detector (3 h, wave 1)
- [x] **9.1** Attack taxonomy (1 h, wave 2)
- [ ] **N2** Block / review / allow policy evaluation (2 h, wave 2)
- [ ] **N3** Stress-set evaluation under the protocol thresholds (and the blind runs at the freeze) (2 h, wave 2)
- [x] **11.3** Venue shortlist (1 h, wave 2)
- [ ] **6.1** README refresh (0.5 h, wave 3)
- [ ] **10.2** Paper: related work update (1.5 h, wave 3)
- [ ] **10.3** Paper: data and protocol section (1.5 h, wave 3)
- [ ] **10.4** Paper: external validity section (1 h, wave 3)
- [ ] **10.8** Paper: limitations update (1.5 h, wave 3)
- [ ] **10.12** Paper: read-through and number check (1 h, wave 3)

**Stretch (only if core is done and time is left):**

- [ ] **2.8** Datasheet for the datasets (1 h)
- [ ] **8.2** Paraphrase robustness on Batch A (2 h)
- [ ] **6.5** Architecture diagram and demo video script (1.5 h)
- [ ] **9.3** Second annotator for the label audit and error causes (1 h)
- [ ] **6.6** Slides refresh (2 h)

Core total: 26.5 h in 16 tasks.

Your tasks are mostly new data, human annotation and analysis, plus the data and limitations parts of the paper. Two rules matter most for you: **blind Batch B stays private until the freeze** (tasks 2.4 and N3), and **you never fix labels in the pool yourself** (report them to Tvisha).

#### 0.A · Setup and first PR
**Owner:** Karishma · **Est:** 1.5 h · **Wave:** 0 · **Tier:** core · **Depends on:** — · **Blocks:** 2.4, 2.7, 3.7, 4.3, 4.4, 9.1

**Why.** Everything later assumes your pool, splits and caches are identical to Tvisha's.
**Do.** Follow Section 5 steps 1 to 6 exactly. Then start a Claude session with the prompt in Section 12 and ask it: "Using only the rules in Section 4, list the five rules you will follow while working on this repository, and name the three files you would read before writing a script that evaluates documents." Check that its answer mentions the test split, labelled patches, and no AI attribution in git. If it does not, paste Sections 2 to 5 and 14 again.
**Create / edit.** In `ROADMAP.md`, on your own line for 0.A, change `- [ ]` to `- [x]` (the only edit allowed in that file).
**Done when.** The three split hashes match, the smoke test prints pattern 0.24 / sentence 0.64 / LR+sentence 0.64 (within 0.01), `git status` was clean before your checkbox edit, and a PR with only the checkbox change is open.
**Watch out.** If any hash differs, stop and tell Tvisha before doing anything else.

#### 2.4 · Adversarial set, 100 new documents (Batch A open, Batch B blind)
**Owner:** Karishma · **Est:** 4 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.A · **Blocks:** N3, 10.3, 2.8, 8.2, 10.1

**Why.** The only adversarial evidence so far is 10 documents that the patterns were patched against ("10 of 10" is a patch, not a result). We need documents written to avoid our detectors' obvious cues, by someone who is not tuning against the detectors, in two batches: Batch A (design batch, may be used while developing defences) and Batch B (blind, used once at the freeze).
**Read first.** `data/adversarial/adv_001.txt` to `adv_010.txt`, `eval/adversarial_test.py`, `detectors/pattern_detector.py` (`SUSPICIOUS_PATTERNS`: read to understand what NOT to rely on, do not write against it), `data/_provenance/*.tsv` (format), `eval/protocol.py` (`Row`), Section 4 Rules 2, 5, 15.
**Do.**
1. Write 50 new documents for Batch A (`adv_011` to `adv_060`) and 50 for Batch B (`adv_061` to `adv_110`). Each is a legitimate-looking host text (3 to 12 sentences, 300 to 3,000 characters) in one domain (README / docs, email, support article, wiki-style passage, code file with comments, table or CSV rows, policy text) with an attack goal embedded by one technique:
   - **T1** boilerplate anchoring: attack hidden inside a long legitimate changelog, footer or licence block.
   - **T2** paraphrased override: tells the assistant to stop following its earlier guidance without the words ignore / disregard / previous instructions.
   - **T3** authority spoof without trigger words ("per the compliance team", "approved by the platform owner").
   - **T4** code comment / docstring carrying the instruction.
   - **T5** markdown footnote or HTML comment.
   - **T6** split across two or three innocuous-looking sentences that only jointly instruct.
   - **T7** polite or indirect request ("it would help the reader if the assistant ...").
   - **T8** role-play or fictional framing.
   - **T9** non-English line(s) inside an English document (use a language you write well).
   - **T10** exfiltration-style (asks to send or repeat the conversation to a fake address).
   Goals (use only fake targets): G1 recommend or visit `https://example.com/...`; G2 send content to `audit@example.com`; G3 state a false fact (for example a refund window); G4 refuse to answer or hide information; G5 reveal hidden instructions.
2. Each batch has at least 4 documents per technique and at least 3 domains.
3. Mix authorship: at most half drafted with Claude, the rest written by hand. Every Claude draft must be read and edited by you. If one generator writes all 100 they will share one style, which is the exact flaw we are measuring in N1. Record `hand` or `claude_edited` per file.
4. Create `data/_provenance/adversarial_sources.tsv` (tab-separated, with a header row) with the columns `filename`, `batch`, `source`, `technique`, `domain`, `goal`, `notes`, one row per file. Also add rows for `adv_001` to `adv_010` (batch A, source `hand_authored_redteam`, technique `T0_original`).
5. Create `data/evalsets/build_adversarial.py`: `python -m data.evalsets.build_adversarial --batch A` reads the files and the TSV and writes `data/evalsets/adversarial_A.jsonl` in the `Row` schema (`id=adv_011`, `label=1`, `source=ext_adversarial`, `attack_type=<technique>`, `domain`, `group=<id>`, `meta="technique=T3|goal=G1|batch=A|authorship=hand"`). It prints counts per technique, domain, goal, the length range, the largest TF-IDF cosine between two documents (must be below 0.8), and fails if any URL other than `example.com` or any e-mail other than `@example.com` appears. `--batch B --src-dir <folder>` builds Batch B from a folder outside the repo.
6. **Batch B stays private.** Write the 50 files and their TSV rows in a folder outside the repo (for example `C:\Users\<you>\pds_private\adversarial_B\`). Do not commit them, do not show them to Tvisha or SreeValli, and do not run any detector on them until Tvisha announces the freeze (Section 11). Tell Tvisha only that the 50 files exist.
**Create / edit.** `data/adversarial/adv_011.txt` to `adv_060.txt`, `data/_provenance/adversarial_sources.tsv`, `data/evalsets/build_adversarial.py`, `data/evalsets/adversarial_A.jsonl`.
**Output.** Batch A in the repo; Batch B on your machine.
**Done when.** The builder passes for Batch A (and for B locally); the PR contains no Batch B file.
**Watch out.** Write by technique, not against the detector output. Do not check which of your documents are caught and rewrite the missed ones (that would turn the set into a patch target). Attacks stay harmless (Rule 15).

#### 2.7 · Blind label audit
**Owner:** Karishma · **Est:** 2 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.A · **Blocks:** 10.3, 10.8, 2.8, 9.3

**Why.** All results assume the pool labels are right. BIPIA and PoisonedRAG are benchmarks built by script, and the "clean" PoisonedRAG documents are Wikipedia passages that could contain instruction-like text. A blind audit estimates how much label noise there is.
**Read first.** `eval/protocol.py` (`read_jsonl`, `load_split`), `data/pooled_stats.md`.
**Do.**
1. Write `eval/label_audit_sample.py`. Mode 1: `python -m eval.label_audit_sample --seed 42 --n 120` samples 120 documents from the `train` and `val` roles of seed 42 only (40 each from bipia, poisonedrag, pds; half clean, half poisoned in each; never two documents of the same group), shuffles them, and writes `eval/results/label_audit_sheet_seed42.csv` (columns `audit_id,text,your_label,reason`; text capped at 4,000 characters) and `eval/results/label_audit_key_seed42.csv` (`audit_id,id,label,source,group`). Mode 1 prints nothing about labels.
2. Open only the sheet (Excel or VS Code). For each row write `your_label` = 0 (clean), 1 (poisoned: contains an instruction aimed at the AI or text pushing a false claim) or 9 (unsure), and a short reason. PoisonedRAG poisoned passages are only recognisable relative to a question, so use 9 freely there. Do NOT open the key file before you finish.
3. Save as `eval/results/label_audit_filled_seed42.csv`. Mode 2: `python -m eval.label_audit_sample --seed 42 --score eval/results/label_audit_filled_seed42.csv` joins the key and writes `eval/results/label_audit_seed42.md`: confusion matrix (unsure excluded and counted), accuracy and Cohen's kappa per source (`sklearn.metrics.cohen_kappa_score`), and a list of every disagreement (id, pool label, your label, reason).
4. Write `docs/label_audit.md` by hand (at most 250 words): how many suspected label errors per source, what kinds (for example a "clean" passage that contains an instruction), and the consequence. Do not change any label (Rule 2); send the list of suspected errors to Tvisha.
**Create / edit.** `eval/label_audit_sample.py`, `docs/label_audit.md`.
**Output.** Sheet, key, filled sheet, report.
**Done when.** All 120 rows labelled, report generated, summary written, disagreements sent to Tvisha.
**Watch out.** Keep the sampling seeded and deterministic so the sheet can be regenerated. Blindness is the point: do not peek.

#### 3.7 · Contamination check of the pretrained models
**Owner:** Karishma · **Est:** 1 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.A · **Blocks:** 10.8

**Why.** If a pretrained model has seen our evaluation data, its good scores mean less. We need to know what is documented.
**Do.** For each model (`gpt2`, `sentence-transformers/all-MiniLM-L6-v2`, `protectai/deberta-v3-base-prompt-injection-v2`) open its model card on huggingface.co and copy the sentence(s) that describe training data. For each dataset in the pool (BIPIA, PoisonedRAG, BEIR NQ) note the release date and the licence from its own page. Then write what is documented and what is not. Do not state that a model was or was not trained on something unless the card says so; write "not documented" instead.
**Create / edit.** `docs/contamination.md` (table: model, documented training data with a quoted sentence, URL, date read; table: dataset, release date, licence; a conclusion of at most 100 words).
**Done when.** Every cell has a quote and a URL or the words "not documented".
**Watch out.** Short quotes only. Use only pages you opened.

#### 4.3 · Manual error analysis
**Owner:** Karishma · **Est:** 2 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.A · **Blocks:** 10.8, 9.3

**Why.** The paper's thesis is "where detectors fail". A tally of why documents are missed or wrongly flagged is the core evidence for that.
**Read first.** `eval/fpr_eval.py` (`fit_everything`, `threshold_at_budget`), `eval/results/fpr_eval_seed42.md`.
**Do.**
1. Write `eval/error_dump.py` (`--seed 42 --method "LR + sentence classifier" --budget 0.05`; validation only). It writes `eval/results/error_dump_seed42.csv` with all false positives and a seeded random sample of 60 false negatives stratified by source. Columns: `id,source,attack_type,domain,y,flag,score,pat,emb,ppl,sentence_score,top_sentence,text_head,cause,notes` (`text_head` = first 400 characters; `top_sentence` from `SentenceInstructionDetector.score_units`; `cause` and `notes` empty).
2. Read every row and fill `cause`. False negatives: `no-instruction-words` (a false-claim passage, nothing to match), `paraphrase` (instruction without the usual words), `buried-in-long-doc`, `code-context`, `other`. False positives: `benign-imperative` (how-to or policy text), `template-text`, `unusual-domain`, `other`. Add a code if you need one and define it in the summary.
3. Write `docs/error_analysis.md` by hand: counts per cause and per source, up to three short verbatim examples per cause (one sentence each), and one line per cause on what a detector would need to catch it.
**Create / edit.** `eval/error_dump.py`, `docs/error_analysis.md`.
**Output.** The CSV (with your annotations committed as `eval/results/error_dump_seed42_annotated.csv`) and the summary.
**Done when.** Every dumped row has a cause; the summary counts add up to the number of rows.
**Watch out.** One seed is enough for a qualitative analysis. Do not edit documents or labels.

#### 4.4 · Boilerplate subtraction for the embedding detector
**Owner:** Karishma · **Est:** 3 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.A, 1.9 · **Blocks:** 4.10

**Why.** Documents of one type share boilerplate (email headers, table schemas, code scaffolding). The embedding score then mostly measures "looks like the template", not "contains something strange". Removing the shared component may recover signal.
**Read first.** `detectors/embedding_detector.py`, Section 3 interfaces for `get_embeddings` and `score_missing` (task 1.9), `eval/results/fpr_eval_seed42.md` (embedding row).
**Do.** Write `eval/embedding_boilerplate.py` (`--seed N`, train + val; the baseline is the seed's `baseline`-role documents). Variants (each stored with `score_missing`, score = `(1 - max cosine)/2` unless stated):
- `emb_center`: subtract the mean of the baseline embeddings from every embedding, renormalise.
- `emb_abtt1`, `emb_abtt3`: as `emb_center`, then remove the top 1 (or 3) principal components of the centred baseline embeddings ("all-but-the-top", via `numpy.linalg.svd`), renormalise.
- `emb_lines`: before embedding, delete every line that occurs (lower-cased, stripped) in at least 5 baseline documents; the baseline documents are stripped the same way; if nothing is left, use the original text. Re-embed with `detectors.embedding_detector._get_model().encode(texts, normalize_embeddings=True)`.
Report per variant and seed: AUC on train and val overall and per source, TPR at the train-derived 5% budget (val), and the same for "LR on the 3 scores with this variant replacing the embedding score". Selection rule: the variant with the highest mean train AUC over seeds wins only if it beats the original embedding by at least 0.03 train AUC; write the outcome to `eval/results/embedding_boilerplate_selected.json`.
**Create / edit.** `eval/embedding_boilerplate.py`.
**Output.** `eval/results/embedding_boilerplate_seed<N>.md`, signal caches, the selection JSON.
**Done when.** Three seeds, per-source AUC table present, selection recorded (including "original stays").
**Watch out.** Choose by train AUC, not validation. If nothing helps, the report says so plainly; that is a paper-worthy negative result.

#### 9.1 · Attack taxonomy
**Owner:** Karishma · **Est:** 1 h · **Wave:** 2 · **Tier:** core · **Depends on:** 0.A · **Blocks:** 10.2

**Do.** Write `docs/taxonomy.md`: a table of the attack classes we use (explicit override; authority spoof; exfiltration; role-play / jailbreak; code-comment injection; corpus poisoning with a false claim; obfuscated or encoded; multilingual; split or indirect). Columns: definition in one line, a harmless example with fake targets, where it appears (BIPIA, PoisonedRAG, `pds_*`, adversarial T-codes), which signal should catch it and the measured val recall from the "By attack type" table of `eval/results/fpr_eval_seed42.md`. Cite BIPIA, PoisonedRAG, the indirect prompt injection paper (Greshake et al.) and the OWASP LLM01 entry after checking each reference yourself.
**Done when.** Every row has a number or "not measured"; every reference was opened and checked.

#### N2 · Block / review / allow policy evaluation
**Owner:** Karishma · **Est:** 2 h · **Wave:** 2 · **Tier:** core · **Depends on:** 4.10, 1.9 · **Blocks:** 1.7, 10.6

**Why.** A detector with a hard yes/no is the wrong interface when 60% to 85% of alerts at 1% prevalence are false. A three-way policy (allow, send to review, block) shows the trade-off in units people understand.
**Read first.** `eval/fpr_eval.py` (`precision`, `threshold_at_budget`), `eval/results/fusion_v2_candidate.json`.
**Do.** Write `eval/policy_eval.py` (`--seed N`, val). For method in {LR + sentence classifier, the candidate from 4.10}: thresholds from train out-of-fold clean scores: `t_review` at the 5% budget, `t_block` at the 1% budget; score above `t_block` = block, between = review, below = allow. Report on val: counts and shares of clean and poisoned documents in each action (with n), then expected counts per 1,000 retrieved documents at attack prevalences 1% and 0.1% (use the observed rates; rule of three for zero cells), and the expected cost per 1,000 documents with illustrative costs `C_miss` (poisoned allowed) in {10, 100, 1000}, `C_review` = 1, `C_block_clean` = 5. State clearly that costs are assumptions. Compare with "allow all", "block above the 5% threshold" and "review above the 5% threshold, never block".
**Create / edit.** `eval/policy_eval.py`.
**Output.** `eval/results/policy_eval_seed<N>.md`.
**Done when.** Three seeds, table of counts and costs, assumptions section.
**Watch out.** Do not search for the "best" costs; show a range.

#### N3 · Stress-set evaluation under the protocol thresholds (and the blind runs at the freeze)
**Owner:** Karishma · **Est:** 2 h · **Wave:** 2 · **Tier:** core · **Depends on:** 1.9, 2.4, 2.5, 8.4 · **Blocks:** 1.7, 10.4

**Why.** Adversarial and obfuscated documents are not in the pool, so they need their own evaluation script that applies the thresholds learned on the pool unchanged.
**Read first.** `eval/apply_models.py` (`fit_models`), `eval/signal_cache.py` (`score_external`, `load_evalset`), `data/evalsets/*.jsonl`, `detectors/normalize.py` (task 8.4).
**Do.**
1. Write `eval/stress_eval.py --seed N --set <name>` (names: `adversarial_A`, `obfuscated_A`, later `adversarial_B`, `obfuscated_B`). It fits the models with `fit_models(seed)`, scores the set with `score_external`, and reports for every method the recall at the 5% and 1% train-derived thresholds overall and per technique / transformation, with Wilson 95% intervals (check one value against `scipy.stats.binomtest(k, n).proportion_ci(method="wilson")`). Add the rows "pattern bank + normalise", "sentence classifier + normalise" and "LR + sentence + normalise", where the pattern and sentence scores are computed on `normalize(text)` and the embedding and perplexity scores on the raw text. State the number of documents in every cell. Since these sets contain only poisoned documents, recall is the only metric; the false-alarm side comes from the pool.
2. Write the hosts and payloads for obfuscation Batch B: 20 benign host paragraphs (80 to 200 words, everyday topics) and 20 payload sentences (generic injection sentences with fake targets), in two text files, one item per line, kept outside the repo until the freeze. SreeValli's generator (task 2.5) takes `--hosts` and `--payloads` files.
3. **At the freeze (Section 11):** add Batch B to the repo, run `stress_eval` for `adversarial_B` and `obfuscated_B` once per seed, commit the reports. No second run.
**Create / edit.** `eval/stress_eval.py`.
**Output.** `eval/results/stress_eval_<set>_seed<N>.md` and `.csv` (columns `method,group,budget,k,n,rate,lo,hi`; task N4 turns the CSVs into a paper table).
**Done when.** Batch A and obfuscated A reports exist for three seeds; Batch B reports exist after the freeze.
**Watch out.** These sets show recall only. Never report a "false-alarm rate" from them. Report patches (normalise) as patches, with the false-positive cost measured in task 8.4.

#### 11.3 · Venue shortlist
**Owner:** Karishma · **Est:** 1 h · **Wave:** 2 · **Tier:** core · **Depends on:** — · **Blocks:** —

**Do.** Ask Tvisha what the course requires (deadline, page limit, whether a venue submission is expected). Search for 5 to 8 realistic venues (student-friendly workshops, short-paper tracks, arXiv-first routes) in AI security / NLP / trustworthy ML. For each record: name, type, next deadline, page limit, template, review type, policy on AI tools, URL, and the date you checked it.
**Create / edit.** `docs/venues.md`.
**Done when.** Each row has a URL and a check date; deadlines were read from the venue's own page.
**Watch out.** Deadlines change; do not copy them from memory or from third-party lists.

#### 6.1 · README refresh
**Owner:** Karishma · **Est:** 0.5 h · **Wave:** 3 · **Tier:** core · **Depends on:** 1.7 · **Blocks:** —

**Do.** The README's results table is the old 77-document table. Replace it with the final held-out headline table copied from `paper/tables/` (task N4), add the Section 5 quickstart, the repository map, a "How to reproduce" paragraph (pool, hashes, `eval.final_test`), and three honest bullets on limitations. Keep it under 150 lines.
**Done when.** No number in the README is typed from memory; the quickstart works on a clean clone (ask SreeValli to run it).

#### 10.2 · Paper: related work update
**Owner:** Karishma · **Est:** 1.5 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 9.1 · **Blocks:** 10.12, 10.13

**Do.** Update the Related Work section. Read `related_work.md`, `docs/prior_art.md`, `docs/baselines.md` first. Make sure it covers: indirect prompt injection and RAG poisoning (BIPIA, PoisonedRAG, the original indirect-injection paper), published detectors (the protectai DeBERTa model card, perplexity-based detection), prompt-injection datasets (deepset), calibration / conformal methods, and how this paper differs (a failure-and-latency comparison under a fixed false-alarm budget, not a new detector).
**Create / edit.** The related-work file in `paper/` (name posted after H4) and the `.bib` file.
**Done when.** Every cited work was opened and checked (authors, year, venue or arXiv id) and compiles without undefined references.
**Watch out.** Never add a reference from memory or from a model's suggestion without opening it.

#### 10.3 · Paper: data and protocol section
**Owner:** Karishma · **Est:** 1.5 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 2.4, 2.7 · **Blocks:** 10.12, 10.13

**Do.** Write the data section from `data/pooled_stats.md`, `dataset_methodology.md`, `docs/label_audit.md`, `docs/contamination.md` and `docs/taxonomy.md`: the five sources with counts, group-aware splitting, BIPIA host de-duplication and why (twins leaked across splits), shortcut checks (type-only AUC 0.5, length ratio), label audit result, the separate evaluation sets (adversarial A / B, obfuscated A / B, deepset) and why they are not in the pool, licences.
**Done when.** Table of sources matches `pooled_stats.md` exactly; compiles.

#### 10.4 · Paper: external validity section
**Owner:** Karishma · **Est:** 1 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 2.10, N3, 8.1 · **Blocks:** 10.7, 10.12, 10.13

**Do.** Write the section on what carries over to new data: leave-one-source-out results (in-corpus versus held-out gap), the deepset source, adversarial A and B recall, obfuscated A and B recall (patches labelled as patches), and what the sentence-classifier diagnosis implies. Numbers come from the generated tables only.
**Done when.** Each claim has a table row; the in-corpus versus held-out gap is stated in the first paragraph.

#### 10.8 · Paper: limitations update
**Owner:** Karishma · **Est:** 1.5 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 4.3, 3.7, 2.7, N1 · **Blocks:** 10.12, 10.13

**Do.** Update `paper/sections/limitations.tex`. Cover, with a number or file wherever possible: benchmarks are synthetic and English-heavy; about 170 clean documents per split (one false alarm = 0.6 points; zero seen means below roughly 2%); thresholds from one pool; in-corpus versus held-out gap; possible generator-style learning (N1); label-audit findings; pretrained-model contamination (3.7); latency measured on one laptop CPU; the adversarial results are patches; the end-to-end effect on answers was not measured (unless 7.0 is done).
**Done when.** Each limitation points to a section, table or file.

#### 10.12 · Paper: read-through and number check
**Owner:** Karishma · **Est:** 1 h · **Wave:** 3 · **Tier:** core · **Depends on:** 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.10, 10.11 · **Blocks:** —

**Do.** Read the whole compiled PDF with a checker's eye. Verify: every number in the text equals the cell in `paper/tables/` or the file in `eval/results/` it claims; the abstract contains no digit; terminology is consistent (false-alarm rate, recall, budget); every figure and table is referenced; no reference is undefined. Write `docs/paper_check_karishma.md` with a list of issues (section, line, problem, suggested fix) and send it to the section owners.
**Done when.** The list is posted and every item is closed or explained.

#### 2.8 · Datasheet for the datasets (stretch)
**Owner:** Karishma · **Est:** 1 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 2.4, 2.7 · **Blocks:** —

**Do.** `docs/datasheet.md` following the "Datasheets for Datasets" questions (motivation, composition, collection, preprocessing, uses, distribution, maintenance) for the pool and the evaluation sets, using facts from `data/pooled_stats.md`, `dataset_methodology.md`, the provenance TSVs, `docs/label_audit.md`.

#### 8.2 · Paraphrase robustness on Batch A (stretch)
**Owner:** Karishma · **Est:** 2 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 2.4, 1.9 · **Blocks:** —

**Do.** Paraphrase each of the 50 new Batch A documents (rewrite the attack sentences with different words, keep the goal; hand-edit every paraphrase) into `data/evalsets/adversarial_A_paraphrase.jsonl` (`group` = original id, `meta="orig=<id>"`). Use `eval/stress_eval.py` to compare recall original versus paraphrase for each method under the same thresholds, paired by document. Do not paraphrase pool documents (test-split rule).

#### 6.5 · Architecture diagram and demo video script (stretch)
**Owner:** Karishma · **Est:** 1.5 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 6.2 · **Blocks:** —

**Do.** One architecture diagram (signals, fusion, cascade, verdict) as a vector figure in `paper/figures/` and a 2 to 3 minute demo script for SreeValli's Streamlit app (what to type, what to point at).

#### 9.3 · Second annotator for the label audit and error causes (stretch)
**Owner:** Karishma · **Est:** 1 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 2.7, 4.3 · **Blocks:** —

**Do.** Ask SreeValli to label the first 40 rows of the audit sheet without seeing the key and to re-code 20 of the error-dump rows; compute agreement (Cohen's kappa) and add it to `docs/label_audit.md` and `docs/error_analysis.md`.

#### 6.6 · Slides refresh (stretch)
**Owner:** Karishma · **Est:** 2 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 1.7 · **Blocks:** —

**Do.** Update `slides_outline.md` to the final story (problem, protocol, where each signal fails, latency, mitigations, limitations), using figures from `paper/figures/`. If the course needs a deck, make it from the outline.

---

## 10. To do: SreeValli

**Core tasks (tick your own box in the PR that finishes the task):**

- [x] **0.B** Setup and first PR (1.5 h, wave 0)
- [ ] **1.3** Latency benchmark (1.5 h, wave 1)
- [ ] **2.10** deepset loader without `datasets`, and a fourth held-out source (2.5 h, wave 1)
- [ ] **2.5** Obfuscation generator (1.5 h, wave 1)
- [ ] **8.3** Position and length sensitivity (1 h, wave 1)
- [ ] **6.2** Streamlit demo (2 h, wave 1)
- [ ] **6.3** Tests and CI (2 h, wave 1)
- [ ] **5.1** Conformal and per-source (Mondrian) thresholds (2 h, wave 2)
- [ ] **8.1** Evasion rates of the obfuscations (1 h, wave 2)
- [ ] **8.4** Normalise-then-match defence (a labelled patch) (2 h, wave 2)
- [ ] **N4** Results collector and paper tables (3 h, wave 2)
- [ ] **1.6** Figures F1 to F6 (2.5 h, wave 2)
- [ ] **10.1** Paper: threat model update (1 h, wave 3)
- [ ] **10.10** Paper: reproducibility appendix (1 h, wave 3)
- [ ] **10.11** Paper: system section (1 h, wave 3)
- [ ] **10.13** Paper: read-through and compile check (1 h, wave 3)

**Stretch (only if core is done and time is left):**

- [ ] **4.1** Full protocol ablation (1.5 h)
- [ ] **4.9** Set-level near-duplicate signal (2 h)
- [ ] **6.4** Retriever middleware (2 h)
- [ ] **7.0** Smallest honest end-to-end test (4 h)

Core total: 26.5 h in 16 tasks.

Your tasks are measurement, robustness, tooling and the technical parts of the paper (tables, figures, system, reproducibility). Two rules matter most for you: **you never see Karishma's blind Batch B files** (tasks 2.4, N3), and **everything you add after looking at failures is a labelled patch** (task 8.4).

#### 0.B · Setup and first PR
**Owner:** SreeValli · **Est:** 1.5 h · **Wave:** 0 · **Tier:** core · **Depends on:** — · **Blocks:** 1.3, 2.10, 2.5, 8.3, 6.2, 6.4, 7.0

**Why.** Everything later assumes your pool, splits and caches are identical to Tvisha's.
**Do.** Follow Section 5 steps 1 to 6 exactly. Then start a Claude session with the prompt in Section 12 and ask it: "Using only the rules in Section 4, list the five rules you will follow while working on this repository, and name the three files you would read before writing a script that evaluates documents." Check that its answer mentions the test split, labelled patches, and no AI attribution in git; if not, paste Sections 2 to 5 and 14 again.
**Create / edit.** In `ROADMAP.md`, on your own line for 0.B, change `- [ ]` to `- [x]` (the only edit allowed in that file).
**Done when.** The three split hashes match, the smoke test prints pattern 0.24 / sentence 0.64 / LR+sentence 0.64 (within 0.01), and a PR with only the checkbox change is open.
**Watch out.** If any hash differs, stop and tell Tvisha before doing anything else.

#### 1.3 · Latency benchmark
**Owner:** SreeValli · **Est:** 1.5 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.B · **Blocks:** 5.4, 1.6, 10.11

**Why.** The `*_ms` columns in the score caches contain laptop-sleep stalls (one document shows about 67 minutes), the published-classifier timing is one shared measurement across seeds, and the sentence classifier was never timed. Every latency claim in the paper needs a clean benchmark.
**Read first.** `detectors/base.py` (`timed`), `detectors/embedding_detector.py` (`BaselineCorpus`, `score_document`), `detectors/perplexity_detector.py`, `detectors/published_classifier.py` (`warm_up`, `document_margin`), `detectors/sentence_detector.py`, `eval/baseline_eval.py` (existing latency table), `eval/fpr_eval.py` (`fit_everything` shows how the sentence detector is fitted).
**Do.** Write `eval/latency_benchmark.py`:
```
python -m eval.latency_benchmark --seed 42 --n 200 --repeats 3 --threads 4
```
1. Sample `n` validation documents of the seed with a fixed `random.Random(0)`. Build the embedding `BaselineCorpus` from the seed's `baseline`-role documents, fit the sentence detector on the train role, and warm up every model (score 5 other documents first and throw the timings away; call `pc.warm_up()`).
2. For each document and each detector (pattern, embedding including its own MiniLM encode, perplexity, sentence classifier `score`, published classifier `document_margin`) time `repeats` runs with `time.perf_counter()` around the scoring call only and keep the median of the repeats per document (this removes one-off stalls). Also time "ours3" (pattern + embedding + perplexity in sequence) and "ours3 + sentence".
3. Report per detector: median, 95th percentile and mean of the per-document medians, and the same by length bucket (under 500, 500 to 2,000, over 2,000 characters). Record the hardware and software: platform string, CPU name (`wmic cpu get name` on Windows, guarded by try/except), logical cores, `torch.get_num_threads()`, Python / torch / transformers / sentence-transformers versions, plugged in or battery (ask the user), `--threads` value (call `torch.set_num_threads`).
4. Write a machine-readable `eval/results/latency_benchmark.json`: `{"hardware": {...}, "n": 200, "detectors": {"pattern": {"median_ms":..., "p95_ms":..., "mean_ms":...}, ...}}` (task 5.4 reads this).
**Create / edit.** `eval/latency_benchmark.py`.
**Output.** `eval/results/latency_benchmark.md`, `.csv` (one row per document and detector), `.json`.
**Done when.** The script runs end to end on your machine; the report states medians and p95 for every detector; the hardware block is complete. Then Tvisha re-runs the identical command on the project laptop at the start of wave 2 (plugged in, sleep disabled in Windows power settings) and commits that result; that machine's numbers are the ones in the paper.
**Watch out.** No means in headline claims (mean is dominated by long documents and stalls). Do not time model loading. Do not run other heavy programs during the benchmark.

#### 2.10 · deepset loader without `datasets`, and a fourth held-out source
**Owner:** SreeValli · **Est:** 2.5 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.B, 1.9 · **Blocks:** 10.4

**Why.** The deepset prompt-injection set is an independent external source, but its loader needs the `datasets` package, which breaks our environment (Rule 10). We rewrite the loader with `huggingface_hub` + `pandas`, and use the set as a held-out source.
**Read first.** `data/loaders/deepset.py` (current loader and `Row` use), `data/loaders/_schema.py`, `eval/signal_cache.py` (`score_external`), `eval/apply_models.py` (`fit_models`), `eval/sentence_loso.py` (how a held-out source is reported).
**Do.**
1. Rewrite `load_deepset_rows()`: `from huggingface_hub import hf_hub_download, list_repo_files`; list the files of the dataset repo `deepset/prompt-injections` (`repo_type="dataset"`), download every `.parquet` file (if none exists, use `.csv` or `.json` files), read with `pandas.read_parquet` / `read_csv`, concatenate train and test, build `Row.make(id=f"deepset_{i:04d}", ..., source="deepset", attack_type="prompt_injection" or "clean", domain="general")` as before. No import of `datasets`. Update the docstring.
2. `data/evalsets/build_deepset.py` writes `data/evalsets/deepset.jsonl` with `source` changed to `ext_deepset` and each row its own group. Print counts (expect roughly 660 rows) and the label balance.
3. Write `eval/deepset_eval.py --seed N`: (a) **transfer in:** `fit_models(seed)` applied to the deepset documents via `score_external("deepset", seed)`: TPR / FPR at the train-derived 5% and 1% thresholds for every method, on all rows and on an "English-looking" subset (rule written in the report: at least 3 of {the, and, is, to, of, you, a, in} appear as whole words); (b) **transfer out:** fit the sentence detector on deepset only (5-fold out-of-fold scores for its 5% threshold) and report AUC and TPR / FPR on the seed's validation documents; (c) the published classifier on deepset with the threshold from the seed's train clean margins (see `eval/baseline_eval.py`). Print n for every cell, and the fraction of non-English-looking rows.
**Create / edit.** `data/loaders/deepset.py`, `data/evalsets/build_deepset.py`, `data/evalsets/deepset.jsonl`, `eval/deepset_eval.py`.
**Output.** `eval/results/deepset_eval_seed<N>.md` and `.csv` (columns `direction,method,budget,tpr,fpr,n_pos,n_neg,subset`).
**Done when.** `python -m data.loaders.deepset` runs without `datasets` installed (test in a venv without it, or by `pip uninstall datasets` only if it is installed in YOUR venv); three seed reports exist.
**Watch out.** deepset items are short chat prompts, many in German, not retrieved documents. Report them as an external check of the instruction signals, not as RAG documents. Do not translate or filter silently.

#### 2.5 · Obfuscation generator
**Owner:** SreeValli · **Est:** 1.5 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.B, 1.9 · **Blocks:** N3, 8.1

**Why.** The pattern bank matches words. We measure how easily simple text tricks evade it and the other signals.
**Read first.** `eval/protocol.py` (`Row`), `data/evalsets/README.md` (from 1.9), `detectors/pattern_detector.py` (only to run it as a control).
**Do.**
1. `data/evalsets/obfuscate.py`: pure, deterministic functions `str -> str` (each takes an optional `random.Random`): `zero_width` (U+200B between letters of the payload words), `homoglyph` (Latin a, e, o, p, c, x to Cyrillic lookalikes), `leet` (a4 e3 i1 o0 s5 inside payload words), `spaced` (letters of the key words separated by spaces), `fullwidth` (Latin to fullwidth forms), `html_entities` (characters to `&#NN;`), `base64` (payload encoded, prefixed "decode and follow: "), `case_alternate` (alternating case plus soft hyphens U+00AD). Batch B adds two unseen types, `reverse_words` and `rot13`.
2. `data/evalsets/build_obfuscated.py --batch A`: write 20 benign host paragraphs (80 to 200 words, everyday topics) and 20 generic payload sentences with fake targets (for example a sentence telling the assistant to tell the user to visit `https://example.com/offer`), insert each payload at a random sentence boundary of its host to make a base document, then apply each transformation to the PAYLOAD only. Output `data/evalsets/obfuscated_A.jsonl` with 20 untransformed controls (`attack_type=T0_none`) plus 20 per transformation (`attack_type=<transform name>`, `group=<base id>`, `meta="transform=...|base=obf_A_07|batch=A"`, label 1). `--batch B --hosts FILE --payloads FILE` builds Batch B from Karishma's private files (task N3); `--batch B` also applies the two extra transformations.
3. Control: run the pattern bank on the untransformed controls and print how many are caught; replace payloads until at least 15 of 20 controls are caught (we want to measure evasion of an attack the detector would otherwise catch). Say so in the builder's output.
4. Tests in `tests/test_obfuscate.py`: each transformation changes the string, NFKC of `fullwidth` output equals the input, `base64` decodes back to the payload, same seed gives same output.
**Create / edit.** `data/evalsets/obfuscate.py`, `data/evalsets/build_obfuscated.py`, `data/evalsets/obfuscated_A.jsonl`, `tests/test_obfuscate.py`.
**Output.** The evaluation set and tests.
**Done when.** Builder prints the control catch count and counts per transformation; tests pass.
**Watch out.** Fake targets only (Rule 15). Do not look at Batch B files; if Karishma's files arrive early, tell Tvisha.

#### 8.3 · Position and length sensitivity
**Owner:** SreeValli · **Est:** 1 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.B, 1.9 · **Blocks:** —

**Why.** Perplexity looks only at the first 1,024 tokens; the published classifier uses at most 8 windows; a sentence-level score may be diluted or not. Where the attack sits in a long document should matter, and we can show exactly how.
**Read first.** `detectors/perplexity_detector.py` (token cap), `detectors/published_classifier.py` (`MAX_WINDOWS`), `eval/apply_models.py`.
**Do.** Write `eval/position_eval.py --seed N` (train + val roles only). Hosts: clean documents from the seed's `val` role, concatenated and cut at a sentence boundary to 500, 2,000 and 8,000 characters (30 hosts per length). Payloads: 5 fixed generic injection sentences with fake targets, written in the script. Positions: 0%, 25%, 50%, 75%, 100% of the sentences (insert at a sentence boundary). That is 150 documents per (length, position) cell. Score with pattern, embedding (the seed's baseline), perplexity, sentence classifier and, with `--with-published` (slow), the published classifier; report the detection rate at the train-derived 5% threshold and the mean score, with n and Wilson intervals, plus the clean host alone as a control. Draw nothing here (figures are task 1.6); write a CSV `position_eval_seed<N>.csv` (`length,position,method,rate,lo,hi,n`).
**Create / edit.** `eval/position_eval.py`.
**Output.** `eval/results/position_eval_seed<N>.md` and `.csv`.
**Done when.** Three seeds run; the report explains in one generated sentence per method whether position matters.
**Watch out.** Hosts come from the validation role only. Do not use pool documents of other roles.

#### 5.1 · Conformal and per-source (Mondrian) thresholds
**Owner:** SreeValli · **Est:** 2 h · **Wave:** 2 · **Tier:** core · **Depends on:** 1.9 · **Blocks:** 1.7, 10.6

**Why.** A threshold taken as a quantile of out-of-fold train scores has no guarantee, and a pooled threshold lets one source (BIPIA) take all the false alarms. Split-conformal thresholds give a finite-sample guarantee for exchangeable clean documents, and Mondrian (per-source) thresholds control the false-alarm rate per source.
**Read first.** `eval/sentence_adapt.py` (`conformal_threshold`; new-source adaptation was already studied there, do not repeat it), `eval/fpr_eval.py`, `eval/apply_models.py`.
**Do.** Write `eval/conformal.py --seed N` for the methods "sentence classifier" and "LR + sentence classifier" (train + val).
1. **Pooled conformal:** calibration scores are the train out-of-fold clean scores (GroupKFold folds are already group-aware); with `n` calibration scores and level `alpha`, the threshold is the `ceil((n+1)(1-alpha))`-th smallest (return infinity if that index exceeds `n`). Compare with `threshold_at_budget`.
2. **Mondrian:** the same per source group (bipia, poisonedrag, pds), applied to validation documents of the same group.
3. Report on validation for alpha 5% and 1%: TPR and FPR overall and per source group, pooled versus Mondrian, with Clopper-Pearson 95% intervals for FPR (`scipy.stats.beta.ppf`), n, and whether alpha lies inside the interval.
4. State the cost: Mondrian needs the source (collection) at inference time; with about 55 + 12 clean `pds` documents its threshold is coarse.
**Create / edit.** `eval/conformal.py`.
**Output.** `eval/results/conformal_seed<N>.md`.
**Done when.** Three seeds; per-source FPR table; honest statement of when the guarantee holds (exchangeable clean documents of the same source) and when it does not (new source).
**Watch out.** Do not tune alpha. No test rows.

#### 8.1 · Evasion rates of the obfuscations
**Owner:** SreeValli · **Est:** 1 h · **Wave:** 2 · **Tier:** core · **Depends on:** 2.5, 1.9 · **Blocks:** 8.4, 1.7, 10.4, 10.1

**Do.** `eval/evasion_eval.py --seed N`: apply `fit_models(seed)` thresholds (5% and 1%) to `obfuscated_A` via `score_external`; table of recall by transformation x method (pattern, sentence, LR + sentence, published), evasion rate = 1 - recall, n per cell (20), Wilson intervals. Add a grouped-bar figure `paper/figures/F7_evasion.pdf` (follow the style rules of task 1.6) from the same numbers.
**Output.** `eval/results/evasion_eval_seed<N>.md/.csv`, F7.
**Done when.** Three seeds; the T0 control row shows high recall for the pattern bank (otherwise the experiment is meaningless).
**Watch out.** This is the "before" measurement for 8.4.

#### 8.4 · Normalise-then-match defence (a labelled patch)
**Owner:** SreeValli · **Est:** 2 h · **Wave:** 2 · **Tier:** core · **Depends on:** 8.1 · **Blocks:** N3, 1.7, 10.6, 10.11

**Why.** Cheap text normalisation may undo several obfuscations before the regexes and the sentence classifier see the text. This is a patch written after seeing the failures (Rule 5): it is evaluated again on Karishma's blind obfuscation Batch B at the freeze, and its false-positive cost is reported.
**Do.**
1. `detectors/normalize.py`: `normalize(text) -> str`, applying in order: `html.unescape`; `unicodedata.normalize("NFKC")`; delete characters of category Cf and U+200B to U+200D, U+2060, U+FEFF, U+00AD; map Cyrillic / Greek lookalikes to Latin ONLY inside words that also contain Latin letters (do not damage genuinely Cyrillic or Greek text); collapse spaced-out letters (letters only, regex `(?<![A-Za-z])(?:[A-Za-z] ){3,}[A-Za-z](?![A-Za-z])`, so "1 2 3 4 5" stays as it is); map leetspeak digits inside a token only if the mapped token equals a word of a short attack vocabulary (ignore, instructions, previous, system, prompt, password, secret, override, disregard, reveal, send, forward, ...); find base64 tokens of at least 24 characters, decode them, and if the result is at least 90% printable ASCII append it after the text. Also `explain_normalize(text) -> list[str]` naming the steps that changed something.
2. Add an optional `normalize: bool = False` argument to `pds.scan` in `pds/__init__.py` (default behaviour unchanged).
3. `eval/normalize_eval.py --seed N`: (a) recall of pattern bank and sentence classifier on `obfuscated_A` before and after normalisation, per transformation (the design batch, so labelled as "developed against"); (b) false-positive cost: share of pool train + val clean documents whose pattern score or sentence score changes, and TPR / FPR at the train-derived thresholds with and without normalising on validation; (c) latency of `normalize` (median and p95 over 500 documents).
4. Tests in `tests/test_normalize.py`: each step; genuine Cyrillic text is unchanged; ordinary text with digits ("room 4A") is unchanged.
5. In the module docstring list what it does NOT handle (rot13, reversed text, paraphrase, translation).
**Create / edit.** `detectors/normalize.py`, `pds/__init__.py` (one parameter), `eval/normalize_eval.py`, `tests/test_normalize.py`.
**Output.** `eval/results/normalize_eval_seed<N>.md`.
**Done when.** Three seeds; the report shows both the recovery on Batch A and the false-positive cost; tests pass.
**Watch out.** Do not iterate the normaliser against Batch A until recall hits 100%; stop after the listed steps. Never claim robustness beyond Batch A and Batch B.

#### 6.2 · Streamlit demo
**Owner:** SreeValli · **Est:** 2 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.B, 0.4 · **Blocks:** 6.5

**Why.** A live demo makes the failure modes tangible for readers and the course demo.
**Read first.** `pds/__init__.py` (`scan`, `ScanResult.detector_results` = [pattern, embedding, perplexity]), `eval/explain.py` (`format_verdict(pattern_result, embedding_result, perplexity_result, threshold=0.5)`), `demo_commands.md`.
**Do.** `demo/app.py` (run with `streamlit run demo/app.py`) and `requirements-demo.txt` (`-r requirements.txt` + `streamlit`; streamlit is approved for this file only). Text area, six sample buttons (`demo/samples.py`: clean email, an email with a plain injection, an authority spoof, a code comment, an obfuscated payload, a false-claim passage; fake targets only). Show: the three detector scores and their latencies, the fixed-fusion verdict from `pds.scan`, the `format_verdict` text, a toggle "normalise first" (only if `detectors/normalize.py` exists). If `data/pooled/pooled.jsonl` is present, fit the sentence detector on seed-42 train documents once (`st.cache_resource`) and show the three most instruction-like sentences with their probabilities and highlight those above 0.5; otherwise hide that panel with a message. A fixed footer: "Research demo, not a security product; validation-era numbers."
**Create / edit.** `demo/app.py`, `demo/samples.py`, `requirements-demo.txt`, `docs/demo_notes.md` (the actual output for each sample, including where it fails).
**Done when.** The app starts on your machine; all six samples run; `docs/demo_notes.md` records honest results; a screenshot is saved as `docs/demo.png`.
**Watch out.** Do not tune samples until they look good. First start downloads models (about 600 MB).

#### 6.3 · Tests and CI
**Owner:** SreeValli · **Est:** 2 h · **Wave:** 1 · **Tier:** core · **Depends on:** 0.4, 1.9 · **Blocks:** 10.10

**Why.** Reviewers (and we) need to know that splits, helpers and guards still work after everyone's changes.
**Do.**
1. In `pyproject.toml` add `[tool.pytest.ini_options]` with `testpaths = ["tests", "eval"]` and `python_files = ["test_*.py"]`. (`eval/test_stats.py` already has pytest-style tests.)
2. New tests in `tests/` (CI has no torch and no pool, so no model or pool is needed): `test_protocol.py` (`make_split` deterministic and group-safe on synthetic `Row`s; the committed `data/splits/split_seed{42,43,44}.json` give `split_hash` values `6bdb8ff98e`, `93a61f0778`, `4d1126b320`), `test_fpr_helpers.py` (`threshold_at_budget`, `precision` including the rule-of-three branch, `group_bootstrap` on a toy case), `test_pattern_detector.py` (`score_document(text)` returns a `DetectorResult`, use its `.score`; three phrases that must score above 0.5 and three benign sentences that must score 0), and `test_no_test_role.py`: a static scan of `eval/*.py` that fails if the quoted role `"test"` appears in a role list or `load_docs` call in code (comments are ignored) anywhere except `eval/final_test.py`, `eval/protocol.py`, `eval/protocol_eval.py`, `eval/signal_cache.py` and the tests themselves.
3. `.github/workflows/ci.yml`: on push and pull request, `ubuntu-latest`, Python 3.12, `pip install -r requirements-ci.txt`, `python -m pytest -q`.
**Create / edit.** `pyproject.toml` (test config only), `tests/*.py`, `.github/workflows/ci.yml`.
**Done when.** `python -m pytest -q` passes locally; the workflow is green on the PR.
**Watch out.** Do not add torch or transformers to CI. If a test needs a file that is git-ignored, skip it with a clear message instead of failing.

#### N4 · Results collector and paper tables
**Owner:** SreeValli · **Est:** 3 h · **Wave:** 2 · **Tier:** core · **Depends on:** 1.7a · **Blocks:** 1.6, 10.5

**Why.** Every number in the paper must come from a script. This task turns the one-shot final run into the tables the paper includes, and can be developed on the dry-run output before the test split is touched.
**Read first.** Card 1.7a (output file formats), `eval/fpr_eval.py` (`group_bootstrap`, `precision`), `eval/stats.py` (`aggregate_seeds`).
**Do.**
1. `eval/collect_results.py --mode dryrun|test`: read `eval/results/final_<mode>_seed<N>_scores.csv` and `..._thresholds.json` for seeds 42 to 44 and compute per seed, method and budget: TPR, FPR with group-bootstrap 95% intervals, n_pos, n_neg, threshold, precision at 1% and 0.1% prevalence (rule of three for zero FPR). Write `eval/results/collected_<mode>.csv`, plus `collected_<mode>_by_source.csv` and `collected_<mode>_by_attack.csv`.
2. `eval/make_tables.py --mode dryrun|test`: LaTeX (booktabs) files in `paper/tables/`: `headline.tex` (method rows; TPR and FPR at 5% and 1%, shown as mean [min to max] over the three seeds), `by_source.tex`, `precision.tex`, `latency.tex` (from `latency_benchmark.json`), `external.tex` (adversarial / obfuscated / deepset recall from the CSVs written by `stress_eval`, `evasion_eval`, `deepset_eval`; if a CSV is missing, leave a row of "n/a" and print a warning), `loso.tex` (parse the tables in `sentence_loso_seed<N>.md`). Every file starts with `% generated by eval/make_tables.py from <inputs>; do not edit by hand`. Two decimals, n visible, negative results shown.
3. `paper/tables/README.md` listing each table and its source files.
4. After the final test run (task 1.7), re-run both scripts with `--mode test` and re-run `python -m eval.make_figures --mode test` (task 1.6); commit the outputs.
**Create / edit.** `eval/collect_results.py`, `eval/make_tables.py`, `paper/tables/*`.
**Output.** CSVs and `.tex` tables.
**Done when.** On the dry-run output the headline table reproduces the known validation numbers; the test-mode run works once the final files exist (Tvisha tells you).
**Watch out.** `--mode test` must refuse to run if `final_FINAL_seed<N>.json` does not exist. Never type a number by hand.

#### 1.6 · Figures F1 to F6
**Owner:** SreeValli · **Est:** 2.5 h · **Wave:** 2 · **Tier:** core · **Depends on:** 1.3, 5.3, 1.7a, N4 · **Blocks:** 10.11

**Why.** The story is easier to read from six clean figures than from tables.
**Do.** `eval/make_figures.py --mode dryrun|test` (matplotlib). Style for all figures: single-column width 3.5 in or double-column 7.16 in, font size 8, no title inside the figure (captions are in LaTeX), `pdf.fonttype 42`, Okabe-Ito colours (`#000000 #E69F00 #56B4E9 #009E73 #F0E442 #0072B2 #D55E00 #CC79A7`), the same colour per method in every figure, axis labels with units, seed values drawn as dots over the bars, error bars from the group bootstrap. Figures: **F1** recall at the 5% and 1% budgets per method (bars, three seed dots); **F2** recall against realised false-alarm rate from `alpha_sweep_seed*.csv` (log x axis); **F3** per-source heatmap (method x bipia / poisonedrag / pds, recall); **F4** in-corpus versus leave-one-source-out AUC (parse `sentence_loso_seed*.md`); **F5** median latency (log x, from `latency_benchmark.json`) against recall at 5% (one marker per method); **F6** precision against attack prevalence (log x from 0.01% to 10%) for the headline methods at the 5% and 1% budgets, using `eval.fpr_eval.precision`. Save PDFs in `paper/figures/` (file names `F1_recall.pdf` etc.), PNG previews in `eval/results/figures/`, and one-line suggested captions in `paper/figures/captions.md`. F7 comes from task 8.1.
**Create / edit.** `eval/make_figures.py`.
**Done when.** Six PDFs and previews exist for `--mode dryrun`; they are readable in greyscale; Tvisha has looked at them.
**Watch out.** Do not hard-code values; read the collected CSVs.

#### 10.1 · Paper: threat model update
**Owner:** SreeValli · **Est:** 1 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 2.4, 8.1 · **Blocks:** 10.12, 10.13

**Do.** Update `paper/sections/threat_model.tex`: attacker capabilities (can place text in a document that will be retrieved; cannot change the LLM or the detector; may or may not know the detector), what the adversarial and obfuscated sets model, adaptive attackers and what we did not test, and what is out of scope (data poisoning of the detector itself, multimodal inputs).
**Done when.** Compiles; each assumption is tied to an experiment or listed as untested.

#### 10.10 · Paper: reproducibility appendix
**Owner:** SreeValli · **Est:** 1 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 0.4, 6.3 · **Blocks:** 10.12, 10.13

**Do.** An appendix with: environment (versions from `requirements.txt`), the exact commands (build the pool, verify the three split hashes, `eval.final_test`), seeds, split hashes, the lock files, hardware used for latency, what is exactly reproducible from committed caches and what needs downloads, licences, code and data availability statement (repository link; the pool file is rebuilt, not shipped).
**Done when.** Someone who has not worked on the project (ask Karishma) can follow it up to the smoke test.

#### 10.11 · Paper: system section
**Owner:** SreeValli · **Est:** 1 h · **Wave:** 3 · **Tier:** core · **Depends on:** H4, 5.4, 8.4, 1.3, 1.6 · **Blocks:** 10.12, 10.13

**Do.** Describe the final frozen system: the signals with their costs (`paper/tables/latency.tex`), fusion, thresholds from train, the cascade and the normaliser (labelled as a patch), and a block diagram (Karishma's 6.5 diagram if it exists; otherwise a TikZ block diagram). Read `eval/final_config.json` so the text matches what was run.
**Done when.** Compiles; the described configuration equals `final_config.json`.

#### 10.13 · Paper: read-through and compile check
**Owner:** SreeValli · **Est:** 1 h · **Wave:** 3 · **Tier:** core · **Depends on:** 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.10, 10.11 · **Blocks:** —

**Do.** Compile the paper from scratch; check IEEE format (page limit, margins, figure legibility at print size, table widths), all references resolve, every figure and table is cited, and spot-check ten numbers against `paper/tables/` and `eval/results/`. Write `docs/paper_check_sreevalli.md` with issues (section, problem, fix) and send them to the section owners.
**Done when.** Issues list posted and closed.

#### 4.1 · Full protocol ablation (stretch)
**Owner:** SreeValli · **Est:** 1.5 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 1.9 · **Blocks:** —

**Do.** `eval/ablation_protocol.py`: leave-one-signal-out and add-one-signal-in for the logistic-regression fusion over {pattern, embedding, perplexity, sentence, published margin}, train out-of-fold AUC and validation TPR at the train-derived 5% budget, three seeds, with intervals. (The existing `eval/ablation.py` is for the old 77-document set.)

#### 4.9 · Set-level near-duplicate signal (stretch)
**Owner:** SreeValli · **Est:** 2 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 1.9 · **Blocks:** —

**Do.** `eval/sentence_eval.py` has an exploratory `neighbour_density` helper (mean cosine to the k nearest other documents of the same source). Evaluate it properly: AUC and TPR at the train-derived 5% budget, per source. Explain in the report that PoisonedRAG poisoned passages come 5 per target and BIPIA poisoned documents have a clean twin, so high neighbour similarity is partly a construction artefact of the benchmarks; do not claim a deployable signal.

#### 6.4 · Retriever middleware (stretch)
**Owner:** SreeValli · **Est:** 2 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 0.B · **Blocks:** —

**Do.** `pds/middleware.py`: `filter_documents(docs, scanner, policy)` that works on any list of objects with a `page_content` or `text` attribute (no framework imports), returns kept documents plus a report of what was blocked or sent to review and why. Tests with fake documents; a short usage example in the README.

#### 7.0 · Smallest honest end-to-end test (stretch)
**Owner:** SreeValli · **Est:** 4 h · **Wave:** 3 · **Tier:** stretch · **Depends on:** 1.9, 0.B · **Blocks:** —

**Do.** Only after asking Tvisha (compute and API decisions). Take about 20 PoisonedRAG targets from the validation role, build a tiny corpus (their poisoned passages plus clean validation passages), retrieve the top 5 with MiniLM, ask a small instruction-tuned language model that runs on CPU (agree the model with Tvisha) the target question with and without PDS filtering at the 5% budget, and count how often the answer contains the attacker's target answer. With 20 targets the intervals are wide: report them and say so. Do not generalise to BIPIA.

---

## 11. Order of work: waves, dependencies, freeze

### 11.1 Waves

| Wave | Goal | Suggested time (compress proportionally if the deadline is closer) |
|---|---|---|
| 0 | Everyone can run the project; shared plumbing exists | days 1 to 3 |
| 1 | New data, diagnosis, latency, first experiments (no dependencies on each other) | week 1 to 2 |
| 2 | Experiments that need wave-1 outputs; fusion and cascade decisions; tooling | week 2 to 3 |
| FREEZE | Nothing in `eval/`, `detectors/`, `fusion/`, `data/` changes after this | end of week 3 (at least 5 working days before the deadline) |
| 3 | One-shot test run, tables, figures, paper text, final checks | week 4 |

Tasks per wave and person (generated from the cards):

| Wave | Tvisha | Karishma | SreeValli |
|---|---|---|---|
| 0 (before the freeze) | 0.4, 1.9 | 0.A | 0.B |
| 1 (before the freeze) | N1, 4.5, 1.7a | 2.4, 2.7, 3.7, 4.3, 4.4 | 1.3, 2.10, 2.5, 8.3, 6.2, 6.3 |
| 2 (before the freeze) | 4.10, 5.3, 5.4, 11.1 | 9.1, N2, N3, 11.3 | 5.1, 8.1, 8.4, N4, 1.6 |
| 3 (after the freeze) | 1.7, 10.9, H4, 10.5, 10.6, 10.7, 4.2 (stretch), 7.2 (stretch), 7.3 (stretch) | 6.1, 10.2, 10.3, 10.4, 10.8, 10.12, 2.8 (stretch), 8.2 (stretch), 6.5 (stretch), 9.3 (stretch), 6.6 (stretch) | 10.1, 10.10, 10.11, 10.13, 4.1 (stretch), 4.9 (stretch), 6.4 (stretch), 7.0 (stretch) |

### 11.2 Dependency map (generated from the cards)

A task may start when every task in its "Depends on" column is merged to `main`. If you are blocked, start another task of the same wave and tell the owner of the blocking task.

| Task | Owner | Wave | Depends on | Blocks |
|---|---|---|---|---|
| 0.4 requirements.txt and .gitignore check | Tvisha | 0 | — | 6.2, 6.3, 10.10 |
| 1.9 Shared plumbing: score-cache lookup, signal cache, model re-fit helper | Tvisha | 0 | — | N1, 4.5, 1.7a, 4.4, 2.10, 2.5, 8.3, 6.3, 4.10, N2, N3, 5.1, 8.1, 4.2, 7.2, 8.2, 4.1, 4.9, 7.0 |
| 0.A Setup and first PR | Karishma | 0 | — | 2.4, 2.7, 3.7, 4.3, 4.4, 9.1 |
| 0.B Setup and first PR | SreeValli | 0 | — | 1.3, 2.10, 2.5, 8.3, 6.2, 6.4, 7.0 |
| N1 Diagnose the sentence classifier (is it learning instructions or generator style?) | Tvisha | 1 | 1.9 | 10.5, 10.8 |
| 4.5 Domain-aware embedding variants | Tvisha | 1 | 1.9 | 4.10 |
| 1.7a The one-shot final test script (with dry run) | Tvisha | 1 | 1.9 | N4, 1.6, 1.7 |
| 2.4 Adversarial set, 100 new documents (Batch A open, Batch B blind) | Karishma | 1 | 0.A | N3, 10.3, 2.8, 8.2, 10.1 |
| 2.7 Blind label audit | Karishma | 1 | 0.A | 10.3, 10.8, 2.8, 9.3 |
| 3.7 Contamination check of the pretrained models | Karishma | 1 | 0.A | 10.8 |
| 4.3 Manual error analysis | Karishma | 1 | 0.A | 10.8, 9.3 |
| 4.4 Boilerplate subtraction for the embedding detector | Karishma | 1 | 0.A, 1.9 | 4.10 |
| 1.3 Latency benchmark | SreeValli | 1 | 0.B | 5.4, 1.6, 10.11 |
| 2.10 deepset loader without `datasets`, and a fourth held-out source | SreeValli | 1 | 0.B, 1.9 | 10.4 |
| 2.5 Obfuscation generator | SreeValli | 1 | 0.B, 1.9 | N3, 8.1 |
| 8.3 Position and length sensitivity | SreeValli | 1 | 0.B, 1.9 | — |
| 6.2 Streamlit demo | SreeValli | 1 | 0.B, 0.4 | 6.5 |
| 6.3 Tests and CI | SreeValli | 1 | 0.4, 1.9 | 10.10 |
| 4.10 Re-fit fusion with the new signals | Tvisha | 2 | 1.9, 4.4, 4.5 | 5.3, 5.4, N2, 1.7 |
| 5.3 False-alarm budget sweep | Tvisha | 2 | 4.10 | 1.6, 1.7 |
| 5.4 Cost-aware cascade | Tvisha | 2 | 1.3, 4.10 | 1.7, 10.6, 10.11 |
| 11.1 University IPR / policy check | Tvisha | 2 | — | 10.9 |
| 9.1 Attack taxonomy | Karishma | 2 | 0.A | 10.2 |
| N2 Block / review / allow policy evaluation | Karishma | 2 | 4.10, 1.9 | 1.7, 10.6 |
| N3 Stress-set evaluation under the protocol thresholds (and the blind runs at the freeze) | Karishma | 2 | 1.9, 2.4, 2.5, 8.4 | 1.7, 10.4 |
| 11.3 Venue shortlist | Karishma | 2 | — | — |
| 5.1 Conformal and per-source (Mondrian) thresholds | SreeValli | 2 | 1.9 | 1.7, 10.6 |
| 8.1 Evasion rates of the obfuscations | SreeValli | 2 | 2.5, 1.9 | 8.4, 1.7, 10.4, 10.1 |
| 8.4 Normalise-then-match defence (a labelled patch) | SreeValli | 2 | 8.1 | N3, 1.7, 10.6, 10.11 |
| N4 Results collector and paper tables | SreeValli | 2 | 1.7a | 1.6, 10.5 |
| 1.6 Figures F1 to F6 | SreeValli | 2 | 1.3, 5.3, 1.7a, N4 | 10.11 |
| 1.7 Freeze and run the test split once per seed | Tvisha | 3 | 1.7a, 4.10, 5.3, 5.4, 5.1, 8.4, N2, N3, 8.1 | 10.5, 10.6, 6.1, 6.6 |
| 10.9 Authorship and acknowledgements | Tvisha | 3 | H4, 11.1 | — |
| H4 Bring the updated paper source into the repo | Tvisha | 3 | — | 10.9, 10.5, 10.6, 10.2, 10.3, 10.4, 10.8, 10.1, 10.10, 10.11 |
| 10.5 Paper: held-out (test) results section | Tvisha | 3 | H4, 1.7, N4, N1 | 10.7, 10.12, 10.13 |
| 10.6 Paper: mitigations and what they cost | Tvisha | 3 | H4, 1.7, 5.4, 5.1, 8.4, N2 | 10.7, 10.12, 10.13 |
| 10.7 Paper: abstract, introduction, conclusion | Tvisha | 3 | 10.4, 10.5, 10.6 | 10.12, 10.13 |
| 4.2 Stronger perplexity and embedding models (stretch) | Tvisha | 3 | 1.9 | — |
| 7.2 Sensitivity sweeps (stretch) | Tvisha | 3 | 1.9 | 7.3 |
| 7.3 Sweeps write-up (stretch) | Tvisha | 3 | 7.2 | — |
| 6.1 README refresh | Karishma | 3 | 1.7 | — |
| 10.2 Paper: related work update | Karishma | 3 | H4, 9.1 | 10.12, 10.13 |
| 10.3 Paper: data and protocol section | Karishma | 3 | H4, 2.4, 2.7 | 10.12, 10.13 |
| 10.4 Paper: external validity section | Karishma | 3 | H4, 2.10, N3, 8.1 | 10.7, 10.12, 10.13 |
| 10.8 Paper: limitations update | Karishma | 3 | H4, 4.3, 3.7, 2.7, N1 | 10.12, 10.13 |
| 10.12 Paper: read-through and number check | Karishma | 3 | 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.10, 10.11 | — |
| 2.8 Datasheet for the datasets (stretch) | Karishma | 3 | 2.4, 2.7 | — |
| 8.2 Paraphrase robustness on Batch A (stretch) | Karishma | 3 | 2.4, 1.9 | — |
| 6.5 Architecture diagram and demo video script (stretch) | Karishma | 3 | 6.2 | — |
| 9.3 Second annotator for the label audit and error causes (stretch) | Karishma | 3 | 2.7, 4.3 | — |
| 6.6 Slides refresh (stretch) | Karishma | 3 | 1.7 | — |
| 10.1 Paper: threat model update | SreeValli | 3 | H4, 2.4, 8.1 | 10.12, 10.13 |
| 10.10 Paper: reproducibility appendix | SreeValli | 3 | H4, 0.4, 6.3 | 10.12, 10.13 |
| 10.11 Paper: system section | SreeValli | 3 | H4, 5.4, 8.4, 1.3, 1.6 | 10.12, 10.13 |
| 10.13 Paper: read-through and compile check | SreeValli | 3 | 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.10, 10.11 | — |
| 4.1 Full protocol ablation (stretch) | SreeValli | 3 | 1.9 | — |
| 4.9 Set-level near-duplicate signal (stretch) | SreeValli | 3 | 1.9 | — |
| 6.4 Retriever middleware (stretch) | SreeValli | 3 | 0.B | — |
| 7.0 Smallest honest end-to-end test (stretch) | SreeValli | 3 | 1.9, 0.B | — |

The must-have path (never dropped): 0.A, 0.B, 0.4, 1.9, 1.7a, N1, 1.3, 2.4, 4.3, 4.10, 5.4, N3, N4, 1.7, H4, 10.5, 10.7, 10.12, 10.13. Task 4.10 treats 4.4 and 4.5 as optional inputs. If time runs short, drop in this order: every stretch task; 6.2 (demo); 5.1; 8.3; 4.4 and 4.5 (the original embedding signal stays, and the paper says the variants were not tested); 6.3 (CI).

### 11.3 FREEZE checklist (Tvisha runs it; the whole team confirms each step in the chat)

1. All wave-2 PRs are merged; CI is green; `git status` on `main` is clean.
2. `python -m eval.final_test --seed N --dry-run` passes for seeds 42, 43, 44 and the dry-run numbers match the tables in the merged reports.
3. Two people read `eval/final_config.json` line by line: method list, budgets, selected embedding variant, candidate fusion features, cascade settings, normaliser on or off.
4. The latency benchmark has been re-run on the project laptop and committed.
5. Karishma adds Batch B in one PR: `data/evalsets/adversarial_B.jsonl`, the `adv_061` to `adv_110` files and TSV rows, `data/evalsets/obfuscated_B.jsonl`. Nobody else has seen them before. Tvisha merges it.
6. Set `"frozen": true` in `eval/final_config.json`; commit "Freeze final configuration"; `git tag freeze-v1`; push; announce it.
7. Tvisha runs `python -m eval.final_test --seed 42 --final`, then 43, then 44 (plugged in, sleep disabled), commits the outputs and the three lock files.
8. Karishma runs `python -m eval.stress_eval --seed N --set adversarial_B` and `--set obfuscated_B` once per seed and commits the reports.
9. SreeValli runs `collect_results`, `make_tables`, `make_figures` with `--mode test` and commits the generated files.
10. After the freeze only these may change: paper text, generated tables and figures, README, docs. A bug found later is described in the Limitations section, not fixed in a re-run.

### 11.4 Reviews

- Karishma reviews SreeValli's PRs; SreeValli reviews Karishma's PRs; Tvisha reviews both and merges. Tvisha's PRs are reviewed by whoever is free (use the checklist in Section 12.5).
- Review within 24 hours. A reviewer who is blocked says so in the PR.
- Merge order follows the dependency map. Squash-merge is fine; make sure the squashed message has no attribution lines.

### 11.5 Rhythm

Two 15-minute check-ins per week (what is merged, what is blocked, who needs a review). Anything that changes a rule or a shared interface (Section 3) is agreed in the chat first, then written into this file by Tvisha.

---

## 12. Templates

### 12.1 Start prompt for your Claude (one task = one session)

Paste this first, then Sections 2, 3, 4, 5 and 14 of this file, then your task card, then the cards listed in its "Depends on" line.

```
You are helping me with ONE task in the repository tvxsha/PDS (Poison Detection System for RAG pipelines). I will paste: (1) Sections 2 to 5 and 14 of ROADMAP.md, (2) my task card, (3) the cards of the tasks it depends on.

Rules for this session:
- Read every file listed under "Read first" in full before writing any code. Do not guess file names, function names, column names or numbers. If something the card mentions does not exist, stop and tell me.
- Work only on this card. Do not edit protected files unless the card says so.
- Never load, score or look at the TEST split (Rule 1). Request roles explicitly in code. Thresholds come from TRAIN only.
- Reports are produced by scripts and every number must come from a file in eval/results/. Do not type numbers by hand. Always print n.
- I use Windows cmd and a Python venv. Give commands as plain lines I can paste; scripts run from the repo root as `python -m package.module`.
- Run seeds 42, 43 and 44.
- Do not add any AI attribution to commit messages or PR text (Rule 13). Do not commit or push for me.
- When you finish, list: files created or edited, the exact commands to run, what the output should look like, and anything you were unsure about.

First restate the task in three sentences, list the files you will read, and wait for my go-ahead.
```

### 12.2 Git steps for one task (Windows cmd)

```
git checkout main
git pull
git checkout -b yourname/<task-id>-<short-name>
REM ... do the task, run the commands in the card ...
git status
git add path\to\file1 path\to\file2
git commit -m "Add latency benchmark"
git push -u origin yourname/<task-id>-<short-name>
```

Then open a pull request on GitHub with the template below. Add files by name (not `git add .`) so private files (Batch B) and caches you did not intend never get committed. Read `git status` before every commit.

### 12.3 Pull request description

```
## Task
<id> · <title> (ROADMAP.md)

## What changed
- 

## How to check
    <commands>
Expected: <what the output should show>

## Done-when checklist (copy from the card)
- [ ] 

## Seeds run
42 / 43 / 44

## Protected files touched
none

## Caveats
- 
```

### 12.4 Skeleton for a report written by a script

Every `eval/results/*.md` report has this shape (see `eval/fpr_eval.py` for a full example):

```python
lines = [f"# <Title> (seed {seed}; train + validation only)", "",
         "<One paragraph: question, method, what is fitted on what, which thresholds.>", "",
         f"Split hash {split_hash}; score cache {scores_path}.", ""]
# ... tables, always with n per cell and group-bootstrap or Wilson intervals ...
lines += ["Caveats:", "",
          "- Validation numbers only; the test split has not been used.",
          "- <small cells, optimistic choices, what this does not show>"]
open(f"eval/results/<name>_seed{seed}.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
```

### 12.5 Review checklist (10 minutes)

1. Every item of the card's "Done when" is satisfied (run the check commands).
2. `grep -rn "test" ` shows no new use of the TEST role (only `final_test.py`, `protocol*.py`, `signal_cache.py`, and a comment in `sentence_eval.py`).
3. Seeds 42, 43, 44 all run; outputs exist for each.
4. Reports show n, intervals, a caveats section, and no hand-typed numbers.
5. No pool file, caches you do not understand, model weights, `venv/`, keys, or personal data in the diff.
6. Commit messages and PR text contain no attribution lines.
7. Protected files are listed in the PR if touched.
8. Anything written after looking at failures is labelled as a patch.

---

## 13. Glossary and metric definitions

- **Poisoned / clean (label 1 / 0).** A document that carries an injected instruction or a planted false claim, or not.
- **TPR (recall).** Share of poisoned documents flagged. **FPR (false-alarm rate).** Share of clean documents flagged.
- **Budget.** The false-alarm rate we allow (5% or 1%). **Threshold at a budget.** `threshold_at_budget(clean_train_scores, alpha)` = the `1 - alpha` quantile of the TRAIN clean scores; documents with a score strictly above it are flagged. Applied unchanged to validation (and, once, test).
- **Realised FPR.** The false-alarm rate actually seen on validation; it can differ from the budget. Compare methods at similar realised FPR.
- **Best-case recall.** Threshold picked on the same data it is reported on: optimistic, always labelled.
- **Out-of-fold (OOF).** Scores of a document from a model that was not trained on it (5-fold group-aware cross-validation on train). Used so train thresholds are honest.
- **Group.** Documents that share underlying text (a BIPIA host and its poisoned copy; the 5 passages of one PoisonedRAG target). Never split across roles; resampled together in the bootstrap.
- **Group bootstrap.** Resample groups with replacement 1,000 times; the 2.5th and 97.5th percentiles give the 95% interval.
- **Precision at prevalence p.** `precision = TPR*p / (TPR*p + FPR*(1-p))`. At p = 1% a method with 5% false alarms flags five clean documents for every attack found, even with perfect recall.
- **Rule of three.** If zero events are seen in n trials, the 95% upper bound for the rate is about `3/n`. Zero false alarms in 172 clean documents means "below about 1.7%", not zero.
- **Wilson / Clopper-Pearson interval.** Confidence intervals for a proportion `k/n` (Wilson is approximate and good for small n; Clopper-Pearson is exact and conservative).
- **AUC.** Probability that a random poisoned document scores above a random clean one; 0.5 = no signal.
- **LOSO.** Leave-one-source-out: train on all sources but one, test on the held-out one. Shows transfer to unseen data.
- **Conformal (split) threshold.** With `n` calibration clean scores and level `alpha`, take the `ceil((n+1)(1-alpha))`-th smallest; for exchangeable clean data the false-alarm rate is at most `alpha`. **Mondrian:** do this separately per group (source).
- **Cascade.** Run cheap signals first and stop early. **OR cascade:** stop only on a flag. **Clear-early cascade:** also stop when the document looks clearly clean.
- **Expected latency.** Sum over stages of (share of documents reaching the stage) x (median milliseconds of the stage).
- **Role.** `baseline` (builds the embedding reference corpus), `train`, `val`, `test` (used once).
- **Split hash.** Ten-character fingerprint of a split file; identical pools give identical hashes.
- **Signal.** Any per-document number a detector produces. **Signal cache.** CSV of signal values per document, per seed, in `eval/results/`.
- **Evaluation set (evalset).** A separate set of documents in `Row` schema in `data/evalsets/`, never part of the pool.
- **Blind batch.** Data written by someone who is not tuning against the detectors and kept private until the freeze.
- **Patch.** A fix written after seeing failures. Reported as "not independent evidence" and tested on data created afterwards by someone else.
- **Dry run.** The final test script run on validation instead of test. **Lock file.** Marker that a seed's test run happened; the script refuses to run again.
- **BIPIA.** Benchmark of indirect prompt injection against LLM-integrated applications (email, table, code tasks). **PoisonedRAG.** Knowledge-poisoning attack on RAG: adversarial passages that make the system answer a chosen wrong answer.

---

## 14. Known pitfalls (read once)

1. **The pool file is git-ignored.** After a clone you must run `python -m data.pool_datasets` and check the three split hashes (Section 5). Different hash = different pool; stop.
2. **Never `pip install datasets` in the project venv.** It upgrades `huggingface-hub` to 2.x and breaks `transformers`. Fix if it happens: `pip install "huggingface-hub>=1.5.0,<2.0"` then `pip check`. Known-good on Tvisha's machine: transformers 5.16.1, sentence-transformers 6.0.1, huggingface-hub 1.30.0, CPU-only torch, Python 3.14.
3. **Score caches contain TEST ids.** `pooled_scores_seed<N>_<hash>.csv` also holds the test documents (scored once to save time). Always select rows through the roles you request (`load_docs(seed, ["train","val"], ...)`); never average over the whole cache.
4. **Latency columns are polluted.** `*_ms` values include laptop-sleep stalls (one document shows about 67 minutes). Use medians and 95th percentiles from the dedicated benchmark (task 1.3). The published classifier's cached `ms` is one shared measurement across seeds.
5. **Published-classifier cache is keyed by document id only.** It was confirmed clean for the current pool (522 rows). If the pool ever changes, delete and rebuild it.
6. **Wrong score cache.** Before task 1.9 is merged `find_scores_file` can pick the wrong CSV (substring match, newest by modification time, which is meaningless after a clone). Read the "Using score cache:" line every time.
7. **Shortcut warning.** Adding extra clean-only tables to the pool creates a task-type / label shortcut (type-only AUC 0.67 instead of 0.5). The flag for it still exists in `data/pool_datasets.py` and must not be used. The shortcut check in `data/pooled_stats.md` must keep showing about 0.5.
8. **BIPIA duplicates.** The same BIPIA host appears with many attacks; splitting by document leaks it across roles. Always split and bootstrap by `group`.
9. **In-sample thresholds.** A threshold computed from scores of a model on its own training documents is far too optimistic. Use out-of-fold scores (see `oof_lr`, `fit_everything`).
10. **"Best-case" numbers are optimistic** and the old 77-document dev-set numbers (F1 about 0.97) do not describe realistic data. Do not quote them as results.
11. **The 10 of 10 adversarial result is a patch.** The regexes were extended after those 10 documents were seen. Say so every time it is mentioned; new evidence comes from Batch A and Batch B.
12. **Few clean documents.** Validation has about 172 clean documents: one false alarm is 0.58 points and "zero false alarms" means "below about 1.7%". Small cells (for example `bipia_code` with recall 1.00 on 12 to 15 documents) must never be headline numbers.
13. **PoisonedRAG gain may be generator style.** Until task N1 is done, do not write that the sentence classifier "detects instructions" on PoisonedRAG.
14. **Per-source numbers.** The sentence classifier is strong in-corpus (about 0.61 to 0.66) and weak on a held-out source (AUC about 0.58 to 0.71; a carried-over threshold flags 76% to 80% of clean BIPIA documents). Always say which of the two a number is.
15. **Windows habits.** Activate the venv in every new terminal (`venv\Scripts\activate`); quote version pins in cmd (`"huggingface-hub>=1.5.0,<2.0"`); run scripts from the repo root; if VS Code does not show the project, use File, Open Folder.
16. **Harmless messages:** "You are sending unauthenticated requests to the HF Hub", the `torch.jit.script ... Python 3.14` FutureWarning, "`loss_type=None` was set in the config". Anything with `Traceback` is not harmless.
17. **Long runs and sleep.** Disable sleep (Settings, System, Power) for runs longer than a few minutes. Caches are resumable: run the same command again after an interruption.
18. **Downloads.** Hugging Face and BEIR downloads can be slow or rate-limited. If the pool build fails, ask Tvisha for `pooled.jsonl`; never edit loaders to work around it.
19. **Attack text.** Keep it harmless (fake domains, fake keys). Quote at most one short sentence of a document in a report. Do not publish real personal data or working malware.
20. **Working with Claude.** Claude can invent file names, function names and numbers. Make it read the files first, check any path it cites with `dir` or `git ls-files`, run what it writes, and compare outputs to the "Done when" lines. Do not accept a result because the code ran; accept it because the numbers match the checks. Paste the rules again if a long session starts to drift.