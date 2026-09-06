# Poison Detection System (PDS)

**Module B -- Extended Research Project**
*Where Poison Detectors Fail: A Comparative Failure & Latency Analysis of RAG Defense Signals*

## What this is

A detection layer for RAG (Retrieval-Augmented Generation) pipelines that
checks incoming documents for knowledge poisoning / prompt-injection before
they're trusted, using three signals:

1. **Instruction pattern detector** -- keyword/regex match for injection phrases
2. **Embedding anomaly detector** -- flags documents statistically unlike a clean baseline
3. **Perplexity / fluency scorer** -- flags text that reads unnaturally

Scores are combined via calibrated fixed-weight fusion and a learned
(logistic regression) fusion, and evaluated via a staged/tiered pipeline
for latency efficiency.

## Key results

| Method | F1 | Notes |
|---|---|---|
| Naive fixed fusion | 0.70 | Baseline, hurt by score-scale mismatch across detectors |
| Learned fusion (raw scores) | 0.88 | Partial fix via regularized logistic regression |
| Calibrated fixed fusion | 0.97 | Manual min-max normalization before averaging |
| Learned fusion (calibrated) | 0.98 | Best accuracy; confirms embedding as strongest signal |
| Staged/tiered pipeline | 0.97 | Same accuracy as calibrated fusion, 72.5% less latency |

## Project structure
pds/
├── detectors/ # the 3 detector modules (shared interface in base.py)
├── data/ # clean + poisoned test documents (see data/README.md and data/_provenance/)
├── fusion/ # fixed + learned score-combination logic
└── eval/ # evaluation, calibration, threshold tuning, staged pipeline

## Running things

From the `pds/` root, with the venv activated:

python -m eval.run_evaluation
python -m eval.calibrated_fusion
python -m eval.train_learned_fusion_calibrated
python -m eval.staged_pipeline

See `ROADMAP.md` for full task breakdown and current project status.