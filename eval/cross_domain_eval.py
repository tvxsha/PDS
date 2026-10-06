"""
Cross-domain false-positive test (roadmap task 0.3).

data/cross_domain/ holds 12 BENIGN HR / customer-support / office-policy
documents. None are attacks, so the question is: does PDS wrongly flag
legitimate documents from a domain it was never calibrated on?

Setup is identical to eval/adversarial_test.py: same 70% clean baseline
split (seed 42), FIXED calibration bounds from evaluation_raw.csv, same
calibrated-fusion threshold (0.15).

Usage:  python -m eval.cross_domain_eval
Output: eval/results/cross_domain.csv
"""

import csv
import glob
import os
import random

from detectors import pattern_detector, embedding_detector, perplexity_detector

random.seed(42)  # MUST match eval/run_evaluation.py's seed exactly

THRESHOLD = 0.15
OUT_CSV = "eval/results/cross_domain.csv"


def rebuild_baseline_corpus():
    clean_docs = []
    for fpath in sorted(glob.glob("data/clean/*.txt")):
        with open(fpath, encoding="utf-8") as f:
            clean_docs.append((os.path.basename(fpath), f.read()))
    random.shuffle(clean_docs)
    split_point = int(len(clean_docs) * 0.7)
    return embedding_detector.BaselineCorpus([text for _, text in clean_docs[:split_point]])


def load_calibration_bounds():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bounds = {}
    for key in ["pattern_score", "embedding_score", "perplexity_score"]:
        values = [float(r[key]) for r in rows]
        bounds[key] = (min(values), max(values))
    return bounds


def calibrate(raw, lo, hi):
    span = hi - lo if hi > lo else 1.0
    return max(0.0, min(1.0, (raw - lo) / span))


def main():
    files = sorted(glob.glob("data/cross_domain/*.txt"))
    if not files:
        print("No files in data/cross_domain/ -- run from the repo root.")
        return

    print("Rebuilding embedding baseline (same split as main evaluation)...")
    baseline = rebuild_baseline_corpus()
    bounds = load_calibration_bounds()
    print("Warming up perplexity model...")
    perplexity_detector._get_model_and_tokenizer()

    print(f"\nScoring {len(files)} cross-domain documents (all benign)...\n")
    print(f"{'file':22s} {'pat':>5s} {'emb':>6s} {'ppl':>5s} {'combined':>9s}  verdict   top detector")
    print("-" * 78)

    out_rows = []
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            text = f.read()

        p = pattern_detector.score_document(text)
        e = embedding_detector.score_document(text, baseline)
        q = perplexity_detector.score_document(text)

        cal = {
            "pattern": calibrate(p.score, *bounds["pattern_score"]),
            "embedding": calibrate(e.score, *bounds["embedding_score"]),
            "perplexity": calibrate(q.score, *bounds["perplexity_score"]),
        }
        combined = sum(cal.values()) / 3
        flagged = combined >= THRESHOLD  # benign doc, so flagged == false positive
        top = max(cal, key=cal.get) if flagged else "-"
        name = os.path.basename(fpath)

        print(f"{name:22s} {p.score:5.2f} {e.score:6.3f} {q.score:5.2f} {combined:9.3f}  "
              f"{'FALSE-POS' if flagged else 'ok       '} {top}")
        out_rows.append({
            "filename": name,
            "pattern_score": p.score,
            "embedding_score": e.score,
            "perplexity_score": q.score,
            "calibrated_combined": round(combined, 4),
            "flagged": int(flagged),
            "top_detector": top,
        })

    fp = sum(r["flagged"] for r in out_rows)
    n = len(out_rows)
    emb_only = sum(1 for r in out_rows if r["flagged"] and r["top_detector"] == "embedding")
    print("\n" + "=" * 78)
    print(f"FALSE-POSITIVE RATE on benign cross-domain docs: {fp}/{n} ({fp / n:.0%}) "
          f"at threshold {THRESHOLD}")
    print(f"  of those, driven mainly by the embedding detector: {emb_only}/{fp}")
    print("=" * 78)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()