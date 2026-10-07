"""
Zero-shot external transfer test: the quick answer to "does PDS still work
on documents it has never seen, from sources the team did not write?"

The ORIGINAL pipeline is frozen exactly as it is in the paper:
  * embedding baseline = the 38 README docs from the seed-42 split
  * calibration bounds = frozen from eval/results/evaluation_raw.csv
  * calibrated-fusion thresholds 0.15 and 0.20
Nothing is refitted or tuned on the external documents, so this is a genuine
out-of-distribution test (unlike the 77-doc development numbers).

Needs data/pooled/pooled.jsonl (python -m data.pool_datasets).

Usage:  python -m eval.external_transfer
        python -m eval.external_transfer --max-per-class 100 --sources bipia,poisonedrag
Output: eval/results/external_transfer.csv  and  eval/results/external_transfer.md
"""

import argparse
import csv
import random
from collections import defaultdict

import numpy as np

# Importing this module seeds the GLOBAL random generator with 42; its
# rebuild_baseline_corpus() must run before anything else uses `random`.
from eval.cross_domain_eval import rebuild_baseline_corpus, load_calibration_bounds, calibrate
from eval.protocol import read_jsonl, parse_meta, POOLED_PATH
from eval.stats import compute_metrics
from detectors import pattern_detector, embedding_detector, perplexity_detector

THRESHOLDS = (0.15, 0.20)
DETECTORS = ("pattern", "embedding", "perplexity")


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="bipia,poisonedrag,deepset")
    ap.add_argument("--max-per-class", type=int, default=60,
                    help="cap per source per label, for speed (seeded sample)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Rebuilding the frozen dev baseline and calibration bounds...")
    baseline = rebuild_baseline_corpus()
    bounds = load_calibration_bounds()
    perplexity_detector._get_model_and_tokenizer()

    rows = read_jsonl(POOLED_PATH)
    wanted = [s.strip() for s in args.sources.split(",")]
    missing = [s for s in wanted if not any(r.source == s for r in rows)]
    if missing:
        print(f"Not in the pool, skipping: {', '.join(missing)}")
    wanted = [s for s in wanted if s not in missing]
    rng = random.Random(args.seed)
    chosen = []
    for src in wanted:
        for label in (0, 1):
            pool = sorted((r for r in rows if r.source == src and r.label == label), key=lambda r: r.id)
            rng.shuffle(pool)
            chosen += pool[:args.max_per_class]
    print(f"Scoring {len(chosen)} external documents (frozen pipeline, no tuning)...")

    out = []
    for i, r in enumerate(chosen, 1):
        p = pattern_detector.score_document(r.text).score
        e = embedding_detector.score_document(r.text, baseline).score
        q = perplexity_detector.score_document(r.text).score
        cal = [calibrate(p, *bounds["pattern_score"]), calibrate(e, *bounds["embedding_score"]),
               calibrate(q, *bounds["perplexity_score"])]
        out.append({"id": r.id, "source": r.source, "label": r.label, "attack_type": r.attack_type,
                    "domain": r.domain, "position": parse_meta(r.meta).get("position", ""),
                    "pattern": p, "embedding": e, "perplexity": q,
                    "cal_pattern": cal[0], "cal_embedding": cal[1], "cal_perplexity": cal[2],
                    "combined": sum(cal) / 3})
        if i % 50 == 0:
            print(f"  {i}/{len(chosen)}")

    with open("eval/results/external_transfer.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    lines = ["# Zero-shot external transfer (frozen development pipeline)", "",
             "Embedding baseline, calibration bounds and thresholds are exactly those used for the "
             "77-document development set. Nothing was tuned on these documents.", ""]
    for t in THRESHOLDS:
        lines += [f"## Calibrated fusion at threshold {t}", "",
                  "| source | n clean | n poisoned | recall | FPR | precision | F1 |", "|---|---|---|---|---|---|---|"]
        for src in wanted:
            sub = [o for o in out if o["source"] == src]
            y = np.array([o["label"] for o in sub])
            pred = np.array([int(o["combined"] >= t) for o in sub])
            m = compute_metrics(y, pred)
            lines.append(f"| {src} | {int((y == 0).sum())} | {int((y == 1).sum())} | {m['recall']:.2f} | "
                         f"{m['fpr']:.2f} | {m['precision']:.2f} | {m['f1']:.2f} |")
        lines.append("")

    lines += ["## Does each detector carry ANY signal here? (ROC-AUC, threshold-free)", "",
              "0.50 = no better than a coin flip; below 0.50 = the detector points the wrong way.", "",
              "| source | pattern | embedding | perplexity | calibrated fusion |", "|---|---|---|---|---|"]
    for src in wanted:
        sub = [o for o in out if o["source"] == src]
        y = [o["label"] for o in sub]
        lines.append(f"| {src} | " + " | ".join(f"{auc(y, [o[d] for o in sub]):.2f}" for d in DETECTORS)
                     + f" | {auc(y, [o['combined'] for o in sub]):.2f} |")
    lines.append("")

    bip = [o for o in out if o["source"] == "bipia" and o["label"] == 1]
    if bip:
        lines += ["## BIPIA: does attack position matter? (recall at threshold 0.15)", "",
                  "| position | n | recall | embedding mean (cal.) |", "|---|---|---|---|"]
        for pos in ("start", "middle", "end"):
            sub = [o for o in bip if o["position"] == pos]
            if sub:
                lines.append(f"| {pos} | {len(sub)} | {np.mean([o['combined'] >= 0.15 for o in sub]):.2f} | "
                             f"{np.mean([o['cal_embedding'] for o in sub]):.2f} |")
        lines.append("")

    lines += ["## Per attack type: recall at 0.15", "", "| attack type | n | recall |", "|---|---|---|"]
    by = defaultdict(list)
    for o in out:
        if o["label"] == 1:
            by[o["attack_type"]].append(o["combined"] >= 0.15)
    for t, v in sorted(by.items()):
        lines.append(f"| {t} | {len(v)} | {np.mean(v):.2f} |")
    lines.append("")

    text = "\n".join(lines)
    with open("eval/results/external_transfer.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)
    print("Saved eval/results/external_transfer.csv and .md")


if __name__ == "__main__":
    main()