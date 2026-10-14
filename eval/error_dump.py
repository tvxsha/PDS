"""Export validation-only false positives and a seeded sample of false negatives.

Example:
    python -m eval.error_dump --seed 42 --method "LR + sentence classifier" --budget 0.05

This script does not load or score the test split. The output's cause and notes fields
are intentionally blank for manual annotation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from eval.fpr_eval import fit_everything, threshold_at_budget
from eval.sentence_eval import find_scores_file, load_docs
from detectors.sentence_detector import SentenceInstructionDetector


def proportional_sample(indices_by_source: dict[str, list[int]], total: int, rng: np.random.Generator) -> list[int]:
    """Sample up to total rows with source-proportional quotas and no replacement."""
    sources = sorted(indices_by_source)
    available = sum(len(indices_by_source[s]) for s in sources)
    target = min(total, available)
    if target == 0:
        return []
    exact = {s: target * len(indices_by_source[s]) / available for s in sources}
    quotas = {s: int(np.floor(exact[s])) for s in sources}
    left = target - sum(quotas.values())
    for s in sorted(sources, key=lambda x: exact[x] - quotas[x], reverse=True):
        if left and quotas[s] < len(indices_by_source[s]):
            quotas[s] += 1
            left -= 1
    chosen = []
    for source in sources:
        pool = indices_by_source[source]
        if quotas[source]:
            chosen.extend(rng.choice(pool, size=quotas[source], replace=False).tolist())
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--method", default="LR + sentence classifier")
    ap.add_argument("--budget", type=float, default=0.05, choices=(0.05, 0.01))
    ap.add_argument("--scores", default=None)
    ap.add_argument("--false-negatives", type=int, default=60)
    args = ap.parse_args()

    score_path = find_scores_file(args.seed, args.scores)
    docs = load_docs(args.seed, ["train", "val"], score_path)
    train, val = docs["train"], docs["val"]
    if not train or not val:
        raise SystemExit("Train/validation score rows are missing. Check the score cache and split files.")
    tr_scores, va_scores = fit_everything(train, val)
    if args.method not in va_scores:
        raise SystemExit(f"Unknown method {args.method!r}. Available: {', '.join(va_scores)}")
    ytr = np.asarray([d["y"] for d in train], dtype=int)
    yv = np.asarray([d["y"] for d in val], dtype=int)
    threshold = threshold_at_budget(tr_scores[args.method][ytr == 0], args.budget)
    scores = np.asarray(va_scores[args.method], dtype=float)
    flags = scores > threshold
    false_positive = np.where((yv == 0) & flags)[0].tolist()
    false_negative = np.where((yv == 1) & ~flags)[0].tolist()

    by_source: dict[str, list[int]] = defaultdict(list)
    for i in false_negative:
        by_source[val[i]["src"]].append(i)
    rng = np.random.default_rng(args.seed)
    selected_fn = proportional_sample(by_source, args.false_negatives, rng)
    selected = sorted(set(false_positive + selected_fn))

    pool = {}
    with open("data/pooled/pooled.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pool[row["id"]] = row
    # Fit one final sentence detector on train solely to generate a human-readable top sentence.
    sentence_model = SentenceInstructionDetector().fit([d["text"] for d in train], ytr)

    out_path = f"eval/results/error_dump_seed{args.seed}.csv"
    os.makedirs("eval/results", exist_ok=True)
    columns = ["id", "source", "attack_type", "domain", "y", "flag", "score", "pat", "emb", "ppl",
               "sentence_score", "top_sentence", "text_head", "cause", "notes"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for i in selected:
            d = val[i]
            p = pool.get(d["id"], {})
            probs, units = sentence_model.score_units(d["text"])
            top_idx = int(np.argmax(probs)) if len(probs) else 0
            top_sentence = units[top_idx] if units else ""
            writer.writerow({
                "id": d["id"], "source": d["src"], "attack_type": p.get("attack_type", "unknown"),
                "domain": p.get("domain", "unknown"), "y": int(d["y"]), "flag": int(flags[i]),
                "score": f"{scores[i]:.8f}", "pat": f"{d['pat']:.8f}", "emb": f"{d['emb']:.8f}",
                "ppl": f"{d['ppl']:.8f}", "sentence_score": f"{probs[top_idx]:.8f}" if len(probs) else "",
                "top_sentence": top_sentence[:400], "text_head": d["text"][:400], "cause": "", "notes": ""
            })
    print(f"Method: {args.method}; seed: {args.seed}; budget: {args.budget:.0%}; train-derived threshold: {threshold:.8f}")
    print(f"Validation false positives exported: {len(false_positive)}")
    print(f"False negatives sampled: {len(selected_fn)} of {len(false_negative)} available")
    print(f"Wrote {len(selected)} rows to {out_path}; annotate cause/notes manually.")
    print("Test split was not loaded.")


if __name__ == "__main__":
    main()
