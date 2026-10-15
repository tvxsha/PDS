"""
Latency benchmark (roadmap task 1.3): milliseconds per document for every detector, measured cleanly.

    python -m eval.latency_benchmark --seed 42 --n 200 --repeats 3 --threads 4 --power plugged

Run it on the project laptop, plugged in, sleep disabled, nothing else heavy running. It takes a few minutes.

What it does
  1. Samples n documents of the seed's VALIDATION role with random.Random(0). Only the roles baseline, train and val are
     read; the held-out role is never requested.
  2. Builds every detector the way the evaluation does (embedding baseline from the baseline role, sentence classifier
     fitted on the train role) and warms every model up on other documents. Loading and fitting are not timed.
  3. For each document and each detector it times `repeats` runs with time.perf_counter() around the scoring call only and
     keeps the MEDIAN of the repeats per document, which removes one-off stalls (the old *_ms columns contain laptop-sleep
     stalls of many minutes).
  4. Reports, per detector, the median, the 95th percentile and the mean of those per-document medians, also by document
     length. Headline claims use the median and the p95, not the mean.

Detectors: pattern, embedding (including its own MiniLM encode), perplexity (GPT-2), sentence classifier, published classifier
(document_margin), plus "ours3" (pattern, embedding, perplexity in sequence) and "ours3 + sentence".

Files: eval/results/latency_benchmark.json (read by eval.cascade_eval and eval.make_paper_tables), .csv (one row per document
and detector) and .md.
"""

import argparse
import csv
import datetime
import json
import os
import platform
import random
import subprocess
import time
from importlib import metadata

import numpy as np

RESULTS_DIR = "eval/results"
STEM = "latency_benchmark"
BUCKETS = (("under 500 characters", 0, 500), ("500 to 2,000 characters", 500, 2000), ("over 2,000 characters", 2000, None))
ORDER = ["pattern", "embedding", "perplexity", "sentence classifier", "published classifier", "ours3", "ours3 + sentence"]
N_WARM = 5


# --------------------------------------------------------------------------
# measuring (no models: these are what the tests exercise)
# --------------------------------------------------------------------------

def sample_docs(docs, n, seed=0):
    """n documents, chosen with random.Random(seed) from the documents sorted by id (so the order of the input is irrelevant)."""
    docs = sorted(docs, key=lambda d: d["id"])
    if n >= len(docs):
        return docs
    rng = random.Random(seed)
    return rng.sample(docs, n)


def time_ms(fn, text, clock=time.perf_counter):
    start = clock()
    fn(text)
    return (clock() - start) * 1000.0


def warm_up(detectors, texts):
    """Score the texts once with every detector and throw the timings away (model load, caches, first-call overhead)."""
    for text in texts:
        for fn in detectors.values():
            fn(text)


def measure(detectors, docs, repeats, clock=time.perf_counter, progress=None):
    """One row per document and detector: {"id", "chars", "detector", "median_ms", "max_ms"}.
    median_ms is the median of `repeats` runs of this detector on this document; max_ms shows whether a stall happened."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    rows = []
    for k, d in enumerate(docs):
        for name, fn in detectors.items():
            runs = [time_ms(fn, d["text"], clock) for _ in range(repeats)]
            rows.append({"id": d["id"], "chars": len(d["text"]), "detector": name,
                         "median_ms": float(np.median(runs)), "max_ms": float(max(runs))})
        if progress and (k + 1) % 20 == 0:
            progress(k + 1, len(docs))
    return rows


def bucket_of(chars):
    for label, low, high in BUCKETS:
        if chars >= low and (high is None or chars < high):
            return label
    return BUCKETS[-1][0]


def _stats(values):
    v = np.asarray(values, dtype=float)
    return {"n": int(len(v)), "median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)), "mean_ms": float(v.mean())}


def summarise(rows):
    """({detector: stats}, {detector: {length bucket: stats}}); detectors in the order of ORDER, then any others."""
    names = [n for n in ORDER if any(r["detector"] == n for r in rows)]
    names += sorted({r["detector"] for r in rows} - set(names))
    overall, by_length = {}, {}
    for name in names:
        mine = [r for r in rows if r["detector"] == name]
        overall[name] = _stats([r["median_ms"] for r in mine])
        by_length[name] = {}
        for label, _, _ in BUCKETS:
            part = [r["median_ms"] for r in mine if bucket_of(r["chars"]) == label]
            if part:
                by_length[name][label] = _stats(part)
    return overall, by_length


# --------------------------------------------------------------------------
# hardware and software
# --------------------------------------------------------------------------

def cpu_name():
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True, timeout=20).stdout
            lines = [x.strip() for x in out.splitlines() if x.strip() and x.strip().lower() != "name"]
            if lines:
                return lines[0]
    except Exception:
        pass
    return platform.processor() or "unknown"


def hardware(threads, power):
    info = {"platform": platform.platform(), "cpu": cpu_name(), "logical_cores": os.cpu_count() or 0,
            "threads_requested": int(threads), "power": power, "python": platform.python_version(),
            "date": datetime.date.today().isoformat()}
    for pkg in ("torch", "transformers", "sentence-transformers", "scikit-learn", "numpy"):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info[pkg] = "not installed"
    try:
        import torch
        info["torch_threads"] = int(torch.get_num_threads())
    except Exception:
        info["torch_threads"] = 0
    return info


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------

def result_dict(seed, n, repeats, hw, overall, by_length):
    return {"hardware": hw, "seed": seed, "n": n, "repeats": repeats,
            "detectors": {k: {"median_ms": v["median_ms"], "p95_ms": v["p95_ms"], "mean_ms": v["mean_ms"]} for k, v in overall.items()},
            "by_length": {k: {b: {"n": s["n"], "median_ms": s["median_ms"], "p95_ms": s["p95_ms"]} for b, s in per.items()}
                          for k, per in by_length.items()}}


def render_md(res, overall, by_length):
    hw = res["hardware"]
    lines = ["# Latency benchmark", "",
             f"{res['n']} validation documents of seed {res['seed']} (sampled with random.Random(0)), {res['repeats']} timed runs per "
             "document and detector, after a warm-up on other documents. Each document's figure is the median of its runs; the "
             "columns below are the median, 95th percentile and mean over documents. Model loading and fitting are not timed.", "",
             "Hardware and software: " + "; ".join(f"{k}: {v}" for k, v in hw.items()) + ".", "",
             "| detector | median ms | p95 ms | mean ms |", "|---|---|---|---|"]
    for name, s in overall.items():
        lines.append(f"| {name} | {s['median_ms']:.2f} | {s['p95_ms']:.2f} | {s['mean_ms']:.2f} |")
    lines += ["", "## By document length (median ms / p95 ms, number of documents)", "",
              "| detector | " + " | ".join(b for b, _, _ in BUCKETS) + " |", "|---|" + "---|" * len(BUCKETS)]
    for name, per in by_length.items():
        cells = [f"{per[b]['median_ms']:.2f} / {per[b]['p95_ms']:.2f} (n={per[b]['n']})" if b in per else "n/a" for b, _, _ in BUCKETS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Use the median and the p95 in the paper, not the mean: the mean is pulled up by long documents. These numbers are "
              "valid for this machine and these settings only."]
    return "\n".join(lines) + "\n"


def write_outputs(res, rows, overall, by_length, out_dir=RESULTS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    paths["json"] = os.path.join(out_dir, STEM + ".json")
    with open(paths["json"], "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    paths["csv"] = os.path.join(out_dir, STEM + ".csv")
    with open(paths["csv"], "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "chars", "detector", "median_ms", "max_ms"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    paths["md"] = os.path.join(out_dir, STEM + ".md")
    with open(paths["md"], "w", encoding="utf-8", newline="\n") as f:
        f.write(render_md(res, overall, by_length))
    return paths


# --------------------------------------------------------------------------
# the real detectors (heavy imports live here)
# --------------------------------------------------------------------------

def load_docs(seed):
    """{"baseline": [...], "train": [...], "val": [...]} as lists of dicts (id, text, y). The held-out role is not requested."""
    from eval.protocol import POOLED_PATH, load_split, read_jsonl
    split = load_split(seed)
    pool = {r.id: r for r in read_jsonl(POOLED_PATH)}
    return {role: [{"id": i, "text": pool[i].text, "y": pool[i].label} for i in split["roles"][role]]
            for role in ("baseline", "train", "val")}


def build_detectors(docs):
    """{name: callable(text)} for the five detectors and the two sequences, all loaded and fitted (not timed here)."""
    from detectors import embedding_detector, pattern_detector, perplexity_detector, sentence_detector
    from detectors import published_classifier as pc
    print("Building the embedding baseline and fitting the sentence classifier (not timed)...", flush=True)
    baseline = embedding_detector.BaselineCorpus([d["text"] for d in docs["baseline"]])
    sentence = sentence_detector.SentenceInstructionDetector().fit([d["text"] for d in docs["train"]], [d["y"] for d in docs["train"]])
    perplexity_detector._get_model_and_tokenizer()
    pc.warm_up()

    def pattern(t):
        return pattern_detector.score_document(t)

    def embedding(t):
        return embedding_detector.score_document(t, baseline)

    def perplexity(t):
        return perplexity_detector.score_document(t)

    def sentence_classifier(t):
        return sentence.score(t)

    def published(t):
        return pc.document_margin(t)

    def ours3(t):
        pattern(t)
        embedding(t)
        perplexity(t)

    def ours3_sentence(t):
        ours3(t)
        sentence_classifier(t)

    return {"pattern": pattern, "embedding": embedding, "perplexity": perplexity, "sentence classifier": sentence_classifier,
            "published classifier": published, "ours3": ours3, "ours3 + sentence": ours3_sentence}


def ask_power(given):
    if given:
        return {"plugged": "plugged in", "battery": "on battery"}[given]
    try:
        answer = input("Is the laptop plugged in? [y/n] ").strip().lower()
    except EOFError:
        return "unknown"
    return "plugged in" if answer.startswith("y") else "on battery" if answer.startswith("n") else "unknown"


def main():
    ap = argparse.ArgumentParser(description="Latency benchmark (see the top of this file).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--power", choices=["plugged", "battery"], default=None, help="asked interactively if omitted")
    args = ap.parse_args()

    power = ask_power(args.power)
    try:
        import torch
        torch.set_num_threads(args.threads)
    except ImportError:
        raise SystemExit("torch is not installed in this environment; run this on the project laptop (the venv with the detectors).")

    docs = load_docs(args.seed)
    sample = sample_docs(docs["val"], args.n)
    ids = {d["id"] for d in sample}
    warm_texts = [d["text"] for d in docs["train"] if d["id"] not in ids][:N_WARM]
    detectors = build_detectors(docs)
    print(f"Warming up on {len(warm_texts)} other documents...", flush=True)
    warm_up(detectors, warm_texts)

    print(f"Timing {len(sample)} documents x {len(detectors)} detectors x {args.repeats} runs...", flush=True)
    t0 = time.perf_counter()
    rows = measure(detectors, sample, args.repeats, progress=lambda k, n: print(f"  {k}/{n} documents", flush=True))
    overall, by_length = summarise(rows)
    res = result_dict(args.seed, len(sample), args.repeats, hardware(args.threads, power), overall, by_length)
    paths = write_outputs(res, rows, overall, by_length)

    print("\n" + render_md(res, overall, by_length))
    print(f"Finished in {time.perf_counter() - t0:.0f} s. Written:", *paths.values(), sep="\n  ")
    print("\nCommit all three files:  git add " + " ".join(p.replace(os.sep, "/") for p in paths.values()))


if __name__ == "__main__":
    main()