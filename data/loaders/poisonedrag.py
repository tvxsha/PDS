"""
Loader for PoisonedRAG corpus-poisoning passages (task 2.1).

Source:  https://github.com/sleeepeer/PoisonedRAG   (Zou et al. 2024, arXiv:2402.07867)
Licence: MIT. Clean passages come from BEIR (https://github.com/beir-cellar/beir);
         NQ is built from Wikipedia text. Downloaded for research use only and
         never committed (data/external/ is git-ignored).

Why this source matters: unlike every other source, PoisonedRAG poison has NO
injected instructions. Each passage is a fluent, GPT-generated paragraph that
simply asserts a wrong answer to one target question ("Chicago Fire season 4
has 24 episodes" when the truth is 23). There is nothing for a regex to match
and nothing garbled for a perplexity scorer to see. It is the realistic hard
case for PDS, and the one most likely to show where the three signals fail.

Poisoned side: the repo ships 100 target questions x 5 passages per dataset
(nq / hotpotqa / msmarco) in results/adv_targeted_results/<ds>.json. We take
`n_targets` of the targets; the 5 passages of a target share one group.

Clean side: random passages from the BEIR corpus, chosen so their LENGTH
distribution matches the poisoned passages one-for-one (see clean_sampler.py).
That needs the BEIR zip for the dataset (~0.5 GB for NQ, one-time download).

Run directly:  python -m data.loaders.poisonedrag
"""

import json
import random
import zipfile

from data.loaders._fetch import fetch
from data.loaders.clean_sampler import match_by_length
from eval.protocol import Row, print_source_counts

POISON_URL = ("https://raw.githubusercontent.com/sleeepeer/PoisonedRAG/main/"
              "results/adv_targeted_results/{ds}.json")
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{ds}.zip"


def load_poisoned_rows(ds="nq", n_targets=40, seed=42):
    path = fetch(POISON_URL.format(ds=ds), f"poisonedrag/{ds}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    targets = random.Random(seed).sample(sorted(data), min(n_targets, len(data)))
    rows = []
    for tid in sorted(targets):
        group = f"prag_{ds}_{tid}"
        for j, text in enumerate(data[tid]["adv_texts"]):
            rows.append(Row.make(f"{group}_{j}", text, 1, "poisonedrag", f"poisonedrag_{ds}",
                                 "wikipedia_qa" if ds != "msmarco" else "web_qa",
                                 group=group, meta=f"target={tid}"))
    return rows


def _beir_candidates(zip_path, ds, lo, hi, want, max_scan):
    """Stream corpus.jsonl out of the zip; keep passages with lo <= len <= hi."""
    cands = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"{ds}/corpus.jsonl") as f:
            for i, line in enumerate(f):
                if i >= max_scan or len(cands) >= want:
                    break
                rec = json.loads(line)
                text = (rec.get("text") or "").strip()
                if lo <= len(text) <= hi:
                    cands.append((rec["_id"], text))
    return cands


def load_clean_rows(poisoned, ds="nq", beir_zip=None, seed=42, max_scan=600_000):
    zip_path = beir_zip or fetch(BEIR_URL.format(ds=ds), f"beir/{ds}.zip")
    lens = [r.length for r in poisoned]
    lo, hi = max(40, min(lens) - 60), max(lens) + 120
    cands = _beir_candidates(zip_path, ds, lo, hi, want=15 * len(lens), max_scan=max_scan)
    if len(cands) < 2 * len(lens):
        print(f"  WARNING: only {len(cands)} clean candidates in the length band "
              f"[{lo},{hi}] for {len(lens)} poisoned docs; matching will be loose")
    picks = match_by_length(cands, lens, seed=seed)
    return [Row.make(f"prag_{ds}_clean_{doc_id}", text, 0, "poisonedrag", "clean",
                     "wikipedia_qa" if ds != "msmarco" else "web_qa")
            for doc_id, text in picks]


def load_poisonedrag_rows(ds="nq", n_targets=40, seed=42, beir_zip=None):
    poisoned = load_poisoned_rows(ds, n_targets, seed)
    return poisoned + load_clean_rows(poisoned, ds, beir_zip=beir_zip, seed=seed)


if __name__ == "__main__":
    rows = load_poisoned_rows(n_targets=5)
    print_source_counts(rows, "poisonedrag (poisoned side only, 5 targets)")