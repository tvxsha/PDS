"""
Loader for BIPIA, the Benchmark for Indirect Prompt Injection Attacks (task 2.1).

Source:  https://github.com/microsoft/BIPIA   (Yi et al. 2023, arXiv:2312.14197)
Licence: MIT for the code and attack strings. The host documents come from
         OpenAI Evals (email), WikiTableQuestions (table, CC BY-SA 4.0) and
         Stack Exchange (code, CC BY-SA). We only DOWNLOAD them for research
         use; nothing from data/external/ is committed to this repo.

What BIPIA is: a set of clean host documents (emails, tables, Stack Overflow
answers) plus a separate list of attack strings. A poisoned document is a
host document with one attack string inserted at the START, MIDDLE or END.
That makes a paired test: the clean and poisoned versions differ ONLY by the
inserted attack.

This loader reproduces BIPIA's own construction (insert_start / insert_middle
/ insert_end from bipia/data/utils.py). One deliberate difference: BIPIA's
"middle" picks a sentence start with NLTK Punkt; we use a regex sentence
splitter so no extra dependency is needed.

The web-QA and summarisation tasks are skipped: their contexts cannot be
redistributed (NewsQA / XSum licences) and must be rebuilt by hand.

Each host document yields ONE clean row plus `attacks_per_context` poisoned
rows, all sharing a group so the protocol never splits them across train/test.
The attack position is recorded in `meta` ("position=end") so the evaluation
can measure position sensitivity.

Run directly:  python -m data.loaders.bipia
"""

import json
import random
import re

from data.loaders._fetch import fetch
from eval.protocol import Row, print_source_counts

BASE = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/"
TASKS = ("email", "table", "code")
ATTACK_FAMILY = {"email": "text", "table": "text", "code": "code"}
POSITIONS = ("start", "middle", "end")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _insert(context, attack, position, rng):
    if position == "start":
        return "\n".join([attack, context])
    if position == "end":
        return "\n".join([context, attack])
    starts = [m.end() for m in _SENTENCE_END.finditer(context)] or [len(context)]
    cut = rng.choice(starts)
    return "\n".join([context[:cut], attack, context[cut:]])


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _context_text(rec):
    c = rec["context"]
    return "\n".join(c) if isinstance(c, list) else str(c)


def _load_attacks(family):
    pool = []
    for split in ("train", "test"):
        path = fetch(f"{BASE}{family}_attack_{split}.json", f"bipia/{family}_attack_{split}.json")
        with open(path, encoding="utf-8") as f:
            for category, strings in json.load(f).items():
                pool.extend((category, s) for s in strings)
    return pool


def load_bipia_rows(n_contexts=60, attacks_per_context=2, seed=42, tasks=TASKS):
    rng = random.Random(seed)
    attacks = {fam: _load_attacks(fam) for fam in {ATTACK_FAMILY[t] for t in tasks}}

    rows = []
    for task in tasks:
        contexts = []
        for split in ("train", "test"):
            path = fetch(f"{BASE}{task}/{split}.jsonl", f"bipia/{task}/{split}.jsonl")
            contexts.extend(_context_text(rec) for rec in _read_jsonl(path))
        contexts = [c for c in contexts if c.strip()]
        picked = rng.sample(range(len(contexts)), min(n_contexts, len(contexts)))
        family = ATTACK_FAMILY[task]

        for n, idx in enumerate(sorted(picked)):
            text = contexts[idx]
            group = f"bipia_{task}_{idx:04d}"
            rows.append(Row.make(f"{group}_clean", text, 0, "bipia", "clean", task, group=group))
            for k in range(attacks_per_context):
                category, attack = rng.choice(attacks[family])
                position = POSITIONS[(n + k) % 3]   # cycle so all three positions are balanced
                rows.append(Row.make(
                    f"{group}_a{k}", _insert(text, attack, position, rng), 1, "bipia",
                    f"bipia_{family}", task, group=group,
                    meta=f"category={category}|position={position}"))
    return rows


if __name__ == "__main__":
    rows = load_bipia_rows(n_contexts=10)
    print_source_counts(rows, "bipia (smoke test, 10 contexts/task)")