"""
Evaluation protocol (roadmap task 1.1): the one shared document schema and a
group-aware, stratified train/val/test split.

Why this file exists
--------------------
Before this, every number in the paper came from tuning thresholds on the
same 77 documents they were reported on. The protocol fixes that:

  * ONE schema (Row) used by every loader and every eval script.
  * ONE split, saved to data/splits/split_seed{N}.json, with four roles:
        baseline : clean docs that ONLY build the embedding detector's
                   reference corpus (never scored, never fitted on)
        train    : fit calibration bounds / learned fusion
        val      : choose thresholds
        test     : touched ONCE at the end (eval/protocol_eval.py --final)
  * Splits are by GROUP, not by document. A BIPIA email and its poisoned
    copies share a group; the 5 PoisonedRAG paraphrases of one target share
    a group. Splitting by document would leak the same underlying text
    across train and test and inflate every score.

Usage:
    python -m eval.protocol --seed 42        # reads data/pooled/pooled.jsonl
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

POOLED_PATH = "data/pooled/pooled.jsonl"
SPLIT_DIR = "data/splits"
ROLES = ("baseline", "train", "val", "test")


@dataclass
class Row:
    id: str            # unique across the whole pool, e.g. "bipia_email_007_a0"
    text: str
    label: int         # 0 = clean, 1 = poisoned
    source: str        # which upstream dataset: pds_dev, bipia, poisonedrag, deepset, ...
    attack_type: str   # "clean" for label 0
    domain: str        # free text: email, table, code, wikipedia_qa, software_docs, ...
    length: int        # len(text) in characters
    group: str = ""    # rows sharing a group are never split apart
    meta: str = ""     # "key=value|key=value" extras, e.g. "category=X|position=end"

    @staticmethod
    def make(id, text, label, source, attack_type, domain="unknown", group=None, meta=""):
        return Row(id=id, text=text, label=int(label), source=source,
                   attack_type=attack_type, domain=domain, length=len(text),
                   group=group or id, meta=meta)

    def to_dict(self):
        return asdict(self)


def parse_meta(meta: str) -> dict:
    out = {}
    for part in (meta or "").split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def print_source_counts(rows, source_name):
    n_total = len(rows)
    n_clean = sum(1 for r in rows if r.label == 0)
    print(f"[{source_name}] {n_total} rows total ({n_clean} clean / {n_total - n_clean} poisoned)")
    by_attack = Counter(r.attack_type for r in rows)
    for attack_type, count in sorted(by_attack.items()):
        print(f"  - {attack_type}: {count}")


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path):
    rows = []
    known = set(Row.__dataclass_fields__)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                rows.append(Row(**{k: v for k, v in d.items() if k in known}))
    return rows


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def make_split(rows, seed=42, baseline_frac=0.25, fractions=(0.6, 0.2, 0.2)):
    """
    Group-aware stratified split. Strata = (source, label-set of the group,
    attack_type for poison-only groups), so every source and every attack
    type appears in train, val and test (when it has enough groups).

    Within each stratum, groups are shuffled (seeded) and dealt out:
      1. baseline_frac of the groups that contain a clean doc -> "baseline"
         (any poisoned twin inside a baseline group is DROPPED, so a clean
         doc in the reference corpus can never sit next to its own poisoned
         copy in the test set)
      2. the rest -> train / val / test in the given fractions.
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r.source, r.group)].append(r)

    strata = defaultdict(list)
    for key, members in groups.items():
        labels = tuple(sorted({m.label for m in members}))
        poison_types = {m.attack_type for m in members if m.label == 1}
        atype = next(iter(poison_types)) if (labels == (1,) and len(poison_types) == 1) else ""
        strata[(key[0], labels, atype)].append(key)

    rng = random.Random(seed)
    role_of_group = {}
    for stratum in sorted(strata):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        has_clean = 0 in stratum[1]
        n_base = round(len(keys) * baseline_frac) if has_clean else 0
        for k in keys[:n_base]:
            role_of_group[k] = "baseline"
        rest = keys[n_base:]
        n = len(rest)
        n_val = round(n * fractions[1])
        n_test = round(n * fractions[2])
        if n >= 3:
            n_val, n_test = max(1, n_val), max(1, n_test)
        for k in rest[:n_test]:
            role_of_group[k] = "test"
        for k in rest[n_test:n_test + n_val]:
            role_of_group[k] = "val"
        for k in rest[n_test + n_val:]:
            role_of_group[k] = "train"

    split = {"seed": seed, "baseline_frac": baseline_frac, "fractions": list(fractions),
             "roles": {r: [] for r in ROLES}, "dropped": []}
    for r in rows:
        role = role_of_group[(r.source, r.group)]
        if role == "baseline" and r.label == 1:
            split["dropped"].append(r.id)
        else:
            split["roles"][role].append(r.id)
    for r in ROLES:
        split["roles"][r].sort()
    split["dropped"].sort()
    split["hash"] = split_hash(split)
    return split


def split_hash(split) -> str:
    """Fingerprint of the role assignment; changes if the pool or seed changes."""
    blob = json.dumps(split["roles"], sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def validate_split(rows, split):
    """Returns a list of problems (empty list = split is sound)."""
    problems = []
    by_id = {r.id: r for r in rows}
    seen = Counter()
    for role in ROLES:
        for i in split["roles"][role]:
            seen[i] += 1
    for i in split["dropped"]:
        seen[i] += 1
    dup = [i for i, c in seen.items() if c > 1]
    missing = [i for i in by_id if i not in seen]
    if dup:
        problems.append(f"{len(dup)} ids appear in more than one role")
    if missing:
        problems.append(f"{len(missing)} ids are in no role")
    group_roles = defaultdict(set)
    for role in ROLES:
        for i in split["roles"][role]:
            r = by_id[i]
            group_roles[(r.source, r.group)].add(role)
    straddle = [g for g, rs in group_roles.items() if len(rs) > 1]
    if straddle:
        problems.append(f"{len(straddle)} groups straddle roles (leakage), e.g. {straddle[0]}")
    if any(by_id[i].label == 1 for i in split["roles"]["baseline"]):
        problems.append("baseline contains poisoned docs")
    for role in ("train", "val", "test"):
        labels = {by_id[i].label for i in split["roles"][role]}
        if labels != {0, 1}:
            problems.append(f"{role} does not contain both labels")
    return problems


def apply_split(rows, split):
    by_id = {r.id: r for r in rows}
    return {role: [by_id[i] for i in split["roles"][role]] for role in ROLES}


def split_path(seed):
    return os.path.join(SPLIT_DIR, f"split_seed{seed}.json")


def save_split(split, path=None):
    path = path or split_path(split["seed"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1)
    return path


def load_split(seed=None, path=None):
    with open(path or split_path(seed), encoding="utf-8") as f:
        return json.load(f)


def print_split_table(rows, split):
    by_id = {r.id: r for r in rows}
    counts = defaultdict(Counter)   # (source, label) -> Counter(role)
    for role in ROLES:
        for i in split["roles"][role]:
            r = by_id[i]
            counts[(r.source, r.label)][role] += 1
    print(f"\n{'source':14s} {'label':>5s} " + " ".join(f"{r:>9s}" for r in ROLES))
    for (source, label), c in sorted(counts.items()):
        print(f"{source:14s} {label:>5d} " + " ".join(f"{c[r]:>9d}" for r in ROLES))
    tot = {r: len(split["roles"][r]) for r in ROLES}
    print(f"{'TOTAL':14s} {'':>5s} " + " ".join(f"{tot[r]:>9d}" for r in ROLES))
    print(f"dropped (poisoned twins of baseline groups): {len(split['dropped'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool", default=POOLED_PATH)
    args = ap.parse_args()

    rows = read_jsonl(args.pool)
    split = make_split(rows, seed=args.seed)
    problems = validate_split(rows, split)
    print_split_table(rows, split)
    print(f"\nsplit hash: {split['hash']}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("Split validated: no id overlap, no group straddles roles, baseline is all clean.")
    print("Saved to", save_split(split))


if __name__ == "__main__":
    main()