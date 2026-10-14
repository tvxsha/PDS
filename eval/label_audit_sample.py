"""Create and score a blinded manual audit sample of pooled labels.

Mode 1 (blind sheet + separate key):
    python -m eval.label_audit_sample --seed 42 --n 120

Mode 2 (after a human has filled the sheet):
    python -m eval.label_audit_sample --seed 42 --score eval/results/label_audit_filled_seed42.csv

The script only samples train/validation documents. It never loads test documents.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

POOL = Path("data/pooled/pooled.jsonl")
RESULTS = Path("eval/results")
SOURCES = ("bipia", "poisonedrag", "pds")


def source_bucket(source: str) -> str | None:
    if source.startswith("bipia"):
        return "bipia"
    if source.startswith("poisonedrag"):
        return "poisonedrag"
    if source.startswith("pds"):
        return "pds"
    return None


def load_pool() -> dict[str, dict]:
    rows = {}
    with POOL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def make_sample(seed: int, n: int) -> list[dict]:
    if n != 120:
        raise SystemExit("The protocol specifies exactly 120 documents (--n 120).")
    pool = load_pool()
    split_path = Path(f"data/splits/split_seed{seed}.json")
    if not split_path.exists():
        raise SystemExit(f"Missing split file: {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    # Only train and val are considered; test is intentionally never accessed.
    eligible_ids = set(split["roles"]["train"]) | set(split["roles"]["val"])
    candidates = [r for rid, r in pool.items() if rid in eligible_ids and source_bucket(r.get("source", ""))]
    rng = random.Random(seed)
    chosen: list[dict] = []
    used_groups: set[tuple[str, str]] = set()
    for source in SOURCES:
        for label in (0, 1):
            cell = [r for r in candidates if source_bucket(r.get("source", "")) == source and int(r["label"]) == label]
            rng.shuffle(cell)
            got = 0
            for row in cell:
                group_key = row.get("group") or row["id"]
                if group_key in used_groups:
                    continue
                used_groups.add(group_key)
                chosen.append(row)
                got += 1
                if got == 20:
                    break
            if got != 20:
                raise SystemExit(
                    f"Cannot make required sample: source={source}, label={label}; "
                    f"found only {got}/20 unique groups in train+val. No files written."
                )
    rng.shuffle(chosen)
    return chosen


def write_blind(seed: int, n: int) -> None:
    rows = make_sample(seed, n)
    RESULTS.mkdir(parents=True, exist_ok=True)
    sheet = RESULTS / f"label_audit_sheet_seed{seed}.csv"
    key = RESULTS / f"label_audit_key_seed{seed}.csv"
    with sheet.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["audit_id", "text", "your_label", "reason"])
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow({"audit_id": f"A{i:03d}", "text": str(row.get("text", ""))[:4000], "your_label": "", "reason": ""})
    with key.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audit_id", "id", "label", "source", "group"])
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            writer.writerow({"audit_id": f"A{i:03d}", "id": row["id"], "label": int(row["label"]),
                             "source": source_bucket(row.get("source", "")), "group": row.get("group") or row["id"]})
    # Do not print labels, class counts, source counts, or key contents in blind mode.
    print(f"Wrote blinded sheet: {sheet}")
    print(f"Wrote separate key:  {key}")
    print("Label audit mode 1 complete. Annotate only the sheet; do not open the key until finished.")


def score_audit(seed: int, score_path: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    key_path = RESULTS / f"label_audit_key_seed{seed}.csv"
    if not key_path.exists():
        raise SystemExit(f"Missing key file {key_path}; create the blinded sample first.")
    with key_path.open(newline="", encoding="utf-8-sig") as f:
        key_rows = {r["audit_id"]: r for r in csv.DictReader(f)}
    with open(score_path, newline="", encoding="utf-8-sig") as f:
        filled = list(csv.DictReader(f))
    required = {"audit_id", "your_label", "reason"}
    if not filled or not required.issubset(filled[0]):
        raise SystemExit(f"Filled sheet must include columns: {', '.join(sorted(required))}")
    by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    disagreements = []
    unsure_by_source = Counter()
    invalid = []
    for row in filled:
        aid = row.get("audit_id", "").strip()
        if aid not in key_rows:
            invalid.append(f"unknown audit_id {aid!r}")
            continue
        try:
            pred = int(row.get("your_label", "").strip())
        except ValueError:
            invalid.append(f"missing/non-integer your_label for {aid}")
            continue
        if pred not in (0, 1, 9):
            invalid.append(f"invalid your_label={pred} for {aid}; expected 0, 1, or 9")
            continue
        k = key_rows[aid]
        gold = int(k["label"])
        source = k["source"]
        if pred == 9:
            unsure_by_source[source] += 1
            continue
        by_source[source].append((gold, pred))
        if gold != pred:
            disagreements.append((k["id"], source, gold, pred, row.get("reason", "").strip()))
    if invalid:
        raise SystemExit("Input validation failed:\n- " + "\n- ".join(invalid[:20]))
    if len(filled) != len(key_rows):
        raise SystemExit(f"Expected {len(key_rows)} annotated rows; found {len(filled)}. No report written.")

    try:
        from sklearn.metrics import cohen_kappa_score, confusion_matrix
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for Cohen's kappa; install project dependencies first.") from exc

    lines = [f"# Blind label audit (seed {seed})", "",
             "The audit sample is from train and validation only. Label 9 (unsure) is excluded from accuracy/kappa and reported separately.", "",
             "| source | reviewed | unsure | pool clean → audited poisoned | pool poisoned → audited clean | accuracy (non-unsure) | Cohen's κ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for source in SOURCES:
        pairs = by_source[source]
        n_unsure = unsure_by_source[source]
        if pairs:
            gold = [p[0] for p in pairs]
            pred = [p[1] for p in pairs]
            cm = confusion_matrix(gold, pred, labels=[0, 1])
            acc = sum(a == b for a, b in pairs) / len(pairs)
            kappa = cohen_kappa_score(gold, pred, labels=[0, 1])
            kappa_text = f"{kappa:.3f}" if kappa == kappa else "undefined"
            lines.append(f"| {source} | {len(pairs)} | {n_unsure} | {cm[0,1]} | {cm[1,0]} | {acc:.3f} | {kappa_text} |")
        else:
            lines.append(f"| {source} | 0 | {n_unsure} | 0 | 0 | n/a | n/a |")
    lines += ["", "## Disagreements", "", "| id | source | pool label | audited label | reason |", "|---|---|---:|---:|---|"]
    for rid, source, gold, pred, reason in disagreements:
        lines.append(f"| {rid} | {source} | {gold} | {pred} | {reason.replace('|', '/').replace(chr(10), ' ')} |")
    lines += ["", "## Interpretation notes", "", "- This is a small seeded audit, not a relabeling pass; pooled labels must not be changed here.",
              "- Send suspected label errors to the project lead for adjudication.", ""]
    out = RESULTS / f"label_audit_seed{seed}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote audit report: {out}")
    print(f"Disagreements: {len(disagreements)}; unsure: {sum(unsure_by_source.values())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--score", help="filled annotation CSV; switches to scoring mode")
    args = parser.parse_args()
    if args.score:
        score_audit(args.seed, args.score)
    else:
        write_blind(args.seed, args.n)


if __name__ == "__main__":
    main()
