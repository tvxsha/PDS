"""Build the human-authored adversarial set in the shared Row schema."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from eval.protocol import Row, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "_provenance" / "adversarial_sources.tsv"
OUT_DIR = ROOT / "evalsets"
URL_RE = re.compile(r"https?://[^\s<>\])}\"']+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
TECHNIQUES = {f"T{i}" for i in range(1, 11)}


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {"filename", "batch", "source", "technique", "domain", "goal", "notes"}
    if not rows or set(rows[0]) != expected:
        raise ValueError(f"{path} must have exactly these TSV columns: {sorted(expected)}")
    return rows


def _check_targets(text: str, filename: str) -> None:
    for target in URL_RE.findall(text):
        if urlparse(target).hostname != "example.com":
            raise ValueError(f"non-example.com URL in {filename}: {target}")
    for address in EMAIL_RE.findall(text):
        if not address.lower().endswith("@example.com"):
            raise ValueError(f"non-example.com e-mail in {filename}: {address}")


def build(batch: str, src_dir: Path | None = None) -> Path:
    batch = batch.upper()
    if batch not in {"A", "B"}:
        raise ValueError("batch must be A or B")
    source_dir = src_dir if batch == "B" else ROOT / "adversarial"
    manifest = (source_dir / "adversarial_sources.tsv") if batch == "B" else PROVENANCE
    if batch == "B" and src_dir is None:
        raise ValueError("Batch B requires --src-dir outside the repository")

    entries = [row for row in _read_manifest(manifest)
               if row["batch"].upper() == batch
               and not (batch == "A" and row["technique"] == "T0_original")]
    if len(entries) != 50:
        raise ValueError(f"Batch {batch} must contain exactly 50 rows; found {len(entries)}")

    rows: list[Row] = []
    texts: list[str] = []
    goals: Counter[str] = Counter()
    for entry in entries:
        path = source_dir / entry["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8").strip()
        if not 300 <= len(text) <= 3000:
            raise ValueError(f"{path.name} has {len(text)} characters; expected 300-3000")
        if not 3 <= len(re.findall(r"[.!?](?:\s|$)", text)) <= 12:
            raise ValueError(f"{path.name} must contain 3-12 sentences")
        _check_targets(text, path.name)
        technique = entry["technique"]
        goal = entry["goal"]
        if entry["source"] not in {
            "hand", "claude_edited", "hand_authored_redteam", "codex_generated"
        }:
            raise ValueError(f"unknown authorship in {path.name}: {entry['source']}")
        if batch == "A" and path.stem.startswith("adv_0") and int(path.stem[4:]) <= 10:
            # The original ten are represented in the manifest, not rebuilt here.
            raise ValueError("original red-team examples cannot be part of Batch A's 50 new rows")
        doc_id = path.stem
        meta = f"technique={technique}|goal={goal}|batch={batch}|authorship={entry['source']}"
        rows.append(Row.make(doc_id, text, 1, "ext_adversarial", technique,
                             domain=entry["domain"], group=doc_id, meta=meta))
        texts.append(text)
        goals[goal] += 1

    counts = Counter(row.attack_type for row in rows)
    domains = Counter(row.domain for row in rows)
    if set(counts) != TECHNIQUES or min(counts.values()) < 4:
        raise ValueError(f"each T1-T10 technique needs at least four documents: {dict(counts)}")
    if len(domains) < 3:
        raise ValueError("each batch must cover at least three document domains")
    if any(not re.fullmatch(r"G[1-5]", goal) for goal in goals):
        raise ValueError("goal codes must be G1-G5")

    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
    similarities = cosine_similarity(matrix)
    np.fill_diagonal(similarities, 0.0)
    max_similarity = float(similarities.max())
    if max_similarity >= 0.8:
        raise ValueError(f"largest pairwise TF-IDF cosine is {max_similarity:.3f}; must be below 0.8")

    output = (OUT_DIR / f"adversarial_{batch}.jsonl" if batch == "A"
              else source_dir / "adversarial_B.jsonl")
    write_jsonl(rows, str(output))
    lengths = [len(text) for text in texts]
    print(f"Batch {batch}: {len(rows)} documents; length range {min(lengths)}-{max(lengths)}")
    print(f"Largest TF-IDF cosine: {max_similarity:.4f}")
    print("Technique counts:", dict(sorted(counts.items())))
    print("Domain counts:", dict(sorted(domains.items())))
    print("Goal counts:", dict(sorted(goals.items())))
    print(f"Wrote {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", choices=("A", "B"), required=True)
    parser.add_argument("--src-dir", type=Path)
    args = parser.parse_args()
    build(args.batch, args.src_dir)


if __name__ == "__main__":
    main()
