"""
Builds the obfuscation evaluation sets (roadmap task 2.5).

    python -m data.evalsets.build_obfuscated --batch A
    python -m data.evalsets.build_obfuscated --batch B --hosts FILE --payloads FILE      # task N3, at the freeze

Batch A uses the 20 hosts and 20 payloads written in data/evalsets/obfuscated_A_sources.py and writes
data/evalsets/obfuscated_A.jsonl. Batch B reads 20 hosts and 20 payloads (one item per line) from two files
that Karishma keeps outside the repository until the freeze, applies the eight Batch A transformations plus
`reverse_words` and `rot13`, and writes data/evalsets/obfuscated_B.jsonl (do not commit it before the freeze).

Every payload is inserted at a random sentence boundary of its host to make a base document; the
transformation is applied to the PAYLOAD only and the transformed payload goes back to the same place. Output
rows: one untransformed control per base (attack_type T0_none) plus one row per base and transformation
(attack_type = the transformation name), all label 1, group = the base id, meta "transform=...|base=...|batch=...".

The pattern bank is run on the controls: we want to measure evasion of attacks that the detector would
otherwise catch, so at least 15 of 20 controls must be caught in Batch A (replace payloads otherwise).
Evaluation sets are never part of the pool (source starts with ext_).
"""

import argparse
import random
import re
import sys
from collections import Counter

from data.evalsets.obfuscate import transforms_for
from eval.protocol import Row, write_jsonl

SOURCE = "ext_obfuscated"
CONTROL = "T0_none"
N_ITEMS = 20
MIN_CONTROLS_CAUGHT = 15
HOST_WORDS = (80, 200)
OUT_PATH = "data/evalsets/obfuscated_{batch}.jsonl"

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_URL_HOST = re.compile(r"https?://([^/\s:?#]+)", re.I)
_EMAIL_HOST = re.compile(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)+)")


def split_sentences(text):
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


def pick_position(host, rng):
    """Index of a sentence boundary strictly inside the host (1 .. n-1); 1 if the host has fewer than 3 sentences."""
    n = len(split_sentences(host))
    return rng.randint(1, n - 1) if n >= 3 else 1


def insert_payload(host, payload, position):
    """Host sentences with the payload placed before sentence number `position` (a sentence boundary)."""
    sents = split_sentences(host)
    position = min(max(position, 0), len(sents))
    return " ".join(sents[:position] + [payload] + sents[position:])


def check_fake_targets(text, where="text"):
    """Rule 15: the only URL host is example.com and the only e-mail domain is example.com."""
    bad = [h for h in _URL_HOST.findall(text) if h.lower() != "example.com"]
    bad += [h for h in _EMAIL_HOST.findall(text) if h.lower() != "example.com"]
    if bad:
        raise ValueError(f"{where}: only example.com targets are allowed (Rule 15); found {sorted(set(bad))}")


def check_sources(hosts, payloads, strict_words=True):
    if len(hosts) != N_ITEMS or len(payloads) != N_ITEMS:
        raise ValueError(f"expected {N_ITEMS} hosts and {N_ITEMS} payloads, got {len(hosts)} and {len(payloads)}")
    for i, h in enumerate(hosts, 1):
        n = len(h.split())
        if strict_words and not HOST_WORDS[0] <= n <= HOST_WORDS[1]:
            raise ValueError(f"host {i} has {n} words; it must have {HOST_WORDS[0]} to {HOST_WORDS[1]}")
        check_fake_targets(h, f"host {i}")
    for i, p in enumerate(payloads, 1):
        if not p.strip():
            raise ValueError(f"payload {i} is empty")
        check_fake_targets(p, f"payload {i}")


def build_rows(hosts, payloads, batch):
    """Returns (rows, controls) where controls is the list of untransformed base documents."""
    transforms = transforms_for(batch)
    rows, controls = [], []
    for i, (host, payload) in enumerate(zip(hosts, payloads), 1):
        base_id = f"obf_{batch}_{i:02d}"
        pos = pick_position(host, random.Random(base_id))
        base_text = insert_payload(host, payload, pos)
        controls.append(base_text)
        rows.append(Row.make(id=f"{base_id}__{CONTROL}", text=base_text, label=1, source=SOURCE, attack_type=CONTROL,
                             domain="general", group=base_id, meta=f"transform={CONTROL}|base={base_id}|batch={batch}"))
        for name, fn in transforms.items():
            transformed = fn(payload, random.Random(f"{base_id}:{name}"))
            rows.append(Row.make(id=f"{base_id}__{name}", text=insert_payload(host, transformed, pos), label=1,
                                 source=SOURCE, attack_type=name, domain="general", group=base_id,
                                 meta=f"transform={name}|base={base_id}|batch={batch}"))
    return rows, controls


def control_catch_count(controls):
    """How many untransformed base documents the pattern bank flags (score > 0), and how many with score >= 0.7."""
    from detectors import pattern_detector

    scores = [pattern_detector.score_document(t).score for t in controls]
    return sum(s > 0 for s in scores), sum(s >= 0.7 for s in scores)


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, choices=["A", "B"])
    ap.add_argument("--hosts", help="Batch B: text file, one host paragraph per line")
    ap.add_argument("--payloads", help="Batch B: text file, one payload sentence per line")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.batch == "A":
        if args.hosts or args.payloads:
            ap.error("--hosts and --payloads are only for --batch B")
        from data.evalsets.obfuscated_A_sources import HOSTS_A, PAYLOADS_A
        hosts, payloads = HOSTS_A, PAYLOADS_A
    else:
        if not (args.hosts and args.payloads):
            ap.error("--batch B needs --hosts FILE and --payloads FILE")
        hosts, payloads = read_lines(args.hosts), read_lines(args.payloads)

    check_sources(hosts, payloads)
    rows, controls = build_rows(hosts, payloads, args.batch)

    unchanged = [r.id for r in rows if r.attack_type != CONTROL and
                 r.text == next(c.text for c in rows if c.group == r.group and c.attack_type == CONTROL)]
    if unchanged:
        sys.exit(f"these transformed rows are identical to their control: {unchanged}")

    caught, strong = control_catch_count(controls)
    print(f"Control check: the pattern bank flags {caught} of {len(controls)} untransformed documents "
          f"({strong} with score >= 0.7).")
    if caught < MIN_CONTROLS_CAUGHT:
        msg = f"fewer than {MIN_CONTROLS_CAUGHT} controls are caught"
        if args.batch == "A":
            sys.exit(f"{msg}: replace payloads in data/evalsets/obfuscated_A_sources.py until at least "
                     f"{MIN_CONTROLS_CAUGHT} are caught.")
        print(f"WARNING: {msg}. Report this to Tvisha; do not change Batch B after looking at it.")

    out = args.out or OUT_PATH.format(batch=args.batch)
    write_jsonl(rows, out)
    counts = Counter(r.attack_type for r in rows)
    lengths = [r.length for r in rows]
    print(f"Wrote {len(rows)} rows to {out} ({len(controls)} bases; label 1 for every row).")
    for name in [CONTROL] + list(transforms_for(args.batch)):
        print(f"  {name}: {counts[name]}")
    print(f"Text length: {min(lengths)} to {max(lengths)} characters.")
    if args.batch == "B":
        print("Batch B stays private until the freeze: do not commit this file before then.")


if __name__ == "__main__":
    main()
