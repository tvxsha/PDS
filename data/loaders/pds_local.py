"""
Loader for the team's own hand-built documents (the original 77-doc set and
its extensions), so they live in the SAME pool and go through the SAME split
as every external source.

Sources produced (kept separate so results can be reported per source):
  pds_dev     data/clean (55) + data/poisoned (60: injection/authority/corpus/perplexity)
  pds_adv     data/adversarial (10 red-team docs, poisoned only)
  pds_xdomain data/cross_domain (12 benign HR/support policies, clean only)

CAVEAT that must stay attached to these sources: the regex rules and the
credential patterns were written after reading these documents, so they are
NOT independent evidence. The external sources are.
"""

import glob
import os

from eval.protocol import Row, print_source_counts


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def load_pds_local_rows():
    rows = []
    for p in sorted(glob.glob("data/clean/*.txt")):
        stem = os.path.splitext(os.path.basename(p))[0]
        rows.append(Row.make(f"pds_{stem}", _read(p), 0, "pds_dev", "clean", "software_docs"))
    for p in sorted(glob.glob("data/poisoned/*.txt")):
        stem = os.path.splitext(os.path.basename(p))[0]
        rows.append(Row.make(f"pds_{stem}", _read(p), 1, "pds_dev",
                             f"pds_{stem.split('_')[0]}", "software_docs"))
    for p in sorted(glob.glob("data/adversarial/*.txt")):
        stem = os.path.splitext(os.path.basename(p))[0]
        rows.append(Row.make(f"pds_{stem}", _read(p), 1, "pds_adv", "pds_adversarial", "software_docs"))
    for p in sorted(glob.glob("data/cross_domain/*.txt")):
        stem = os.path.splitext(os.path.basename(p))[0]
        rows.append(Row.make(f"pds_{stem}", _read(p), 0, "pds_xdomain", "clean", "hr_support_policy"))
    return rows


if __name__ == "__main__":
    rows = load_pds_local_rows()
    print_source_counts(rows, "pds_local")