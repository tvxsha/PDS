"""
Builds the deepset evaluation set (roadmap task 2.10): data/evalsets/deepset.jsonl.

    python -m data.evalsets.build_deepset

Downloads deepset/prompt-injections through data.loaders.deepset (huggingface_hub + pandas, no
`datasets` package), changes the source to "ext_deepset" (evaluation sets are never pool sources,
Rule 2) and gives every row its own group. The set is NOT added to the pool and the frozen splits
are not touched.

deepset items are short chat prompts, many of them German; they are not retrieved documents. They
are used as an external check of the instruction signals only (see eval/deepset_eval.py).
Nothing is translated or filtered here.
"""

from collections import Counter

from data.loaders.deepset import load_deepset_rows
from eval.protocol import Row, write_jsonl

OUT_PATH = "data/evalsets/deepset.jsonl"
SOURCE = "ext_deepset"


def to_evalset_rows(rows):
    """Loader rows (source 'deepset', ids deepset_NNNN) -> evaluation-set rows (source 'ext_deepset')."""
    out = []
    for r in rows:
        new_id = r.id.replace("deepset_", f"{SOURCE}_", 1)
        out.append(Row.make(id=new_id, text=r.text, label=r.label, source=SOURCE, attack_type=r.attack_type,
                            domain=r.domain, group=new_id, meta=r.meta))
    return out


def main():
    rows = to_evalset_rows(load_deepset_rows())
    write_jsonl(rows, OUT_PATH)
    n_pos = sum(r.label for r in rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}: {n_pos} injection (label 1), {len(rows) - n_pos} legitimate (label 0).")
    print("By file split:", dict(Counter(r.meta for r in rows)))
    print("Expect roughly 660 rows in total. Evaluation set only: not part of the pool.")


if __name__ == "__main__":
    main()
