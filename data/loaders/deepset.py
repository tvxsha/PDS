"""
Loader for deepset/prompt-injections (task 2.2).

Source: https://huggingface.co/datasets/deepset/prompt-injections
License: Apache 2.0 (open, no gating, no access request needed).
Schema on HF: two columns, `text` (string) and `label` (0 = legit prompt,
1 = injection), ~662 rows total across train+test splits combined.

This is the simplest of the three 2.2 sources -- already binary labeled,
no attack-combination logic needed (unlike opi.py).

Requires: `pip install datasets` (HuggingFace `datasets` library). Not
run/verified in this session -- no network-installable ML deps were
available here tonight (see docs/baselines.md for the same caveat on 3.1).
Run this file directly to both verify it works and print the 2.2 "done
when" counts.
"""

from ._schema import Row, print_source_counts


def load_deepset_rows():
    from datasets import load_dataset, concatenate_datasets

    ds = load_dataset("deepset/prompt-injections")
    # Combine all available splits (train + test) into one pool; 2.6
    # (pooling/splitting across all sources) is a separate, later task.
    all_splits = concatenate_datasets([ds[split] for split in ds.keys()])

    rows = []
    for i, ex in enumerate(all_splits):
        label = int(ex["label"])
        rows.append(
            Row.make(
                id=f"deepset_{i:04d}",
                text=ex["text"],
                label=label,
                source="deepset",
                attack_type="prompt_injection" if label == 1 else "clean",
                domain="general",
            )
        )
    return rows


if __name__ == "__main__":
    rows = load_deepset_rows()
    print_source_counts(rows, "deepset/prompt-injections")
