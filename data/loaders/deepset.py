"""
Loader for deepset/prompt-injections (roadmap task 2.10; first written in task 2.2).

Source:  https://huggingface.co/datasets/deepset/prompt-injections
Licence: Apache 2.0.
Columns: `text` (string) and `label` (0 = legitimate prompt, 1 = injection); about 660 rows in a
         train and a test file. Many prompts are German, and they are short chat prompts, not
         retrieved documents.

This loader does NOT import the `datasets` package (Rule 10: installing it upgrades
`huggingface-hub` to 2.x and breaks `transformers`). It lists the files of the dataset repository
with `huggingface_hub`, downloads every `.parquet` file (or, if there is none, the `.csv` /
`.json` / `.jsonl` files) and reads them with pandas.

    python -m data.loaders.deepset        # prints the counts per source (needs internet once)

Rows are returned with source "deepset" (the loader convention). The evaluation set
`data/evalsets/deepset.jsonl` (built by data/evalsets/build_deepset.py) renames the source to
"ext_deepset", because evaluation sets are never part of the pool (Rule 2).
"""

import os

from ._schema import Row, print_source_counts

REPO_ID = "deepset/prompt-injections"
SOURCE = "deepset"
FORMATS = (".parquet", ".csv", ".json", ".jsonl")      # in order of preference
SPLIT_ORDER = {"train": 0, "validation": 1, "val": 1, "test": 2}


def split_of(path):
    """'data/train-00000-of-00001.parquet' -> 'train'; 'test.csv' -> 'test'; otherwise the file stem."""
    stem = os.path.basename(path).split(".")[0]
    for name in SPLIT_ORDER:
        if stem.lower().startswith(name):
            return name
    return stem


def choose_files(repo_files):
    """Files to read, in a fixed order (train, validation, test, then the rest, each alphabetically)."""
    for ext in FORMATS:
        chosen = [f for f in repo_files if f.lower().endswith(ext)]
        if chosen:
            return sorted(chosen, key=lambda f: (SPLIT_ORDER.get(split_of(f), 3), f))
    raise RuntimeError(f"no {'/'.join(FORMATS)} file found in the dataset repository {REPO_ID}")


def read_table(path):
    import pandas as pd

    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported file type: {path}")


def frames_to_rows(frames):
    """frames: list of (split_name, DataFrame with columns text and label) -> list[Row]. Ids follow the file order."""
    rows, skipped = [], 0
    for split, df in frames:
        missing = [c for c in ("text", "label") if c not in df.columns]
        if missing:
            raise ValueError(f"{REPO_ID} ({split}): expected columns text and label, missing {missing}; "
                             f"found {list(df.columns)}")
        for text, label in zip(df["text"], df["label"]):
            if not isinstance(text, str) or not text.strip() or label != label:   # null text or label
                skipped += 1
                continue
            label = int(label)
            if label not in (0, 1):
                raise ValueError(f"{REPO_ID} ({split}): label must be 0 or 1, found {label!r}")
            rows.append(Row.make(id=f"{SOURCE}_{len(rows):04d}", text=text, label=label, source=SOURCE,
                                 attack_type="prompt_injection" if label == 1 else "clean",
                                 domain="general", meta=f"split={split}"))
    if skipped:
        print(f"[{REPO_ID}] skipped {skipped} rows with an empty text or label", flush=True)
    return rows


def load_deepset_rows():
    from huggingface_hub import hf_hub_download, list_repo_files

    files = choose_files(list_repo_files(REPO_ID, repo_type="dataset"))
    frames = []
    for f in files:
        path = hf_hub_download(repo_id=REPO_ID, filename=f, repo_type="dataset")
        frames.append((split_of(f), read_table(path)))
    return frames_to_rows(frames)


if __name__ == "__main__":
    print_source_counts(load_deepset_rows(), REPO_ID)
