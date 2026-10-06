"""
Loader for HackAPrompt, "direct injection" slice (task 2.2).

Source: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset
License: MIT.

**CORRECTION vs. earlier research notes:** this dataset is currently
**gated** on HuggingFace ("You need to agree to share your contact
information to access this dataset") -- it is not an open-access download.
Whoever runs this needs their own HF account, to accept the dataset's terms
on its page, and a logged-in `huggingface-cli login` (or `HF_TOKEN` env var)
first, the same manual step already documented for Meta Prompt Guard in
docs/baselines.md. This was not flagged correctly in earlier session notes
tonight -- confirmed directly against the HF page just now.

**Confirmed columns** (from the HF dataset viewer, not guessed):
  level, user_input, prompt, completion, model, expected_completion,
  token_count, correct, error, score, dataset, timestamp
(`score` only populated for submissions-platform rows; `timestamp` only for
playground rows.)

**"Direct injection" slice:** the roadmap asks for a *separate* slice,
distinct from the other two 2.2 sources, rather than the whole competition
dump (which spans many levels of increasing defense difficulty --
translation tasks, output-format constraints, etc., not just plain
override attempts). HackAPrompt's levels increase in defense complexity;
level 1 is the original competition's most basic "get the model to say
exactly 'I have been PWNED'" direct-override task, which is the cleanest
match for a "direct injection" slice. **This level numbering is from
memory of the public HackAPrompt competition, not re-verified against the
gated data in this session** -- confirm `df['level'].unique()` once you
have access, and adjust `levels` below if level 1 isn't what you want.

`correct == True` means the submitted user_input successfully broke the
target model's instructions for that level -- i.e. a *successful* prompt
injection, which is label 1 here. `correct == False` rows are attempted-
but-failed injections; attempted and failed is still an injection attempt
in intent, so they are also label 1 (attack_type is tagged so you can
filter either way downstream -- see `label_mode` below if you'd rather only
keep successful attacks as positives).

Requires: `pip install datasets` + a HF login with dataset access accepted.
Not run/verified in this session (gated + no ML deps installed tonight).
"""

from ._schema import Row, print_source_counts


def load_hackaprompt_rows(levels=(1,), label_mode="any_attempt"):
    """
    levels: which HackAPrompt competition levels to include in the "direct
        injection" slice. Defaults to level 1 only -- see module docstring.
    label_mode:
        "any_attempt"  -> every row in `levels` is label 1 (it's a prompt
                          injection attempt, successful or not)
        "success_only" -> only rows where correct == True are label 1;
                          failed attempts are dropped entirely (use this if
                          you want a cleaner "this actually worked" slice)
    """
    from datasets import load_dataset

    ds = load_dataset("hackaprompt/hackaprompt-dataset")
    # Competition data lives in a single split in most HF snapshots of this
    # dataset; concatenate whatever splits exist to be safe.
    all_splits = ds.keys()

    rows = []
    i = 0
    for split in all_splits:
        for ex in ds[split]:
            if ex.get("level") not in levels:
                continue
            text = ex.get("user_input")
            if not text:
                continue
            is_success = bool(ex.get("correct", False))
            if label_mode == "success_only" and not is_success:
                continue

            attack_type = (
                "prompt_injection_success" if is_success else "prompt_injection_attempt"
            )
            rows.append(
                Row.make(
                    id=f"hackaprompt_{i:05d}",
                    text=text,
                    label=1,
                    source="hackaprompt",
                    attack_type=attack_type,
                    domain=f"hackaprompt_level_{ex.get('level')}",
                )
            )
            i += 1
    return rows


if __name__ == "__main__":
    rows = load_hackaprompt_rows()
    print_source_counts(rows, "hackaprompt (direct injection slice, level 1)")
