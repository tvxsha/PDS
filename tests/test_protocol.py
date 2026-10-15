"""
Tests for eval/protocol.py (roadmap task 6.3). No pool, no models: synthetic
Row objects only, plus a check of the committed split files' hashes.

    python -m pytest tests/test_protocol.py
"""

from eval.protocol import Row, make_split, validate_split, split_hash, load_split


def make_pool():
    """Two sources, 12 groups each, every group holding one clean + one
    poisoned row -- so every role that gets a group automatically gets
    both labels, and group-straddling is easy to detect."""
    rows = []
    for src in ("alpha", "beta"):
        for i in range(12):
            grp = f"{src}_grp{i:02d}"
            rows.append(Row.make(
                id=f"{grp}_clean", text=f"{src} clean document number {i} " * 5,
                label=0, source=src, attack_type="clean", domain="unit_test", group=grp,
            ))
            rows.append(Row.make(
                id=f"{grp}_poison", text=f"{src} poisoned document number {i} " * 5,
                label=1, source=src, attack_type="injection", domain="unit_test", group=grp,
            ))
    return rows


def test_make_split_is_deterministic():
    rows = make_pool()
    split_a = make_split(rows, seed=7)
    split_b = make_split(rows, seed=7)
    assert split_a["roles"] == split_b["roles"]
    assert split_a["dropped"] == split_b["dropped"]
    assert split_a["hash"] == split_b["hash"]


def test_make_split_is_group_safe():
    rows = make_pool()
    split = make_split(rows, seed=7)
    problems = validate_split(rows, split)
    assert problems == [], problems


def test_make_split_assigns_every_row_exactly_once():
    rows = make_pool()
    split = make_split(rows, seed=3)
    assigned = [i for role in ("baseline", "train", "val", "test") for i in split["roles"][role]]
    assigned += split["dropped"]
    assert sorted(assigned) == sorted(r.id for r in rows)
    assert len(assigned) == len(set(assigned))  # no id appears twice


def test_split_hash_is_a_pure_function_of_roles():
    rows = make_pool()
    split = make_split(rows, seed=11)
    assert split_hash(split) == split["hash"]
    # recomputing from the same roles (a fresh dict) gives the same hash
    same_roles = {"roles": split["roles"]}
    assert split_hash(same_roles) == split["hash"]


def test_validate_split_catches_a_group_straddle():
    rows = make_pool()
    split = make_split(rows, seed=7)
    # move one id from train into val by hand, without moving its group twin
    some_train_id = split["roles"]["train"][0]
    split["roles"]["train"].remove(some_train_id)
    split["roles"]["val"].append(some_train_id)
    problems = validate_split(rows, split)
    assert any("straddle" in p for p in problems)


def test_committed_split_hashes_match_the_roadmap():
    # data/pooled/pooled.jsonl is git-ignored (rebuilt locally), but the
    # split files themselves are committed -- this is the check every
    # teammate is told to run by hand in Section 5 step 5, made automatic.
    expected = {42: "6bdb8ff98e", 43: "93a61f0778", 44: "4d1126b320"}
    for seed, want in expected.items():
        split = load_split(seed=seed)
        assert split["hash"] == want, f"seed {seed}: expected {want}, got {split['hash']}"
        assert split_hash(split) == want
