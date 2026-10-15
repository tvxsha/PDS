"""
Static guard (roadmap task 6.3, Rule 1): no eval/*.py script may request the
held-out "test" role, except the handful of files that are explicitly
allowed to touch it. This turns "nobody peeks at the test split before the
freeze" from a rule people have to remember into something CI checks.

Uses the AST (not a text/regex scan) so string literals inside comments are
never considered -- a comment can say "test" freely, only code cannot.

    python -m pytest tests/test_no_test_role.py
"""

import ast
import glob
import os

# Files allowed to reference the "test" role by name, per the roadmap card.
# "the tests themselves" (eval/test_*.py) are exempted separately below,
# since a test ABOUT the test-role guard legitimately needs to spell it out.
ALLOWED = {
    "eval/final_test.py",
    "eval/protocol.py",
    "eval/protocol_eval.py",
    "eval/signal_cache.py",
}


def _is_test_file(path):
    return os.path.basename(path).startswith("test_")


def _string_literals(path):
    """Yield (lineno, value) for every string-literal AST node in the file.
    Using the AST (rather than a line-based # scan) means a comment
    containing the word "test" is never mistaken for code touching the
    held-out role."""
    with open(path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _eval_py_files():
    return sorted(glob.glob("eval/*.py"))


def test_no_eval_script_requests_the_test_role():
    offenders = []
    for path in _eval_py_files():
        if path in ALLOWED or _is_test_file(path):
            continue
        for lineno, value in _string_literals(path):
            if value == "test":
                offenders.append(f"{path}:{lineno}")
    assert offenders == [], (
        "these eval/*.py files reference the quoted role \"test\" outside the files "
        f"allowed to touch the held-out split (Rule 1): {offenders}"
    )


def test_allowed_list_names_files_that_actually_exist():
    # if one of these is ever renamed, the guard above would silently stop
    # checking it instead of failing loudly -- catch that here.
    missing = [p for p in ALLOWED if not os.path.isfile(p)]
    assert missing == [], f"ALLOWED lists files that don't exist: {missing}"


def test_guard_itself_detects_a_planted_violation():
    # a meta-test: prove the AST scan actually catches the thing it exists
    # to catch, using a string that is never written to disk.
    planted_source = 'from eval.sentence_eval import load_docs\n' \
                      'data = load_docs(42, ["train", "test"], "scores.csv")\n'
    tree = ast.parse(planted_source)
    hits = [n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == "test"]
    assert hits, "the AST scan failed to notice a role list containing \"test\""
