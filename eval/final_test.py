"""
The one-shot final evaluation (roadmap task 1.7a).

    python -m eval.final_test --seed 42 --dry-run     # rehearsal on the VALIDATION split; repeat as often as you like
    python -m eval.final_test --seed 42 --final       # the held-out TEST split; once per seed, only after the freeze

Both commands run the same function, evaluate(seed, eval_role). The only differences are which documents are
scored (validation or held-out), the file prefix (final_dryrun_* or final_test_*) and the safety checks that
guard the held-out run.

What is evaluated is listed in eval/final_config.json; how each method is built lives in eval/final_methods.py.
Everything is fitted on TRAIN only. Each method's threshold is the TRAIN clean-score quantile at the stated
false-alarm budget (out-of-fold scores for fitted methods), applied unchanged to the evaluation documents.
There is no refit on train + validation (a conservative choice; say so in the paper).

Files written to eval/results/ (mode is "dryrun" or "test"):
    final_<mode>_seed<N>_scores.csv       id,label,source,attack_type,group,method,score   (evaluation documents only)
    final_<mode>_seed<N>_thresholds.json  {"seed","split_hash","role","thresholds":{method:{"0.05":t,"0.01":t}}}
    final_<mode>_seed<N>_<extra>.csv      one per file returned by an extra (for example the cascade of card 5.4)
    final_<mode>_seed<N>.md               the report
    final_FINAL_seed<N>.json              the lock (--final only)

The --final run refuses unless ALL of these hold; it checks them before any test document is loaded:
    * "frozen": true in eval/final_config.json
    * the seed is listed in the config
    * eval/results/final_FINAL_seed<N>.json does not exist (the split was not run before)
    * eval/results/protocol_FINAL_seed<N>.json does not exist (the old protocol_eval --final was not used)
    * git: code and data are committed (eval/results is not checked) and HEAD carries a tag "freeze-..."
It then scores and computes everything IN MEMORY, writes the lock file, and only then writes or prints a result.
A crash before the lock exists may be fixed and the command run again. Once the lock exists, never run it again.
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import time
from importlib import metadata

import numpy as np

from eval import final_methods as registry
from eval.fpr_eval import SRC_GROUPS, group_bootstrap, precision, source_group, threshold_at_budget
from eval.protocol import POOLED_PATH, load_split, read_jsonl, split_hash
from eval.sentence_eval import find_scores_file, load_docs

CONFIG_PATH = "eval/final_config.json"
RESULTS_DIR = "eval/results"
HELD_OUT = "test"
MODES = {"val": "dryrun", HELD_OUT: "test"}           # eval_role -> file prefix
FREEZE_TAG_PREFIX = "freeze-"
GIT_CHECKED = ["detectors", "fusion", "eval", "data", ":(exclude)eval/results"]
RESERVED_KEYS = {"scores", "thresholds"}
_KEY_OK = re.compile(r"[A-Za-z0-9_.\-]+")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def load_config():
    """(config dict, sha256 of the file's bytes). Exits with a plain message if the file is wrong."""
    if not os.path.exists(CONFIG_PATH):
        raise SystemExit(f"{CONFIG_PATH} not found. Run from the repo root (the folder that contains eval/ and data/).")
    with open(CONFIG_PATH, "rb") as f:
        raw = f.read()
    try:
        cfg = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise SystemExit(f"{CONFIG_PATH} is not valid JSON: {e}")
    problems = []
    if not isinstance(cfg, dict):
        raise SystemExit(f"{CONFIG_PATH} must contain a JSON object.")
    for key in ("frozen", "seeds", "budgets", "prevalences", "bootstrap_reps", "methods", "extras"):
        if key not in cfg:
            problems.append(f"missing key {key!r}")
    if not problems:
        if not isinstance(cfg["frozen"], bool):
            problems.append('"frozen" must be true or false')
        if not (isinstance(cfg["seeds"], list) and cfg["seeds"] and all(type(s) is int for s in cfg["seeds"])):
            problems.append('"seeds" must be a non-empty list of integers')
        for key in ("budgets", "prevalences"):
            v = cfg[key]
            if not (isinstance(v, list) and v and len(set(v)) == len(v)
                    and all(type(x) in (int, float) and 0 < x < 1 for x in v)):
                problems.append(f'"{key}" must be a non-empty list of different numbers between 0 and 1')
        if not (type(cfg["bootstrap_reps"]) is int and cfg["bootstrap_reps"] >= 1):
            problems.append('"bootstrap_reps" must be a positive integer')
        for key, nonempty in (("methods", True), ("extras", False)):
            v = cfg[key]
            if not (isinstance(v, list) and all(isinstance(x, str) for x in v) and len(set(v)) == len(v)
                    and (v or not nonempty)):
                problems.append(f'"{key}" must be a list of different names' + (" (not empty)" if nonempty else ""))
    if problems:
        raise SystemExit(f"{CONFIG_PATH} is not valid:\n  - " + "\n  - ".join(problems))
    return cfg, hashlib.sha256(raw).hexdigest()


def check_registry(cfg):
    """Every name in the config must be registered in eval/final_methods.py."""
    problems = []
    for key, reg in (("methods", registry.METHODS), ("extras", registry.EXTRAS)):
        unknown = [n for n in cfg[key] if n not in reg]
        if unknown:
            problems.append(f'"{key}" in {CONFIG_PATH} names {unknown}, which eval/final_methods.py does not register. '
                            f"Registered: {sorted(reg)}")
    if problems:
        raise SystemExit("\n".join(problems))


# --------------------------------------------------------------------------
# the checks that guard the held-out run
# --------------------------------------------------------------------------

def lock_path(seed):
    return os.path.join(RESULTS_DIR, f"final_FINAL_seed{seed}.json")


def _refuse(*lines):
    raise SystemExit("\n".join(["REFUSED. The held-out documents were not opened, nothing was written."] + list(lines)))


def _git(*args):
    try:
        p = subprocess.run(["git", *args], capture_output=True, text=True)
    except OSError:
        _refuse("git is not available on PATH. The lock file records the commit, so git is needed for --final.")
    if p.returncode != 0:
        _refuse(f"'git {' '.join(args)}' failed: {p.stderr.strip() or p.stdout.strip()}")
    return p.stdout.strip()


def final_preconditions(seed, cfg):
    """Returns {"git_head", "git_tags"}. Raises SystemExit with the reason if the held-out run is not allowed."""
    if cfg["frozen"] is not True:
        _refuse(f'{CONFIG_PATH} says "frozen": false.',
                "The test split is run only after the freeze (ROADMAP.md, section 11.3).",
                f"For a rehearsal use:  python -m eval.final_test --seed {seed} --dry-run")
    if seed not in cfg["seeds"]:
        _refuse(f"Seed {seed} is not in the config's seeds {cfg['seeds']}.")
    if os.path.exists(lock_path(seed)):
        _refuse(f"The test split was already evaluated for seed {seed} ({lock_path(seed)}).",
                "Running it again would be a second look at the test data, so it is not allowed.",
                "A bug found later is described in the paper's limitations, not fixed by a re-run.")
    old = os.path.join(RESULTS_DIR, f"protocol_FINAL_seed{seed}.json")
    if os.path.exists(old):
        _refuse(f"The test split of seed {seed} was already evaluated by the old script ({old}).",
                "This would be a second look. Tell the team; the paper has to say how the test split was used.")
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", *GIT_CHECKED)
    if dirty:
        shown = dirty.splitlines()
        _refuse("These files in detectors/, fusion/, eval/ or data/ are not committed:",
                *["  " + line for line in shown[:15]], *(["  ..."] if len(shown) > 15 else []),
                "Commit them (and the frozen config) first. eval/results is not checked.")
    tags = [t for t in _git("tag", "--points-at", "HEAD").splitlines() if t.startswith(FREEZE_TAG_PREFIX)]
    if not tags:
        _refuse(f'HEAD has no tag starting with "{FREEZE_TAG_PREFIX}". Section 11.3: commit "Freeze final configuration", then',
                "  git tag freeze-v1", "and run this again.")
    return {"git_head": head, "git_tags": tags}


def _versions():
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "scikit-learn", "pandas", "torch", "transformers", "sentence-transformers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _create_lock(seed, payload):
    """Exclusive create: fails if the file exists, even if two commands start at the same moment."""
    path = lock_path(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        f = open(path, "x", encoding="utf-8", newline="\n")
    except FileExistsError:
        _refuse(f"{path} appeared while this command was running. Another run is using the test split. Nothing was written.")
    with f:
        json.dump(payload, f, indent=1)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return path


# --------------------------------------------------------------------------
# scoring and thresholds (identical for both roles)
# --------------------------------------------------------------------------

def compute(seed, eval_role, cfg):
    check_registry(cfg)
    scores_path = find_scores_file(seed)
    split = load_split(seed)
    h = split_hash(split)
    data = load_docs(seed, ["train", eval_role], scores_path)
    train, ev = data["train"], data[eval_role]
    for role, docs in (("train", train), (eval_role, ev)):
        want = len(split["roles"][role])
        if len(docs) != want:
            raise SystemExit(f"{role}: the score cache {scores_path} has {len(docs)} of the {want} documents of this split. "
                             f"Fill it with:  python -m eval.protocol_eval --seed {seed}   (it scores every role; "
                             "it evaluates the test split only with --final)")
    pool = {r.id: r for r in read_jsonl(POOLED_PATH)}
    for d in train + ev:
        d["attack"] = pool[d["id"]].attack_type
    print(f"seed {seed}: {len(train)} train documents, {len(ev)} {eval_role} documents, split hash {h}", flush=True)

    ctx = registry.Context(seed=seed, eval_role=eval_role, final=(eval_role == HELD_OUT), train=train,
                           eval_docs=ev, scores_path=scores_path, split_hash=h, config=cfg)
    ytr = ctx.y_train
    train_scores, eval_scores, thresholds = {}, {}, {}
    for name in cfg["methods"]:
        tr, es = registry.METHODS[name](ctx)
        tr, es = np.asarray(tr, dtype=float), np.asarray(es, dtype=float)
        if tr.shape != (len(train),) or es.shape != (len(ev),):
            raise SystemExit(f"method {name!r} returned scores of shape {tr.shape} and {es.shape}; "
                             f"expected ({len(train)},) and ({len(ev)},)")
        if not (np.isfinite(tr).all() and np.isfinite(es).all()):
            raise SystemExit(f"method {name!r} returned a score that is not a finite number")
        train_scores[name], eval_scores[name] = tr, es
        thresholds[name] = {b: threshold_at_budget(tr[ytr == 0], b) for b in cfg["budgets"]}

    extras = {}
    for name in cfg["extras"]:
        out = registry.EXTRAS[name](ctx)
        if not isinstance(out, dict):
            raise SystemExit(f"extra {name!r} must return a dict {{file_key: DataFrame}}")
        for key, frame in out.items():
            if not (isinstance(key, str) and _KEY_OK.fullmatch(key)) or key in RESERVED_KEYS:
                raise SystemExit(f"extra {name!r}: bad file key {key!r} (letters, digits, _ - . only; "
                                 f"not {sorted(RESERVED_KEYS)})")
            if key in extras:
                raise SystemExit(f"two extras return the file key {key!r}")
            if not hasattr(frame, "to_csv"):
                raise SystemExit(f"extra {name!r}, key {key!r}: expected a DataFrame")
            extras[key] = frame

    y = ctx.y_eval
    flags = {(b, m): eval_scores[m] > thresholds[m][b] for b in cfg["budgets"] for m in cfg["methods"]}
    return dict(seed=seed, role=eval_role, split_hash=h, cfg=cfg, n_train=len(train), docs=ev, y=y,
                grp=np.array([d["grp"] for d in ev]), srcg=np.array([source_group(d["src"]) for d in ev]),
                atk=np.array([d["attack"] for d in ev]), eval_scores=eval_scores, thresholds=thresholds,
                flags=flags, extras=extras)


# --------------------------------------------------------------------------
# files (all rendered as text in memory first)
# --------------------------------------------------------------------------

def _pct(p):
    return f"{p * 100:g}%"


def _rate(flags, mask):
    return float(flags[mask].mean()) if mask.any() else float("nan")


def _cell(x):
    return "n/a" if x != x else f"{x:.2f}"


def render_scores_csv(res):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "label", "source", "attack_type", "group", "method", "score"])
    for m in res["cfg"]["methods"]:
        for d, s in zip(res["docs"], res["eval_scores"][m]):
            w.writerow([d["id"], d["y"], d["src"], d["attack"], d["grp"], m, repr(float(s))])
    return buf.getvalue()


def render_thresholds(res):
    thr = {m: {str(b): float(t) for b, t in per.items()} for m, per in res["thresholds"].items()}
    return json.dumps({"seed": res["seed"], "split_hash": res["split_hash"], "role": res["role"], "thresholds": thr},
                      indent=1) + "\n"


def render_report(res):
    cfg, y, grp = res["cfg"], res["y"], res["grp"]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    reps, methods = cfg["bootstrap_reps"], cfg["methods"]
    final = res["role"] == HELD_OUT
    if final:
        head = [f"# Final evaluation on the held-out TEST split (seed {res['seed']})", "",
                "The test documents were scored once for this seed, after the configuration was frozen. Fitting used TRAIN only "
                "(no refit on train + validation, a conservative choice). Each method's threshold is the TRAIN clean-score quantile "
                "at the stated false-alarm budget (out-of-fold scores for fitted methods), applied unchanged."]
    else:
        head = [f"# Dry run on the VALIDATION split (seed {res['seed']}): rehearsal of the final test", "",
                "This is the same code as the final test run, with validation documents in place of the held-out ones. "
                "It is not the result. Thresholds are the TRAIN clean-score quantile at the stated false-alarm budget "
                "(out-of-fold scores for fitted methods), applied unchanged to validation."]
    lines = head + ["",
                    f"Split hash {res['split_hash']}. Evaluation documents: {len(y)} ({n_pos} poisoned, {n_neg} clean); "
                    f"{res['n_train']} train documents. Config sha256 {res['config_sha256'][:12]}. "
                    f"Intervals are 95% group-bootstrap intervals ({reps} resamples of whole groups: the same email, "
                    f"question or twin pair). With only {n_neg} clean documents, one false alarm is {1 / n_neg:.1%}, so FPR is coarse.",
                    ""]
    prev_head = " / ".join(_pct(p) for p in cfg["prevalences"])
    for b in cfg["budgets"]:
        lines += [f"## False-alarm budget {_pct(b)}", "",
                  f"| method | TPR (95% CI) | FPR (95% CI) | precision at {prev_head} poison | false alarms per 10,000 clean |",
                  "|---|---|---|---|---|"]
        for m in methods:
            fl = res["flags"][(b, m)]
            tpr, fpr = float(fl[y == 1].mean()), float(fl[y == 0].mean())
            (tl, th), (fl_lo, fl_hi) = group_bootstrap(fl, y, grp, reps=reps)
            prec = " / ".join(f"{precision(tpr, fpr, p, n_neg):{'.1%' if p >= 0.01 else '.2%'}}" for p in cfg["prevalences"])
            fa = f"{fpr * 10000:.0f}" if fpr > 0 else f"0 seen (up to ~{3.0 / n_neg * 10000:.0f})"
            lines.append(f"| {m} | {tpr:.2f} [{tl:.2f}, {th:.2f}] | {fpr:.3f} [{fl_lo:.3f}, {fl_hi:.3f}] | {prec} | {fa} |")
        lines.append("")
    lines += ["Precision is computed from the TPR and FPR above for the stated shares of poisoned documents; it is not measured on "
              f"a stream with that share. Where no false alarm was seen it uses the rule-of-three upper bound (3/{n_neg} = "
              f"{3.0 / n_neg:.1%}) instead of zero, so zero-false-alarm methods are not flattered.", ""]

    present = set(res["srcg"])
    sources = [g for g in SRC_GROUPS if g in present] + sorted(present - set(SRC_GROUPS))
    types = sorted({t for t in res["atk"][y == 1]})
    for b in cfg["budgets"]:
        lines += [f"## By source (budget {_pct(b)}): TPR / FPR", "",
                  "| method | " + " | ".join(f"{g} ({int(((res['srcg'] == g) & (y == 1)).sum())} pois / "
                                             f"{int(((res['srcg'] == g) & (y == 0)).sum())} clean)" for g in sources) + " |",
                  "|---|" + "---|" * len(sources)]
        for m in methods:
            fl = res["flags"][(b, m)]
            cells = [f"{_cell(_rate(fl, (res['srcg'] == g) & (y == 1)))} / {_cell(_rate(fl, (res['srcg'] == g) & (y == 0)))}"
                     for g in sources]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines += ["", f"## By attack type (budget {_pct(b)}): TPR among poisoned documents", "",
                  "| attack type | n | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
        for t in types:
            k = (res["atk"] == t) & (y == 1)
            lines.append(f"| {t} | {int(k.sum())} | " + " | ".join(_cell(_rate(res["flags"][(b, m)], k)) for m in methods) + " |")
        lines.append("")

    lines += ["Caveats:", ""]
    if not final:
        lines += ["- Validation numbers. Any design choice made after looking at them makes them optimistic; the final test run is the "
                  "only unbiased use of the held-out documents.",
                  "- Nothing here is locked: this command can be repeated, and the files are overwritten."]
    else:
        lines += ["- This run is locked: methods, thresholds and configuration were fixed before the test documents were scored, "
                  "and it was run once for this seed. A bug found later is reported as a limitation, not fixed by a re-run.",
                  "- Several methods are compared on the same documents; no correction for multiple comparisons is applied."]
    lines += ["- The published classifier is not fitted on our data; the train split only fixes its cut-off.",
              "- Small cells (an attack type with a handful of documents, one false alarm = a whole percentage point) are noisy. "
              "Read the intervals.",
              "- Out-of-fold train scores are slightly less confident than scores from a model fitted on all of train, so the "
              "realised FPR can sit a little off the target. Compare methods at similar realised FPR: a method that overshoots its "
              "budget gets extra recall for free.",
              "- 'Zero false alarms' on a few dozen clean documents only says the rate is below a few percent; it does not mean zero.",
              "- Hand-written adversarial and obfuscated sets are evaluated separately (stress_eval); they are not part of these tables."]
    return "\n".join(lines) + "\n"


def render_outputs(res):
    """{filename: text}. Everything that can go wrong while formatting goes wrong here, before any lock exists."""
    prefix = f"final_{MODES[res['role']]}_seed{res['seed']}"
    out = {f"{prefix}_scores.csv": render_scores_csv(res), f"{prefix}_thresholds.json": render_thresholds(res)}
    for key, frame in res["extras"].items():
        out[f"{prefix}_{key}.csv"] = frame.to_csv(index=False, lineterminator="\n")
    out[f"{prefix}.md"] = render_report(res)
    return out


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_outputs(outputs):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    paths = []
    for name, text in outputs.items():
        path = os.path.join(RESULTS_DIR, name)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------
# the one function both commands call
# --------------------------------------------------------------------------

def evaluate(seed, eval_role):
    """eval_role "val": rehearsal, writes final_dryrun_*. eval_role "test": the one-shot run, writes final_test_* and the lock."""
    if eval_role not in MODES:
        raise ValueError(f"eval_role must be one of {sorted(MODES)}, not {eval_role!r}")
    cfg, cfg_sha = load_config()
    git = None
    if eval_role == HELD_OUT:
        git = final_preconditions(seed, cfg)          # raises before any held-out document is loaded
    t0 = time.perf_counter()
    res = compute(seed, eval_role, cfg)
    res["config_sha256"] = cfg_sha
    outputs = render_outputs(res)

    paths = []
    if eval_role == HELD_OUT:
        lock = {"evaluated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                "seed": seed, "split_hash": res["split_hash"], "role": eval_role, **git,
                "config_sha256": cfg_sha, "methods": cfg["methods"], "extras": cfg["extras"],
                "n_train": res["n_train"], "n_eval": len(res["docs"]), "versions": _versions(),
                "outputs": {name: _sha256_text(text) for name, text in outputs.items()},
                "seconds": round(time.perf_counter() - t0, 1)}
        paths.append(_create_lock(seed, lock))   # from here on the split counts as used
    paths += write_outputs(outputs)

    print("\n" + outputs[f"final_{MODES[eval_role]}_seed{seed}.md"])
    print("Written:", *paths, sep="\n  ")
    if eval_role == HELD_OUT:
        print("\nThe test split is now used for this seed. Do NOT run --final again for it.")
        print("Commit these files (all of them, including the lock):\n  git add " + " ".join(p.replace(os.sep, "/") for p in paths))
    else:
        print("\nRehearsal only: validation documents, nothing locked. Repeat as often as you like.")
    res["files"] = paths
    return res


def main():
    ap = argparse.ArgumentParser(description="One-shot final evaluation (see the top of this file).")
    ap.add_argument("--seed", type=int, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="rehearse on the validation split (repeatable)")
    mode.add_argument("--final", action="store_true", help="run the held-out split ONCE (after the freeze)")
    args = ap.parse_args()
    evaluate(args.seed, HELD_OUT if args.final else "val")


if __name__ == "__main__":
    main()