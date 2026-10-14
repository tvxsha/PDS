"""
Task 5.4: cost-aware cascade (VALIDATION ONLY)

    python -m eval.cascade_eval --seed 42
    python -m eval.cascade_eval --all-seeds                      # seeds 42, 43, 44
    python -m eval.cascade_eval --seed 42 --latency PATH         # another latency file than eval/results/latency_benchmark.json

Running every signal on every document costs the sum of their times. Most traffic is clean, so a cascade only saves time if
it can stop early on documents that look clean. Two designs are measured on the same stages, ordered by their cost:

  Design A, OR cascade      A stage flags a document when its score is above that stage's flag threshold; otherwise the document
                            goes on to the next stage; after the last stage it is allowed. A clean document therefore pays for
                            every stage (unless a stage flags it by mistake), so the saving on clean traffic is about zero.
                            It is measured anyway: it is the honest baseline.
  Design B, clear-early     As A, and a stage may also ALLOW a document at once when its score is below that stage's clear
                            threshold: the q-quantile (q = 1%, 2%, 5%) of that stage's score among the POISONED TRAIN documents,
                            so each stage clears at most about q of the poisoned documents. A flag decision has priority over
                            a clear decision. The flag thresholds are those of design A.

Flag thresholds: the budget (5% and 1%) is split across the stages in steps of one tenth of the budget; every split is tried
and the one with the highest recall on the TRAIN documents wins (ties: the lower realised train FPR, then more budget for the
cheaper stages). Each stage's threshold is the quantile of its out-of-fold TRAIN clean scores for its share of the budget, so the
realised false-alarm rate of the whole cascade is at most the sum of the shares (union bound). Nothing is chosen on validation.

Stages are the signals of the 4.10 fusion candidate (eval/results/fusion_v2_candidate.json). Their cost is the median
milliseconds per document of eval/results/latency_benchmark.json (task 1.3), so the milliseconds below are EXPECTATIONS built
from medians, not an end-to-end measurement. "Run everything + LR" is the reference: every stage on every document, then the
fusion candidate at the same budget.

Reads   data/pooled/pooled.jsonl, data/splits/split_seed{seed}.json, the score cache from protocol_eval, the cached
        published-classifier scores, eval/results/fusion_v2_candidate.json, eval/results/latency_benchmark.json
Writes  eval/results/cascade_eval_seed{seed}.md, cascade_eval_seed{seed}.csv (one row per design and budget) and
        cascade_eval_seed{seed}_stages.csv (one row per design, budget and stage)
The held-out role is never requested here (eval.final_test runs the same code as the extra "cascade"). No torch import.
"""

import argparse
import json
import math
import os
import re

import numpy as np

SEEDS = (42, 43, 44)
BUDGETS = (0.05, 0.01)
CLEAR_QUANTILES = (0.01, 0.02, 0.05)
UNITS = 10                                       # the budget is split in steps of budget / UNITS
PREVALENCES = (0.01, 0.001)
REPS = 1000
LATENCY = "eval/results/latency_benchmark.json"
CARD_ORDER = ["pattern", "sentence", "embedding", "perplexity", "published"]
NAMES = {"pattern": ("pattern", "pattern bank", "pattern detector"),
         "sentence": ("sentence", "sentence classifier", "sentence detector"),
         "embedding": ("embedding", "embedding detector", "minilm"),
         "perplexity": ("perplexity", "perplexity detector"),
         "published": ("published", "published classifier")}
PLAIN_STAGES = ("pattern", "sentence", "perplexity", "published")      # every other fusion column is the embedding stage
RUN_ALL = "run everything + LR"
DESIGN_A = "A: OR cascade"
COLUMNS = (["design", "q", "budget", "stages", "split", "train_recall", "train_fpr", "tpr", "tpr_lo", "tpr_hi", "fpr", "fpr_lo",
            "fpr_hi", "clean_exit_early", "ms_clean", "ms_poisoned"]
           + [f"ms_prev_{p:g}" for p in PREVALENCES] + ["ms_all", "saving_clean", "n_pos", "n_neg"])
STAGE_COLUMNS = ["design", "q", "budget", "stage", "ms", "flag_thr", "clear_thr", "reach_clean", "reach_poisoned"]


def design_b(q):
    return f"B: clear-early, q = {q:.0%}"


# --------------------------------------------------------------------------
# the stages and their cost (no models, no data files: these are what the tests exercise)
# --------------------------------------------------------------------------

def stage_of(feature):
    """Stage of a fusion column. The original embedding and any embedding variant are one stage (one MiniLM encode)."""
    return feature if feature in PLAIN_STAGES else "embedding"


def stages_of(features):
    stages = []
    for f in features:
        s = stage_of(f)
        if s in stages:
            raise SystemExit(f"The fusion candidate has two embedding columns ({features}); a cascade with two embedding stages "
                             "is not defined. Ask Tvisha before running this.")
        stages.append(s)
    return stages


def _norm(name):
    return re.sub(r"[\s_\-]+", " ", str(name).strip().lower())


def read_latency(path):
    """({stage: median ms}, hardware text, {stage: p95 ms or None}, sorted detector keys) from the benchmark JSON of task 1.3.
    Keys are matched by name (pattern, sentence, embedding, perplexity, published; 'classifier' / 'bank' / 'detector' endings
    are accepted). A stage whose key is not found is simply absent from the result."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist. Task 1.3 (SreeValli) writes it:  python -m eval.latency_benchmark --seed 42 "
                         "--n 200 --repeats 3 --threads 4.  Pass --latency PATH to use another file.")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    det = data.get("detectors")
    if not isinstance(det, dict) or not det:
        raise SystemExit(f"{path} has no 'detectors' object (keys found: {sorted(data)}).")
    by_norm = {_norm(k): k for k in det}
    ms, p95 = {}, {}
    for stage, names in NAMES.items():
        key = next((by_norm[n] for n in names if n in by_norm), None)
        if key is None:
            continue
        entry = det[key]
        value = entry.get("median_ms") if isinstance(entry, dict) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise SystemExit(f"{path}: detectors[{key!r}]['median_ms'] is {value!r}, not a positive number.")
        ms[stage] = float(value)
        q = entry.get("p95_ms")
        p95[stage] = float(q) if isinstance(q, (int, float)) and not isinstance(q, bool) and math.isfinite(q) else None
    hw = data.get("hardware")
    hardware = "; ".join(f"{k}: {v}" for k, v in hw.items() if isinstance(v, (str, int, float))) if isinstance(hw, dict) else ""
    return ms, hardware, p95, sorted(det)


def check_stages(stages, ms, found_keys):
    missing = [s for s in stages if s not in ms]
    if missing:
        raise SystemExit(f"The latency file has no median for the stage(s) {missing}. Detector keys in the file: {found_keys}. "
                         "Ask the author of task 1.3 which key is which, or add the alias to NAMES in eval/cascade_eval.py.")


def order_by_cost(stages, ms):
    """Cheapest first by median ms; ties keep the order pattern, sentence, embedding, perplexity, published."""
    return sorted(stages, key=lambda s: (ms[s], CARD_ORDER.index(s)))


# --------------------------------------------------------------------------
# thresholds, the grid of budget splits, the cascade itself
# --------------------------------------------------------------------------

def threshold_at_budget(clean_scores, alpha):
    """The rule of eval.fpr_eval.threshold_at_budget: at most alpha of the clean scores lie strictly above it."""
    return float(np.quantile(clean_scores, 1 - alpha))


def compositions(total, parts):
    """All ways to write total as an ordered sum of `parts` non-negative integers."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def choose_split(train_scores, ytr, order, budget, units=UNITS):
    """Split `budget` over the stages (in steps of budget / units): the split with the highest recall on the TRAIN documents.
    Ties: lower realised train FPR, then more budget for the cheaper (earlier) stages. train_scores: {stage: array}."""
    ytr = np.asarray(ytr)
    pos, neg = ytr == 1, ytr == 0
    flags = {}
    for s in order:
        clean = np.asarray(train_scores[s], dtype=float)[neg]
        flags[s] = [np.asarray(train_scores[s], dtype=float) > threshold_at_budget(clean, budget * u / units)
                    for u in range(units + 1)]
    best = None
    for split in compositions(units, len(order)):
        hit = np.zeros(len(ytr), dtype=bool)
        for s, u in zip(order, split):
            hit |= flags[s][u]
        recall, fpr = float(hit[pos].mean()), float(hit[neg].mean())
        key = (-recall, fpr, tuple(-u for u in split))
        if best is None or key < best[0]:
            best = (key, split, recall, fpr)
    _, split, recall, fpr = best
    thr = {s: threshold_at_budget(np.asarray(train_scores[s], dtype=float)[neg], budget * u / units)
           for s, u in zip(order, split)}
    share = {s: budget * u / units for s, u in zip(order, split)}
    return {"split": split, "flag_thr": thr, "share": share, "train_recall": recall, "train_fpr": fpr}


def clear_thresholds(train_scores, ytr, order, q):
    """q-quantile of each stage's score among the POISONED train documents. The last stage needs none (it allows anyway)."""
    ytr = np.asarray(ytr)
    return {s: float(np.quantile(np.asarray(train_scores[s], dtype=float)[ytr == 1], q)) for s in order[:-1]}


def simulate(scores, flag_thr, clear_thr, order):
    """Run the cascade on documents. scores: {stage: array}. Returns (flagged bool array, exit stage index per document).
    A document leaves at the first stage that flags it, or (design B) clears it; the others reach the last stage, where they
    are flagged or allowed. A flag has priority over a clear. The exit index is the last stage that was RUN."""
    n = len(np.asarray(scores[order[0]]))
    alive = np.ones(n, dtype=bool)
    flagged = np.zeros(n, dtype=bool)
    exit_idx = np.full(n, len(order) - 1)
    for i, s in enumerate(order):
        x = np.asarray(scores[s], dtype=float)
        hit = alive & (x > flag_thr[s])
        flagged |= hit
        exit_idx[hit] = i
        alive &= ~hit
        if clear_thr is not None and i < len(order) - 1:
            clear = alive & (x < clear_thr[s])
            exit_idx[clear] = i
            alive &= ~clear
    return flagged, exit_idx


def expected_ms(exit_idx, y, order, ms, prevalences=PREVALENCES):
    """Expected milliseconds per document from the per-stage medians: a document that exits at stage i paid for stages 0..i.
    Returns (clean documents, poisoned documents, {prevalence: mix}, cost of running every stage)."""
    cum = np.cumsum([ms[s] for s in order])
    cost = cum[np.asarray(exit_idx)]
    y = np.asarray(y)
    clean, pois = float(cost[y == 0].mean()), float(cost[y == 1].mean())
    return clean, pois, {p: (1 - p) * clean + p * pois for p in prevalences}, float(cum[-1])


def split_text(order, share):
    return ", ".join(f"{s} {share[s] * 100:g}%" for s in order)


# --------------------------------------------------------------------------
# one budget: the designs side by side
# --------------------------------------------------------------------------

def _row(design, q, budget, order, ms, flagged, exit_idx, y, grp, reps, info=None, train=None):
    from eval.fpr_eval import group_bootstrap
    y = np.asarray(y)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    tpr, fpr = float(flagged[y == 1].mean()), float(flagged[y == 0].mean())
    (tl, th), (fl, fh) = group_bootstrap(flagged, y, np.asarray(grp), reps=reps)
    if fpr == 0:
        fh = max(float(fh), 3.0 / n_neg)                       # rule of three
    clean, pois, mix, ms_all = expected_ms(exit_idx, y, order, ms)
    last = len(order) - 1
    row = {"design": design, "q": "" if q is None else q, "budget": budget, "stages": "+".join(order),
           "split": "" if info is None else split_text(order, info["share"]),
           "train_recall": "" if train is None else train[0], "train_fpr": "" if train is None else train[1],
           "tpr": tpr, "tpr_lo": float(tl), "tpr_hi": float(th), "fpr": fpr, "fpr_lo": float(fl), "fpr_hi": float(fh),
           "clean_exit_early": float((exit_idx[y == 0] < last).mean()), "ms_clean": clean, "ms_poisoned": pois,
           "ms_all": ms_all, "saving_clean": max(0.0, round(1.0 - clean / ms_all, 12)), "n_pos": n_pos, "n_neg": n_neg}
    for p, v in mix.items():
        row[f"ms_prev_{p:g}"] = v
    return row


def _stage_rows(design, q, budget, order, ms, flag_thr, clear_thr, exit_idx, y):
    y = np.asarray(y)
    out = []
    for i, s in enumerate(order):
        out.append({"design": design, "q": "" if q is None else q, "budget": budget, "stage": s, "ms": ms[s],
                    "flag_thr": "" if flag_thr is None else flag_thr[s],
                    "clear_thr": "" if clear_thr is None or s not in clear_thr else clear_thr[s],
                    "reach_clean": float((exit_idx[y == 0] >= i).mean()), "reach_poisoned": float((exit_idx[y == 1] >= i).mean())})
    return out


def run_designs(train_scores, ytr, eval_scores, yev, gev, order, ms, budgets=BUDGETS, reps=REPS, lr=None, quantiles=CLEAR_QUANTILES):
    """Rows (list of dicts, columns COLUMNS) and stage rows (STAGE_COLUMNS) for every budget.
    train_scores / eval_scores: {stage: array}; lr: optional (train scores, eval scores) of the fusion candidate."""
    ytr, yev = np.asarray(ytr), np.asarray(yev)
    rows, stage_rows = [], []
    last = len(order) - 1
    for budget in budgets:
        if lr is not None:
            thr = threshold_at_budget(np.asarray(lr[0], dtype=float)[ytr == 0], budget)
            flagged = np.asarray(lr[1], dtype=float) > thr
            all_exit = np.full(len(yev), last)
            rows.append(_row(RUN_ALL, None, budget, order, ms, flagged, all_exit, yev, gev, reps))
            stage_rows += _stage_rows(RUN_ALL, None, budget, order, ms, None, None, all_exit, yev)
        info = choose_split(train_scores, ytr, order, budget)
        designs = [(DESIGN_A, None, None)] + [(design_b(q), q, clear_thresholds(train_scores, ytr, order, q)) for q in quantiles]
        for name, q, clear in designs:
            tr_flag, _ = simulate(train_scores, info["flag_thr"], clear, order)
            train = (float(tr_flag[ytr == 1].mean()), float(tr_flag[ytr == 0].mean()))
            flagged, exit_idx = simulate(eval_scores, info["flag_thr"], clear, order)
            rows.append(_row(name, q, budget, order, ms, flagged, exit_idx, yev, gev, reps, info=info, train=train))
            stage_rows += _stage_rows(name, q, budget, order, ms, info["flag_thr"], clear, exit_idx, yev)
    return rows, stage_rows


# --------------------------------------------------------------------------
# the evaluation behind eval.final_test (an extra) and behind this script's command line
# --------------------------------------------------------------------------

def run_cascade_ctx(ctx, latency_path=LATENCY):
    """The whole task for an eval.final_methods Context (validation in a dry run, the held-out role in the final run).
    Stage scores come from the same columns as the fusion candidate, so the cascade and the LR describe the same signals."""
    from eval import fusion_v2
    cand = fusion_v2.load_candidate()
    stages = stages_of(cand["features"])
    ms, hardware, p95, found = read_latency(latency_path)
    check_stages(stages, ms, found)
    order = order_by_cost(stages, ms)
    tr_cols, ev_cols = fusion_v2.candidate_columns(ctx, cand)
    train_scores = {stage_of(f): np.asarray(tr_cols[f], dtype=float) for f in cand["features"]}
    eval_scores = {stage_of(f): np.asarray(ev_cols[f], dtype=float) for f in cand["features"]}
    lr = fusion_v2.candidate_scores(ctx)                                 # refuses a provisional candidate in the final run
    budgets = tuple(ctx.config.get("budgets", BUDGETS))
    reps = int(ctx.config.get("bootstrap_reps", REPS))
    gev = np.array([d["grp"] for d in ctx.eval])
    rows, stage_rows = run_designs(train_scores, ctx.y_train, eval_scores, ctx.y_eval, gev, order, {s: ms[s] for s in order},
                                   budgets=budgets, reps=reps, lr=lr)
    return {"rows": rows, "stage_rows": stage_rows, "order": order, "ms": {s: ms[s] for s in order},
            "p95": {s: p95[s] for s in order}, "hardware": hardware, "cand": cand, "budgets": budgets,
            "n_pos": int((ctx.y_eval == 1).sum()), "n_neg": int((ctx.y_eval == 0).sum()), "n_eval": len(ctx.eval),
            "n_train": len(ctx.train), "latency_path": latency_path}


def to_frames(res):
    import pandas as pd
    return {"cascade": pd.DataFrame(res["rows"], columns=COLUMNS),
            "cascade_stages": pd.DataFrame(res["stage_rows"], columns=STAGE_COLUMNS)}


def cascade_extra(ctx):
    """eval.final_methods registers this as the extra "cascade": {file key: DataFrame}."""
    return to_frames(run_cascade_ctx(ctx))


def build_val_context(seed):
    from eval import final_methods as registry
    from eval.protocol import load_split, split_hash
    from eval.sentence_eval import find_scores_file, load_docs
    scores_path = find_scores_file(seed)
    split = load_split(seed)
    data = load_docs(seed, ["train", "val"], scores_path)                # the held-out role is never requested
    train, val = data["train"], data["val"]
    for role, docs in (("train", train), ("val", val)):
        if len(docs) != len(split["roles"][role]):
            raise SystemExit(f"{role}: the score cache has {len(docs)} of the {len(split['roles'][role])} documents of this split.")
    return registry.Context(seed=seed, eval_role="val", final=False, train=train, eval_docs=val, scores_path=scores_path,
                            split_hash=split_hash(split), config={})


def run_cascade(seed, eval_role="val", latency_path=LATENCY):
    """The cascade for one seed on the VALIDATION documents. The held-out role is run only by eval.final_test --final, which
    calls the registered extra "cascade" with its own context."""
    if eval_role != "val":
        raise SystemExit("run_cascade only runs the validation role. The held-out documents are scored by "
                         "python -m eval.final_test --final (the extra 'cascade' in eval/final_methods.py).")
    return run_cascade_ctx(build_val_context(seed), latency_path)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _ci(v, lo, hi, nd=2):
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def _rows_at(res, budget):
    return [r for r in res["rows"] if r["budget"] == budget]


def sentences(res, budget):
    """What the table says, generated from its numbers."""
    rows = {r["design"]: r for r in _rows_at(res, budget)}
    ref, a = rows.get(RUN_ALL), rows[DESIGN_A]
    out = []
    ms_all = a["ms_all"]
    out.append(f"Running every stage on every document costs {ms_all:.1f} ms (stages in order: {a['stages'].replace('+', ', ')}).")
    out.append(f"Design A stops early on {a['clean_exit_early']:.1%} of the clean validation documents (only the ones a stage flags "
               f"by mistake), so on clean traffic it costs {a['ms_clean']:.1f} ms per document, a saving of {a['saving_clean']:.1%}, "
               f"and it flags {a['tpr']:.2f} of the poisoned documents at an FPR of {a['fpr']:.3f}.")
    for r in rows.values():
        if not r["design"].startswith("B"):
            continue
        out.append(f"{r['design']}: {r['clean_exit_early']:.1%} of clean documents stop before the last stage, clean traffic costs "
                   f"{r['ms_clean']:.1f} ms per document (saving {r['saving_clean']:.1%}), at 1% poison {r['ms_prev_0.01']:.1f} ms; "
                   f"it flags {r['tpr']:.2f} of the poisoned documents ({r['tpr'] - a['tpr']:+.2f} against design A) at an FPR of {r['fpr']:.3f}.")
    if ref is not None:
        out.append(f"For reference the fusion candidate (all stages, then the LR) flags {ref['tpr']:.2f} at an FPR of {ref['fpr']:.3f}.")
    return out


def _signed(x):
    x = round(x, 1)
    return f"{(0.0 if x == 0 else x):+.1f}"


def conclusion(res, budget=None):
    """ONE sentence generated from the numbers of one budget (the first one, 5%, unless another is given)."""
    budget = res["budgets"][0] if budget is None else budget
    rows = _rows_at(res, budget)
    a = next(r for r in rows if r["design"] == DESIGN_A)
    bs = [r for r in rows if r["design"].startswith("B")]
    qs = ", ".join("{:.0%}".format(r["q"]) for r in bs)
    ms_b = ", ".join("{:.1f}".format(r["ms_clean"]) for r in bs)
    saving_b = ", ".join("{:.1%}".format(r["saving_clean"]) for r in bs)
    change_b = ", ".join(_signed((r["tpr"] - a["tpr"]) * 100) for r in bs)
    return (f"At the {budget:.0%} budget design A needs {a['ms_clean']:.1f} ms per clean document, only {a['saving_clean']:.1%} less "
            f"than the {a['ms_all']:.1f} ms of running every stage (a clean document passes every stage), while design B with "
            f"q = {qs} needs {ms_b} ms ({saving_b} less than running every stage) and changes the share of poisoned "
            f"documents flagged by {change_b} points against design A.")


def render(res, seed):
    n_pos, n_neg, order = res["n_pos"], res["n_neg"], res["order"]
    cand = res["cand"]
    lines = [f"# Cost-aware cascade, seed {seed} (task 5.4, validation only)", "",
             f"Validation: {res['n_eval']} documents ({n_pos} poisoned, {n_neg} clean); thresholds come from the {res['n_train']} TRAIN "
             "documents (out-of-fold scores for the sentence classifier) and are applied unchanged. Nothing here is chosen on "
             "validation. Intervals (95%) resample groups, not single documents.", "",
             "**The milliseconds are expectations built from per-stage medians (task 1.3), not an end-to-end measurement.** A document "
             "that leaves the cascade at a stage has paid for that stage and every earlier one. The LR on top of the stages is "
             "counted as free.", "",
             "**Design A saves nothing on clean traffic by construction:** a clean document passes every stage, so its expected "
             "time equals the full pipeline, except for the clean documents that a stage flags by mistake and that leave early "
             "(the column 'clean documents that stop early': for design A it is the share of clean documents flagged before the last stage). "
             "Only design B can save time on clean documents, and it pays for it with poisoned documents it allows too early.", "",
             f"**Conclusion (generated from the numbers below).** {conclusion(res)}", "",
             f"The stages are the signals of the fusion candidate, feature set {cand.get('feature_set', '?')} of task 4.10 "
             f"({', '.join(cand['features'])})"
             + ("" if cand.get("complete", True) else "; **the candidate is provisional** (" + ", ".join(cand.get("missing_inputs", []))
                + " not merged yet), so rerun this script after task 4.10 is rerun") + ".", "",
             "## Stages (cheapest first)", "", "| stage | median ms | 95th percentile ms |", "|---|---|---|"]
    for s in order:
        p95 = res["p95"][s]
        p95_text = "" if p95 is None else f"{p95:g}"
        lines.append(f"| {s} | {res['ms'][s]:g} | {p95_text} |")
    lines += ["", f"Cost file: `{res['latency_path']}`" + (f". Hardware: {res['hardware']}." if res["hardware"] else "."), ""]
    for budget in res["budgets"]:
        rows = _rows_at(res, budget)
        lines += [f"## False-alarm budget {budget:.0%}", "", conclusion(res, budget), "",
                  "| design | TPR (95% CI) | FPR (95% CI) | clean documents that stop early | ms per clean document | ms per poisoned "
                  "document | ms at 1% / 0.1% poison | saving on clean traffic |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['design']} | {_ci(r['tpr'], r['tpr_lo'], r['tpr_hi'])} | {_ci(r['fpr'], r['fpr_lo'], r['fpr_hi'], 3)} | "
                         f"{r['clean_exit_early']:.1%} | {r['ms_clean']:.1f} | {r['ms_poisoned']:.1f} | "
                         f"{r['ms_prev_0.01']:.1f} / {r['ms_prev_0.001']:.1f} | {r['saving_clean']:.1%} |")
        a = next(r for r in rows if r["design"] == DESIGN_A)
        lines += ["", f"Budget split chosen on TRAIN (flag thresholds of designs A and B): {a['split']}. On the TRAIN documents this split "
                  f"has recall {a['train_recall']:.2f} at a false-alarm rate of {a['train_fpr']:.3f} (the shares add up to {budget:.1%}; "
                  "the realised rate can differ by a document or two per stage because a threshold falls between two documents).", "",
                  "Share of documents that reach each stage:", "",
                  "| design | stage | flag threshold | clear threshold | clean documents | poisoned documents |", "|---|---|---|---|---|---|"]
        for r in res["stage_rows"]:
            if r["budget"] != budget or r["design"] == RUN_ALL:
                continue
            ft = "" if r["flag_thr"] == "" else f"{r['flag_thr']:.4f}"
            ct = "" if r["clear_thr"] == "" else f"{r['clear_thr']:.4f}"
            lines.append(f"| {r['design']} | {r['stage']} | {ft} | {ct} | {r['reach_clean']:.1%} | {r['reach_poisoned']:.1%} |")
        lines += ["", "In words (generated from the table above):", ""]
        lines += [f"- {s}" for s in sentences(res, budget)]
        lines.append("")
    lines += ["Caveats:", "",
              f"- With {n_neg} clean validation documents one false alarm is {1 / n_neg:.1%}, so FPR is coarse; intervals show it. "
              "Compare designs at similar realised FPR.",
              "- The cascade flags with one threshold per stage (an OR of the stages); it does not use the LR. Its recall and false-alarm "
              "rate therefore differ from the LR row, and only the milliseconds are directly comparable.",
              "- Design B clears at most about q of the poisoned TRAIN documents at each stage; over several stages the loss can add up. "
              "The measured TPR difference against design A is what counts, not q.",
              "- No q is chosen here. If the paper reports one q as the headline, name it before the held-out run; do not pick it "
              "by looking at the held-out numbers.",
              "- Stages with a share of 0% have a flag threshold at the largest TRAIN clean score: they flag only what no clean TRAIN "
              "document reached.",
              "- Milliseconds use per-stage medians from the benchmark machine. Shared work (tokenising the same text twice, loading "
              "models), batching and the cost of the LR are not modelled.",
              "- Validation numbers only; the held-out split was not used. A choice made after reading these tables makes them optimistic."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-seeds", action="store_true", help="seeds 42, 43 and 44")
    ap.add_argument("--latency", default=LATENCY, help="latency benchmark JSON (task 1.3)")
    args = ap.parse_args()
    os.makedirs("eval/results", exist_ok=True)
    for seed in (SEEDS if args.all_seeds else (args.seed,)):
        res = run_cascade(seed, "val", args.latency)
        base = f"eval/results/cascade_eval_seed{seed}"
        frames = to_frames(res)
        frames["cascade"].to_csv(base + ".csv", index=False, lineterminator="\n")
        frames["cascade_stages"].to_csv(base + "_stages.csv", index=False, lineterminator="\n")
        text = render(res, seed)
        with open(base + ".md", "w", encoding="utf-8") as f:
            f.write(text)
        print("\n" + text)
        print(f"Saved {base}.md\nSaved {base}.csv\nSaved {base}_stages.csv")


if __name__ == "__main__":
    main()