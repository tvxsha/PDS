"""
Task 4.10: re-fit the fusion with the new signals (TRAIN + VALIDATION only).

    python -m eval.fusion_v2                 # seeds 42, 43, 44, then the choice -> fusion_v2_candidate.json
    python -m eval.fusion_v2 --seed 42       # one seed's report only (the choice needs all three seeds)

Feature sets (every one is a logistic regression, fitted and scored exactly like the "LR" rows of eval.fpr_eval):
    F0  pattern + embedding + perplexity
    F1  F0 + sentence classifier (out-of-fold on train, as fit_everything does)
    F2  F1 with the embedding replaced by the variant selected in task 4.5   (skipped if 4.5 kept the original)
    F3  F1 with the embedding replaced by the variant selected in task 4.4   (skipped if 4.4 kept the original,
                                                                              or if its result file does not exist yet)
    F4  F1 + the published classifier's margin (squashed like eval.baseline_eval)
    F5  F1 with the embedding replaced by BOTH variants (two columns)        (needs both)

The choice is made BEFORE validation is looked at and uses TRAIN numbers only: the highest mean out-of-fold TRAIN AUC
over seeds 42 to 44; a set whose mean is less than 0.005 below the best is a tie, and among tied sets the one with
fewer features and cheaper signals wins (order F0, F1, F2, F3, F5, F4). If nothing beats F1, F1 stays.

Reads   data/pooled/pooled.jsonl, data/splits/split_seed{seed}.json, the score cache from protocol_eval,
        eval/results/embedding_variants_selected.json (4.5), eval/results/embedding_boilerplate_selected.json (4.4,
        optional), the variant caches signal_<name>_seed<N>_<hash>.csv, the published-classifier cache.
Writes  eval/results/fusion_v2_seed{seed}.md and eval/results/fusion_v2_candidate.json
The test split is never loaded. No torch import.
"""

import argparse
import json
import os

import numpy as np

SEEDS = (42, 43, 44)
TIE = 0.005
SEL_45 = "eval/results/embedding_variants_selected.json"
SEL_44 = "eval/results/embedding_boilerplate_selected.json"
CANDIDATE = "eval/results/fusion_v2_candidate.json"
ORIGINAL = "original"
BASE_FEATURES = ("pattern", "embedding", "perplexity", "sentence", "published")
SET_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5"]
PREFERENCE = ["F0", "F1", "F2", "F3", "F5", "F4"]       # tie-break: fewer features, then cheaper signals
DESCRIPTION = {"F0": "pattern + embedding + perplexity",
               "F1": "F0 + sentence classifier",
               "F2": "F1 with the task 4.5 embedding variant",
               "F3": "F1 with the task 4.4 embedding variant",
               "F4": "F1 + published classifier",
               "F5": "F1 with both embedding variants"}
RULE = (f"highest mean out-of-fold TRAIN AUC over seeds {SEEDS[0]} to {SEEDS[-1]}; sets less than {TIE} below the best "
        "are tied and the one with fewer features and cheaper signals wins (F0, F1, F2, F3, F5, F4)")


# --------------------------------------------------------------------------
# inputs, feature sets, selection (no models, no data files: these are what the tests exercise)
# --------------------------------------------------------------------------

def read_selection(path):
    """('missing', None) | ('original', None) | ('variant', name) from a selection JSON of task 4.4 or 4.5."""
    if not os.path.exists(path):
        return "missing", None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    name = next((data[k] for k in ("selected", "selected_variant", "variant", "name") if isinstance(data.get(k), str)), None)
    if name is None:
        raise SystemExit(f"{path} has no text value under 'selected', 'selected_variant', 'variant' or 'name' "
                         f"(keys found: {sorted(data)}). Ask its author which key holds the choice.")
    if data.get("uses_variant") is False or name.strip().lower() in ("", "original", "original stays", "none"):
        return "original", None
    return "variant", name.strip()


def feature_list(set_name, v45=None, v44=None):
    """Column names of a feature set, or None if the set cannot be built (a variant it needs does not exist)."""
    if set_name == "F0":
        return ["pattern", "embedding", "perplexity"]
    if set_name == "F1":
        return ["pattern", "embedding", "perplexity", "sentence"]
    if set_name == "F2":
        return None if v45 is None else ["pattern", v45, "perplexity", "sentence"]
    if set_name == "F3":
        return None if v44 is None else ["pattern", v44, "perplexity", "sentence"]
    if set_name == "F4":
        return ["pattern", "embedding", "perplexity", "sentence", "published"]
    if set_name == "F5":
        return None if v45 is None or v44 is None else ["pattern", v45, v44, "perplexity", "sentence"]
    raise ValueError(set_name)


def squash(margin):
    return np.clip(margin, -10, 10) / 10.0      # the same as eval.baseline_eval.squash


def build_matrix(features, tr, ev):
    """Design matrices (train, eval) for a list of feature names. tr / ev: {name: array}.
    The published margin is squashed; an embedding variant is first mapped linearly to the ORIGINAL embedding's TRAIN
    mean and spread (statistics of the train documents only), so the L2 penalty of the logistic regression treats a
    variant like the signal it replaces."""
    o_mu, o_sd = float(np.mean(tr["embedding"])), float(np.std(tr["embedding"]))
    cols_tr, cols_ev = [], []
    for f in features:
        a, b = np.asarray(tr[f], dtype=float), np.asarray(ev[f], dtype=float)
        if f == "published":
            a, b = squash(a), squash(b)
        elif f not in BASE_FEATURES:
            mu, sd = float(a.mean()), max(float(a.std()), 1e-12)
            a, b = (a - mu) / sd * o_sd + o_mu, (b - mu) / sd * o_sd + o_mu
        cols_tr.append(a)
        cols_ev.append(b)
    return np.column_stack(cols_tr), np.column_stack(cols_ev)


def fit_and_score(features, tr, ev, y, groups):
    """(out-of-fold train scores, eval scores) of the logistic regression on these features."""
    from eval.fpr_eval import lr_fit, oof_lr
    Xtr, Xev = build_matrix(features, tr, ev)
    return oof_lr(Xtr, y, groups), lr_fit(Xtr, y).predict_proba(Xev)[:, 1]


def select_feature_set(auc):
    """auc: {set: [out-of-fold TRAIN AUC per seed]} for the sets that were run. Only TRAIN numbers can enter here."""
    mean = {k: float(np.mean(v)) for k, v in auc.items()}
    best_auc = max(mean.values())
    best = min((k for k in mean if mean[k] == best_auc), key=PREFERENCE.index)
    tied = [k for k in PREFERENCE if k in mean and best_auc - mean[k] < TIE]
    return {"selected": tied[0], "best_by_auc": best, "tied_with_best": tied, "mean_train_auc": mean,
            "train_auc_per_seed": {k: list(map(float, v)) for k, v in auc.items()}, "rule": RULE}


def conclusion(sel):
    m, ch, best = sel["mean_train_auc"], sel["selected"], sel["best_by_auc"]
    if ch == best:
        text = (f"{ch} ({DESCRIPTION[ch]}) has the highest mean out-of-fold TRAIN AUC, {m[ch]:.3f}, and is selected.")
    else:
        text = (f"{best} ({DESCRIPTION[best]}) has the highest mean out-of-fold TRAIN AUC, {m[best]:.3f}, but "
                f"{ch} ({DESCRIPTION[ch]}) is less than {TIE} below it ({m[ch]:.3f}) and is simpler or cheaper, so {ch} is selected.")
    if ch == "F1":
        text += " Nothing beat F1, so F1 stays."
    elif "F1" in m:
        text += f" F1 has {m['F1']:.3f}."
    return text


def candidate_json(sel, v45, v44, status44):
    feats = feature_list(sel["selected"], v45, v44)
    variants = [f for f in feats if f not in BASE_FEATURES]
    missing = ["4.4"] if status44 == "missing" else []
    return {"features": feats, "embedding_variant": " + ".join(variants) if variants else ORIGINAL,
            "uses_published": "published" in feats, "feature_set": sel["selected"], "embedding_variants": variants,
            "complete": not missing, "missing_inputs": missing, "seeds": list(SEEDS), "rule": RULE,
            "mean_train_auc": sel["mean_train_auc"],
            "note": ("PROVISIONAL: task 4.4 had not delivered its result file when this was written. Run "
                     "`python -m eval.fusion_v2` again after it is merged." if missing else conclusion(sel))}


# --------------------------------------------------------------------------
# one seed
# --------------------------------------------------------------------------

def analyse(seed, v45, v44):
    from sklearn.metrics import roc_auc_score
    from detectors import published_classifier as pc
    from eval import baseline_eval as be
    from eval.apply_models import fit_models
    from eval.fpr_eval import BUDGETS, PREVALENCES, best_case_recall, group_bootstrap, precision, threshold_at_budget
    from eval.sentence_eval import find_scores_file, load_docs
    from eval.signal_cache import load_signal

    scores_path = find_scores_file(seed)
    data = load_docs(seed, ["train", "val"], scores_path)                  # the held-out role is never requested
    train, val = data["train"], data["val"]
    fm = fit_models(seed, scores_path)
    if [d["id"] for d in val] != fm.val_ids:
        raise SystemExit("fit_models and load_docs list the validation documents in a different order.")
    ytr, yva = np.array([d["y"] for d in train]), np.array([d["y"] for d in val])
    gtr, gva = np.array([d["grp"] for d in train]), np.array([d["grp"] for d in val])
    n_pos, n_neg = int(yva.sum()), int((1 - yva).sum())

    tr = {"pattern": np.array([d["pat"] for d in train]), "embedding": np.array([d["emb"] for d in train]),
          "perplexity": np.array([d["ppl"] for d in train]), "sentence": fm.train_scores["sentence classifier"]}
    ev = {"pattern": np.array([d["pat"] for d in val]), "embedding": np.array([d["emb"] for d in val]),
          "perplexity": np.array([d["ppl"] for d in val]), "sentence": fm.val_scores["sentence classifier"]}

    cache = be.read_cache(be.cache_path(pc.DEFAULT_MODEL, pc.MAX_WINDOWS))
    lost = [d["id"] for d in train + val if d["id"] not in cache]
    if lost:
        raise SystemExit(f"{len(lost)} train/validation documents have no published-classifier score in the cache. "
                         f"Run  python -m eval.baseline_eval --seed {seed}  first (it needs torch).")
    tr["published"] = np.array([cache[d["id"]]["margin"] for d in train])
    ev["published"] = np.array([cache[d["id"]]["margin"] for d in val])

    for name in (v45, v44):
        if name is None:
            continue
        sig = load_signal(name, seed)
        lost = [d["id"] for d in train + val if d["id"] not in sig]
        if lost:
            raise SystemExit(f"The signal {name} (seed {seed}) has no score for {len(lost)} train/validation documents. "
                             f"Run the script of the task that makes it for this seed first.")
        tr[name] = np.array([sig[d["id"]][0] for d in train])
        ev[name] = np.array([sig[d["id"]][0] for d in val])

    sets = {}
    for set_name in SET_ORDER:
        feats = feature_list(set_name, v45, v44)
        if feats is None:
            continue
        oof, pva = fit_and_score(feats, tr, ev, ytr, gtr)
        rows = {}
        for alpha in BUDGETS:
            thr = threshold_at_budget(oof[ytr == 0], alpha)
            flags = pva > thr
            tpr, fpr = float(flags[yva == 1].mean()), float(flags[yva == 0].mean())
            (tl, th), (fl, fh) = group_bootstrap(flags, yva, gva)
            rows[alpha] = {"thr": thr, "tpr": tpr, "tpr_ci": (float(tl), float(th)), "fpr": fpr,
                           "fpr_ci": (float(fl), float(fh)), "best": best_case_recall(pva, yva, alpha),
                           "prec": [precision(tpr, fpr, p, n_neg) for p in PREVALENCES]}
        sets[set_name] = {"features": feats, "auc_tr": float(roc_auc_score(ytr, oof)),
                          "auc_va": float(roc_auc_score(yva, pva)), "rows": rows}
        # F0 and F1 are the "LR" rows of fpr_eval: they must reproduce the models fitted by apply_models
        ref = {"F0": "LR on the 3 current scores", "F1": "LR + sentence classifier"}.get(set_name)
        if ref:
            worst = max(float(np.abs(oof - fm.train_scores[ref]).max()), float(np.abs(pva - fm.val_scores[ref]).max()))
            if worst > 1e-9:
                raise SystemExit(f"{set_name} differs from '{ref}' of eval.apply_models by {worst:.2e}: "
                                 "fusion_v2 and fit_models have drifted apart.")
    return {"seed": seed, "n_train": len(train), "n_val": len(val), "n_pos": n_pos, "n_neg": n_neg, "sets": sets}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def render(res, results, sel, inputs, budgets):
    n_neg = res["n_neg"]
    lines = [f"# Fusion re-fit with the new signals, seed {res['seed']} (task 4.10, validation only)", "",
             "Each feature set is a logistic regression (class-balanced, C = 1) fitted on TRAIN. Its threshold at each "
             "false-alarm budget is the quantile of the out-of-fold TRAIN scores of the clean documents, applied unchanged "
             f"to validation ({res['n_val']} documents: {res['n_pos']} poisoned, {n_neg} clean). The sentence-classifier "
             "feature is out-of-fold on train, as in eval.fpr_eval. F0 and F1 reproduce the two LR rows of fpr_eval "
             "(checked to 1e-9 against eval.apply_models).", "",
             "## Inputs", ""]
    lines += [f"- {line}" for line in inputs]
    lines += ["", "## The rule (fixed before any validation number was looked at)", "",
              f"The feature set is chosen by {RULE}. Validation numbers below are shown for information; they play no part "
              "in the choice. If nothing beats F1, F1 stays.", ""]
    ran = [s for s in SET_ORDER if s in res["sets"]]
    if sel is not None:
        seeds = [r["seed"] for r in results]
        lines += ["## Out-of-fold TRAIN AUC (the numbers the choice used)", "",
                  "| feature set | features | " + " | ".join(f"seed {s}" for s in seeds) + " | mean |",
                  "|---|---|" + "---|" * (len(seeds) + 1)]
        for s in ran:
            mark = " **<- selected**" if s == sel["selected"] else ""
            lines.append(f"| {s}: {DESCRIPTION[s]}{mark} | {', '.join(res['sets'][s]['features'])} | "
                         + " | ".join(f"{r['sets'][s]['auc_tr']:.3f}" for r in results) + f" | {sel['mean_train_auc'][s]:.3f} |")
        lines += ["", conclusion(sel),
                  "The three seeds share most of their documents, so the mean over seeds is a steadier number than one seed, "
                  "not three independent measurements.", ""]
    else:
        lines += ["## Out-of-fold TRAIN AUC", "", "| feature set | features | this seed |", "|---|---|---|"]
        for s in ran:
            lines.append(f"| {s}: {DESCRIPTION[s]} | {', '.join(res['sets'][s]['features'])} | {res['sets'][s]['auc_tr']:.3f} |")
        lines += ["", "Only one seed was run, so no feature set was chosen (that needs seeds 42, 43 and 44).", ""]
    missing = [s for s in SET_ORDER if s not in res["sets"]]
    if missing:
        lines += [f"Not run: {', '.join(missing)} (see Inputs).", ""]
    for alpha in budgets:
        lines += [f"## False-alarm budget {alpha:.0%} (validation)", "",
                  "| feature set | TPR (95% CI) | FPR (95% CI) | best-case TPR at this budget | precision at 1% / 0.1% poison | "
                  "false alarms per 10,000 clean | validation AUC |", "|---|---|---|---|---|---|---|"]
        for s in ran:
            d = res["sets"][s]
            r = d["rows"][alpha]
            fa = f"{r['fpr'] * 10000:.0f}" if r["fpr"] > 0 else f"0 seen (up to ~{3.0 / n_neg * 10000:.0f})"
            lines.append(f"| {s}: {DESCRIPTION[s]} | {r['tpr']:.2f} [{r['tpr_ci'][0]:.2f}, {r['tpr_ci'][1]:.2f}] | "
                         f"{r['fpr']:.3f} [{r['fpr_ci'][0]:.3f}, {r['fpr_ci'][1]:.3f}] | {r['best']:.2f} | "
                         f"{r['prec'][0]:.1%} / {r['prec'][1]:.2%} | {fa} | {d['auc_va']:.3f} |")
        lines.append("")
    lines += ["Caveats:", "",
              f"- With {n_neg} clean validation documents one false alarm is {1 / n_neg:.1%}, so FPR is coarse. Differences of a few "
              "points between feature sets are usually inside the intervals.",
              "- 'best-case TPR' picks the threshold on validation itself (optimistic); the first TPR column is the honest one.",
              "- Precision uses the observed FPR, or the rule-of-three bound 3/n where no false alarm was seen.",
              "- F5 replaces the one embedding column by two (one per variant); every other set has the columns listed in the table.",
              "- The published margin comes from the shared cache made by eval.baseline_eval; the classifier is not fitted on our data.",
              "- Validation was not used to choose anything here. Any later change made after reading these tables makes them optimistic."]
    return "\n".join(lines) + "\n"


def describe_inputs(status45, v45, status44, v44):
    lines = []
    if status45 == "variant":
        lines.append(f"Task 4.5 selected the embedding variant `{v45}`: F2 uses it.")
    elif status45 == ORIGINAL:
        lines.append("Task 4.5 kept the original embedding (eval/results/embedding_variants_selected.json): F2 is identical to F1 and is not run.")
    else:
        lines.append("Task 4.5 selection file not found: F2 is not run.")
    if status44 == "variant":
        lines.append(f"Task 4.4 selected the embedding variant `{v44}`: F3 uses it.")
    elif status44 == ORIGINAL:
        lines.append("Task 4.4 kept the original embedding (eval/results/embedding_boilerplate_selected.json): F3 is identical to F1 and is not run.")
    else:
        lines.append("**Task 4.4 has not delivered eval/results/embedding_boilerplate_selected.json yet: F3 and F5 are not run, and the "
                     "candidate file is marked provisional.** Run this script again once 4.4 is merged.")
    if status45 == "variant" and status44 == "variant":
        lines.append("Both variants exist: F5 uses both.")
    lines.append("The published classifier's margins come from the cache of eval.baseline_eval (F4).")
    return lines


# --------------------------------------------------------------------------
# the candidate as a method of the final test (eval/final_methods.py calls this)
# --------------------------------------------------------------------------

def load_candidate(path=CANDIDATE):
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist. Run  python -m eval.fusion_v2  (task 4.10) first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _variant_scorer(ctx, name):
    """Scorer(row) for a task 4.5 variant, used only for documents that are not in the signal cache (the held-out ones)."""
    from eval import embedding_variants as ev45
    if name not in ev45.VARIANTS:
        raise NotImplementedError(f"The candidate uses the embedding variant {name!r}, which is not a task 4.5 variant. "
                                  "Add a scorer for it in eval/fusion_v2.py (_variant_scorer) before the final run: "
                                  "the held-out documents are not in its cache.")
    state = {}

    def scorer(row):
        if not state:
            from eval.protocol import POOLED_PATH, load_split, read_jsonl
            from eval.signal_cache import get_embeddings
            pool = {r.id: r for r in read_jsonl(POOLED_PATH)}
            base_ids = load_split(ctx.seed)["roles"]["baseline"]
            ids, E = get_embeddings(ctx.seed, ["baseline", "train", ctx.eval_role], final=ctx.final)
            pos = {i: k for k, i in enumerate(ids)}
            state.update(ref=ev45.Reference(E[[pos[i] for i in base_ids]], [pool[i].domain for i in base_ids]), E=E, pos=pos)
        return float(state["ref"].score(name, state["E"][state["pos"][row.id]][None, :], [row.domain])[0])
    return scorer


def candidate_scores(ctx):
    """(train out-of-fold scores, eval scores) of the 4.10 candidate for an eval.final_methods Context."""
    cand = load_candidate()
    if ctx.final and not cand.get("complete", True):
        raise SystemExit("eval/results/fusion_v2_candidate.json is provisional (missing: "
                         + ", ".join(cand.get("missing_inputs", [])) + "). Re-run  python -m eval.fusion_v2  after those tasks are merged.")
    feats = cand["features"]
    fm = ctx.fitted()
    tr = {"pattern": np.array([d["pat"] for d in ctx.train]), "embedding": np.array([d["emb"] for d in ctx.train]),
          "perplexity": np.array([d["ppl"] for d in ctx.train]), "sentence": fm.train_scores["sentence classifier"]}
    ev = {"pattern": np.array([d["pat"] for d in ctx.eval]), "embedding": np.array([d["emb"] for d in ctx.eval]),
          "perplexity": np.array([d["ppl"] for d in ctx.eval]), "sentence": ctx.base_scores()["sentence classifier"]}
    if "published" in feats:
        tr["published"], ev["published"] = ctx.published()
    for name in cand.get("embedding_variants", []):
        tr[name], ev[name] = ctx.signal(name, _variant_scorer(ctx, name))
    groups = np.array([d["grp"] for d in ctx.train])
    return fit_and_score(feats, tr, ev, ctx.y_train, groups)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="one seed only (no choice is made); default: 42, 43 and 44")
    args = ap.parse_args()
    seeds = SEEDS if args.seed is None else (args.seed,)

    status45, v45 = read_selection(SEL_45)
    status44, v44 = read_selection(SEL_44)
    inputs = describe_inputs(status45, v45, status44, v44)
    print("\n".join(inputs), flush=True)

    results = []
    for seed in seeds:
        print(f"\n[seed {seed}] fitting...", flush=True)
        results.append(analyse(seed, v45, v44))

    sel = None
    if tuple(seeds) == SEEDS:
        sel = select_feature_set({s: [r["sets"][s]["auc_tr"] for r in results] for s in results[0]["sets"]})
    from eval.fpr_eval import BUDGETS
    os.makedirs("eval/results", exist_ok=True)
    for res in results:
        text = render(res, results, sel, inputs, BUDGETS)
        path = f"eval/results/fusion_v2_seed{res['seed']}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved {path}")
    print("\n" + render(results[0], results, sel, inputs, BUDGETS))
    if sel is not None:
        cand = candidate_json(sel, v45, v44, status44)
        with open(CANDIDATE, "w", encoding="utf-8") as f:
            json.dump(cand, f, indent=1)
        print(f"Saved {CANDIDATE}")
        print("SELECTED:", sel["selected"], "|", cand["note"])


if __name__ == "__main__":
    main()