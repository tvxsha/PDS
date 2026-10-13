"""
Domain-aware embedding variants (roadmap task 4.5). TRAIN and VALIDATION only.

    python -m eval.embedding_variants                  # seeds 42, 43, 44: signals, three reports, selection file
    python -m eval.embedding_variants --seeds 42       # one seed: report only (the selection needs all three seeds)

Why
  The original embedding score is (1 - max cosine to any baseline document) / 2. With a baseline that mixes
  emails, tables, code and encyclopedia passages, it mostly says "the baseline covers this topic". The variants
  test whether averaging over more neighbours, or comparing only inside the document's own domain, helps.
  For every variant a higher score means more anomalous.

Variants (all computed from the same MiniLM embeddings; none of them uses a label)
  emb_top3, emb_top5   (1 - mean of the 3 or 5 highest cosines to the baseline) / 2
  emb_dom_max          (1 - max cosine) / 2, with the baseline restricted to documents of the same domain
  emb_dom_z            minus the z-score of the document's max cosine (same domain restriction) against the
                       leave-one-out nearest-neighbour cosines of that domain's baseline documents
  A domain with fewer than 5 baseline documents falls back to the whole baseline (for emb_dom_z also to the
  whole baseline's leave-one-out distribution).

Files
  eval/results/signal_<variant>_seed<N>_<splithash>.csv   cached scores of train + validation (id,score,ms)
  eval/results/embedding_variants_seed<N>.md              the report
  eval/results/embedding_variants_selected.json           the selected variant (or "original"), three seeds

Notes
  * The ms column of a variant is only the extra time after the document is embedded (well under
    a millisecond). The full cost of a variant is the embedding time of the original signal plus this.
  * Reads train, validation and the baseline documents only. The held-out role is never requested.
  * The numeric core (Reference) needs only numpy, so it can also score documents of the final run.
"""

import argparse
import json
import os
import sys

import numpy as np

MIN_DOMAIN = 5                  # fewer baseline documents than this in a domain: use the whole baseline
SD_FLOOR = 1e-3                 # smallest spread used for a z-score (guards a domain whose documents are near-identical)
VARIANTS = ("emb_top3", "emb_top5", "emb_dom_max", "emb_dom_z")
ORIGINAL = "original"
SEEDS = (42, 43, 44)
BUDGET = 0.05
MARGIN = 0.03                   # the best variant must beat the original by this much mean TRAIN AUC
SELECTED_PATH = "eval/results/embedding_variants_selected.json"


# --------------------------------------------------------------------------
# numeric core (numpy only)
# --------------------------------------------------------------------------

def _anomaly(similarity):
    """The original scaling: cosine similarity -> anomaly score in [0, 1]."""
    return np.clip((1.0 - np.asarray(similarity, dtype=float)) / 2.0, 0.0, 1.0)


class Reference:
    """The baseline-role documents: normalised embeddings plus their domains.

    Reference(emb, domains).score(name, E, doms) scores the rows of E (L2-normalised embeddings) whose
    domains are `doms`. name is one of VARIANTS, or "emb_max" (the original score, recomputed here)."""

    def __init__(self, emb, domains):
        self.B = np.asarray(emb, dtype=np.float64)
        self.domains = [str(d) for d in domains]
        n = len(self.domains)
        if self.B.ndim != 2 or self.B.shape[0] != n:
            raise ValueError("emb must be a matrix with one row per baseline document")
        if n < 2:
            raise ValueError("the baseline needs at least 2 documents")
        self.all_idx = np.arange(n)
        self.index = {}                                   # domain -> baseline row numbers (domains with enough documents)
        for d in sorted(set(self.domains)):
            idx = np.array([i for i, x in enumerate(self.domains) if x == d])
            if len(idx) >= MIN_DOMAIN:
                self.index[d] = idx
        # nearest-neighbour cosine of every baseline document to the OTHER baseline documents of its pool
        S = self.B @ self.B.T
        np.fill_diagonal(S, -np.inf)
        self.stats = {}                                   # pool key -> (mean, spread); key None = whole baseline
        for key, idx in [(None, self.all_idx)] + sorted(self.index.items()):
            loo = S[np.ix_(idx, idx)].max(axis=1)
            self.stats[key] = (float(loo.mean()), max(float(loo.std(ddof=1)), SD_FLOOR))

    def pool_key(self, domain):
        """The domain itself if it has enough baseline documents, else None (the whole baseline)."""
        domain = str(domain)
        return domain if domain in self.index else None

    def fallback_domains(self, domains):
        """Domains (of the given list) that fall back to the whole baseline."""
        return sorted({str(d) for d in domains if self.pool_key(d) is None})

    def score(self, name, E, domains):
        E = np.asarray(E, dtype=np.float64).reshape(-1, self.B.shape[1])
        domains = [str(d) for d in domains]
        if len(domains) != len(E):
            raise ValueError("one domain per row of E")
        S = E @ self.B.T
        if name == "emb_max":
            return _anomaly(S.max(axis=1))
        if name in ("emb_top3", "emb_top5"):
            k = min(int(name[-1]), S.shape[1])
            return _anomaly(np.sort(S, axis=1)[:, -k:].mean(axis=1))
        if name in ("emb_dom_max", "emb_dom_z"):
            keys = [self.pool_key(d) for d in domains]
            m = np.array([S[i, self.all_idx if k is None else self.index[k]].max() for i, k in enumerate(keys)])
            if name == "emb_dom_max":
                return _anomaly(m)
            mu = np.array([self.stats[k][0] for k in keys])
            sd = np.array([self.stats[k][1] for k in keys])
            return -(m - mu) / sd                          # sign flipped: far from the domain's neighbours = high
        raise ValueError(f"unknown variant {name!r}")


# --------------------------------------------------------------------------
# one seed: signals and tables
# --------------------------------------------------------------------------

def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _within_domain_auc(y, s, domains, min_each=5):
    """Unweighted mean AUC over the domains that have at least `min_each` documents of each label, and how many that is.
    This is the score's ability to separate inside a domain, with no help from differences between domains."""
    domains = np.asarray(domains)
    vals = []
    for d in sorted(set(domains.tolist())):
        m = domains == d
        if (y[m] == 0).sum() >= min_each and (y[m] == 1).sum() >= min_each:
            vals.append(_auc(y[m], s[m]))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def _budget_row(s_tr, y_tr, s_va, y_va, g_va):
    """Threshold from the TRAIN clean scores at BUDGET, applied unchanged to validation."""
    from eval.fpr_eval import group_bootstrap, threshold_at_budget
    thr = threshold_at_budget(s_tr[y_tr == 0], BUDGET)
    flags = s_va > thr
    (tl, th), (fl, fh) = group_bootstrap(flags, y_va, g_va)
    return {"thr": thr, "tpr": float(flags[y_va == 1].mean()), "tpr_ci": (float(tl), float(th)),
            "fpr": float(flags[y_va == 0].mean()), "fpr_ci": (float(fl), float(fh))}


def store_signals(seed, ref, emb_of):
    """Score train + validation with every variant (cached by eval.signal_cache) and return {name: {id: score}}."""
    from eval.signal_cache import load_signal, score_missing
    for name in VARIANTS:
        def scorer(row, name=name):
            return float(ref.score(name, emb_of[row.id][None, :], [row.domain])[0])
        score_missing(name, seed, ["train", "val"], scorer)
    return {name: {i: v[0] for i, v in load_signal(name, seed).items()} for name in VARIANTS}


def check_cache(seed, ref, docs, E, domains, signals):
    """The cached scores must equal a fresh recomputation (guards a stale cache after the code changed)."""
    for name in VARIANTS:
        fresh = ref.score(name, E, domains)
        cached = np.array([signals[name][d["id"]] for d in docs])
        worst = float(np.abs(fresh - cached).max())
        if worst > 1e-9:
            raise SystemExit(f"The cached signal {name} (seed {seed}) differs from a fresh computation by {worst:.2e}. "
                             f"The cache is stale. Delete eval/results/signal_{name}_seed{seed}_*.csv and run again.")


def analyse(seed):
    from eval.fpr_eval import SRC_GROUPS, lr_fit, oof_lr, source_group
    from eval.protocol import POOLED_PATH, load_split, read_jsonl
    from eval.sentence_eval import find_scores_file, load_docs
    from eval.signal_cache import get_embeddings

    scores_path = find_scores_file(seed)
    print(f"[seed {seed}] score cache: {scores_path}", flush=True)
    data = load_docs(seed, ["train", "val"], scores_path)                 # the held-out role is never requested
    train, val = data["train"], data["val"]
    docs = train + val
    split = load_split(seed)
    pool = {r.id: r for r in read_jsonl(POOLED_PATH)}
    base_ids = split["roles"]["baseline"]
    dom = {d["id"]: pool[d["id"]].domain for d in docs}
    base_dom = [pool[i].domain for i in base_ids]

    ids, E = get_embeddings(seed, ["baseline", "train", "val"])
    pos = {i: k for k, i in enumerate(ids)}
    emb_of = {d["id"]: E[pos[d["id"]]] for d in docs}
    ref = Reference(E[[pos[i] for i in base_ids]], base_dom)

    signals = store_signals(seed, ref, emb_of)
    Edocs = np.array([emb_of[d["id"]] for d in docs])
    check_cache(seed, ref, docs, Edocs, [dom[d["id"]] for d in docs], signals)

    # the original score is read from the existing cache; recomputing it here is only a consistency check
    recomputed = ref.score("emb_max", Edocs, [dom[d["id"]] for d in docs])
    orig_diff = float(np.abs(recomputed - np.array([d["emb"] for d in docs])).max())

    ytr = np.array([d["y"] for d in train]); yva = np.array([d["y"] for d in val])
    gtr = np.array([d["grp"] for d in train]); gva = np.array([d["grp"] for d in val])
    srctr = np.array([source_group(d["src"]) for d in train]); srcva = np.array([source_group(d["src"]) for d in val])
    s_tr = {ORIGINAL: np.array([d["emb"] for d in train])}
    s_va = {ORIGINAL: np.array([d["emb"] for d in val])}
    for name in VARIANTS:
        s_tr[name] = np.array([signals[name][d["id"]] for d in train])
        s_va[name] = np.array([signals[name][d["id"]] for d in val])
    names = [ORIGINAL] + list(VARIANTS)
    dom_tr = np.array([dom[d["id"]] for d in train]); dom_va = np.array([dom[d["id"]] for d in val])

    single, lr, by_src = {}, {}, {}
    pat_tr = np.array([d["pat"] for d in train]); ppl_tr = np.array([d["ppl"] for d in train])
    pat_va = np.array([d["pat"] for d in val]); ppl_va = np.array([d["ppl"] for d in val])
    o_mu, o_sd = s_tr[ORIGINAL].mean(), s_tr[ORIGINAL].std()
    for name in names:
        wd_tr, k_tr = _within_domain_auc(ytr, s_tr[name], dom_tr)
        wd_va, k_va = _within_domain_auc(yva, s_va[name], dom_va)
        wd_counts = (k_tr, k_va)
        single[name] = {"auc_tr": _auc(ytr, s_tr[name]), "auc_va": _auc(yva, s_va[name]), "wd_tr": wd_tr, "wd_va": wd_va,
                        **_budget_row(s_tr[name], ytr, s_va[name], yva, gva)}
        by_src[name] = {g: (_auc(ytr[srctr == g], s_tr[name][srctr == g]), _auc(yva[srcva == g], s_va[name][srcva == g]))
                        for g in SRC_GROUPS}
        # LR on (pattern, this embedding signal, perplexity). The signal is first mapped linearly to the original's
        # TRAIN mean and spread, so the L2 penalty of the logistic regression treats every variant alike.
        if name == ORIGINAL:
            rescale = lambda v: v
        else:
            v_mu, v_sd = s_tr[name].mean(), max(s_tr[name].std(), 1e-12)
            rescale = lambda v, v_mu=v_mu, v_sd=v_sd: (v - v_mu) / v_sd * o_sd + o_mu
        Xtr = np.c_[pat_tr, rescale(s_tr[name]), ppl_tr]
        Xva = np.c_[pat_va, rescale(s_va[name]), ppl_va]
        oof = oof_lr(Xtr, ytr, gtr)
        pva = lr_fit(Xtr, ytr).predict_proba(Xva)[:, 1]
        lr[name] = {"auc_tr": _auc(ytr, oof), "auc_va": _auc(yva, pva), **_budget_row(oof, ytr, pva, yva, gva)}

    # domain x label table (train) and the baseline's domains
    doms = sorted({dom[d["id"]] for d in train} | set(base_dom))
    table = []
    for dname in doms:
        n0 = sum(1 for d in train if dom[d["id"]] == dname and d["y"] == 0)
        n1 = sum(1 for d in train if dom[d["id"]] == dname and d["y"] == 1)
        table.append({"domain": dname, "clean": n0, "poisoned": n1, "baseline": base_dom.count(dname)})
    return {"seed": seed, "n_train": len(train), "n_val": len(val), "n_clean_va": int((yva == 0).sum()),
            "n_pos_va": int(yva.sum()), "n_baseline": len(base_ids), "orig_diff": orig_diff, "names": names,
            "single": single, "lr": lr, "by_src": by_src, "table": table,
            "fallback": ref.fallback_domains(doms), "src_groups": list(SRC_GROUPS), "wd_counts": wd_counts}


# --------------------------------------------------------------------------
# selection and report
# --------------------------------------------------------------------------

def select(results):
    """Highest mean TRAIN AUC over the seeds wins, but only if it beats the original by at least MARGIN."""
    seeds = [r["seed"] for r in results]
    per_seed = {n: [r["single"][n]["auc_tr"] for r in results] for n in results[0]["names"]}
    mean = {n: float(np.mean(v)) for n, v in per_seed.items()}
    best = max(VARIANTS, key=lambda n: mean[n])                    # ties: the earlier variant in VARIANTS
    gain = mean[best] - mean[ORIGINAL]
    per_seed_gain = [a - b for a, b in zip(per_seed[best], per_seed[ORIGINAL])]
    chosen = best if gain >= MARGIN else ORIGINAL
    if chosen == ORIGINAL:
        noise = (f"No variant is clearly better than the original: the best one ({best}) gains {gain:+.3f} mean TRAIN AUC, "
                 f"below the required {MARGIN:.2f}. The original embedding stays.")
    elif min(per_seed_gain) > 0:
        noise = (f"{best} beats the original by {gain:+.3f} mean TRAIN AUC (per seed: "
                 + ", ".join(f"{g:+.3f}" for g in per_seed_gain) + "), better in every seed, so it is larger than the seed-to-seed "
                 "wobble. The seeds share most documents, so this is weaker evidence than three independent runs.")
    else:
        noise = (f"{best} beats the original by {gain:+.3f} mean TRAIN AUC but not in every seed (per seed: "
                 + ", ".join(f"{g:+.3f}" for g in per_seed_gain) + "), so the gain is within seed-to-seed noise.")
    return {"selected": chosen, "best_variant": best, "uses_variant": chosen != ORIGINAL, "seeds": seeds,
            "rule": f"highest mean TRAIN AUC over the seeds; the original stays unless the best variant beats it by {MARGIN:.2f}",
            "mean_train_auc": mean, "train_auc_per_seed": per_seed, "gain_of_best": gain, "note": noise}


def _f(x, nd=2):
    return "n/a" if x != x else f"{x:.{nd}f}"


def render(res, sel):
    n_clean, n_pos = res["n_clean_va"], res["n_pos_va"]
    L = [f"# Domain-aware embedding variants (seed {res['seed']}, train + validation only)", "",
         "The original embedding score is (1 - max cosine to any baseline document) / 2. The variants average over more neighbours "
         "(emb_top3, emb_top5) or compare only with baseline documents of the same domain (emb_dom_max, emb_dom_z). "
         "For every score, higher means more anomalous. No variant uses a label; the baseline is the split's baseline role. "
         f"Thresholds come from the TRAIN clean scores at the {BUDGET:.0%} budget and are applied unchanged to validation "
         f"({n_pos} poisoned, {n_clean} clean validation documents; one false alarm = {1 / n_clean:.1%}).", "",
         "Domain labels are metadata of the benchmark. In a deployment the domain would be the collection a document comes from "
         "(the mailbox, the wiki, the table store), which is usually known, but it would have to be recorded.", "",
         "## Setup", "",
         f"- Baseline documents: {res['n_baseline']}. Domains with fewer than {MIN_DOMAIN} baseline documents fall back to the whole "
         f"baseline: {', '.join(res['fallback']) if res['fallback'] else 'none'}.",
         f"- Consistency check: the original score recomputed from these embeddings differs from the cached original by at most "
         f"{res['orig_diff']:.1e}" + (" (batch and single encoding differ in the last digits; fine)." if res["orig_diff"] <= 1e-3
                                      else ". WARNING: larger than 1e-3, so the baseline used here is not the one that scored the cache."),
         "- The ms column of the cached variant signals is only the time after the embedding exists; add the original embedding's ms for the full cost.", "",
         "### Domain x label (train)", "",
         "| domain | clean | poisoned | baseline docs | note |", "|---|---|---|---|---|"]
    risks = []
    for t in res["table"]:
        tot = t["clean"] + t["poisoned"]
        major = max(t["clean"], t["poisoned"]) / tot if tot else 0.0
        note = ""
        if tot and major > 0.90:
            note = f"{major:.0%} {'clean' if t['clean'] >= t['poisoned'] else 'poisoned'}: shortcut risk"
            risks.append(t["domain"])
        if t["baseline"] < MIN_DOMAIN:
            note = (note + "; " if note else "") + "falls back to the whole baseline"
        L.append(f"| {t['domain']} | {t['clean']} | {t['poisoned']} | {t['baseline']} | {note} |")
    L += ["", ("Shortcut risk: " + ", ".join(risks) + (" has" if len(risks) == 1 else " have") + " more than 90% of one label in train, "
               "so a score that depends on the domain could partly encode the label. Read those domains with care." if risks else
               "No domain holds more than 90% of one label in train."), ""]

    def row(name, d):
        return (f"| {name} | {_f(d['auc_tr'])} | {_f(d['auc_va'])} | {d['tpr']:.2f} [{d['tpr_ci'][0]:.2f}, {d['tpr_ci'][1]:.2f}] | "
                f"{d['fpr']:.3f} [{d['fpr_ci'][0]:.3f}, {d['fpr_ci'][1]:.3f}] |")

    hdr = ["| signal | AUC train | AUC validation | TPR (95% CI) | FPR (95% CI) |", "|---|---|---|---|---|"]
    hdr1 = ["| signal | AUC train | AUC validation | AUC inside a domain, train / validation | TPR (95% CI) | FPR (95% CI) |",
            "|---|---|---|---|---|---|"]
    L += [f"## Each signal alone (threshold at the {BUDGET:.0%} budget)", "", *hdr1]
    for n in res["names"]:
        d = res["single"][n]
        L.append(f"| {n if n != ORIGINAL else 'original embedding'} | {_f(d['auc_tr'])} | {_f(d['auc_va'])} | "
                 f"{_f(d['wd_tr'])} / {_f(d['wd_va'])} | {d['tpr']:.2f} [{d['tpr_ci'][0]:.2f}, {d['tpr_ci'][1]:.2f}] | "
                 f"{d['fpr']:.3f} [{d['fpr_ci'][0]:.3f}, {d['fpr_ci'][1]:.3f}] |")
    k_tr, k_va = res["wd_counts"]
    L += ["", "AUC below 0.50 means poisoned documents look MORE like the baseline than clean ones do. 'Inside a domain' is the unweighted "
              f"mean AUC over the domains with at least 5 documents of each label ({k_tr} domains in train, {k_va} in validation): it "
              "shows whether a variant separates better within a domain. If the overall AUC rises but this column does not, the gain "
              "comes from putting the domains on a common scale, not from a better look at each document. "
              "A false-alarm rate far above the budget means the train threshold did not carry over, and the recall is partly free.", "",
          "## LR on (pattern, embedding signal, perplexity), the signal replacing the original embedding", "",
          "AUC train is out-of-fold. The 'original embedding' row is the 'LR on the 3 current scores' method of `fpr_eval`.", "", *hdr]
    L += [row(n if n != ORIGINAL else "original embedding", res["lr"][n]) for n in res["names"]]
    gs = res["src_groups"]
    L += ["", "## AUC by source group (train / validation)", "",
          "| signal | " + " | ".join(gs) + " |", "|---|" + "---|" * len(gs)]
    for n in res["names"]:
        L.append(f"| {n if n != ORIGINAL else 'original embedding'} | "
                 + " | ".join(f"{_f(res['by_src'][n][g][0])} / {_f(res['by_src'][n][g][1])}" for g in gs) + " |")
    L += ["", "## Selection (rule fixed in advance, uses TRAIN AUC of the signal alone, all three seeds)", ""]
    if sel is None:
        L += ["Not computed: the selection needs seeds 42, 43 and 44 in one run.", ""]
    else:
        L += ["| signal | " + " | ".join(f"train AUC seed {s}" for s in sel["seeds"]) + " | mean |",
              "|---|" + "---|" * (len(sel["seeds"]) + 1)]
        for n in [ORIGINAL] + list(VARIANTS):
            L.append(f"| {n if n != ORIGINAL else 'original embedding'} | "
                     + " | ".join(f"{v:.3f}" for v in sel["train_auc_per_seed"][n]) + f" | {sel['mean_train_auc'][n]:.3f} |")
        L += ["", f"Rule: {sel['rule']}.", "", f"**Selected: {sel['selected']}.** {sel['note']}", ""]
    L += ["Caveats:", "",
          "- Validation numbers are a preview only. The choice above was made on TRAIN AUC and not on anything shown for validation.",
          "- Four variants were compared, so the best of them is optimistic even on train; that is why the rule asks for a margin.",
          "- A negative result is a result: if the original stays, the paper says the domain-aware variants were tested and did not help."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")                 # Windows consoles cannot print every character
    except (AttributeError, ValueError):
        pass
    results = [analyse(s) for s in args.seeds]
    sel = select(results) if sorted(args.seeds) == sorted(SEEDS) else None
    os.makedirs("eval/results", exist_ok=True)
    for res in results:
        report = render(res, sel)
        path = f"eval/results/embedding_variants_seed{res['seed']}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved {path}")
    if sel is not None:
        with open(SELECTED_PATH, "w", encoding="utf-8") as f:
            json.dump(sel, f, indent=1)
        print(f"Saved {SELECTED_PATH}")
        print("\nSELECTED: " + sel["selected"] + "  |  " + sel["note"])
    else:
        print("\nOnly part of the seeds were run, so no selection file was written.")


if __name__ == "__main__":
    main()