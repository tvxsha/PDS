"""
Check the claims that the paper's prose makes against the generated numbers.

    python paper/check_claims.py            # dry run (validation files); to test the script
    python paper/check_claims.py --final    # held-out files, after the three --final runs

Reads the same files as eval/make_paper_tables.py (final_<mode>_seed<N>_scores.csv, _thresholds.json, the
cascade CSVs and eval/results/latency_benchmark.json) and prints one line per claim: PASS, CHECK (the sentence
should be reworded) or INFO (a number to read). It writes nothing. Each claim names the sentence it belongs to.
Numbers that are typed into the prose (per-source figures, test-versus-validation) are re-computed here and must
round to the typed value: a CHECK means the prose and the generated tables disagree.
Run it from the repository root. It lives under paper/, so it is not part of the freeze check.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from eval import final_test                                       # noqa: E402
from eval.make_paper_tables import LATENCY, RESULTS_DIR, load_seed, mean, seed_stats   # noqa: E402

PAT, EMB, PPL, AVG = "pattern bank", "embedding", "perplexity", "naive average of the 3"
SEN, LR3, LRS, PUB = "sentence classifier", "LR on the 3 current scores", "LR + sentence classifier", "published classifier"
OWN = [PAT, EMB, PPL, SEN]
B5, B1 = 0.05, 0.01
rows = []


def say(verdict, where, claim, detail=""):
    rows.append((verdict, where, claim, detail))


def check(ok, where, claim, detail=""):
    say("PASS" if ok else "CHECK", where, claim, detail)


def typed(ok_value, got, where, claim):
    """A number typed into the prose must equal the computed value rounded to two decimals."""
    ok = abs(got - ok_value) <= 0.0051
    say("PASS" if ok else "CHECK", where, claim, f"prose says {ok_value:.2f}, tables give {got:.3f}")


def load(prefix, cfg):
    seeds = [load_seed(s, prefix, cfg["methods"]) for s in cfg["seeds"]]
    stats = [seed_stats(sd, cfg) for sd in seeds]
    return seeds, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    cfg, _ = final_test.load_config()
    cfg = dict(cfg)
    cfg["bootstrap_reps"] = 20                      # intervals are not checked here
    role = final_test.HELD_OUT if args.final else "val"
    prefix = f"final_{final_test.MODES[role]}"
    seeds, stats = load(prefix, cfg)

    def T(m, b):
        return mean([s[(m, b)]["tpr"] for s in stats])

    def F(m, b):
        return mean([s[(m, b)]["fpr"] for s in stats])

    def S(m, b, g):
        return (mean([s[(m, b)]["source"].get(g, (np.nan, np.nan))[0] for s in stats]),
                mean([s[(m, b)]["source"].get(g, (np.nan, np.nan))[1] for s in stats]))

    def P(m, b, p):
        return mean([s[(m, b)]["prec"][p] for s in stats])

    srcs = sorted({str(g) for s in stats for q in s.values() for g in q["source"]})
    n_neg = [int((1 - sd["y"]).sum()) for sd in seeds]
    print(f"mode: {'FINAL (held-out test)' if args.final else 'DRY RUN (validation, not results)'}; seeds {cfg['seeds']}; "
          f"clean documents per seed {n_neg}; sources {srcs}\n")

    # ---- results: held-out detection --------------------------------------------------------------------------
    share = T(SEN, B5) / T(LRS, B5)
    check(share >= 0.8, "results/heldout", '"the sentence classifier supplies most of the fused recall"',
          f"sentence {T(SEN, B5):.2f} of LR+sentence {T(LRS, B5):.2f} = {share:.0%} (rule: >= 80%)")
    check(F(PAT, B5) == 0, "results/heldout", '"pattern bank reaches X with no false alarms"', f"pattern FPR {F(PAT, B5):.3f}, TPR {T(PAT, B5):.2f}")
    top3 = max(T(PAT, B5), T(EMB, B5), T(PPL, B5))
    check(top3 < 0.5, "results/heldout", '"The three original signals are weak"',
          f"pattern {T(PAT, B5):.2f}, embedding {T(EMB, B5):.2f}, perplexity {T(PPL, B5):.2f} (rule: all < 0.50)")
    others = {m: T(m, B5) for m in cfg["methods"] if m != LRS}
    check(T(LRS, B5) >= max(others.values()), "results + conclusion", '"best method is LR + sentence classifier"',
          "; ".join(f"{m} {v:.2f}" for m, v in sorted(others.items(), key=lambda kv: -kv[1])[:3]) + f" vs LR+sentence {T(LRS, B5):.2f}")
    check(T(LRS, B5) >= 0.5, "abstract", '"catches most attacks at a five percent budget"', f"LR+sentence TPR {T(LRS, B5):.2f}, FPR {F(LRS, B5):.3f}")
    check(T(LRS, B1) < T(LRS, B5), "results/heldout", '"Tightening the budget to 1% lowers the fused model"', f"{T(LRS, B5):.2f} -> {T(LRS, B1):.2f}")
    say("INFO", "results/heldout", "LR on the 3 scores, naive average", f"LR3 {T(LR3, B5):.2f}, naive average {T(AVG, B5):.2f}")
    say("INFO", "results/heldout", "published classifier", f"TPR {T(PUB, B5):.2f}, FPR {F(PUB, B5):.3f}")
    check(0.03 <= F(LRS, B5) <= 0.07, "results/heldout", "realised FPR close to the 5% budget (reword 'can differ' if not)",
          f"FPR {F(LRS, B5):.3f}; per seed {[round(s[(LRS, B5)]['fpr'], 3) for s in stats]}")
    check(0.075 <= F(LR3, B5) <= 0.125, "results/heldout", '"LR on the 3 signals: realised FPR about twice the budget"',
          f"LR3 FPR {F(LR3, B5):.3f}; per seed {[round(s[(LR3, B5)]['fpr'], 3) for s in stats]}")

    # ---- results: where the detectors fail ----------------------------------------------------------------------
    prag = next((g for g in srcs if "poisoned" in g.lower()), None)
    bip = next((g for g in srcs if "bipia" in g.lower()), None)
    if prag and bip:
        w = "results/fail"
        typed(0.00, S(PAT, B5, prag)[0], w, "pattern bank recall on PoisonedRAG at 5% (prose: flags none)")
        typed(0.00, S(PPL, B5, prag)[0], w, "perplexity recall on PoisonedRAG at 5% (prose: flags none)")
        typed(0.16, S(EMB, B5, prag)[0], w, "embedding recall on PoisonedRAG at 5%")
        typed(0.15, S(EMB, B5, prag)[1], w, "embedding false-alarm rate on PoisonedRAG at 5%")
        typed(0.23, S(LR3, B5, prag)[0], w, "LR on 3 signals: recall on PoisonedRAG at 5%")
        typed(0.22, S(LR3, B5, prag)[1], w, "LR on 3 signals: false-alarm rate on PoisonedRAG at 5%")
        typed(0.40, S(SEN, B5, prag)[0], w, "sentence classifier recall on PoisonedRAG at 5%")
        typed(0.61, S(LRS, B5, prag)[0], w, "LR+sentence recall on PoisonedRAG at 5%")
        typed(0.03, S(SEN, B1, prag)[0], w, "sentence classifier recall on PoisonedRAG at 1%")
        typed(0.09, S(LRS, B1, prag)[0], w, "LR+sentence recall on PoisonedRAG at 1%")
        typed(0.75, S(LRS, B5, bip)[0], w, "LR+sentence recall on BIPIA at 5%")
        typed(0.68, S(LRS, B1, bip)[0], w, "LR+sentence recall on BIPIA at 1%")
        typed(0.07, S(LRS, B5, bip)[1], w, "LR+sentence false-alarm rate on BIPIA at 5%")
        typed(0.02, S(LRS, B5, prag)[1], w, "LR+sentence false-alarm rate on PoisonedRAG at 5%")
        typed(0.00, S(LR3, B5, bip)[1], w, "LR on 3 signals: false-alarm rate on BIPIA at 5%")
        check(S(LR3, B5, prag)[1] >= 4 * B5, w, '"more than four times the budget"', f"LR3 on PoisonedRAG FPR {S(LR3, B5, prag)[1]:.3f} vs 4 x {B5} = {4 * B5:.2f}")
        check(S(LRS, B1, prag)[0] <= 0.15 and S(LRS, B1, bip)[0] >= 0.5, "abstract + conclusion + results/why-1%",
              '"at 1% the fusion keeps most instruction-style attacks but catches almost none of the fluent passages"',
              f"BIPIA {S(LRS, B1, bip)[0]:.2f}, PoisonedRAG {S(LRS, B1, prag)[0]:.2f} (rules: >= 0.50 and <= 0.15)")
        gap = {m: S(m, B5, prag)[0] - S(m, B5, prag)[1] for m in (PAT, EMB, PPL, AVG, LR3, PUB)}
        check(max(gap.values()) <= 0.05, "abstract", '"the original signals flag fluent passages no more often than clean ones"',
              ", ".join(f"{m} {v:+.2f}" for m, v in gap.items()) + " (recall minus false-alarm rate on PoisonedRAG at 5%; rule: all <= 0.05)")
        say("INFO", "results/fail", "false-alarm rate by source, LR+sentence at 5%", ", ".join(f"{g} {S(LRS, B5, g)[1]:.3f}" for g in srcs))

    # ---- precision ----------------------------------------------------------------------------------------------
    p01, p001 = P(LRS, B5, 0.01), P(LRS, B5, 0.001)
    check(p001 < 0.5, "results/precision + abstract + conclusion", '"precision falls quickly with prevalence / an alert is often a false alarm"',
          f"LR+sentence at 5%: {p01:.3f} at 1% poisoned, {p001:.3f} at 0.1%; at 1% budget {P(LRS, B1, 0.01):.3f} / {P(LRS, B1, 0.001):.3f}")

    # ---- latency and cascade --------------------------------------------------------------------------------------
    if os.path.exists(LATENCY):
        det = json.load(open(LATENCY, encoding="utf-8"))["detectors"]
        med = {k: det[k]["median_ms"] for k in det}
        lp, le, lq, ls = med["pattern"], med["embedding"], med["perplexity"], med["sentence classifier"]
        own_sum = lp + le + lq + ls
        check(lq / own_sum > 0.5, "abstract + results/cost + conclusion", '"perplexity dominates latency / most of the detector time"',
              f"perplexity {lq:.1f} ms of {own_sum:.1f} ms ({lq / own_sum:.0%})")
        check(ls < 5 and lp < ls, "results/cost", '"sentence classifier nearly free, pattern bank cheaper still"',
              f"sentence {ls:.2f} ms, pattern {lp:.2f} ms")
        strongest = max(OWN, key=lambda m: T(m, B5))
        check(strongest == SEN, "results/cost", '"the sentence classifier, the strongest single signal"',
              f"strongest own signal at 5%: {strongest} ({T(strongest, B5):.2f}); sentence {T(SEN, B5):.2f}")
        weakest = min(OWN, key=lambda m: T(m, B5))
        check(weakest == PPL, "results/cost", '"perplexity ... and the weakest"', f"weakest own signal at 5%: {weakest} ({T(weakest, B5):.2f}); perplexity {T(PPL, B5):.2f}")
    cpaths = [os.path.join(RESULTS_DIR, f"{prefix}_seed{s}_cascade.csv") for s in cfg["seeds"]]
    if all(os.path.exists(p) for p in cpaths):
        c = pd.concat([pd.read_csv(p) for p in cpaths], ignore_index=True)
        for b in (B5, B1):
            sub = c[np.isclose(c["budget"].astype(float), b)]
            base = sub[sub["design"].str.startswith("run everything")]
            bq = sub[sub["design"].str.contains("q = 5%")]
            if len(base) and len(bq):
                save, dt = bq["saving_clean"].mean(), base["tpr"].mean() - bq["tpr"].mean()
                typed(0.58 if b == B5 else 0.55, save, "results/cost + conclusion", f"cascade q=5%: saving on clean traffic at the {b:.0%} budget")
                typed(0.68 if b == B5 else 0.36, base["tpr"].mean(), "results/cost", f"run everything: recall at the {b:.0%} budget")
                typed(0.60 if b == B5 else 0.35, bq["tpr"].mean(), "results/cost", f"cascade q=5%: recall at the {b:.0%} budget")
                check(save > 0.5, "abstract + results/cost + conclusion", f'"saves more than half of the time on clean traffic" (budget {b:.0%})',
                      f"saving {save:.0%}; recall loss {dt:.2f} (prose: 0.08 at 5%, 0.01 at 1%)")
            for dsg, grp in sub.groupby("design"):
                say("INFO", "results/cost", f"cascade, budget {b:.0%}: {dsg}",
                    f"recall {grp['tpr'].mean():.3f}, FPR {grp['fpr'].mean():.3f}, ms per clean doc {grp['ms_clean'].mean():.1f}, saving on clean {grp['saving_clean'].mean():.0%}")
            orc = sub[sub["design"].str.startswith("A:")]
            if len(orc):
                say("INFO", "system/cascade", f'OR cascade realised FPR at {b:.0%} budget', f"{orc['fpr'].mean():.3f} (the text says it can exceed the budget)")
    else:
        say("INFO", "results/cost", "cascade files missing", "no cascade claims checked")

    # ---- limitations -----------------------------------------------------------------------------------------------
    nn = n_neg[0]
    check(150 <= nn <= 200, "limitations", '"about 0.6 percentage points per false alarm; below roughly 2%"',
          f"{nn} clean documents: one alarm = {100 / nn:.2f} points; rule of three = {300 / nn:.1f}%")

    # ---- test versus validation (only with --final) -----------------------------------------------------------------
    if args.final:
        try:
            _, vstats = load("final_dryrun", cfg)
            vt = {m: mean([s[(m, B5)]["tpr"] for s in vstats]) for m in (LRS, SEN, PUB)}
            typed(0.68, T(LRS, B5), "results (opening)", "test recall, LR+sentence at 5% (prose: 0.68)")
            typed(0.68, vt[LRS], "results (opening)", "validation recall, LR+sentence at 5% (prose: 0.68)")
            typed(0.59, T(SEN, B5), "results (opening)", "test recall, sentence classifier at 5% (prose: 0.59)")
            typed(0.64, vt[SEN], "results (opening)", "validation recall, sentence classifier at 5% (prose: 0.64)")
            typed(0.07, T(PUB, B5), "results (opening)", "test recall, published classifier at 5% (prose: 0.07)")
            typed(0.11, vt[PUB], "results (opening)", "validation recall, published classifier at 5% (prose: 0.11)")
        except SystemExit as e:
            say("INFO", "results", "dry-run files not found", str(e))

    w = max(len(r[1]) for r in rows)
    for v, where, claim, detail in rows:
        print(f"{v:5s} | {where:<{w}s} | {claim}\n      {detail}")
    n_check = sum(1 for r in rows if r[0] == "CHECK")
    print(f"\n{sum(1 for r in rows if r[0] == 'PASS')} PASS, {n_check} CHECK, {sum(1 for r in rows if r[0] == 'INFO')} INFO")


if __name__ == "__main__":
    main()