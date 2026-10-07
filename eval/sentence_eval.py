"""
Task 4.6 evaluation: does a sentence-level instruction classifier help? (VALIDATION ONLY)

    python -m eval.sentence_eval            # seed 42
    python -m eval.sentence_eval --seed 43

Reads   data/pooled/pooled.jsonl
        data/splits/split_seed{seed}.json
        the score cache from protocol_eval (CSV starting "id,pattern_score,embedding_score,...";
        found automatically under eval/ and data/, or pass --scores PATH)
Writes  eval/results/sentence_eval_seed{seed}.md

Fits only on the train split, evaluates only on the validation split. The test split is
never loaded into a document list, so nothing here can touch it. Thresholds are picked on
validation, so every number below is optimistic until the frozen test run.
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

from detectors.sentence_detector import SentenceInstructionDetector

ALPHA = 0.05  # false-positive budget for the headline metric


SCORE_HEADER = "id,pattern_score,embedding_score,perplexity_score"


def find_scores_file(seed, override=None):
    """The score cache written by protocol_eval. Found by its column header, so the exact
    file name does not matter. Use --scores PATH to point at it by hand."""
    if override:
        return override
    guess = f"eval/results/pooled_scores_seed{seed}.csv"
    if os.path.exists(guess):
        return guess
    hits = []
    for root in ("eval", "data"):
        for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
            try:
                with open(path, encoding="utf-8") as f:
                    if f.readline().strip().startswith(SCORE_HEADER):
                        hits.append(path)
            except (OSError, UnicodeDecodeError):
                pass
    if not hits:
        raise SystemExit("Could not find the score cache (a CSV starting with '" + SCORE_HEADER +
                         "'). Run: dir /s /b *.csv   and pass the right one with --scores PATH")
    pick = [h for h in hits if str(seed) in os.path.basename(h)] or hits
    pick.sort(key=os.path.getmtime, reverse=True)
    if len(pick) > 1:
        print("Several score files found, using the newest:", *pick, sep="\n  ")
    return pick[0]


def load_docs(seed, roles, scores_path):
    pool = {}
    with open("data/pooled/pooled.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pool[r["id"]] = r
    scores = {}
    with open(scores_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scores[r["id"]] = r
    out = {}
    for role in roles:  # only the roles we are asked for; "test" is never requested
        docs = []
        for i in json.load(open(f"data/splits/split_seed{seed}.json"))["roles"][role]:
            if i in scores and i in pool:
                p, s = pool[i], scores[i]
                docs.append(dict(id=i, text=p["text"], y=int(p["label"]), src=p["source"], grp=p.get("group", i),
                                 pat=float(s["pattern_score"]), emb=float(s["embedding_score"]),
                                 ppl=float(s["perplexity_score"])))
        out[role] = docs
    return out

def recall_at_fpr(score, y, alpha=ALPHA):
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, score)
    return float(tpr[fpr <= alpha].max())


def bootstrap(score, y, fn, reps=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            vals.append(fn(score[idx], y[idx]))
    return np.percentile(vals, [2.5, 97.5])


def neighbour_density(vec, docs, k=3):
    """Unlabelled, corpus-level signal: mean cosine to the k nearest OTHER documents of the
    same source. Exploratory: it only makes sense if the evaluated corpus has realistic
    topical neighbours among the clean documents (see the caveat in the report)."""
    out = np.zeros(len(docs))
    for src in {d["src"] for d in docs}:
        idx = [i for i, d in enumerate(docs) if d["src"] == src]
        X = vec.transform([docs[i]["text"] for i in idx])
        sim = (X @ X.T).toarray()
        np.fill_diagonal(sim, 0)
        out[idx] = np.sort(sim, axis=1)[:, -k:].mean(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scores", default=None, help="path to the score cache CSV (found automatically if omitted)")
    args = ap.parse_args()
    seed = args.seed
    scores_path = find_scores_file(seed, args.scores)
    print("Using score cache:", scores_path)

    data = load_docs(seed, ["train", "val"], scores_path)
    train, val = data["train"], data["val"]
    ytr = np.array([d["y"] for d in train])
    yv = np.array([d["y"] for d in val])
    gtr = np.array([d["grp"] for d in train])
    src_v = np.array([d["src"] for d in val])
    print(f"train {len(train)} docs, validation {len(val)} docs "
          f"({int(yv.sum())} poisoned / {int((1 - yv).sum())} clean). Test split not loaded.")

    # 1) sentence classifier: fit on train, score validation
    model = SentenceInstructionDetector().fit([d["text"] for d in train], ytr)
    sen_v = np.array([model.score(d["text"]) for d in val])

    # 2) out-of-fold sentence scores on train (so the fusion model never sees its own fit)
    sen_oof = np.zeros(len(train))
    for tr_idx, te_idx in GroupKFold(5).split(train, ytr, gtr):
        m = SentenceInstructionDetector().fit([train[i]["text"] for i in tr_idx], ytr[tr_idx])
        sen_oof[te_idx] = [m.score(train[i]["text"]) for i in te_idx]

    # 3) neighbour density (exploratory)
    tf = TfidfVectorizer(sublinear_tf=True).fit([d["text"] for d in train])
    nb_tr, nb_v = neighbour_density(tf, train), neighbour_density(tf, val)

    base_tr = np.c_[[d["pat"] for d in train], [d["emb"] for d in train], [d["ppl"] for d in train]]
    base_v = np.c_[[d["pat"] for d in val], [d["emb"] for d in val], [d["ppl"] for d in val]]

    def lr(Xtr, Xv):
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        return m.predict_proba(Xv)[:, 1]

    S = {
        "pattern bank": base_v[:, 0],
        "embedding": base_v[:, 1],
        "perplexity": base_v[:, 2],
        "sentence classifier (4.6)": sen_v,
        "LR on the 3 current scores": lr(base_tr, base_v),
        "LR + sentence classifier": lr(np.c_[base_tr, sen_oof], np.c_[base_v, sen_v]),
        "LR + sentence + neighbour density (exploratory)": lr(np.c_[base_tr, sen_oof, nb_tr], np.c_[base_v, sen_v, nb_v]),
    }
    groups = {"bipia": src_v == "bipia", "poisonedrag": src_v == "poisonedrag",
              "pds": np.isin(src_v, ["pds_dev", "pds_adv", "pds_xdomain"])}

    lines = ["# Sentence classifier evaluation (validation only, thresholds picked on validation: optimistic)", "",
             f"train {len(train)} docs, validation {len(val)} docs ({int(yv.sum())} poisoned, {int((1 - yv).sum())} clean). "
             f"Recall is measured at a false-positive rate of at most {ALPHA:.0%}; with only "
             f"{int((1 - yv).sum())} clean validation documents that is a handful of false alarms, so intervals are wide.", "",
             "| method | AUC | recall @ FPR<=5% (95% CI) | bipia | poisonedrag | pds |", "|---|---|---|---|---|---|"]
    for name, s in S.items():
        lo, hi = bootstrap(s, yv, recall_at_fpr)
        cells = " | ".join(f"{recall_at_fpr(s[m], yv[m]):.2f}" for m in groups.values())
        lines.append(f"| {name} | {roc_auc_score(yv, s):.2f} | {recall_at_fpr(s, yv):.2f} [{lo:.2f}, {hi:.2f}] | {cells} |")
    lines += ["", "Per-source ROC-AUC:", "", "| method | bipia | poisonedrag | pds |", "|---|---|---|---|"]
    for name, s in S.items():
        cells = " | ".join(f"{roc_auc_score(yv[m], s[m]):.2f}" for m in groups.values())
        lines.append(f"| {name} | {cells} |")

    lines += ["", "Prevalence-adjusted precision at the same operating point (task 1.5):", "",
              "| method | TPR | FPR | precision at 1% | at 0.1% | at 0.01% | false alarms per 10,000 docs |",
              "|---|---|---|---|---|---|---|"]
    for name in ["LR on the 3 current scores", "LR + sentence classifier"]:
        fpr, tpr, _ = roc_curve(yv, S[name])
        k = np.where(fpr <= ALPHA)[0].max()
        t, f = tpr[k], fpr[k]
        prec = [t * p / (t * p + f * (1 - p)) for p in (0.01, 0.001, 0.0001)]
        lines.append(f"| {name} | {t:.2f} | {f:.3f} | {prec[0]:.1%} | {prec[1]:.2%} | {prec[2]:.3%} | {f * 10000:.0f} |")

    lines += ["", "Caveats:", "",
              "- Thresholds were picked on validation, so these numbers are optimistic. The test split has not been used.",
              "- The sentence classifier learns attack wording from the train split. Its BIPIA gain is within-benchmark; "
              "task 4.6 still needs the leave-one-source-out check before any claim about other attack styles.",
              "- Neighbour density is only meaningful if clean documents also have topical neighbours. In this pool the clean "
              "PoisonedRAG passages are random NQ passages while the poisoned ones come in groups of five, so its PoisonedRAG "
              "result is probably inflated. Treat it as a hint, not a result."]
    report = "\n".join(lines)
    print("\n" + report)
    path = f"eval/results/sentence_eval_seed{seed}.md"
    os.makedirs("eval/results", exist_ok=True)
    open(path, "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()