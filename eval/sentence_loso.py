"""
Task 4.6b: leave-one-source-out check for the sentence classifier (train + validation only).

    python -m eval.sentence_loso            # seed 42
    python -m eval.sentence_loso --seed 43

For each source group (bipia, poisonedrag, pds) in turn: train on every OTHER group, then test
on the held-out group. The held-out group's attack wording, topics and templates are never
seen, which is the closest thing we have to "a new attack style". Reads the same files as
eval.sentence_eval and writes eval/results/sentence_loso_seed{seed}.md.

The test split is never loaded. Nothing here is tuned; the only choice is the 5% false-alarm
budget, and the "transferred" threshold is set on the TRAINING sources only.
"""

import argparse
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from detectors.sentence_detector import SentenceInstructionDetector
from eval.sentence_eval import ALPHA, bootstrap, find_scores_file, load_docs, recall_at_fpr


def source_group(src):
    return "pds" if src.startswith("pds") else src


def lr_fit_predict(Xtr, ytr, Xte):
    m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def oof_lr(X, y, groups):
    """Out-of-fold predictions on the training sources (used only to set the threshold)."""
    out = np.zeros(len(y))
    for a, b in GroupKFold(5).split(X, y, groups):
        out[b] = lr_fit_predict(X[a], y[a], X[b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scores", default=None, help="path to the score cache CSV (found automatically if omitted)")
    args = ap.parse_args()
    seed = args.seed
    scores_path = find_scores_file(seed, args.scores)
    print("Using score cache:", scores_path)

    data = load_docs(seed, ["train", "val"], scores_path)  # the test role is never requested
    docs = data["train"] + data["val"]
    for d in docs:
        d["g"] = source_group(d["src"])
    print(f"{len(docs)} documents from train + validation. Test split not loaded.")

    rows = []
    for held in ["bipia", "poisonedrag", "pds"]:
        tr = [d for d in docs if d["g"] != held]
        ho = [d for d in docs if d["g"] == held]
        ytr = np.array([d["y"] for d in tr])
        yho = np.array([d["y"] for d in ho])
        gtr = np.array([d["grp"] for d in tr])
        base_tr = np.c_[[d["pat"] for d in tr], [d["emb"] for d in tr], [d["ppl"] for d in tr]]
        base_ho = np.c_[[d["pat"] for d in ho], [d["emb"] for d in ho], [d["ppl"] for d in ho]]
        print(f"held out {held}: train on {len(tr)} docs from the other sources, test on {len(ho)} "
              f"({int(yho.sum())} poisoned / {int((1 - yho).sum())} clean)")

        model = SentenceInstructionDetector().fit([d["text"] for d in tr], ytr)
        sen_ho = np.array([model.score(d["text"]) for d in ho])
        sen_oof = np.zeros(len(tr))
        for a, b in GroupKFold(5).split(tr, ytr, gtr):
            m = SentenceInstructionDetector().fit([tr[i]["text"] for i in a], ytr[a])
            sen_oof[b] = [m.score(tr[i]["text"]) for i in b]

        X_tr_s, X_ho_s = np.c_[base_tr, sen_oof], np.c_[base_ho, sen_ho]
        methods = {
            "pattern bank": (base_ho[:, 0], base_tr[:, 0]),
            "LR on the 3 current scores": (lr_fit_predict(base_tr, ytr, base_ho), oof_lr(base_tr, ytr, gtr)),
            "sentence classifier": (sen_ho, sen_oof),
            "LR + sentence classifier": (lr_fit_predict(X_tr_s, ytr, X_ho_s), oof_lr(X_tr_s, ytr, gtr)),
        }
        for name, (s_ho, s_tr) in methods.items():
            auc = roc_auc_score(yho, s_ho)
            lo, hi = bootstrap(s_ho, yho, lambda s, y: roc_auc_score(y, s))
            thr = np.quantile(s_tr[ytr == 0], 1 - ALPHA)  # 5% FPR on the TRAINING sources' clean documents
            flagged = s_ho > thr
            rows.append((held, name, auc, lo, hi, recall_at_fpr(s_ho, yho),
                         float(flagged[yho == 1].mean()), float(flagged[yho == 0].mean()),
                         int(yho.sum()), int((1 - yho).sum())))

    lines = ["# Leave-one-source-out check for the sentence classifier (train + validation only)", "",
             "Each block trains on the other source groups and tests on the held-out one, so the held-out attack style, "
             "wording and topics are never seen. Nothing was tuned on the held-out data.", "",
             "- AUC: threshold-free. 0.50 means no signal.",
             f"- best-case recall: recall at FPR <= {ALPHA:.0%} using the best threshold ON the held-out source (optimistic).",
             f"- transferred TPR / FPR: threshold set so that {ALPHA:.0%} of the TRAINING sources' clean documents are flagged, "
             "then applied unchanged to the held-out source. This is what a new deployment would actually see.", ""]
    for held in ["bipia", "poisonedrag", "pds"]:
        sub = [r for r in rows if r[0] == held]
        lines += [f"## Held out: {held} ({sub[0][8]} poisoned / {sub[0][9]} clean)", "",
                  "| method | AUC (95% CI) | best-case recall | transferred TPR | transferred FPR |", "|---|---|---|---|---|"]
        for r in sub:
            lines.append(f"| {r[1]} | {r[2]:.2f} [{r[3]:.2f}, {r[4]:.2f}] | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} |")
        lines.append("")
    lines += ["Caveats:", "",
              "- Only three source groups exist, so this is a coarse test of generalisation, not a guarantee about unseen attacks.",
              "- Intervals come from resampling documents; documents from the same BIPIA email or PoisonedRAG question are related, "
              "so the true uncertainty is somewhat larger.",
              "- The test split has not been used."]
    report = "\n".join(lines)
    print("\n" + report)
    path = f"eval/results/sentence_loso_seed{seed}.md"
    os.makedirs("eval/results", exist_ok=True)
    open(path, "w", encoding="utf-8").write(report + "\n")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()