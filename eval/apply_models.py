"""
Apply the models fitted on TRAIN, with their TRAIN-derived thresholds, to any documents (roadmap task 1.9).

    from eval.apply_models import fit_models
    fm = fit_models(42)
    fm.thresholds["LR + sentence classifier"][0.05]      # threshold from TRAIN clean scores (out-of-fold for learned methods)
    scores = fm.score(docs)                              # docs: list of dict(id, text, pat, emb, ppl) -> {method: np.ndarray}

`fit_models` repeats the fitting done inside eval.fpr_eval.fit_everything but keeps the fitted objects, so
the same models can score documents that are not in the pool (adversarial, obfuscated, deepset ...).
Method names are those of fit_everything.

    python -m eval.apply_models --seed 42            # fit and print the thresholds
    python -m eval.apply_models --seed 42 --check    # prints OK only if the validation scores equal fit_everything's

Only the train and validation roles are ever loaded here. No torch import (CI has none).
"""

import argparse
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

from detectors.sentence_detector import SentenceInstructionDetector
from eval.fpr_eval import BUDGETS, fit_everything, lr_fit, oof_lr, threshold_at_budget
from eval.sentence_eval import find_scores_file, load_docs

METHODS = ["pattern bank", "embedding", "perplexity", "naive average of the 3", "sentence classifier",
           "LR on the 3 current scores", "LR + sentence classifier"]


@dataclass
class FittedModels:
    seed: int
    sentence_model: object      # SentenceInstructionDetector fitted on all of TRAIN
    lr3: object                 # logistic regression on (pattern, embedding, perplexity)
    lr_sent: object             # logistic regression on (pattern, embedding, perplexity, sentence score)
    thresholds: dict            # {method: {0.05: t, 0.01: t}} from TRAIN clean scores (out-of-fold for learned methods)
    train_scores: dict          # {method: np.ndarray} out-of-fold scores of the TRAIN documents
    val_scores: dict            # {method: np.ndarray} for the seed's validation documents
    val_ids: list               # ids in the order of val_scores
    methods: list

    def score(self, docs):
        """docs: list of dict(id, text, pat, emb, ppl) -> {method: np.ndarray}, one value per document."""
        B = np.array([[d["pat"], d["emb"], d["ppl"]] for d in docs], dtype=float).reshape(-1, 3)
        sen = np.array([self.sentence_model.score(d["text"]) for d in docs], dtype=float)
        return {"pattern bank": B[:, 0], "embedding": B[:, 1], "perplexity": B[:, 2],
                "naive average of the 3": B.mean(1) if len(B) else np.zeros(0),
                "sentence classifier": sen,
                "LR on the 3 current scores": self.lr3.predict_proba(B)[:, 1] if len(B) else np.zeros(0),
                "LR + sentence classifier": self.lr_sent.predict_proba(np.c_[B, sen])[:, 1] if len(B) else np.zeros(0)}

    def flags(self, scores, budget):
        """{method: bool array}: score strictly above the TRAIN-derived threshold of that budget."""
        return {m: scores[m] > self.thresholds[m][budget] for m in scores}


def fit_models(seed, scores_path=None):
    scores_path = find_scores_file(seed, scores_path)
    print("Using score cache:", scores_path)
    data = load_docs(seed, ["train", "val"], scores_path)      # the held-out role is never requested
    train, val = data["train"], data["val"]
    ytr = np.array([d["y"] for d in train])
    gtr = np.array([d["grp"] for d in train])
    btr = np.c_[[d["pat"] for d in train], [d["emb"] for d in train], [d["ppl"] for d in train]]

    sentence_model = SentenceInstructionDetector().fit([d["text"] for d in train], ytr)
    sen_oof = np.zeros(len(train))
    for a, b in GroupKFold(5).split(train, ytr, gtr):
        m = SentenceInstructionDetector().fit([train[i]["text"] for i in a], ytr[a])
        sen_oof[b] = [m.score(train[i]["text"]) for i in b]
    xs_tr = np.c_[btr, sen_oof]

    lr3, lr_sent = lr_fit(btr, ytr), lr_fit(xs_tr, ytr)
    train_scores = {"pattern bank": btr[:, 0], "embedding": btr[:, 1], "perplexity": btr[:, 2],
                    "naive average of the 3": btr.mean(1), "sentence classifier": sen_oof,
                    "LR on the 3 current scores": oof_lr(btr, ytr, gtr),
                    "LR + sentence classifier": oof_lr(xs_tr, ytr, gtr)}
    thresholds = {m: {a: threshold_at_budget(train_scores[m][ytr == 0], a) for a in BUDGETS} for m in METHODS}
    fm = FittedModels(seed=seed, sentence_model=sentence_model, lr3=lr3, lr_sent=lr_sent, thresholds=thresholds,
                      train_scores=train_scores, val_scores={}, val_ids=[d["id"] for d in val], methods=list(METHODS))
    fm.val_scores = fm.score(val)
    return fm


def check(seed):
    """Compare fit_models with fit_everything (catches drift if fpr_eval.py is edited)."""
    fm = fit_models(seed)
    data = load_docs(seed, ["train", "val"], find_scores_file(seed))
    ytr = np.array([d["y"] for d in data["train"]])
    tr, va = fit_everything(data["train"], data["val"])
    worst = 0.0
    print(f"{'method':30s} {'max |val diff|':>15s} {'max |train diff|':>17s} {'max |thr diff|':>15s}")
    for m in METHODS:
        dv = float(np.abs(fm.val_scores[m] - va[m]).max())
        dt = float(np.abs(fm.train_scores[m] - tr[m]).max())
        dh = max(abs(fm.thresholds[m][a] - threshold_at_budget(tr[m][ytr == 0], a)) for a in BUDGETS)
        worst = max(worst, dv, dt, dh)
        print(f"{m:30s} {dv:15.2e} {dt:17.2e} {dh:15.2e}")
    if worst <= 1e-9:
        print("OK")
    else:
        print(f"MISMATCH (largest difference {worst:.2e}): fit_models and fit_everything have drifted apart")
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scores", default=None, help="path to the score cache CSV (found automatically if omitted)")
    ap.add_argument("--check", action="store_true", help="compare with eval.fpr_eval.fit_everything; prints OK")
    args = ap.parse_args()
    if args.check:
        check(args.seed)
        return
    fm = fit_models(args.seed, args.scores)
    print(f"seed {fm.seed}: {len(fm.val_ids)} validation documents scored")
    print(f"{'method':30s} " + " ".join(f"thr@{a:.0%}".rjust(10) for a in BUDGETS))
    for m in fm.methods:
        print(f"{m:30s} " + " ".join(f"{fm.thresholds[m][a]:10.4f}" for a in BUDGETS))


if __name__ == "__main__":
    main()