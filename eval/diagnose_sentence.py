"""
Task N1: is the sentence classifier detecting instructions, or the style of the attack generator?
(TRAIN + VALIDATION only; nothing here is tuned; deterministic)

    python -m eval.diagnose_sentence                 # seed 42
    python -m eval.diagnose_sentence --seed 43

Why: the sentence classifier is our best signal (about 0.64 recall at 5% false alarms), but most of its
in-corpus gain is on PoisonedRAG, where the pattern bank scores 0.00. PoisonedRAG poisoned passages are
machine-written and the clean ones are Wikipedia passages, so a model could separate them without ever
looking at an instruction. The controls below are simple models that see only length, style or character
patterns. If a control reaches most of the classifier's recall on PoisonedRAG, the gain is consistent with
generator style.

The classifier is the one fitted by eval.apply_models.fit_models (train only), with its train-derived 5%
threshold, so the numbers in section 2 equal the sentence-classifier rows of eval/results/fpr_eval_seed<N>.md.
The controls are fitted on TRAIN only; their 5% thresholds come from out-of-fold TRAIN scores.

Writes eval/results/diagnose_sentence_seed<N>.md. The test split is never loaded.
"""

import argparse
import os
import re
import string
import sys
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from detectors.sentence_detector import split_units
from eval.apply_models import fit_models
from eval.fpr_eval import SRC_GROUPS, group_bootstrap, source_group, threshold_at_budget
from eval.protocol import POOLED_PATH, read_jsonl
from eval.sentence_eval import find_scores_file, load_docs

BUDGET = 0.05
TOP_FEATURES = 25
TOP_SENTENCES = 15
RULE_RATIO = 0.80            # decision rule of the card: control 3 TPR >= 80% of the classifier's TPR on PoisonedRAG
SENTENCE = "sentence classifier"
HAND_NAMES = ['hand: ends with "?"', "hand: has you/your", "hand: has modal word", "hand: has output word",
              "hand: has code fence", "hand: shorter than 40 words"]
STYLE_NAMES = ["log characters", "mean words per sentence", "share of capitalised words",
               "punctuation per 100 characters", "digit share", "type-token ratio"]
_WORD = re.compile(r"\w+")
_PUNCT = set(string.punctuation)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def frac(k, n):
    return f"{k / n:.2f} ({k}/{n})" if n else "n/a (0)"


def auc_or_nan(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def fmt_auc(x):
    return "n/a" if x != x else f"{x:.2f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def clean_sentence(s, limit=200):
    """One line, at most `limit` characters, safe inside a markdown table cell."""
    s = " ".join(s.split())
    if len(s) > limit:
        s = s[:limit - 3] + "..."
    return s.replace("|", "\\|").replace("`", "'")


def ci_text(flags, y, grp):
    """95% group-bootstrap interval of the TPR for these documents (resamples whole groups)."""
    if y.sum() == 0 or (1 - y).sum() == 0:
        return "n/a"
    (lo, hi), _ = group_bootstrap(flags, y, grp)
    return f"[{lo:.2f}, {hi:.2f}]"


def style_features(text):
    """Six document-level style features that say nothing about instructions."""
    n_chars = max(len(text), 1)
    words = _WORD.findall(text)
    n_words = max(len(words), 1)
    n_sent = len(split_units(text))
    return [float(np.log(n_chars)),
            n_words / n_sent,
            sum(w[0].isupper() for w in words) / n_words,
            100.0 * sum(c in _PUNCT for c in text) / n_chars,
            sum(c.isdigit() for c in text) / n_chars,
            len({w.lower() for w in words}) / n_words]


def lr_pipeline_scaled():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))


def char_ngram_pipeline():
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=50000),
                         LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))


def fit_control(make_model, Xtr, ytr, gtr, Xva):
    """(out-of-fold TRAIN scores, validation scores). Everything, including a TF-IDF vocabulary, is fitted inside each fold."""
    final = make_model().fit(Xtr, ytr)
    oof = np.zeros(len(ytr))
    for a, b in GroupKFold(5).split(Xtr, ytr, gtr):
        oof[b] = make_model().fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
    return oof, final.predict_proba(Xva)[:, 1]


# --------------------------------------------------------------------------
# section 1: features
# --------------------------------------------------------------------------

def feature_report(model, train, groups):
    names = list(model.vec.get_feature_names_out()) + HAND_NAMES
    coef = model.clf.coef_[0]
    if len(coef) != len(names):
        raise SystemExit(f"{len(coef)} coefficients but {len(names)} feature names: the detector layout changed")
    # which training documents contain each feature in at least one of their sentences
    rows_i, cols_i = [], []
    for i, d in enumerate(train):
        cols = np.unique(model._matrix(split_units(d["text"])).tocsr().indices)
        rows_i += [i] * len(cols)
        cols_i += cols.tolist()
    present = coo_matrix((np.ones(len(rows_i)), (rows_i, cols_i)), shape=(len(train), len(names))).tocsr()
    y = np.array([d["y"] for d in train])
    g = np.array([source_group(d["src"]) for d in train])

    def line(rank, j):
        pos = int(present[y == 1][:, j].sum())
        neg = int(present[y == 0][:, j].sum())
        by_src = " / ".join(str(int(present[(y == 1) & (g == s)][:, j].sum())) for s in groups)
        return [rank, f"`{names[j]}`", f"{coef[j]:+.3f}", f"{pos} / {neg}", by_src]

    order = np.argsort(coef)
    head = ["rank", "feature", "coefficient", "train docs with it (poisoned / clean)",
            "poisoned train docs by source (" + " / ".join(groups) + ")"]
    lines = ["## 1. Features", "",
             f"The {TOP_FEATURES} most positive and {TOP_FEATURES} most negative coefficients of the logistic regression "
             "(sentence level; TF-IDF words and word pairs, then 6 hand features). A feature is counted in a document if any "
             "of its sentences contains it. Features that occur almost only in one source's poisoned documents point to "
             "source style rather than instructions.", "",
             "### Most positive (pushes a sentence towards 'poisoned')", ""]
    lines += table(head, [line(r + 1, j) for r, j in enumerate(order[::-1][:TOP_FEATURES])])
    lines += ["### Most negative (pushes a sentence towards 'clean')", ""]
    lines += table(head, [line(r + 1, j) for r, j in enumerate(order[:TOP_FEATURES])])
    return lines


# --------------------------------------------------------------------------
# section 6: top sentences
# --------------------------------------------------------------------------

def top_sentences(model, val, thr, title, mask):
    """One row per distinct best sentence among the selected validation documents, highest probability first."""
    best = []
    for d, keep in zip(val, mask):
        if not keep:
            continue
        probs, units = model.score_units(d["text"])
        i = int(np.argmax(probs))
        best.append((float(probs[i]), units[i], d))
    best.sort(key=lambda t: (-t[0], t[2]["id"]))
    seen = Counter(t[1] for t in best)
    rows, shown = [], set()
    for p, sent, d in best:
        if sent in shown:
            continue
        shown.add(sent)
        rows.append([len(rows) + 1, f"{p:.3f}", "yes" if p > thr else "no", d["src"], d["attack"], d["y"], seen[sent],
                     clean_sentence(sent)])
        if len(rows) == TOP_SENTENCES:
            break
    head = ["#", "probability", "above 5% threshold", "source", "attack type", "label", "docs with this best sentence",
            "sentence (cut at 200 characters)"]
    return [f"### {title} ({len(best)} validation documents)", ""] + table(head, rows)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def run(seed, scores_path):
    fm = fit_models(seed, scores_path)
    model = fm.sentence_model
    data = load_docs(seed, ["train", "val"], find_scores_file(seed, scores_path))     # the held-out role is never requested
    train, val = data["train"], data["val"]
    attack = {r.id: r.attack_type for r in read_jsonl(POOLED_PATH)}
    for d in train + val:
        d["attack"] = attack[d["id"]]
        d["g"] = source_group(d["src"])
    if [d["id"] for d in val] != list(fm.val_ids):
        raise SystemExit("validation documents are not in the order fit_models used")

    ytr = np.array([d["y"] for d in train])
    gtr = np.array([d["grp"] for d in train])
    y = np.array([d["y"] for d in val])
    grp = np.array([d["grp"] for d in val])
    gsrc = np.array([d["g"] for d in val])
    atk = np.array([d["attack"] for d in val])
    present = {d["g"] for d in train + val}
    groups = [s for s in SRC_GROUPS if s in present] + sorted(present - set(SRC_GROUPS))
    cells = [("all documents", np.ones(len(val), dtype=bool))] + [(s, gsrc == s) for s in groups]

    # ---- the classifier and its train-derived threshold
    sen = np.asarray(fm.val_scores[SENTENCE], dtype=float)
    thr_sen = fm.thresholds[SENTENCE][BUDGET]
    flags_sen = sen > thr_sen

    # ---- controls (train only)
    print("control 2 (style features)...", flush=True)
    Xs_tr = np.array([style_features(d["text"]) for d in train])
    Xs_va = np.array([style_features(d["text"]) for d in val])
    oof2, va2 = fit_control(lr_pipeline_scaled, Xs_tr, ytr, gtr, Xs_va)
    thr2 = threshold_at_budget(oof2[ytr == 0], BUDGET)
    print("control 3 (character n-grams; the slowest step)...", flush=True)
    T_tr = np.array([d["text"] for d in train], dtype=object)
    T_va = np.array([d["text"] for d in val], dtype=object)
    oof3, va3 = fit_control(char_ngram_pipeline, T_tr, ytr, gtr, T_va)
    thr3 = threshold_at_budget(oof3[ytr == 0], BUDGET)
    methods = {SENTENCE: (sen, flags_sen),
               "control 2: style features": (va2, va2 > thr2),
               "control 3: character n-grams": (va3, va3 > thr3)}

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    lines = [f"# Diagnosis of the sentence classifier (seed {seed}; train + validation only)", "",
             f"Train: {len(train)} documents. Validation: {len(val)} documents ({n_pos} poisoned, {n_neg} clean). "
             f"The test split was not loaded and nothing was tuned. The classifier is the one fitted on TRAIN by "
             f"`fit_models`; its threshold ({thr_sen:.4f}) is the 5% train out-of-fold clean-score quantile, as in the other reports. "
             "The controls are fitted on TRAIN only; their 5% thresholds also come from out-of-fold TRAIN scores. "
             "Cells show a rate with its counts (hits/n), so every n is visible.", ""]

    # ---- decision rule (computed here, quoted by hand in docs/sentence_diagnosis.md)
    pr = gsrc == "poisonedrag"
    decision = ["## Summary and decision rule", ""]
    if pr.any() and y[pr].sum() and (1 - y[pr]).sum():
        tpr = {m: float(f[pr & (y == 1)].mean()) for m, (_, f) in methods.items()}
        fpr = {m: float(f[pr & (y == 0)].mean()) for m, (_, f) in methods.items()}
        c3 = "control 3: character n-grams"
        ratio = tpr[c3] / tpr[SENTENCE] if tpr[SENTENCE] > 0 else float("nan")
        if ratio != ratio:
            verdict = "no verdict: the classifier flags no poisoned PoisonedRAG validation document"
        elif ratio >= RULE_RATIO:
            verdict = ("CONSISTENT WITH GENERATOR STYLE: the PoisonedRAG gain is consistent with learning generator style "
                       "and is not evidence of instruction detection")
        else:
            verdict = "CONTROLS EXPLAIN ONLY PART OF IT: the controls explain only part of the PoisonedRAG gain"
        rule = (f"Decision rule (fixed in advance): if control 3 reaches at least {RULE_RATIO:.0%} of the classifier's TPR "
                f"on PoisonedRAG, the PoisonedRAG gain is consistent with generator style. "
                f"Here: control 3 TPR {tpr[c3]:.2f} against classifier TPR {tpr[SENTENCE]:.2f}, ratio "
                f"{'n/a' if ratio != ratio else f'{ratio:.2f}'} (threshold {RULE_RATIO:.2f}). Outcome: **{verdict}**.")
        decision += [rule, "",
                     f"False-alarm rates on PoisonedRAG clean validation documents: classifier {fpr[SENTENCE]:.3f}, "
                     f"control 2 {fpr['control 2: style features']:.3f}, control 3 {fpr[c3]:.3f}. "
                     "A control that overshoots 5% gets recall for free, so read TPR and FPR together.", ""]
        console = (f"seed {seed}: PoisonedRAG TPR classifier {tpr[SENTENCE]:.2f}, control 3 {tpr[c3]:.2f}, "
                   f"ratio {'n/a' if ratio != ratio else f'{ratio:.2f}'} -> {verdict.split(':')[0]}")
    else:
        decision += ["Decision rule: not computable (no PoisonedRAG documents with both labels in validation).", ""]
        console = f"seed {seed}: decision rule not computable"
    head = ["method"] + [f"{name} ({int((m & (y == 1)).sum())} pois / {int((m & (y == 0)).sum())} clean)" for name, m in cells]
    rows = []
    for name, (_, fl) in methods.items():
        rows.append([name] + [f"{_rate(fl, m & (y == 1))} / {_rate(fl, m & (y == 0))}" for _, m in cells])
    decision += ["TPR / FPR at the 5% budget (threshold from TRAIN), by source group:", ""] + table(head, rows)
    lines += decision

    # ---- 1. features
    lines += feature_report(model, train, groups)

    # ---- 2. where the recall is
    lines += ["## 2. Where the recall is", "",
              f"Validation TPR / FPR of the sentence classifier at the 5% threshold. These rows must equal the sentence-classifier "
              f"rows of `eval/results/fpr_eval_seed{seed}.md`.", ""]
    rows = []
    for name, m in cells:
        kp, np_ = int(flags_sen[m & (y == 1)].sum()), int((m & (y == 1)).sum())
        kn, nn = int(flags_sen[m & (y == 0)].sum()), int((m & (y == 0)).sum())
        rows.append([name, frac(kp, np_), ci_text(flags_sen[m], y[m], grp[m]), frac(kn, nn)])
    lines += table(["source group", "TPR (hits/poisoned)", "TPR 95% CI (groups resampled)", "FPR (false alarms/clean)"], rows)
    caught_total = int(flags_sen[y == 1].sum())
    rows = []
    for t in sorted(set(atk[y == 1])):
        m = (atk == t) & (y == 1)
        k = int(flags_sen[m].sum())
        rows.append([t, int(m.sum()), frac(k, int(m.sum())), f"{k / caught_total:.2f}" if caught_total else "n/a"])
    lines += ["By attack type (poisoned validation documents only):", ""]
    lines += table(["attack type", "n poisoned", "TPR (hits/n)", "share of all caught poisoned documents"], rows)

    # ---- 3. control 1: length
    lines += ["## 3. Control 1: length", "",
              "AUC on validation of a single number used as the score (0.50 = no signal). Above 0.50 means poisoned documents "
              "are longer; below 0.50 means they are shorter, so an AUC far from 0.50 in either direction is a length cue.", ""]
    n_chars = np.array([len(d["text"]) for d in val], dtype=float)
    n_units = np.array([len(split_units(d["text"])) for d in val], dtype=float)
    rows = []
    for name, m in cells:
        row = [name, f"{int((m & (y == 1)).sum())} / {int((m & (y == 0)).sum())}"]
        for s in (n_chars, n_units):
            a = auc_or_nan(y[m], s[m])
            row += [fmt_auc(a), "n/a" if a != a else ("poisoned longer" if a > 0.5 else "poisoned shorter" if a < 0.5 else "none")]
        rows.append(row)
    lines += table(["source group", "n poisoned / clean", "AUC characters", "direction", "AUC sentences", "direction"], rows)

    # ---- 4 and 5. controls 2 and 3
    for number, title, key, text_ in (
            (4, "Control 2: six style features", "control 2: style features",
             "Logistic regression (class_weight balanced, standardised inputs) on: " + ", ".join(STYLE_NAMES) + ". Fitted on TRAIN."),
            (5, "Control 3: bag of style (character 3 to 5-grams)", "control 3: character n-grams",
             "Document-level TF-IDF on character 3 to 5-grams (`char_wb`, at most 50000 features) and logistic regression "
             "(class_weight balanced). Fitted on TRAIN; the vocabulary is rebuilt inside every out-of-fold split.")):
        score, fl = methods[key]
        lines += [f"## {number}. {title}", "", text_ + " Threshold: 5% of the out-of-fold TRAIN clean scores.", ""]
        rows = []
        for name, m in cells:
            kp, np_ = int(fl[m & (y == 1)].sum()), int((m & (y == 1)).sum())
            kn, nn = int(fl[m & (y == 0)].sum()), int((m & (y == 0)).sum())
            rows.append([name, fmt_auc(auc_or_nan(y[m], score[m])), frac(kp, np_), ci_text(fl[m], y[m], grp[m]), frac(kn, nn)])
        lines += table(["source group", "AUC (validation)", "TPR (hits/poisoned)", "TPR 95% CI (groups resampled)",
                        "FPR (false alarms/clean)"], rows)
        lines += ["For comparison, the sentence classifier's document AUC on validation: " +
                  ", ".join(f"{name}: {fmt_auc(auc_or_nan(y[m], sen[m]))}" for name, m in cells) + ".", ""]

    # ---- 6. top sentences
    lines += ["## 6. Top sentences", "",
              "For each validation document the sentence with the highest probability is its best sentence. The tables list the 15 "
              "highest-scoring distinct best sentences (identical sentences are shown once, with the number of documents that "
              "share them). A template that repeats across documents suggests a learned phrase. Sentences are cut at 200 characters.", ""]
    lines += top_sentences(model, val, thr_sen, "Poisoned PoisonedRAG documents", (gsrc == "poisonedrag") & (y == 1))
    lines += top_sentences(model, val, thr_sen, "Clean documents (the highest ones are the false alarms)", y == 0)
    lines += top_sentences(model, val, thr_sen, "Poisoned BIPIA documents", (gsrc == "bipia") & (y == 1))

    # ---- 7. caveats
    lines += ["## 7. Caveats: what the controls can and cannot show", "",
              "- A control that matches the classifier shows that generator style is *enough* to get this recall in this pool. "
              "It does not prove the classifier uses the same cues.",
              "- A control that falls short does not prove the classifier detects instructions. The controls are simple, document-level models "
              "and may be weaker than a sentence-level model that sees the same style cues; the leave-one-source-out check "
              "(`eval/results/sentence_loso_seed<N>.md`) is the better evidence about transfer.",
              "- The controls are fitted on the same sources as the classifier, so they measure in-corpus separability, not transfer.",
              "- In PoisonedRAG the poisoned passages are machine-written and the clean ones are Wikipedia passages, so any "
              "model that tells machine text from Wikipedia text will separate them. This confound is part of the data.",
              "- The decision rule compares point estimates. With 100 poisoned PoisonedRAG validation documents a TPR has an interval "
              "of roughly +/-0.10 (see the intervals), so a ratio near 0.80 is not decisive.",
              "- Small cells (the pds group has 14 poisoned and 10 clean validation documents) are noisy; read the counts.",
              "- Validation numbers only. This is a diagnosis: no model or threshold was changed because of it.", ""]
    return lines, console


def _rate(flags, mask):
    return frac(int(flags[mask].sum()), int(mask.sum()))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")          # sentences from the pool may contain characters a console cannot show
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scores", default=None, help="path to the score cache CSV (found automatically if omitted)")
    args = ap.parse_args()
    lines, console = run(args.seed, args.scores)
    report = "\n".join(lines)
    print("\n" + report)
    path = f"eval/results/diagnose_sentence_seed{args.seed}.md"
    os.makedirs("eval/results", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report + "\n")
    print(f"\nSaved {path}\n\nDECISION RULE  {console}")


if __name__ == "__main__":
    main()