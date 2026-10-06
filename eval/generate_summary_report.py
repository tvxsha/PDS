"""
Runs the adversarial test (with results now actually saved, unlike before)
and writes ONE consolidated markdown report pulling together every key
result from the whole evaluation -- calibration comparison, failure map,
staged pipeline, adversarial red-teaming, and the cross-domain
false-positive test. This is the file to have open during a review,
instead of re-running scripts live and hoping the numbers match what you
remember.

Naive and calibrated fusion F1 are COMPUTED from eval/results/evaluation_raw.csv
so they cannot go stale. Learned-fusion and staged-pipeline numbers come
from their own scripts (they need sklearn / separate runs) and are written
in by hand below -- update them if you re-run those scripts.

Usage: python -m eval.generate_summary_report
"""

import csv
import glob
import os
import random

from detectors import pattern_detector, embedding_detector, perplexity_detector

random.seed(42)


def rebuild_baseline_corpus():
    clean_paths = sorted(glob.glob("data/clean/*.txt"))
    clean_docs = []
    for fpath in clean_paths:
        with open(fpath, encoding="utf-8") as f:
            clean_docs.append((os.path.basename(fpath), f.read()))
    random.shuffle(clean_docs)
    split_point = int(len(clean_docs) * 0.7)
    baseline_clean = clean_docs[:split_point]
    baseline_texts = [text for _, text in baseline_clean]
    return embedding_detector.BaselineCorpus(baseline_texts)


def load_calibration_bounds():
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bounds = {}
    for key in ["pattern_score", "embedding_score", "perplexity_score"]:
        values = [float(r[key]) for r in rows]
        bounds[key] = (min(values), max(values))
    return bounds


def calibrate(raw_score, lo, hi):
    span = hi - lo if hi > lo else 1.0
    normalized = (raw_score - lo) / span
    return max(0.0, min(1.0, normalized))


def run_adversarial_test(baseline_corpus, bounds):
    adv_files = sorted(glob.glob("data/adversarial/*.txt"))
    results = []
    for fpath in adv_files:
        with open(fpath, encoding="utf-8") as f:
            text = f.read()
        pattern_result = pattern_detector.score_document(text)
        embedding_result = embedding_detector.score_document(text, baseline_corpus)
        perplexity_result = perplexity_detector.score_document(text)

        cal_pattern = calibrate(pattern_result.score, *bounds["pattern_score"])
        cal_embedding = calibrate(embedding_result.score, *bounds["embedding_score"])
        cal_perplexity = calibrate(perplexity_result.score, *bounds["perplexity_score"])
        combined = (cal_pattern + cal_embedding + cal_perplexity) / 3
        verdict = "CAUGHT" if combined >= 0.15 else "EVADED"

        results.append({
            "filename": os.path.basename(fpath),
            "content": text,
            "pattern_score": pattern_result.score,
            "embedding_score": embedding_result.score,
            "perplexity_score": perplexity_result.score,
            "calibrated_combined": combined,
            "verdict": verdict,
        })
    return results


def _f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def naive_fusion_f1(rows):
    """F1 of naive fixed fusion straight from evaluation_raw.csv (Flag/Reject counts as detection)."""
    tp = sum(1 for r in rows if r["true_label"] == "1" and r["fixed_fusion_verdict"] != "Allow")
    fp = sum(1 for r in rows if r["true_label"] == "0" and r["fixed_fusion_verdict"] != "Allow")
    fn = sum(1 for r in rows if r["true_label"] == "1" and r["fixed_fusion_verdict"] == "Allow")
    return _f1(tp, fp, fn)


def calibrated_fusion_best(rows):
    """Best F1 of min-max calibrated fixed fusion over a 0.05..0.95 threshold sweep."""
    keys = ["pattern_score", "embedding_score", "perplexity_score"]
    cols = {k: [float(r[k]) for r in rows] for k in keys}

    def norm(k, v):
        lo, hi = min(cols[k]), max(cols[k])
        return (v - lo) / ((hi - lo) or 1.0)

    scores = [sum(norm(k, float(r[k])) for k in keys) / 3 for r in rows]
    best = (-1.0, None, None, None)
    for t in [i / 20 for i in range(1, 20)]:
        tp = sum(1 for r, s in zip(rows, scores) if r["true_label"] == "1" and s >= t)
        fp = sum(1 for r, s in zip(rows, scores) if r["true_label"] == "0" and s >= t)
        fn = sum(1 for r, s in zip(rows, scores) if r["true_label"] == "1" and s < t)
        p, rc, f1 = _f1(tp, fp, fn)
        if f1 > best[0] + 1e-12:
            best = (f1, t, p, rc)
    return best


def cross_domain_section():
    """Summarizes eval/results/cross_domain.csv if eval.cross_domain_eval has been run."""
    path = "eval/results/cross_domain.csv"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    fp = sum(int(r["flagged"]) for r in rows)
    emb = sum(1 for r in rows if int(r["flagged"]) and r["top_detector"] == "embedding")
    return [
        "## Cross-domain false-positive test",
        "",
        f"**{fp}/{n} benign HR / support-policy documents were wrongly flagged ({fp / n:.0%}); "
        f"{emb} of those were driven mainly by the embedding detector.**",
        "",
        "These documents (contributed by Karishma) are all legitimate, so every flag is a false "
        "positive. The embedding baseline is built only from open-source software READMEs, so any "
        "document written in a different style or domain looks anomalous even when it is harmless. "
        "This indicates the embedding detector partly measures distance from the baseline domain, "
        "not malicious intent, so the in-domain F1 above should not be read as domain-general.",
        "",
    ]


def main():
    print("Rebuilding baseline and running adversarial test...")
    baseline_corpus = rebuild_baseline_corpus()
    bounds = load_calibration_bounds()
    adv_results = run_adversarial_test(baseline_corpus, bounds)

    # save adversarial results as CSV too, not just in the markdown report
    os.makedirs("eval/results", exist_ok=True)
    with open("eval/results/adversarial_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=adv_results[0].keys())
        writer.writeheader()
        writer.writerows(adv_results)

    evaded = [r for r in adv_results if r["verdict"] == "EVADED"]
    caught = [r for r in adv_results if r["verdict"] == "CAUGHT"]
    pattern_hits = [r["filename"] for r in adv_results if r["pattern_score"] > 0]

    # load main evaluation results for the summary
    with open("eval/results/evaluation_raw.csv", encoding="utf-8") as f:
        main_rows = list(csv.DictReader(f))

    naive_p, naive_r, naive_f1 = naive_fusion_f1(main_rows)
    cal_f1, cal_t, cal_p, cal_r = calibrated_fusion_best(main_rows)

    report_lines = []
    report_lines.append("# PDS Evaluation Summary Report")
    report_lines.append("")
    report_lines.append("Generated by `eval/generate_summary_report.py` -- consolidates all")
    report_lines.append("key results in one place for review/presentation purposes.")
    report_lines.append("")
    report_lines.append("> **Caveat:** all fusion numbers below come from the same 77-document development "
                         "set that was also used to choose thresholds, so they are optimistic. "
                         "A held-out train/val/test protocol is planned (see ROADMAP.md).")
    report_lines.append("")

    report_lines.append("## Fusion method comparison")
    report_lines.append("")
    report_lines.append("| Method | F1 | Notes |")
    report_lines.append("|---|---|---|")
    report_lines.append(f"| Naive fixed fusion | {naive_f1:.2f} | Baseline, hurt by score-scale mismatch "
                         f"(precision {naive_p:.2f}, recall {naive_r:.2f}); computed from evaluation_raw.csv |")
    report_lines.append("| Learned fusion (raw scores) | 0.88 | Partial fix via logistic regression, 5-fold CV out-of-fold |")
    report_lines.append(f"| Calibrated fixed fusion | {cal_f1:.2f} | Min-max normalization before averaging "
                         f"(best threshold {cal_t:.2f}, precision {cal_p:.2f}, recall {cal_r:.2f}); "
                         f"computed from evaluation_raw.csv |")
    report_lines.append("| Learned fusion (calibrated) | 0.98 | Best accuracy at threshold 0.65; embedding is the strongest feature |")
    report_lines.append("| Staged/tiered pipeline | 0.97 | Same F1 as calibrated fusion (precision 0.94, recall 1.00), "
                         "70.5% less latency |")
    report_lines.append("")

    report_lines.append("## Adversarial red-teaming results")
    report_lines.append("")
    report_lines.append(f"**Current: {len(caught)}/{len(adv_results)} deliberately crafted evasion attempts "
                         f"caught ({len(caught)/len(adv_results):.0%}).**")
    report_lines.append("")
    report_lines.append("**History:** the first red-teaming run caught 9/10. The one evasion, adv_010 "
                         "(combined score 0.106, threshold 0.15), is described below. We then added two "
                         "credential-exposure patterns to the pattern detector. Those patterns were written "
                         "after reading adv_010's exact wording, so the post-patch result is a fix for a gap "
                         "that red-teaming found, NOT independent evidence of 100% robustness. Fresh, unseen "
                         "adversarial documents are needed to measure that.")
    report_lines.append("")
    report_lines.append("All 10 attacks were written to avoid the ORIGINAL pattern-trigger phrases and to read as "
                         "fluent README-style text. "
                         + (f"After the patch, {len(pattern_hits)} of them now match the credential patterns "
                            f"({', '.join(pattern_hits)})." if pattern_hits else "None match the pattern detector."))
    report_lines.append("")
    report_lines.append("| File | Pattern | Embedding | Perplexity | Combined | Verdict |")
    report_lines.append("|---|---|---|---|---|---|")
    for r in adv_results:
        report_lines.append(
            f"| {r['filename']} | {r['pattern_score']:.2f} | {r['embedding_score']:.3f} | "
            f"{r['perplexity_score']:.2f} | {r['calibrated_combined']:.3f} | {r['verdict']} |"
        )
    report_lines.append("")

    adv010 = next((r for r in adv_results if r["filename"] == "adv_010.txt"), None)
    if adv010 is not None:
        report_lines.append("### The original evasion: adv_010.txt")
        report_lines.append("")
        report_lines.append(f"> {adv010['content']}")
        report_lines.append("")
        report_lines.append("**Finding: \"boilerplate anchoring\"** -- opening a document with maximally "
                             "generic, ubiquitous legitimate text (e.g. standard license language) dilutes "
                             "the whole-document embedding average enough for a malicious clause later in "
                             "the same document to slip under threshold. The embedding and perplexity "
                             "detectors alone still miss it; the fix works only because a pattern now matches "
                             "its wording.")
        report_lines.append("")

    report_lines.extend(cross_domain_section())

    report_lines.append("## Dataset summary")
    report_lines.append("")
    n_clean = sum(1 for r in main_rows if r["true_label"] == "0")
    n_poisoned = sum(1 for r in main_rows if r["true_label"] == "1")
    report_lines.append(f"- {n_clean} clean documents (held-out test split) + {n_poisoned} poisoned documents "
                         f"evaluated in the main run")
    report_lines.append("- 55 total clean docs (real READMEs from 10 open-source projects), "
                         "60 total poisoned docs (15 per category: injection, authority, corpus, perplexity)")
    report_lines.append("- 10 adversarial red-teaming documents (hand-crafted)")
    report_lines.append("- 12 cross-domain test documents (HR/support policy text, contributed by Karishma)")

    report_text = "\n".join(report_lines)
    with open("eval/results/summary_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\nSaved: eval/results/summary_report.md")
    print("Saved: eval/results/adversarial_results.csv")
    print(f"\nEvasion rate: {len(evaded)}/{len(adv_results)} evaded ({len(evaded)/len(adv_results):.0%})")


if __name__ == "__main__":
    main()