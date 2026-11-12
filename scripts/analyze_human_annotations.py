#!/usr/bin/env python3
"""
analyze_human_annotations.py
==============================
RMG Human Annotation Suite — Analysis
Mike Singleton — v0.9.5

Merges completed human annotation scores with the private key file,
then reruns the main behavioral test (geometry margin → human-rated
correctness) and compares keyword scoring to human scoring.

Supports Google Forms CSV export format:
    annotation_id, reviewer_id, score, optional_note

Also supports merged single-file format (one row per item with
multiple score columns).

Usage:
    python3 scripts/analyze_human_annotations.py

    # Explicit paths
    python3 scripts/analyze_human_annotations.py \\
        --responses  annotation/responses_R01.csv \\
                     annotation/responses_R02.csv \\
                     annotation/responses_R03.csv \\
        --key        annotation/annotation_key_private.csv \\
        --output     results/human_annotation_validation.json \\
        --threshold  median

    # Strict threshold (only score=2 counts as correct)
    python3 scripts/analyze_human_annotations.py --threshold strict

    # Lenient threshold (mean >= 1.5)
    python3 scripts/analyze_human_annotations.py --threshold lenient
"""

import os, json, csv, argparse
from pathlib import Path
from collections import defaultdict

ROOT      = Path(__file__).resolve().parents[1]
ANNOT_DIR = ROOT / "annotation"
RESULTS   = ROOT / "results"

DEFAULT_KEY    = ANNOT_DIR / "annotation_key_private.csv"
DEFAULT_OUTPUT = RESULTS   / "human_annotation_validation.json"


# ── Statistics ─────────────────────────────────────────────────────────────────
def mannwhitney(a, b):
    from scipy.stats import mannwhitneyu
    if not a or not b:
        return None, None
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def roc_auc(margins, labels):
    from scipy.stats import rankdata
    if len(set(labels)) < 2:
        return None
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranked = rankdata(margins)
    rank_sum = sum(r for r, l in zip(ranked, labels) if l == 1)
    return float((rank_sum - n_pos*(n_pos+1)/2) / (n_pos*n_neg))


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return None, None
    r, p = spearmanr(x, y)
    return float(r), float(p)


def cohens_kappa(r1, r2):
    n = len(r1)
    if n == 0: return None
    cats = sorted(set(r1)|set(r2))
    po = sum(a==b for a,b in zip(r1,r2)) / n
    pe = sum((r1.count(c)/n)*(r2.count(c)/n) for c in cats)
    return (po-pe)/(1-pe) if pe != 1 else 1.0


def krippendorff_alpha(data, level="ordinal"):
    """
    Krippendorff's alpha for ordinal data.
    data: list of lists — data[i][j] = score from rater j for item i.
    Missing values represented as None.
    """
    import numpy as np
    units   = len(data)
    raters  = max(len(row) for row in data)

    # Flatten to pairable values
    def metric(a, b):
        if level == "ordinal":
            # For ordinal: d^2 = (rank_a - rank_b)^2
            return (a - b) ** 2
        return (a - b) ** 2  # interval fallback

    # Observed disagreement D_o
    D_o_num = 0; D_o_den = 0
    for row in data:
        valid = [v for v in row if v is not None]
        n_u = len(valid)
        if n_u < 2: continue
        for i in range(len(valid)):
            for j in range(i+1, len(valid)):
                D_o_num += metric(valid[i], valid[j])
                D_o_den += 1
    D_o = D_o_num / D_o_den if D_o_den else 0

    # Expected disagreement D_e
    all_vals = [v for row in data for v in row if v is not None]
    n = len(all_vals)
    D_e_num = 0
    for i in range(n):
        for j in range(i+1, n):
            D_e_num += metric(all_vals[i], all_vals[j])
    D_e = D_e_num / (n*(n-1)/2) if n > 1 else 0

    if D_e == 0: return 1.0
    return 1 - D_o / D_e


def logistic_coef(margins, labels):
    try:
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        X = np.array(margins).reshape(-1, 1)
        y = np.array(labels)
        if len(set(y)) < 2: return None
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        return float(clf.coef_[0][0])
    except ImportError:
        return None


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_key(path):
    """Load private key CSV → dict keyed by annotation_id."""
    key = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key[row["annotation_id"]] = row
    return key


def load_responses(paths):
    """
    Load one or more response CSVs.
    Supports two formats:
      A) Google Forms style: annotation_id, reviewer_id, score, optional_note
      B) Single file with multiple score columns: annotation_id, score_R01, score_R02, ...
    Returns: dict[annotation_id] -> list of (reviewer_id, score)
    """
    responses = defaultdict(list)

    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"  ⚠  Not found: {path}")
            continue

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []

            # Detect format
            if "reviewer_id" in fields:
                # Format A: one row per reviewer per item
                for row in reader:
                    aid = row.get("annotation_id", "").strip()
                    rid = row.get("reviewer_id", path.stem).strip()
                    raw = row.get("score", row.get("human_score", "")).strip()
                    if raw in ("0","1","2"):
                        responses[aid].append((rid, int(raw)))
            elif any(f.startswith("score_") for f in fields):
                # Format B: wide format with score_R01, score_R02 columns
                score_cols = [f for f in fields if f.startswith("score_")]
                for row in reader:
                    aid = row.get("annotation_id","").strip()
                    for col in score_cols:
                        rid = col.replace("score_", "")
                        raw = row.get(col,"").strip()
                        if raw in ("0","1","2"):
                            responses[aid].append((rid, int(raw)))
            else:
                # Format C: simple — annotation_id, score (single rater)
                rater_name = path.stem
                for row in reader:
                    aid = row.get("annotation_id","").strip()
                    raw = row.get("score", row.get("human_score","")).strip()
                    if raw in ("0","1","2"):
                        responses[aid].append((rater_name, int(raw)))

        print(f"  Loaded responses from {path.name}")

    return dict(responses)


def apply_threshold(scores, mode):
    """
    Convert list of scores to binary human_correct label.
    mode: 'median' (default) — correct if median == 2
          'strict'           — correct only if all scores >= 2
          'lenient'          — correct if mean >= 1.5
    """
    if not scores: return 0
    if mode == "median":
        scores_sorted = sorted(scores)
        n = len(scores_sorted)
        median = scores_sorted[n//2] if n % 2 else (scores_sorted[n//2-1]+scores_sorted[n//2])/2
        return 1 if median >= 2 else 0
    elif mode == "strict":
        return 1 if all(s >= 2 for s in scores) else 0
    else:  # lenient
        return 1 if sum(scores)/len(scores) >= 1.5 else 0


def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Analyze RMG human annotation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--responses", nargs="+",
                        default=[str(ANNOT_DIR / "annotation_responses.csv")],
                        help="Response CSV file(s). Accepts 1+ files.")
    parser.add_argument("--key",       default=str(DEFAULT_KEY))
    parser.add_argument("--output",    default=str(DEFAULT_OUTPUT))
    parser.add_argument("--threshold", default="median",
                        choices=["median", "strict", "lenient"],
                        help="How to convert scores to binary (default: median)")
    args = parser.parse_args()

    print(f"\nRMG Human Annotation Analysis")
    print(f"{'═'*62}")

    # ── Check files exist ──────────────────────────────────────────────────────
    key_path = Path(args.key)
    if not key_path.exists():
        print(f"  ✗  Key file not found: {key_path}")
        return

    # ── Load ──────────────────────────────────────────────────────────────────
    key       = load_key(key_path)
    responses = load_responses(args.responses)

    if not responses:
        print(f"\n  ✗  No responses found in: {args.responses}")
        print(f"     Complete annotation and export CSV, then rerun.")
        return

    print(f"\n  Key items:       {len(key)}")
    print(f"  Scored items:    {len(responses)}")
    print(f"  Threshold:       {args.threshold}")

    # ── Build merged records ───────────────────────────────────────────────────
    merged = []
    all_rater_ids = set()
    for aid, score_pairs in responses.items():
        if aid not in key:
            continue
        k      = key[aid]
        scores = [s for _, s in score_pairs]
        raters = [r for r, _ in score_pairs]
        all_rater_ids.update(raters)

        margin = None
        try:
            margin = float(k["geometry_margin"]) if k.get("geometry_margin") else None
        except (ValueError, TypeError):
            pass

        kw = None
        try:
            kw = int(k["keyword_correct"]) if k.get("keyword_correct") not in (None,"","None") else None
        except (ValueError, TypeError):
            pass

        mean_score   = sum(scores)/len(scores)
        human_correct = apply_threshold(scores, args.threshold)

        merged.append({
            "annotation_id":  aid,
            "model":          k.get("model",""),
            "expected_class": k.get("expected_class",""),
            "geometry_margin": margin,
            "keyword_correct": kw,
            "behavior_label":  k.get("behavior_label",""),
            "human_scores":    scores,
            "raters":          raters,
            "mean_score":      round(mean_score, 3),
            "human_correct":   human_correct,
        })

    n_raters = len(all_rater_ids)
    print(f"  Raters:          {n_raters}  ({', '.join(sorted(all_rater_ids))})")
    print(f"  Merged items:    {len(merged)}")

    # ── Inter-rater agreement ─────────────────────────────────────────────────
    agreement_report = {}
    if n_raters >= 2:
        print(f"\n{'─'*62}")
        print(f"  INTER-RATER AGREEMENT")
        print(f"{'─'*62}")

        # Build matrix: rows=items, cols=raters (in consistent order)
        rater_order = sorted(all_rater_ids)
        # Only items scored by all raters
        full_items = [r for r in merged if len(r["human_scores"]) == n_raters]
        print(f"  Items scored by all {n_raters} raters: {len(full_items)}")

        if full_items:
            matrix = [item["human_scores"][:n_raters] for item in full_items]

            # Krippendorff's alpha (ordinal)
            k_alpha = krippendorff_alpha(matrix, level="ordinal")
            print(f"  Krippendorff's α (ordinal): {k_alpha:.3f}  ", end="")
            if   k_alpha >= 0.80: print("(strong)")
            elif k_alpha >= 0.67: print("(good)")
            elif k_alpha >= 0.40: print("(moderate — consider rubric revision)")
            else:                 print("(weak — rubric may need tightening)")

            # Pairwise Cohen's kappa (all pairs)
            if n_raters == 2:
                r1 = [item["human_scores"][0] for item in full_items]
                r2 = [item["human_scores"][1] for item in full_items]
                kappa = cohens_kappa(r1, r2)
                pct   = sum(a==b for a,b in zip(r1,r2))/len(r1)
                print(f"  Cohen's κ:                  {kappa:.3f}")
                print(f"  Exact agreement:            {pct*100:.1f}%")
                agreement_report = {
                    "n_raters": 2, "n_full_items": len(full_items),
                    "krippendorff_alpha": round(k_alpha,3),
                    "cohens_kappa": round(kappa,3),
                    "pct_exact_agreement": round(pct,3),
                }
            else:
                # Pairwise kappas for all pairs
                pairs = []
                for i in range(n_raters):
                    for j in range(i+1, n_raters):
                        r1 = [item["human_scores"][i] for item in full_items]
                        r2 = [item["human_scores"][j] for item in full_items]
                        k  = cohens_kappa(r1, r2)
                        pairs.append((rater_order[i], rater_order[j], round(k,3)))
                        print(f"  Cohen's κ ({rater_order[i]}/{rater_order[j]}):            {k:.3f}")
                agreement_report = {
                    "n_raters": n_raters, "n_full_items": len(full_items),
                    "krippendorff_alpha": round(k_alpha,3),
                    "pairwise_kappa": pairs,
                }

            # Class-level agreement
            print(f"\n  Agreement by class:")
            by_class = defaultdict(list)
            for item in full_items:
                by_class[item["expected_class"]].append(item["human_scores"])
            for cls in sorted(by_class):
                items = by_class[cls]
                pct = sum(len(set(s))==1 for s in items)/len(items)
                print(f"    {cls:<12}  {pct*100:.0f}% exact  (n={len(items)})")

    # ── Human accuracy ────────────────────────────────────────────────────────
    n_total   = len(merged)
    n_correct = sum(r["human_correct"] for r in merged)
    print(f"\n{'─'*62}")
    print(f"  HUMAN ACCURACY  (threshold={args.threshold})")
    print(f"{'─'*62}")
    print(f"  {n_correct}/{n_total} = {n_correct/n_total*100:.1f}% rated as relation-preserving")

    # ── Behavioral test: geometry margin → human correctness ─────────────────
    valid = [r for r in merged if r["geometry_margin"] is not None]
    margins        = [r["geometry_margin"] for r in valid]
    binary_labels  = [r["human_correct"]   for r in valid]
    mean_scores    = [r["mean_score"]       for r in valid]

    correct_m   = [r["geometry_margin"] for r in valid if r["human_correct"]==1]
    incorrect_m = [r["geometry_margin"] for r in valid if r["human_correct"]==0]

    auc       = roc_auc(margins, binary_labels)
    u, p_mw   = mannwhitney(correct_m, incorrect_m)
    sp_r,sp_p = spearman(margins, mean_scores)   # continuous spearman vs mean score
    log_c     = logistic_coef(margins, binary_labels)

    print(f"\n{'─'*62}")
    print(f"  GEOMETRY MARGIN → HUMAN CORRECTNESS")
    print(f"{'─'*62}")
    print(f"  N (with margin):        {len(valid)}")
    print(f"  Mean margin (correct):  {fmt(sum(correct_m)/len(correct_m))}"   if correct_m   else "")
    print(f"  Mean margin (wrong):    {fmt(sum(incorrect_m)/len(incorrect_m))}" if incorrect_m else "")
    print(f"  AUC:                    {fmt(auc)}")
    print(f"  Mann-Whitney p:         {fmt(p_mw,6)}"  if p_mw  else "  Mann-Whitney: N/A")
    print(f"  Spearman r (vs 0-2):    {fmt(sp_r)}  p={fmt(sp_p,6)}" if sp_r else "")
    print(f"  Logistic coeff:         {fmt(log_c)}" if log_c else "")

    print(f"\n  Interpretation:")
    if auc is not None:
        if   auc >= 0.70: verdict = f"✓  STRONG — geometry predicts human-rated correctness (AUC={auc:.3f})"
        elif auc >= 0.60: verdict = f"~  USEFUL — positive but below strong threshold (AUC={auc:.3f})"
        else:             verdict = f"✗  WEAK   — geometry does not predict human labels (AUC={auc:.3f})"
        print(f"  {verdict}")

    # ── Per-model breakdown ───────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  PER-MODEL RESULTS")
    print(f"{'─'*62}")
    by_model = defaultdict(list)
    for r in valid:
        by_model[r["model"]].append(r)

    per_model_results = {}
    for model in sorted(by_model):
        items = by_model[model]
        m_margins = [r["geometry_margin"] for r in items]
        m_labels  = [r["human_correct"]   for r in items]
        m_correct = [r["geometry_margin"] for r in items if r["human_correct"]==1]
        m_wrong   = [r["geometry_margin"] for r in items if r["human_correct"]==0]
        m_auc     = roc_auc(m_margins, m_labels)
        _, m_p    = mannwhitney(m_correct, m_wrong)
        acc       = sum(m_labels)/len(m_labels) if m_labels else 0
        print(f"  {model:<14}  n={len(items):>3}  acc={acc*100:.0f}%  "
              f"AUC={fmt(m_auc,3)}  MW p={fmt(m_p,4)}")
        per_model_results[model] = {
            "n": len(items), "accuracy": round(acc,3),
            "auc": round(m_auc,3) if m_auc else None,
            "mannwhitney_p": round(m_p,6) if m_p else None,
        }

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  PER-CLASS HUMAN ACCURACY")
    print(f"{'─'*62}")
    by_class = defaultdict(list)
    for r in merged:
        by_class[r["expected_class"]].append(r["human_correct"])
    per_class = {}
    for cls in sorted(by_class):
        scores = by_class[cls]
        acc = sum(scores)/len(scores) if scores else 0
        per_class[cls] = round(acc,3)
        print(f"  {cls:<12}  {acc*100:.0f}%  ({sum(scores)}/{len(scores)})")

    # ── Keyword vs human ──────────────────────────────────────────────────────
    kw_valid = [r for r in merged if r["keyword_correct"] is not None]
    kw_report = {}
    if kw_valid:
        tp = sum(1 for r in kw_valid if r["keyword_correct"]==1 and r["human_correct"]==1)
        tn = sum(1 for r in kw_valid if r["keyword_correct"]==0 and r["human_correct"]==0)
        fp = sum(1 for r in kw_valid if r["keyword_correct"]==1 and r["human_correct"]==0)
        fn = sum(1 for r in kw_valid if r["keyword_correct"]==0 and r["human_correct"]==1)
        n  = len(kw_valid)
        agree = (tp+tn)/n
        prec  = tp/(tp+fp) if (tp+fp) else 0
        rec   = tp/(tp+fn) if (tp+fn) else 0

        print(f"\n{'─'*62}")
        print(f"  KEYWORD CLASSIFIER vs HUMAN")
        print(f"{'─'*62}")
        print(f"  Agreement:  {agree*100:.1f}%")
        print(f"  Precision:  {prec:.3f}   Recall: {rec:.3f}")
        print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        if   agree >= 0.80: verdict = "✓  Keyword was a good proxy"
        elif agree >= 0.65: verdict = "~  Keyword was noisy but directionally useful"
        else:               verdict = "✗  Keyword was a poor proxy"
        print(f"  {verdict}")

        kw_report = {"n": n, "agreement": round(agree,3),
                     "precision": round(prec,3), "recall": round(rec,3),
                     "tp":tp,"tn":tn,"fp":fp,"fn":fn}

    # ── Paper sentences ───────────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  SUGGESTED PAPER SENTENCES")
    print(f"{'─'*62}")
    if auc is not None:
        if auc >= 0.70:
            print(f"\n  Best outcome:")
            print(f"  Replacing keyword scoring with blind human annotation")
            print(f"  preserved the geometry-behavior relationship: continuations")
            print(f"  rated as relation-preserving by human annotators had higher")
            print(f"  geometry margins (AUC={auc:.3f}, Mann-Whitney p={fmt(p_mw,4)}).")
        elif auc >= 0.60:
            print(f"\n  Mixed outcome:")
            print(f"  Human annotation reduced effect size but preserved the")
            print(f"  positive direction (AUC={auc:.3f}), suggesting the keyword")
            print(f"  classifier was noisy but not spurious.")
        else:
            print(f"\n  Null outcome:")
            print(f"  The geometry-behavior link did not survive human annotation")
            print(f"  (AUC={auc:.3f}); the behavioral result should be treated as")
            print(f"  a keyword-proxy artifact pending further investigation.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "threshold": args.threshold,
        "n_total": n_total, "n_correct": n_correct,
        "human_accuracy": round(n_correct/n_total,4) if n_total else None,
        "inter_rater_agreement": agreement_report,
        "behavioral_test": {
            "n_valid": len(valid),
            "mean_margin_correct":   round(sum(correct_m)/len(correct_m),4)   if correct_m   else None,
            "mean_margin_incorrect": round(sum(incorrect_m)/len(incorrect_m),4) if incorrect_m else None,
            "auc":          round(auc,4)   if auc   else None,
            "mannwhitney_p": round(p_mw,6) if p_mw  else None,
            "spearman_r":    round(sp_r,4) if sp_r  else None,
            "spearman_p":    round(sp_p,6) if sp_p  else None,
            "logistic_coef": round(log_c,4) if log_c else None,
        },
        "per_model":         per_model_results,
        "per_class":         per_class,
        "keyword_vs_human":  kw_report,
        "merged_data":       merged,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'═'*62}")
    print(f"  Results → {out_path}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
