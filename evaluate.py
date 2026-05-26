"""
evaluate.py
Evaluation for the Misviz misleading-chart detection task.

Metrics follow Section 5.2 of:
  "Is this chart lying to me? Automating the detection of misleading visualizations"
  Tonglet et al., ACL 2026  (arXiv:2508.21675)

  ┌─────────────────────────────────────────────────────────────────────┐
  │  Binary classification (all samples)                                │
  │    Acc  – overall accuracy (misleading vs. not)                     │
  │    Pre  – precision on the *misleading* class                       │
  │    Rec  – recall    on the *misleading* class                       │
  │    F1   – macro-F1  (average of misleading and non-misleading F1)   │
  │                                                                     │
  │  Misleader identification (misleading samples only)                 │
  │    EM   – Exact Match:   predicted set == ground-truth set          │
  │    PM   – Partial Match: predicted set ⊆ ground-truth set          │
  └─────────────────────────────────────────────────────────────────────┘

  Nota importante su EM/PM: il paper li calcola solo sui campioni effettivamente misleading 
  (non su tutti), e confronta i set di misleader escludendo il label "no misleader". 
  Lo script rispetta entrambe queste regole.

Usage
-----
  python evaluate.py --results results.json
  python evaluate.py --results results.json --output_report report.txt
"""


import argparse
import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

ALL_LABELS = [
    "no misleader",
    "discretized continuous variable",
    "dual axis",
    "inappropriate axis range",
    "inappropriate item order",
    "inappropriate use of line chart",
    "inappropriate use of pie chart",
    "inconsistent binning size",
    "inconsistent tick intervals",
    "inverted axis",
    "misrepresentation",
    "truncated axis",
    "3d",
]

MISLEADING_LABELS = [l for l in ALL_LABELS if l != "no misleader"]


# ── Parsing helpers ──────────────────────────────────────────────────────────


import re

def parse_prediction(pred_str: str) -> set[str]:
    clean = pred_str.lower().strip()
    found = set()
    for label in ALL_LABELS:
        # word-boundary match per evitare substring incrociati
        pattern = r'(?<![a-z0-9])' + re.escape(label) + r'(?![a-z0-9])'
        if re.search(pattern, clean):
            found.add(label)
    # Se ha trovato sia "no misleader" che altre cose reali, scarta "no misleader"
    if found - {"no misleader"}:
        found.discard("no misleader")
    return found if found else {"no misleader"}


def get_true_set(sample: dict) -> set[str]:
    """
    Return ground-truth labels as a normalised set.
    Supports both 'true_misleader' (eval format) and 'misleader' (raw dataset format).
    """
    raw = sample.get("true_misleader") or sample.get("misleader") or []
    labels = {l.lower().strip() for l in raw}
    return labels if labels else {"no misleader"}


def is_misleading(label_set: set[str]) -> int:
    """1 if the sample is misleading (contains at least one real misleader)."""
    return int(bool(label_set - {"no misleader"}))


# ── Label distribution ───────────────────────────────────────────────────────

def label_distribution(data: list[dict]) -> dict:
    """
    For every misleader label (excluding 'no misleader') count:
      - true_count   : how many samples carry it in ground truth
      - pred_count   : how many samples carry it in predictions
      - precision    : TP / (TP + FP)  for that label
      - recall       : TP / (TP + FN)  for that label
      - f1           : harmonic mean of precision and recall
    Also counts non-misleading samples separately.
    """
    from collections import defaultdict

    true_counts  = defaultdict(int)
    pred_counts  = defaultdict(int)
    tp_counts    = defaultdict(int)

    n_true_clean = 0   # ground-truth non-misleading
    n_pred_clean = 0   # predicted  non-misleading

    for sample in data:
        true_set = get_true_set(sample) - {"no misleader"}
        pred_set = parse_prediction(
            sample.get("predicted_misleader", "")) - {"no misleader"}

        if not true_set:
            n_true_clean += 1
        if not pred_set:
            n_pred_clean += 1

        for label in true_set:
            true_counts[label] += 1
        for label in pred_set:
            pred_counts[label] += 1
        for label in true_set & pred_set:
            tp_counts[label] += 1

    stats = {}
    for label in MISLEADING_LABELS:
        tc = true_counts[label]
        pc = pred_counts[label]
        tp = tp_counts[label]
        pre = tp / pc if pc else 0.0
        rec = tp / tc if tc else 0.0
        f1  = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
        stats[label] = {
            "true":  tc,
            "pred":  pc,
            "pre":   pre * 100,
            "rec":   rec * 100,
            "f1":    f1  * 100,
        }

    stats["__clean__"] = {
        "true": n_true_clean,
        "pred": n_pred_clean,
    }
    return stats


# ── Metrics (paper §5.2) ─────────────────────────────────────────────────────

def binary_metrics(data: list[dict]) -> dict:
    """
    Acc, Pre, Rec, macro-F1 — binary classification: misleading vs. not.
    Pre/Rec are computed for the *misleading* (positive) class only.
    F1 is macro-averaged over both classes, as described in the paper.
    """
    y_true = []
    y_pred = []
    for sample in data:
        y_true.append(is_misleading(get_true_set(sample)))
        y_pred.append(is_misleading(parse_prediction(
            sample.get("predicted_misleader", ""))))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "Acc": float(accuracy_score(y_true, y_pred)) * 100,
        "Pre": float(precision_score(y_true, y_pred, pos_label=1,
                                     zero_division=0)) * 100,
        "Rec": float(recall_score(y_true, y_pred, pos_label=1,
                                  zero_division=0)) * 100,
        "F1":  float(f1_score(y_true, y_pred, average="macro",
                               zero_division=0)) * 100,
    }


def misleader_metrics(data: list[dict]) -> dict:
    """
    EM (Exact Match) and PM (Partial Match) — computed on misleading samples only.

    EM = 1  iff  predicted_set == true_set   (ignoring 'no misleader')
    PM = 1  iff  predicted_set ⊆ true_set    (ignoring 'no misleader')
    """
    em_scores = []
    pm_scores = []

    misleading_samples = [
        s for s in data
        if is_misleading(get_true_set(s))
    ]

    for sample in misleading_samples:
        true_set = get_true_set(sample) - {"no misleader"}
        pred_set = parse_prediction(
            sample.get("predicted_misleader", "")) - {"no misleader"}

        em_scores.append(1.0 if pred_set == true_set else 0.0)
        pm_scores.append(1.0 if pred_set.issubset(true_set) else 0.0)

    n = len(misleading_samples)
    return {
        "EM":              float(np.mean(em_scores)) * 100 if n else float("nan"),
        "PM":              float(np.mean(pm_scores)) * 100 if n else float("nan"),
        "n_misleading":    n,
        "n_total":         len(data),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--output_report", default=None,
                        help="Optional path to save the report as a .txt file")
    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    modes = {s.get("mode", "unknown") for s in data}
    mode_tag = ", ".join(sorted(modes))

    bin_m  = binary_metrics(data)
    mis_m  = misleader_metrics(data)
    dist   = label_distribution(data)

    lines = []
    lines.append("=" * 62)
    lines.append("  Misviz Evaluation Report  (metrics: Tonglet et al., 2026)")
    lines.append(f"  Samples : {mis_m['n_total']}  |  Mode: {mode_tag}")
    lines.append(f"  of which misleading: {mis_m['n_misleading']}")
    lines.append("=" * 62)
    lines.append("")

    # ── Binary classification ────────────────────────────────────────────────
    lines.append("  BINARY CLASSIFICATION  (misleading vs. not)")
    lines.append("  " + "-" * 48)
    lines.append(f"  {'Accuracy (Acc)':<30} {bin_m['Acc']:>7.2f} %")
    lines.append(f"  {'Precision on misleading (Pre)':<30} {bin_m['Pre']:>7.2f} %")
    lines.append(f"  {'Recall on misleading (Rec)':<30} {bin_m['Rec']:>7.2f} %")
    lines.append(f"  {'Macro-F1 (F1)':<30} {bin_m['F1']:>7.2f} %")
    lines.append("")

    # ── Misleader identification ─────────────────────────────────────────────
    lines.append("  MISLEADER IDENTIFICATION  (misleading samples only)")
    lines.append("  " + "-" * 48)
    if mis_m["n_misleading"] == 0:
        lines.append("  No misleading samples found — EM/PM not applicable.")
    else:
        lines.append(f"  {'Exact Match  (EM)':<30} {mis_m['EM']:>7.2f} %")
        lines.append(f"  {'Partial Match (PM)':<30} {mis_m['PM']:>7.2f} %")
    lines.append("")
    # ── Label distribution ───────────────────────────────────────────────────
    lines.append("")
    lines.append("  LABEL DISTRIBUTION  (misleading labels only)")
    lines.append("  " + "-" * 72)
    lines.append(
        f"  {'Label':<38} {'True':>5}  {'Pred':>5}  {'Pre%':>6}  {'Rec%':>6}  {'F1%':>6}"
    )
    lines.append("  " + "-" * 72)

    n_total = mis_m["n_total"]
    for label in MISLEADING_LABELS:
        s = dist[label]
        true_pct = s["true"] / n_total * 100 if n_total else 0.0
        pred_pct = s["pred"] / n_total * 100 if n_total else 0.0
        lines.append(
            f"  {label:<38} "
            f"{s['true']:>3} ({true_pct:4.1f}%)  "
            f"{s['pred']:>3} ({pred_pct:4.1f}%)  "
            f"{s['pre']:>5.1f}  "
            f"{s['rec']:>5.1f}  "
            f"{s['f1']:>5.1f}"
        )

    lines.append("  " + "-" * 72)
    clean = dist["__clean__"]
    clean_true_pct = clean["true"] / n_total * 100 if n_total else 0.0
    clean_pred_pct = clean["pred"] / n_total * 100 if n_total else 0.0
    lines.append(
        f"  {'no misleader (non-misleading)':<38} "
        f"{clean['true']:>3} ({clean_true_pct:4.1f}%)  "
        f"{clean['pred']:>3} ({clean_pred_pct:4.1f}%)"
    )
    lines.append("")
    lines.append("  Columns: True = ground-truth count | Pred = predicted count")
    lines.append("           Pre/Rec/F1 = per-label precision/recall/F1 (%)")
    lines.append("")

    lines.append("=" * 62)
    lines.append("")
    lines.append("  Metric definitions (§5.2):")
    lines.append("  Acc  – fraction of correctly classified samples (binary)")
    lines.append("  Pre  – precision for the misleading class")
    lines.append("  Rec  – recall    for the misleading class")
    lines.append("  F1   – macro-F1 averaged over both classes")
    lines.append("  EM   – 1 iff predicted misleader set == ground-truth set")
    lines.append("  PM   – 1 iff predicted misleader set ⊆ ground-truth set")
    lines.append("=" * 62)

    report = "\n".join(lines)
    print(report)

    if args.output_report:
        with open(args.output_report, "w") as f:
            f.write(report)
        print(f"\nReport saved to {args.output_report}")


if __name__ == "__main__":
    main()