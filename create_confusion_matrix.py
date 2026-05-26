"""
create_confusion_matrix.py

for MODEL in internvl3_finetuned internvl3-8B qwen2.5vl-7B linter_gt tinychart_encoder_only_123; do   for CHART in "line chart" "bar chart" "scatter plot" "map" "pie chart"; do     for DATASET in misviz_synth_test misviz_test; do       FILE="results/$MODEL/${DATASET}.json";       if [ -f "$FILE" ]; then         python create_confusion_matrix.py --results_file "$FILE" --chart_type "$CHART";       fi;     done;   done; done

"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
ALL_MISLEADING_LABELS = [
    "3d",
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
    "no misleader",
    "truncated axis",
    "unparsed",
]

SHORT_LABELS = [
    "3d",
    "discr. cont. var.",
    "dual axis",
    "inapp. axis range",
    "inapp. item order",
    "inapp. line chart",
    "inapp. pie chart",
    "incons. bin size",
    "incons. tick int.",
    "inverted axis",
    "misrepresentation",
    "no misleader",
    "truncated axis",
    "unparsed",
]

DATASET_PATHS = {
    "misviz_synth": "data/misviz_synth/misviz_synth.json",
    "misviz":       "data/misviz/misviz.json",
}


def post_process_pred(pred):
    if pred is None:
        return ["unparsed"]
    if isinstance(pred, list):
        pred = ", ".join(str(p) for p in pred)
    pred = str(pred).strip()
    if not pred:
        return ["unparsed"]
    if "no misleader" in pred.lower():
        return ["no misleader"]
    if "," in pred:
        return [p.strip() for p in pred.lower().replace("\n", "").split(",") if p.strip()]
    return [pred.lower().replace("\n", "").strip()]


def enrich_with_chart_type(data, results_file):
    """Se i record non hanno chart_type, lo recupera dal dataset originale via image_path."""
    if data and all(d.get("chart_type") for d in data):
        return data

    # Determina quale dataset usare dal nome del file
    fname = os.path.basename(results_file)
    if "misviz_synth" in fname:
        dataset_key = "misviz_synth"
    elif "misviz" in fname:
        dataset_key = "misviz"
    else:
        print(f"Attenzione: dataset non riconoscibile da '{fname}', chart_type non disponibile")
        return data

    dataset_path = DATASET_PATHS[dataset_key]
    if not os.path.isfile(dataset_path):
        print(f"Attenzione: dataset non trovato in {dataset_path}, chart_type non disponibile")
        return data

    dataset = json.load(open(dataset_path, encoding="utf-8"))
    lookup = {d["image_path"]: d.get("chart_type", []) for d in dataset}

    for d in data:
        ip = d.get("image_path", "")
        if ip in lookup:
            d["chart_type"] = lookup[ip]
        else:
            d["chart_type"] = []

    return data


def build_confusion_matrix(data):
    label_to_idx = {label: i for i, label in enumerate(ALL_MISLEADING_LABELS)}
    n = len(ALL_MISLEADING_LABELS)
    matrix = np.zeros((n, n), dtype=int)
    for d in data:
        true_labels = d["true_misleader"] if d["true_misleader"] else ["no misleader"]
        pred_labels = post_process_pred(d["predicted_misleader"])
        # nel loop:
        for tl in true_labels:
            tl = tl.lower().strip()
            if tl not in label_to_idx:
                continue
            ti = label_to_idx[tl]
            for pl in pred_labels:
                pl = pl.lower().strip()
                pi = label_to_idx.get(pl, label_to_idx["unparsed"])
                matrix[ti][pi] += 1
    return matrix


def plot_confusion_matrix(matrix, output_path, title):
    n = len(ALL_MISLEADING_LABELS)
    fig, ax = plt.subplots(figsize=(14, 12))

    display = np.fliplr(matrix)
    x_labels = SHORT_LABELS[::-1]

    im = ax.imshow(display, interpolation="nearest", cmap="viridis")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(SHORT_LABELS, fontsize=9)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13, pad=14)

    vmax = display.max()
    thresh = vmax / 2.0
    for i in range(n):
        for j in range(n):
            count = display[i, j]
            if count == 0:
                continue
            color = "white" if count < thresh else "black"
            ax.text(j, i, str(count), ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Salvata: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--chart_type", type=str, default=None,
                        help="Filtra per tipo di grafico, es. 'bar chart', 'pie chart', 'line chart', 'scatter plot', 'map'")
    parser.add_argument("--normalize", action="store_true",
                    help="Normalizza ogni riga (recall per classe)")
    args = parser.parse_args()

    if not os.path.isfile(args.results_file):
        raise FileNotFoundError(f"File non trovato: {args.results_file}")

    data = json.load(open(args.results_file, encoding="utf-8"))

    if args.chart_type:
        data = enrich_with_chart_type(data, args.results_file)
        filtered = [d for d in data if args.chart_type.lower() in [ct.lower() for ct in d.get("chart_type", [])]]
        if not filtered:
            print(f"Nessuna istanza trovata con chart_type='{args.chart_type}'")
            print(f"Tipi disponibili: {sorted(set(ct for d in data for ct in d.get('chart_type', [])))}")
            exit(1)
        print(f"Filtrato per '{args.chart_type}': {len(filtered)}/{len(data)} istanze")
        data = filtered

    # ---- sanity check GT ----

    gt_raw = Counter(
        tl.lower().strip()
        for d in data
        for tl in (d.get("true_misleader") or ["no misleader"])
    )
    label_set = set(ALL_MISLEADING_LABELS)
    unknown_gt = {k: v for k, v in gt_raw.items() if k not in label_set}
    if unknown_gt:
        print(f"⚠ GT labels fuori vocabolario: {unknown_gt}")
    # --------------------------
    known_gt = {k: v for k, v in gt_raw.items() if k in label_set}
    print(f"Distribuzione GT: {dict(sorted(known_gt.items(), key=lambda x: -x[1]))}")

    matrix = build_confusion_matrix(data)

    if args.normalize: #normalizza per riga (recall)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums != 0)
        # Per visualizzare meglio, moltiplichiamo per 100 e arrotondiamo a 1 decimale
        matrix = np.round(matrix * 100, 1)

    base = os.path.splitext(args.results_file)[0] 
    suffix = f"_{args.chart_type.replace(' ', '_')}" if args.chart_type else ""
    output_path = base + suffix + "_confusion_matrix.png"

    parts = args.results_file.replace("\\", "/").split("/")
    title = f"{parts[-2]} — {parts[-1].replace('.json', '')}"
    if args.chart_type:
        title += f" [{args.chart_type}]"

    plot_confusion_matrix(matrix, output_path, title)
