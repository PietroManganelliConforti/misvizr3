from collections import defaultdict
import json
import argparse


ALL_MISLEADING_LABELS = [
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


def post_process_pred(pred):
    if 'no misleader' in pred.lower():
        return []
    elif ',' in pred:
        return [p.strip() for p in pred.lower().replace('\n', '').split(',')]
    else:
        return [pred.lower().replace('\n', '')]


def label_distribution(data):
    true_counts  = defaultdict(int)
    pred_counts  = defaultdict(int)
    tp_counts    = defaultdict(int)
    n_true_clean = 0
    n_pred_clean = 0

    for d in data:
        true_set = set(d['true_misleader'])
        pred_set = set(post_process_pred(d['predicted_misleader']))

        if len(true_set) == 0:
            n_true_clean += 1
        if len(pred_set) == 0:
            n_pred_clean += 1

        for label in true_set:
            true_counts[label] += 1
        for label in pred_set:
            pred_counts[label] += 1
        for label in true_set & pred_set:
            tp_counts[label] += 1

    n_total = len(data)

    print(f"\n  {'Label':<38} {'True':>5}  {'Pred':>5}  {'Pre%':>6}  {'Rec%':>6}  {'F1%':>6}")
    print("  " + "-" * 72)

    for label in ALL_MISLEADING_LABELS:
        tc = true_counts[label]
        pc = pred_counts[label]
        tp = tp_counts[label]
        pre = (tp / pc * 100) if pc else 0.0
        rec = (tp / tc * 100) if tc else 0.0
        f1  = (2 * pre * rec / (pre + rec)) if (pre + rec) else 0.0
        print(
            f"  {label:<38} "
            f"{tc:>3} ({tc/n_total*100:4.1f}%)  "
            f"{pc:>3} ({pc/n_total*100:4.1f}%)  "
            f"{pre:>5.1f}  {rec:>5.1f}  {f1:>5.1f}"
        )

    print("  " + "-" * 72)
    print(
        f"  {'no misleader (non-misleading)':<38} "
        f"{n_true_clean:>3} ({n_true_clean/n_total*100:4.1f}%)  "
        f"{n_pred_clean:>3} ({n_pred_clean/n_total*100:4.1f}%)"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='misviz')
    parser.add_argument('--split',   type=str, default='test')
    parser.add_argument('--model',   type=str, required=True)
    args = parser.parse_args()

    print('-----------------------------------------')
    print(f"{args.model} - {args.dataset}")
    dataset_filename = args.dataset.replace('-', '_')
    results = json.load(open(f"results/{args.model}/{dataset_filename}_{args.split}.json", encoding="utf-8"))

    label_distribution(results)
    print('-----------------------------------------')
    print("\n--- 20 ESEMPI ---\n")
    for d in results[180:200]:
        true  = d['true_misleader']
        pred_raw = d['predicted_misleader']
        pred = post_process_pred(pred_raw)
        print(f"  TRUE : {true}")
        print(f"  PRED LABEL : {pred}")
        print(f"  PRED RAW   : {pred_raw}")
        print()