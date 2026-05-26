"""
inference_finetuned.py

Runs inference with a fine-tuned InternVL3 LoRA model on Misviz or Misviz-synth.
Output format is identical to the zero-shot inference (compatible with evaluate.py).

Usage:
    # Inference su misviz_synth test
    python inference_finetuned.py \
        --base_model_path internvl3/8B/ \
        --adapter_path output/internvl3_finetuned/best_model \
        --dataset misviz_synth \
        --split test \
        --output_dir results/internvl3_finetuned/

    # Inference su misviz test
    python inference_finetuned.py \
        --base_model_path internvl3/8B/ \
        --adapter_path output/internvl3_finetuned/best_model \
        --dataset misviz \
        --split test \
        --dataset_path data/misviz/ \
        --output_dir results/internvl3_finetuned/

    # Valutazione
    python src/evaluate.py --model internvl3_finetuned --dataset misviz_synth --split test
    python src/evaluate.py --model internvl3_finetuned --dataset misviz --split test
"""

import argparse
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

# ── Image preprocessing (same as training) ──────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    best = min(target_ratios, key=lambda r: abs(aspect_ratio - r[0] / r[1]))
    target_width = image_size * best[0]
    target_height = image_size * best[1]
    blocks = best[0] * best[1]
    resized = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % best[0]) * image_size,
            (i // best[0]) * image_size,
            ((i % best[0]) + 1) * image_size,
            ((i // best[0]) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images]).to(torch.bfloat16)
    return pixel_values


# ── Prompt (same as paper) ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in data visualization analysis. Your task is to identify misleaders present in the given visualization.

Please carefully examine the visualization and detect its misleaders. Provide all relevant misleaders, up to three, as a comma separated list. In most cases only one misleader is relevant. If you detect none of the above types of misleaders in the visualization, respond with "no misleader".

The available misleaders to select are, by alphabetical order:
- discretized continuous variable: a map displays a continuous variable transformed into a categorical variable by cutting it into discrete categories, thus exaggerating the difference between boundary cases.
- dual axis: there are two independent y-axis, one on the left and one on the right, with different scales.
- inappropriate axis range: the axis range is too broad or too narrow.
- inappropriate item order: instances of a variable along an axis are in an unconventional, non-linear or non-chronological order.
- inappropriate use of line chart: a line chart is used in inappropriate or unconventional ways, e.g., using a line chart with categorical variables, or encoding the time dimension on the y-axis.
- inappropriate use of pie chart: a pie chart does not display data in a part-to-whole relationship, e.g., its shares do not sum to 100%.
- inconsistent binning size: a variable, such as years or ages, is grouped in unevenly sized bins.
- inconsistent tick intervals: the ticks values in one axis are evenly spaced but their values are not, e.g., the tick value sequence is 10, 20, 40, 45.
- inverted axis: an axis is displayed in a direction opposite to conventions, e.g., the y-axis displays values increasing from top to bottom or the x-axis displays values increasing from right to left.
- misrepresentation: the value labels displayed do not match the size of their visual encodings, e.g., bars may be drawn disproportionate to the corresponding numerical value.
- truncated axis: an axis does not start from zero, resulting in a visual exaggeration of changes in the dependent variable with respect to the independent variable.
- 3d: the visualization includes three-dimensional effects.

Provide only the final answer, without additional explanation."""


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Add model path to sys.path for trust_remote_code
    sys.path.insert(0, args.base_model_path)

    # Load base model with 4-bit quantization
    print(f"Loading base model from {args.base_model_path}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, trust_remote_code=True
    )
    base_model = AutoModel.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    # Verifica che gli adapter siano attivi
    print(f"Active adapter: {model.active_adapter}")   # dovrebbe stampare 'default'

    # Per la chat, usa il base unwrapped — i moduli LoRA iniettati sono comunque
    # nella sua struttura e si attivano automaticamente nei forward.
    chat_model = model.base_model.model


    # Determine dataset metadata path
    if args.dataset == "misviz_synth":
        meta_path = os.path.join(args.dataset_path, "misviz_synth.json")
    else:
        meta_path = os.path.join(args.dataset_path, "misviz.json")

    with open(meta_path) as f:
        all_metadata = json.load(f)


    metadata = [d for d in all_metadata if d["split"] == args.split]
    print(f"Running inference on {len(metadata)} instances ({args.dataset} [{args.split}])")

    # Sanity check: verifica che la GT sia leggibile
    n_with_gt = sum(
        1 for d in metadata
        if (d.get("misleader") or d.get("true_misleader"))
    )
    n_misleading = sum(
        1 for d in metadata
        if (d.get("misleader") or d.get("true_misleader") or [])
    )
    print(f"GT check: {n_misleading}/{len(metadata)} samples have at least one misleader label")
    if n_misleading == 0:
        print("⚠ Warning: nessun sample ha misleader nella GT. "
            f"Chiavi disponibili nel primo entry: {list(metadata[0].keys())}")

    results = []

    
    with torch.no_grad():
        for entry in tqdm(metadata):
            img_path = os.path.join(args.dataset_path, entry["image_path"])
            misleaders = entry.get("misleader", [])

            try:
                pixel_values = load_image(img_path).to(device)
                response = chat_model.chat(
                    tokenizer,
                    pixel_values,
                    SYSTEM_PROMPT,
                    generation_config={"max_new_tokens": 50, "do_sample": False},
                )
            except Exception as e:
                print(f"Error on {img_path}: {e}")
                response = "no misleader"

            results.append({
                "image_path": entry["image_path"],
                "true_misleader": misleaders,
                "predicted_misleader": response.strip(),
            })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.dataset}_{args.split}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="internvl3/8B/",
                        help="Path to base InternVL3 model")
    parser.add_argument("--adapter_path", type=str, default="output/internvl3_finetuned/best_model",
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--dataset", type=str, default="misviz_synth",
                        choices=["misviz_synth", "misviz"],
                        help="Dataset to run inference on")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split")
    parser.add_argument("--dataset_path", type=str, default="data/misviz_synth/",
                        help="Path to dataset folder")
    parser.add_argument("--output_dir", type=str, default="results/internvl3_finetuned/",
                        help="Where to save results JSON")
    args = parser.parse_args()
    main(args)