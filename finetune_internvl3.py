"""
finetune_internvl3.py

Fine-tunes InternVL3-8B on Misviz-synth `train` split using LoRA.

Key features vs. the original script:
  * Class-balanced sampling (WeightedRandomSampler, weight = 1 / count of
    rarest misleader in the sample).
  * Vectorized batching: a custom collator builds proper batched tensors
    (padded input_ids/labels/attention_mask + concatenated pixel_values +
    image_flags), so the GPU actually parallelizes across batch_size.
  * Mid-epoch evaluation + early stopping based on optimizer steps.

Usage:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python finetune_internvl3.py \
        --model_path internvl3/8B/ \
        --dataset_path data/misviz_synth/ \
        --output_dir output/internvl3_finetuned/ \
        --epochs 3 \
        --batch_size 2 \
        --grad_accum 16 \
        --lr 2e-5 \
        --eval_every 500 \
        --eval_subset_size 300 \
        --patience 4
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/home/pietro/Research/ROMA3_grafici/acl2026-misviz/internvl3/8B')

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import f1_score
from conversation import get_conv_template
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

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


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_w, orig_h = image.size
    ar = orig_w / orig_h
    target_ratios = sorted({
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }, key=lambda x: x[0] * x[1])
    best = min(target_ratios, key=lambda r: abs(ar - r[0] / r[1]))
    tw, th = image_size * best[0], image_size * best[1]
    blocks = best[0] * best[1]
    resized = image.resize((tw, th))
    out = []
    for i in range(blocks):
        box = (
            (i % best[0]) * image_size,
            (i // best[0]) * image_size,
            ((i % best[0]) + 1) * image_size,
            ((i // best[0]) + 1) * image_size,
        )
        out.append(resized.crop(box))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images]).to(torch.bfloat16)
    return pixel_values


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_misleaders(entry):
    m = entry.get("misleader", [])
    if isinstance(m, str):
        return [m] if m else []
    return list(m) if m else []


def label_to_target(misleaders):
    if not misleaders:
        return "no misleader"
    return ", ".join(misleaders)


def post_process_pred(pred):
    if "no misleader" in pred.lower():
        return []
    if "," in pred:
        return [p.strip().lower() for p in pred.replace("\n", "").split(",")]
    return [pred.lower().replace("\n", "").strip()]


def compute_binary_f1(preds, trues):
    bp = [1 if len(post_process_pred(p)) > 0 else 0 for p in preds]
    bt = [1 if len(t) > 0 else 0 for t in trues]
    return f1_score(bt, bp, average="macro", zero_division=0)


# ─────────────────────────────────────────────────────────────────────────────
# Class-balanced sampling
# ─────────────────────────────────────────────────────────────────────────────
def compute_sample_weights(metadata, verbose=True):
    """
    For each sample, compute weight = 1 / count_of_rarest_class_in_sample.
    """
    counts = Counter()
    for entry in metadata:
        labels = get_misleaders(entry)
        if not labels:
            counts["no_misleader"] += 1
        else:
            for lab in labels:
                counts[lab] += 1

    if not counts:
        raise ValueError("compute_sample_weights: dataset vuoto, nessuna label trovata.")

    weights = []
    for entry in metadata:
        labels = get_misleaders(entry)
        if not labels:
            # Se non ci sono esempi 'no_misleader' nel set, usa il fallback più sicuro
            # (la classe più rara), in modo da non crashare.
            c = counts.get("no_misleader", min(counts.values()))
            w = 1.0 / c
        else:
            w = max(1.0 / counts[lab] for lab in labels)
        weights.append(w)

    if verbose:
        print("\nClass distribution in training set:")
        total = sum(counts.values())
        for lab, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {lab:40s}  {c:6d}  ({100*c/total:5.2f}%)")
        print(f"  Total label occurrences: {total}\n")

    return weights, counts

# ─────────────────────────────────────────────────────────────────────────────
# Dataset & collator
# ─────────────────────────────────────────────────────────────────────────────
class MisvizDataset(Dataset):
    """Minimal dataset; heavy lifting happens in the collator."""

    def __init__(self, metadata, image_root):
        self.metadata = metadata
        self.image_root = image_root

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        misleaders = get_misleaders(entry)
        return {
            "image_path": os.path.join(self.image_root, entry["image_path"]),
            "target": label_to_target(misleaders),
            "misleaders": misleaders,
        }


class MisvizCollator:
    """
    Vectorized collator. Builds proper batched tensors so the GPU sees a real
    batch instead of a sample-by-sample loop.

    Output (one dict per batch):
        pixel_values  : (sum_patches, 3, H, W)   — concatenated across batch
        image_flags   : (sum_patches,)            — all ones
        input_ids     : (B, max_len)              — right-padded
        attention_mask: (B, max_len)
        labels        : (B, max_len)              — -100 on prompt + padding
        misleaders    : list[list[str]]           — for downstream eval
    """

    def __init__(self, tokenizer, template_name, system_message,
                 num_image_token, max_num=6):
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.system_message = system_message
        self.num_image_token = num_image_token
        self.max_num = max_num
        self.pad_token_id = (tokenizer.pad_token_id
                             if tokenizer.pad_token_id is not None
                             else tokenizer.eos_token_id)

    def _build_one(self, item):
        """Tokenize a single sample. Returns None on image load failure."""


        try:
            pixel_values = load_image(item["image_path"], max_num=self.max_num)
        except Exception as e:
            return None

        num_patches = pixel_values.shape[0]
        image_tokens = (IMG_START_TOKEN
                        + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                        + IMG_END_TOKEN)
        template = get_conv_template(self.template_name)
        template.system_message = self.system_message
        template.append_message(template.roles[0], '<image>\n' + SYSTEM_PROMPT)
        template.append_message(template.roles[1], None)   # placeholder vuoto
        prompt_prefix = template.get_prompt().replace('<image>', image_tokens, 1)

        # poi rifai con il target vero
        template2 = get_conv_template(self.template_name)
        template2.system_message = self.system_message
        template2.append_message(template2.roles[0], '<image>\n' + SYSTEM_PROMPT)
        template2.append_message(template2.roles[1], item["target"])
        prompt_full = template2.get_prompt().replace('<image>', image_tokens, 1)

        prefix_ids = self.tokenizer(prompt_prefix, return_tensors='pt')['input_ids'][0]
        input_ids  = self.tokenizer(prompt_full, return_tensors='pt')['input_ids'][0]

        labels = input_ids.clone()
        labels[:len(prefix_ids)] = -100   # maschera tutto il prefisso
    
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
        }

    def __call__(self, batch):
        prepared = []
        misleaders_out = []
        for item in batch:
            built = self._build_one(item)
            if built is None:
                continue
            prepared.append(built)
            misleaders_out.append(item["misleaders"])

        if not prepared:
            return None

        max_len = max(p["input_ids"].size(0) for p in prepared)
        pad_id = self.pad_token_id

        input_ids = torch.full((len(prepared), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(prepared), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(prepared), max_len), dtype=torch.long)

        for i, p in enumerate(prepared):
            L = p["input_ids"].size(0)
            input_ids[i, :L] = p["input_ids"]
            labels[i, :L] = p["labels"]
            attention_mask[i, :L] = 1

        pixel_values = torch.cat([p["pixel_values"] for p in prepared], dim=0)
        image_flags = torch.ones(pixel_values.shape[0], dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "misleaders": misleaders_out,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, tokenizer, dataset, device, max_samples=None, desc="Eval"):
    model.eval()
    preds, trues = [], []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    indices = list(range(n))

    m = model.module if hasattr(model, "module") else model

    for i in tqdm(indices, desc=desc):
        item = dataset[i]
        try:
            pixel_values = load_image(item["image_path"]).to(device)
        except Exception:
            preds.append("no misleader")
            trues.append(item["misleaders"])
            continue

        response = m.chat(
            tokenizer,
            pixel_values,
            SYSTEM_PROMPT,
            generation_config={"max_new_tokens": 50, "do_sample": False},
        )
        preds.append(response)
        trues.append(item["misleaders"])

    f1 = compute_binary_f1(preds, trues)
    return f1, preds, trues


# ─────────────────────────────────────────────────────────────────────────────
# Training loop with mid-epoch eval + early stopping
# ─────────────────────────────────────────────────────────────────────────────
def train(model, tokenizer, train_loader, val_dataset, optimizer, device, args):
    """
    Trains with mid-epoch eval. Returns (best_f1, best_adapter_state).
    Saves the best checkpoint to {output_dir}/best_model whenever val F1 improves.
    """


    base = model.module if hasattr(model, "module") else model
    unwrapped = base.base_model.model if hasattr(base, "base_model") else base
    unwrapped.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    best_f1 = 0.0
    best_adapter_state = None        # ← NUOVO
    patience_counter = 0
    global_step = 0
    micro_step = 0
    accumulated_loss = 0.0
    should_stop = False

    optimizer.zero_grad()   

    def snapshot_trainable():
        return {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    for epoch in range(1, args.epochs + 1):
        if should_stop:
            break
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device)
            image_flags = batch["image_flags"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = unwrapped.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                labels=labels,
            )

            if outputs.loss is None:
                continue

            loss = outputs.loss / args.grad_accum
            loss.backward()
            accumulated_loss += loss.item() * args.grad_accum
            micro_step += 1

            # Optimizer step
            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix({
                    "step": global_step,
                    "loss": f"{accumulated_loss / args.grad_accum:.4f}",
                })
                accumulated_loss = 0.0

                # Mid-epoch evaluation
                if global_step % args.eval_every == 0:
                    print(f"\n[step {global_step}] Running mid-epoch eval "
                          f"(subset={args.eval_subset_size})...")
                    val_f1, _, _ = evaluate(
                        model, tokenizer, val_dataset, device,
                        max_samples=args.eval_subset_size,
                        desc=f"Eval@{global_step}",
                    )
                    print(f"[step {global_step}] Val F1 (subset): {val_f1:.4f}")

                    if val_f1 > best_f1 + 1e-3:
                        best_f1 = val_f1
                        patience_counter = 0
                        best_adapter_state = snapshot_trainable()   # ← NUOVO
                        out_path = os.path.join(args.output_dir, "best_model")
                        model.save_pretrained(out_path)
                        tokenizer.save_pretrained(out_path)
                        print(f"  ✓ New best (F1={best_f1:.4f}) saved to {out_path}")
                    else:
                        patience_counter += 1
                        print(f"  No improvement ({patience_counter}/{args.patience})")
                        if patience_counter >= args.patience:
                            print("Early stopping triggered.")
                            should_stop = True
                            break

                    model.train()  # back to training mode after eval

        # End-of-epoch eval (full subset, same size as mid-epoch for consistency)
        if not should_stop:
            print(f"\n[epoch {epoch}] End-of-epoch eval...")
            val_f1, _, _ = evaluate(
                model, tokenizer, val_dataset, device,
                max_samples=args.eval_subset_size,
                desc=f"Eval@epoch{epoch}",
            )
            print(f"[epoch {epoch}] Val F1 (subset): {val_f1:.4f}")
            if val_f1 > best_f1 + 1e-3:
                best_f1 = val_f1
                patience_counter = 0
                best_adapter_state = snapshot_trainable()           # ← NUOVO
                out_path = os.path.join(args.output_dir, "best_model")
                model.save_pretrained(out_path)
                tokenizer.save_pretrained(out_path)
                print(f"  ✓ New best (F1={best_f1:.4f}) saved")

    return best_f1, best_adapter_state


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load metadata ───────────────────────────────────────────────────────
    with open(os.path.join(args.dataset_path, "misviz_synth.json")) as f:
        all_metadata = json.load(f)

    train_meta = [d for d in all_metadata if d["split"] == args.train_split]
    val_meta = [d for d in all_metadata if d["split"] == "val"]
    print(f"Train ({args.train_split}): {len(train_meta)} | Val: {len(val_meta)}")

    image_root = args.dataset_path

    # ── Sample weights for class balancing ──────────────────────────────────
    weights, _ = compute_sample_weights(train_meta, verbose=True)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_meta),     # one "epoch" = len(train) draws
        replacement=True,
    )

    # ── Load model & tokenizer ──────────────────────────────────────────────
    print(f"Loading {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # Force dtype consistency (some buffers come back as float32)
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)
    for param in model.vision_model.parameters():
        param.data = param.data.to(torch.bfloat16)
    for buffer in model.vision_model.buffers():
        buffer.data = buffer.data.to(torch.bfloat16)

    # Snapshot template info BEFORE wrapping with PEFT (cleaner attribute access)
    template_name = model.template
    system_message = model.system_message
    num_image_token = model.num_image_token

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Data loaders ────────────────────────────────────────────────────────
    train_dataset = MisvizDataset(train_meta, image_root)
    val_dataset = MisvizDataset(val_meta, image_root)

    collator = MisvizCollator(
        tokenizer=tokenizer,
        template_name=template_name,
        system_message=system_message,
        num_image_token=num_image_token,
        max_num=6,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    # ── Train ───────────────────────────────────────────────────────────────
    best_f1, best_adapter_state = train(
        model, tokenizer, train_loader, val_dataset,
        optimizer, device, args,
    )

    # ── Final full val eval on the BEST checkpoint ──────────────────────────
    print(f"\nTraining done. Best subset val F1: {best_f1:.4f}")

    if best_adapter_state is None:
        print("⚠ No best checkpoint was saved during training "
              "(F1 never improved over the initial 0). "
              "Skipping final eval.")
        return

    print("Restoring best adapter weights for final eval...")
    with torch.no_grad():
        missing = []
        for n, p in model.named_parameters():
            if n in best_adapter_state:
                p.data.copy_(best_adapter_state[n].to(p.device, dtype=p.dtype))
            elif p.requires_grad:
                missing.append(n)
        if missing:
            print(f"⚠ {len(missing)} trainable params not found in snapshot "
                  f"(first 3: {missing[:3]})")

    print("Running final FULL val eval on the BEST model...")
    final_f1, _, _ = evaluate(
        model, tokenizer, val_dataset, device,
        max_samples=None, desc="FinalVal",
    )
    print(f"Final FULL val F1 (best checkpoint): {final_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="internvl3/8B/")
    parser.add_argument("--dataset_path", type=str, default="data/misviz_synth/")
    parser.add_argument("--output_dir", type=str, default="output/internvl3_finetuned/")
    parser.add_argument("--train_split", type=str, default="train",
                        help="Which split to train on: 'train' (full) or 'train_small'")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Max epochs. Early stopping likely fires sooner on the full train set.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--patience", type=int, default=4,
                        help="Number of consecutive non-improving evals before stopping.")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Run eval every N optimizer steps.")
    parser.add_argument("--eval_subset_size", type=int, default=300,
                        help="Use this many val samples for eval (for speed). Set to a large "
                             "number to use all of val. Final eval after training is always full.")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)