# Esperimenti — acl2026-misviz (ROMA3)

Stato aggiornato: maggio 2026.

---

## Ambiente

```bash
conda create --name lying_charts python=3.10
conda activate lying_charts
pip install -r requirements.txt
```

---

## Struttura risultati

```
results/
├── internvl3-8B/
│   ├── misviz_synth_test.json       # zero-shot, 2343 samples
│   └── misviz_test.json             # zero-shot, 2048 samples (-29 img mancanti)
├── qwen2.5vl-7B/
│   └── misviz_synth_test.json       # zero-shot, 2343 samples
├── linter_gt/
│   └── misviz_synth_test.json       # linter con ground truth axis, 2343 samples
├── tinychart_encoder_only_123/
    ├── misviz_synth_test.json       # classifier image-only, 2343 samples
    └── misviz_test.json             # classifier image-only, 2048 samples

```


**Metriche:** Acc = accuracy binaria, Prec/Rec/F1 = sulla classe "misleading", EM = exact match misleader type, PM = partial match misleader type.

---

## Come rilanciarli

### Scaricare le immagini di misviz (una tantum)

```bash
python data/download_misviz_images.py --use_wayback 0
```

### Zero-shot MLLM

```bash
python src/mllm_inference/misleader_detection_MLLM.py \
  --datasets misviz_synth \
  --split test \
  --model internvl3/8B/ \
  --max_tokens 200

python src/mllm_inference/misleader_detection_MLLM.py \
  --datasets misviz_synth \
  --split test \
  --model qwen2.5-vl/7B/ \
  --max_tokens 200
```

> Per girare su misviz reale sostituire `--datasets misviz_synth` con `--datasets misviz`.
> I risultati vengono salvati in `results/<model-name>/`.

### Linter rule-based (ground truth axis)

```bash
python src/rule_based_linter/linter.py \
  --datasets misviz_synth \
  --split test \
  --use_predicted_axis 0
```

> Il linter su misviz reale richiede axis predetti (`--use_predicted_axis 1`), che a loro volta richiedono il fine-tuning di DePlot (vedi README originale).

### Evaluation

```bash
# Metriche binarie e multiclass
python src/evaluate.py --model internvl3-8B --dataset misviz_synth --split test
python src/evaluate.py --model internvl3-8B --dataset misviz --split test
python src/evaluate.py --model tinychart_encoder_only_123 --dataset misviz_synth --split test

# Distribuzione per label
python src/second_evaluate.py --model internvl3-8B --dataset misviz_synth --split test

# Confusion matrix (salvata accanto al JSON)
python create_confusion_matrix.py --results_file results/internvl3-8B/misviz_synth_test.json
```

