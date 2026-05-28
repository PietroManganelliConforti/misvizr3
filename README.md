# Esperimenti — acl2026-misviz (ROMA3)

Stato aggiornato: maggio 2026.

---

## Ambiente

```bash
conda create --name lying_charts python=3.10
conda activate lying_charts
pip install -r requirements.txt
```

## Come rilanciarli

### Scaricare le immagini di misviz (una tantum)

```bash
python data/download_misviz_images.py --use_wayback 0
```

### 1. Zero-shot MLLM

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

### 2. Linter rule-based (ground truth axis)

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
python src/evaluate.py --model tinychart_encoder_only --dataset misviz_synth --split test

# Distribuzione per label
python src/second_evaluate.py --model internvl3-8B --dataset misviz_synth --split test

# Confusion matrix (salvata accanto al JSON)
python create_confusion_matrix.py --results_file results/internvl3-8B/misviz_synth_test.json
```

## 3. TinyChart classifier (image-only) (da fixare)
 
### Pipeline
 
```bash
# Step 1: Precompute embeddings (già fatto, in data/precomp/)
python src/model_tuning/01_precomputation/01_precompute_all_img_encodings.py \
  --models tinychart --datasets misviz_synth misviz \
  --datasetpaths data/misviz_synth/ data/misviz/ \
  --outputpath data/precomp/ --batchsize 16
 
# Step 2: Train classifier head
python src/model_tuning/03_deplot_axis_extraction_classifier/03_run_all_experiments.py \
  --base_models tinychart --experiment_types encoder_only
 
# Step 3: Inference
python src/model_tuning/03_deplot_axis_extraction_classifier/04_inference_classifier.py
 
# Step 4: Evaluate
python src/evaluate.py --model tinychart_encoder_only_123 --dataset misviz_synth --split test
```
### ⚠ Problema di riproducibilità
- Il file `encoding_strategies.py` nel repo upstream è **incompleto**: la classe `TinyChartEncodingStrategyBase` non implementa i metodi astratti `_initialize_encoder`, `encode`, `get_name`.
- Abbiamo scritto la nostra implementazione (`TinyChartOneEncoderEncodingStrategy`) ma gli embeddings risultanti non sono discriminativi:
  - Cosine similarity within-class: **0.81**
  - Cosine similarity between-class: **0.77**
- Gli autori usavano probabilmente un'implementazione diversa (Visual Token Merging, CLS token, layer intermedio) non rilasciata nel repo.
  
## 4. Fine-tuning InternVL3-8B (LoRA) (per ora dataset-small)
 
### Configurazione
 
- **Modello base:** InternVL3-8B con quantizzazione 4-bit (NF4)
- **LoRA:** r=8, alpha=32, target: q/k/v/o_proj, dropout=0.05
- **Training set:** Misviz-synth train_small (5,522 immagini)
- **Validation set:** Misviz-synth val (1,858 immagini)
- **Optimizer:** AdamW, lr=2e-5, weight_decay=0.01
- **Batch:** batch_size=2, grad_accum=16 (effective batch 32)
- **Early stopping:** patience=4, eval ogni 100 optimizer step
- **Class balancing:** WeightedRandomSampler (weight = 1/count classe più rara)
- **Prompt:** Simile al paper (Appendix K)
- **Seed:** 42
### Lanciare il training
 
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python finetune_internvl3.py \
        --model_path internvl3/8B/ \
        --dataset_path data/misviz_synth/ \
        --output_dir output/internvl3_finetuned_small/ \
        --train_split train_small \
        --epochs 5 \
        --batch_size 2 \
        --grad_accum 16 \
        --lr 2e-5 \
        --eval_every 100 \
        --eval_subset_size 300 \
        --patience 4
```
 
L'adapter (~20MB) viene salvato in `output/internvl3_finetuned_small/best_model/`.
 
### Lanciare inference
 
```bash
python inference_finetuned.py \
    --base_model_path internvl3/8B/ \
    --adapter_path output/internvl3_finetuned_small/best_model \
    --dataset misviz_synth \
    --split test \
    --output_dir results/internvl3_finetuned_small/
```
 
### Lanciare valutazione
 
```bash
python src/evaluate.py --model internvl3_finetuned_small --dataset misviz_synth --split test
```
 
