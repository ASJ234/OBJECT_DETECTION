# TBX11K Object Detection Benchmark

## Overview

This project trains and evaluates **4 distinct object detection architectures** on the [TBX11K Tuberculosis X-ray dataset](https://www.kaggle.com/datasets/). The dataset contains 11,702 chest X-ray images (512×512) with bounding box annotations for two classes:

| Class | Abbreviation | Description | Boxes |
|-------|-------------|-------------|-------|
| ActiveTuberculosis | ActiveTB | Active tuberculosis lesions | 972 |
| ObsoletePulmonaryTuberculosis | ObsoleteTB | Healed/obsolete TB scarring | 239 |

## Dataset Classes

### Why 2 Classes?

Object detection models require **bounding box annotations** for each class they detect. The TBX11K dataset only provides bounding box annotations for 2 classes: `ActiveTuberculosis` and `ObsoletePulmonaryTuberculosis`. Without bounding box labels, a model cannot learn to localize and classify objects.

### Image-Level Tags vs. Bounding Box Classes

The dataset contains two types of annotations:

| Type | Examples | Used for Training | Purpose |
|------|----------|-------------------|---------|
| **Bounding box classes** | ActiveTuberculosis, ObsoletePulmonaryTuberculosis | Yes | Define what the model learns to detect |
| **Image-level tags** | `healthy`, `sick_but_non-tb`, `active_tb`, `latent_tb`, `active&latent_tb` | No | Metadata indicating image source/context |

The image-level tags help understand data distribution but are **not** used as training classes. Images with tags like `healthy` or `sick_but_non-tb` have **zero bounding boxes** and serve as **negative samples** during training — they teach the model what "no detection" looks like, reducing false positives.

### num_classes Across Models

| Model | `num_classes` | Reason |
|-------|---------------|--------|
| FCOS | 2 | TorchVision handles background implicitly via score threshold |
| RetinaNet | 2 | TorchVision handles background implicitly via score threshold |
| DETR | 3 | HuggingFace DETR requires explicit background class (class 0) for Hungarian matching |
| EfficientDet | 3 | `effdet` library expects background as class 0 |

## Architectures

| # | Model | Type | Framework | Batch | Epochs | Optimizer | Head LR | Backbone LR |
|   |-------|------|-----------|-------|--------|-----------|---------|-------------|
| 1 | **FCOS** | Anchor-free one-stage | TorchVision | 8 | 50 | AdamW | 5e-4 (cosine) | 1e-5 |
| 2 | **RetinaNet** | One-stage focal loss | TorchVision | 8 | 50 | AdamW | 5e-4 (cosine) | 1e-5 |
| 3 | **EfficientDet-D2** | BiFPN multi-scale | `effdet` + `timm` | 4 | 50 | AdamW | 5e-3 (cosine) | 1e-4 |
| 4 | **DETR** | Transformer-based (end-to-end) | HuggingFace | 4 | 50 | AdamW | 5e-4 (cosine) | 1e-5 |

## Project Structure

```
tb-detection/
├── convert.py                  # Supervisely JSON → COCO format
├── eda.py                      # Exploratory data analysis + visualizations
├── requirements.txt            # Dependencies
├── utils/
│   ├── __init__.py
│   ├── coco_dataset.py         # PyTorch Dataset for COCO-format data
│   ├── engine.py               # Training loop, eval, confusion matrix
│   ├── transforms.py           # Shared augmentation transforms (rotation, translation, scale, etc.)
│   ├── ema.py                  # Exponential Moving Average
│   └── huggingface_hub.py      # HF Hub push integration
├── explain/
│   ├── __init__.py
│   ├── gradcam.py              # Grad-CAM for CNN-based models
│   ├── detr_attention.py       # Cross-attention extraction for DETR
│   └── visualize.py            # Heatmap overlay, comparison grid
├── train_fcos.py               # FCOS training pipeline
├── train_efficientdet.py       # EfficientDet-D2 training pipeline
├── train_retinanet.py          # RetinaNet training pipeline
├── train_detr.py               # DETR training pipeline
├── run_all.py                  # Sequential runner for all 4 pipelines
├── run_kaggle.ipynb            # Kaggle notebook entry point
└── DOCUMENTATION.md            # This file
```

## Setup

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 effdet timm pycocotools pillow tqdm matplotlib seaborn
```

## Usage

### 1. Run EDA
```bash
python eda.py
```
Generates tag distributions, bbox spatial heatmaps, size analysis, sample grids → `results/eda/`

### 2. Convert annotations
```bash
python convert.py --data-root Images --output-dir dataset
```
Creates:
- `dataset/coco/{train,val}.json` + images (for FCOS, RetinaNet, DETR, EfficientDet)

### 3. Train models
```bash
python train_fcos.py          # FCOS — 100 epochs, batch 8
python train_efficientdet.py  # EfficientDet-D2 — 100 epochs, batch 8
python train_retinanet.py     # RetinaNet — 100 epochs, batch 8
python train_detr.py          # DETR — 100 epochs, batch 4
```

Each script:
- Trains the model with best checkpoint selection by val mAP
- Runs validation with COCO metrics
- Computes and saves confusion matrix
- Runs test inference (predictions only — no GT boxes in test set)
- Generates XAI explanations (Grad-CAM or attention maps)

### 4. Compare results
```python
# results/comparison.csv contains all metrics
# results/comparison_barplot.png shows side-by-side
```

## Output Structure

```
results/
├── eda/                           # EDA figures
│   ├── tag_distribution.png
│   ├── bbox_class_distribution.png
│   ├── bbox_spatial_heatmap.png
│   ├── bbox_size_distribution.png
│   ├── bbox_aspect_ratio.png
│   ├── boxes_per_image.png
│   ├── sample_grid.png
│   └── dataset_summary.txt
├── fcos/
│   ├── weights/best_model.pth      # Trained weights
│   ├── metrics.json                # mAP scores
│   ├── confusion_matrix.png        # 3×3 confusion matrix
│   ├── val_preds.json              # Validation predictions
│   ├── test_preds.json             # Test predictions
│   └── explain/                    # Grad-CAM overlays
├── efficientdet/                   # Same structure
├── retinanet/                      # Same structure
├── detr/
│   └── explain/                    # Cross-attention maps (not Grad-CAM)
├── comparison.csv                  # All metrics in one table
└── comparison_barplot.png          # Visual comparison
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| mAP@0.5:0.95 | COCO primary metric (averaged over IoU 0.50–0.95) |
| mAP@0.5 | PASCAL VOC metric (IoU = 0.50) |
| mAP@0.75 | Strict localization (IoU = 0.75) |
| AP_ActiveTB | Per-class AP for Active Tuberculosis |
| AP_ObsoleteTB | Per-class AP for Obsolete Tuberculosis |
| AR@1, AR@10, AR@100 | Average recall at 1/10/100 detections |
| Confusion Matrix | 3×3 (Background, ActiveTB, ObsoleteTB) at IoU ≥ 0.5 |

## EDA Findings (key design decisions)

| Finding | Decision |
|---------|----------|
| 91% of images have no boxes | Keep in dataset with sampler weight 0.05 — model learns background suppression |
| 4:1 ActiveTB to ObsoleteTB ratio | Frequency-based class priors + focal loss α=0.156/0.633 + 5× minority oversampling |
| 95.4% of boxes in top image half | Natural for chest X-rays — no intervention needed |
| Mean aspect ratio 0.97 (roughly square) | Default anchor boxes are appropriate |
| Test set has no ground-truth boxes | Inference-only on test; evaluation on val set |
| `healthy`/`sick_but_non-tb` tags have zero boxes | Include as negative samples (weight 0.05 via sampler) to reduce false positives |
| Image-level tags (5 types) exist alongside bounding boxes | Tags are informational only — not used as training classes |

## XAI Methods

| Architecture | Method | How it works |
|-------------|--------|-------------|
| FCOS | Grad-CAM | Gradient-weighted activations from ResNet-50 layer4 |
| EfficientDet-D2 | Grad-CAM | Gradient-weighted activations from EfficientNet-B2 conv_head |
| RetinaNet | Grad-CAM | Same backbone approach |
| DETR | Cross-attention | Decoder's last multi-head attention weights reshaped to spatial map |

Each model generates explanations for 5 test samples, saved to `results/{model}/explain/`.

## Kaggle

Upload to Kaggle and run `run_kaggle.ipynb` for the full pipeline in one notebook. Ensure the dataset is available at the standard Kaggle input path or adjust paths accordingly.

## Training Optimizations

The following fixes were applied after the initial training run showed mAP stuck at ~0 (epochs 1–14) due to several issues:

### 1. Keep Negative Samples (Critical for mAP)

**Problem**: Filtering out images without annotations (91% of training data) meant validation only evaluated on positive images, making mAP meaningless.

**Fix**: Keep all 6600 images in the dataset. The `WeightedRandomSampler` gives empty images a small probability (0.05) of being sampled, teaching the model to suppress false positives while still focusing on annotated images.

### 2. Class Prior and Alpha for the Classification Head

**Problem**: Two failure modes. First, a hardcoded prior of 0.05 for both classes forced the model to learn the base rate from scratch. Then, per-class frequency priors (ActiveTB +1.39, ObsoleteTB -1.39) created a 2.8-logit gap — the minority head started in a "danger zone" and background gradient mass (5,400 background anchors vs 1-2 positive per image) drowned it below the 0.05 firing threshold, producing **zero** ObsoleteTB detections after 29 epochs.

**Fix**: Neutral bias init — `pi = 0.01` (logit = -4.60) for every class, so no class starts ahead of or behind the other. Class balance comes from focal-loss alpha + oversampling instead of the bias. Minority (ObsoleteTB) alpha raised to 0.75 (FCOS/RetinaNet per-class; EfficientDet scalar `config.alpha`), weakening background pressure on its channel ((1-alpha) from 0.367 → 0.25).

### 3. Higher Head Learning Rate

**Problem**: The randomly-initialized classification head uses the same LR as the pretrained backbone, converging too slowly.

**Fix**: Head LR = 5× base LR (5e-4), backbone LR = 0.01× base LR (1e-5). This 500:1 ratio lets the head learn quickly while preserving pretrained features.

### 4. Balanced Sampler (Majority Capped at 200)

**Problem**: Oversampling the minority (boost multipliers) swung the batch composition too far — minority-only images became ~44% of every epoch while majority-only were ~25%, starving the ActiveTB head (ActiveTB AR stayed 0.000 while ObsoleteTB improved). Boosting one class just starves the other.

**Fix**: Instead of boosting, **cap each group's per-epoch draws**. The `WeightedRandomSampler` is built with per-group targets (empty=200, ActiveTB-only=200, ObsoleteTB-only=200, mixed=200) — the majority class contributes exactly as much as the minority. Both class heads see balanced gradient exposure, and augmentations make repeated draws valuable. Epoch becomes 800 samples (100 batches at batch size 8).

### 5. Increased Gradient Clipping Threshold

**Problem**: `clip_norm = 5.0` discarded ~38% of gradient signal (grad norm ~8).

**Fix**: `clip_norm = 10.0` allows more gradient information through during early training.

### 6. Geometric Augmentations

**Problem**: Static training data, no spatial diversity.

**Fix**: Added rotation (±10°), translation (±10%), and scaling (±30%) via shared `utils/transforms.SharedAugmentedTransform`, with filtering of degenerate (zero-area) boxes after transforms.

### 7. Fixed LR Logging

**Problem**: Step logs showed `LR: 1e-5` (backbone LR), misleading users about the actual training LR (5e-4 for head).

**Fix**: Both per-step and per-epoch logs now show `max(g["lr"] for g in optimizer.param_groups)`, reflecting the effective head LR.

### 8. Environment Compatibility

| Issue | Fix |
|-------|-----|
| Shared memory (SHM) full on mlbox-gpu1 | `num_workers: 0` default in all configs |
| GPU OOM cascade in `run_all.py` | Added `torch.cuda.empty_cache()` between runs, reduced batch sizes to 8/4 |
| Missing `transformers` for DETR | Added dependency check in `run_all.py` |

### Parameter Summary

| Parameter | Before | After |
|-----------|--------|-------|
| Empty image filter | Removed from dataset | Kept with sampler weight 0.05 |
| Class priors | Frequency-based (0.80/0.20) | Neutral (pi=0.01, logit -4.60 all classes) |
| Minority focal alpha | 0.633 | 0.75 (scalar 0.75 for EfficientDet) |
| Head:backbone LR ratio | 10:1 | 500:1 (head 5e-4, backbone 1e-5) |
| Minority sampling | 3× → 5× → 8× boost | Balanced cap: 200 draws/group/epoch |
| Gradient clip | 5.0 | 10.0 |
| Batch size (FCOS/RetinaNet) | 16 | 8 |
| Batch size (EfficientDet/DETR) | 16/8 | 4 |
| num_workers | 2 | 0 |
| Transforms | Per-file duplicates | Shared `SharedAugmentedTransform` with rotation/translation/scale |
