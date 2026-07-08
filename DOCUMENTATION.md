# TBX11K Object Detection Benchmark

## Overview

This project trains and evaluates **4 distinct object detection architectures** on the [TBX11K Tuberculosis X-ray dataset](https://www.kaggle.com/datasets/). The dataset contains 11,702 chest X-ray images (512×512) with bounding box annotations for two classes:

| Class | Abbreviation | Description | Boxes |
|-------|-------------|-------------|-------|
| ActiveTuberculosis | ActiveTB | Active tuberculosis lesions | 972 |
| ObsoletePulmonaryTuberculosis | ObsoleteTB | Healed/obsolete TB scarring | 239 |

## Architectures

| # | Model | Type | Framework | Batch | Epochs | Optimizer | LR |
|---|-------|------|-----------|-------|--------|-----------|-----|
| 1 | **FCOS** | Anchor-free one-stage | TorchVision | 8 | 100 | SGD | 0.01 (cosine) |
| 2 | **EfficientDet-D2** | BiFPN multi-scale | `effdet` + `timm` | 8 | 100 | AdamW | 1e-4 (cosine) |
| 3 | **RetinaNet** | One-stage focal loss | TorchVision | 8 | 100 | SGD | 0.005 (cosine) |
| 4 | **DETR** | Transformer-based (end-to-end) | TorchVision | 4 | 100 | AdamW | 1e-4 (cosine) |

## Project Structure

```
tb-detection/
├── convert.py                  # Supervisely JSON → COCO format
├── eda.py                      # Exploratory data analysis + visualizations
├── requirements.txt            # Dependencies
├── utils/
│   ├── __init__.py
│   ├── coco_dataset.py         # PyTorch Dataset for COCO-format data
│   └── engine.py               # Training loop, eval, confusion matrix
├── explain/
│   ├── __init__.py
│   ├── gradcam.py              # Grad-CAM for CNN-based models
│   ├── detr_attention.py       # Cross-attention extraction for DETR
│   └── visualize.py            # Heatmap overlay, comparison grid
├── train_fcos.py               # FCOS training pipeline
├── train_efficientdet.py       # EfficientDet-D2 training pipeline
├── train_retinanet.py          # RetinaNet training pipeline
├── train_detr.py               # DETR training pipeline
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
| 6.8% of images have boxes | Oversample positives 4× in dataloader |
| 4:1 ActiveTB to ObsoleteTB ratio | Monitor per-class AP, no special weighting (focal loss handles it) |
| 95.4% of boxes in top image half | Natural for chest X-rays — no intervention needed |
| Mean aspect ratio 0.97 (roughly square) | Default anchor boxes are appropriate |
| Test set has no ground-truth boxes | Inference-only on test; evaluation on val set |
| `healthy`/`sick_but_non-tb` tags have zero boxes | Include as negative samples to reduce false positives |

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
