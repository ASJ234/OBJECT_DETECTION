# FCOS Training Pipeline

## Overview
- **Model**: FCOS (Fully Convolutional One-Stage) - ResNet50 backbone + FPN
- **Architecture**: Anchor-free, dense prediction
- **Pretrained Weights**: `fcos_resnet50_fpn` (COCO-pretrained)
- **Key Innovation**: Anchor-free object detection with centerness prediction

---

## Step-by-Step Training Process

### 1. Data Loading & Preprocessing
- Load COCO-format annotations from `dataset/coco/train.json`
- Filter out empty images: **6,600 → 599 annotated images**
- Apply augmentations:
  - Horizontal flip (50% probability)
  - Brightness adjustment ±30%
  - Contrast adjustment ±30%
  - Gamma correction (0.8-1.2)
  - Gaussian noise (σ=0.05)
- **Normalization**: Built-in `GeneralizedRCNNTransform` (ImageNet normalization)
- Collate into batches of 16

### 2. Model Setup
- Load FCOS-ResNet50-FPN pretrained on COCO
- Replace classification head: 91 COCO classes → 2 classes (ActiveTB, ObsoleteTB)
- **Class Priors**: Initialize classification bias with π=0.05
- **Centerness Branch**: Predicts centerness score to down-weight low-quality predictions

### 3. Training Configuration
- **Optimizer**: AdamW with two LR groups:
  - Backbone (ResNet50 + FPN): LR = 1e-5 (10× smaller)
  - Head (classification + regression + centerness): LR = 1e-4
- **LR Schedule**: Cosine annealing from initial LR → 1e-7 over 50 epochs
- **Gradient Clipping**: Max norm = 5.0
- **EMA**: Exponential moving average (decay=0.99)

### 4. Sampling Strategy
- WeightedRandomSampler with minority class boost:
  - ObsoleteTB-only images: 3× weight
  - Mixed images (both classes): 2.5× weight
  - ActiveTB-only images: 1× weight
  - Empty images: 0× weight (filtered out)

### 5. Loss Function
- **Focal Loss**: For classification (handles class imbalance)
  - Alpha: Dynamic per-class (ObsoleteTB gets higher alpha)
  - Gamma: 2.0 (default)
- **GIoU Loss**: For box regression
- **Centerness Loss**: Binary cross-entropy for centerness prediction
- **No Anchors**: FCOS predicts boxes directly from feature map locations

### 6. Per-Epoch Training Loop
- Forward pass → compute focal loss + GIoU loss + centerness loss
- Backward pass → gradient clipping → optimizer step
- Update EMA model
- Log training loss and learning rates

### 7. Validation (per epoch)
- Run EMA model on validation set
- Apply score threshold (0.05) + NMS (IoU=0.5)
- Compute COCO mAP metrics
- Log to Weights & Biases

### 8. Checkpointing & Early Stopping
- Save best model (by mAP@0.5:0.95)
- Early stopping: stop if no improvement for 30 epochs

### 9. Post-Training
- Final evaluation on test set (with and without TTA)
- Save predictions, confusion matrix, training curves
- Push to Hugging Face Hub (optional)

---

## Key Differences from Other Models

| Aspect | FCOS | DETR | RetinaNet | EfficientDet |
|--------|------|------|-----------|--------------|
| **Detection** | Anchor-free | Query-based | Anchor-based | Anchor-based |
| **Loss** | Focal + GIoU + centerness | CE + L1 + GIoU | Focal + Smooth L1 | Focal + Huber |
| **NMS** | Yes (IoU=0.5) | No | Yes (IoU=0.5) | Yes (IoU=0.5) |
| **Anchors** | No | No | Yes (multi-scale) | Yes (multi-scale) |
| **Centerness** | Yes | No | No | No |
| **Normalization** | Built-in | Manual | Built-in | Manual |
| **Input Size** | Flexible | Flexible | Flexible | Fixed 768×768 |
| **Batch Size** | 16 | 8 (grad accum = 4) | 16 | 16 |
| **Learning Rates** | 2 groups (1e-5, 1e-4) | 3 groups | 2 groups (1e-5, 1e-4) | 2 groups (5e-4, 5e-3) |

---

## Expected Behavior
- **Epochs 1-10**: Loss should decrease, mAP should rise above 0
- **Epochs 10-30**: mAP plateaus, learning rate continues decreasing
- **Epochs 30-50**: Marginal gains, early stopping may trigger
- **Final mAP**: Expect 10-25 mAP (limited by dataset size)

---

## Why Centerness?
FCOS introduces a centerness branch to address a key issue in anchor-free detection:
- Without anchors, each feature map location predicts a box
- Locations far from object centers produce low-quality predictions
- Centerness score = √(left × right × top × bottom) / (max × min)
- Range [0, 1]: 1.0 = perfect center, 0.0 = far edge
- Applied as a weight to classification scores during inference
- Reduces false positives from low-quality edge predictions
