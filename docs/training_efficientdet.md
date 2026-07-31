# EfficientDet Training Pipeline

## Overview
- **Model**: EfficientDet-D2 (EfficientNet-B3 backbone + BiFPN)
- **Architecture**: Anchor-based with bi-directional feature pyramid network
- **Pretrained Weights**: `tf_efficientdet_d2` (COCO-pretrained)
- **Key Innovation**: Scalable architecture with compound scaling

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
- **Normalization**: Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Resize**: All images resized to 768×768 (fixed input size)
- Custom collate function scales bounding boxes to match resized images

### 2. Model Setup
- Load EfficientDet-D2 pretrained on COCO
- Replace classification head: 90 COCO classes → 3 classes (background + ActiveTB + ObsoleteTB)
- **BiFPN**: Bi-directional Feature Pyramid Network for multi-scale feature fusion
- **Anchors**: Default anchor boxes at 5 feature levels

### 3. Training Configuration
- **Optimizer**: AdamW with two LR groups:
  - Backbone (EfficientNet-B3): LR = 5e-4
  - Head (classification + box): LR = 5e-3
- **LR Schedule**: Cosine annealing from initial LR → 1e-8 over 50 epochs
- **Gradient Clipping**: Max norm = 5.0
- **EMA**: Exponential moving average (decay=0.99)

### 4. Sampling Strategy
- WeightedRandomSampler with majority undersampling:
  - ActiveTB-only images kept only until the 200-annotation cap (724 → 200); excess get weight 0 (excluded)
  - ObsoleteTB-only and mixed images: 1× weight (all kept, ~178 annotations)
  - Empty images: 0.05× weight (background suppression)
  - One pass over the reduced pool per epoch (~200 ActiveTB vs ~178 ObsoleteTB annotations)

### 5. Loss Function
- **Focal Loss**: For classification (handles class imbalance)
  - Alpha: Scalar 0.25 (symmetric, undersampler balances class counts)
  - Gamma: 1.5 (default)
- **Huber Loss**: For box regression
- **Anchor Assignment**: Based on IoU with ground truth

### 6. Per-Epoch Training Loop
- Forward pass through `DetBenchTrain` wrapper
- Compute focal loss + box regression loss
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

| Aspect | EfficientDet | DETR | FCOS/RetinaNet |
|--------|-------------|------|----------------|
| **Backbone** | EfficientNet-B3 | ResNet50 | ResNet50 |
| **Neck** | BiFPN | None | FPN |
| **Detection** | Anchor-based | Query-based | Anchor-free (FCOS) / Anchor-based (RetinaNet) |
| **Input Size** | Fixed 768×768 | Flexible | Flexible |
| **Normalization** | Manual (in transform) | Manual (in transform) | Built-in |
| **Loss** | Focal + Huber | CE + L1 + GIoU | Focal + GIoU + centerness |
| **Batch Size** | 16 | 8 (grad accum = 4) | 16 |
| **Learning Rates** | 2 groups (5e-4, 5e-3) | 3 groups (1e-5, 1e-4, 1e-4) | 2 groups (1e-5, 1e-4) |

---

## Expected Behavior
- **Epochs 1-10**: Loss should decrease rapidly, mAP should rise above 0
- **Epochs 10-30**: mAP plateaus, learning rate continues decreasing
- **Epochs 30-50**: Marginal gains, early stopping may trigger
- **Final mAP**: Expect 12-28 mAP (EfficientNet backbone may generalize better than ResNet)
