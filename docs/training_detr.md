# DETR Training Pipeline

## Overview
- **Model**: DETR (DEtection TRansformer) - ResNet50 backbone + Transformer encoder-decoder
- **Architecture**: Transformer-based, anchor-free, uses Hungarian matching for set prediction
- **Pretrained Weights**: HuggingFace `facebook/detr-resnet-50` (COCO-pretrained)
- **Key Innovation**: Treats object detection as a set prediction problem

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
- Collate into batches of 8 (effective batch size = 32 with gradient accumulation)

### 2. Model Setup
- Load HuggingFace DETR-ResNet50 pretrained on COCO
- Replace classification head: 91 COCO classes → 2 classes (ActiveTB, ObsoleteTB)
- Model wrapper converts between:
  - Training: xyxy absolute → cxcywh normalized
  - Evaluation: cxcywh normalized → xyxy absolute
- **100 object queries** (learnable embeddings that predict objects)

### 3. Training Configuration
- **Optimizer**: AdamW with three LR groups:
  - Backbone (ResNet50): LR = 1e-5
  - Transformer (encoder + decoder): LR = 1e-4
  - Head (classification + box): LR = 1e-4
- **LR Schedule**: Cosine annealing from initial LR → 1e-8 over 50 epochs
- **Gradient Clipping**: Max norm = 5.0
- **Gradient Accumulation**: 4 steps (effective batch = 32)
- **EMA**: Exponential moving average (decay=0.99) for smoother validation

### 4. Sampling Strategy
- WeightedRandomSampler with majority undersampling:
  - ActiveTB-only images kept only until the 200-annotation cap (724 → 200); excess get weight 0 (excluded)
  - ObsoleteTB-only and mixed images: 1× weight (all kept, ~178 annotations)
  - Empty images: 0.05× weight (background suppression)
  - One pass over the reduced pool per epoch (~200 ActiveTB vs ~178 ObsoleteTB annotations)

### 5. Loss Function
- **Hungarian Matching**: Matches 100 predictions to GT objects (one-to-one)
- **Classification Loss**: Cross-entropy on matched pairs
- **Box Loss**: L1 loss + GIoU loss on matched pairs
- **No Object Class**: Background predictions (model learns to predict "nothing" for empty regions)

### 6. Per-Epoch Training Loop
- Forward pass → compute Hungarian matching loss
- Backward pass → gradient clipping → optimizer step
- Update EMA model
- Log training loss and learning rates

### 7. Validation (per epoch)
- Run EMA model on validation set
- Apply score threshold (0.05)
- Compute COCO mAP metrics (no NMS needed - one-to-one matching)
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

| Aspect | DETR | FCOS/RetinaNet/EfficientDet |
|--------|------|----------------------------|
| **Architecture** | Transformer (encoder-decoder) | CNN + FPN |
| **Object Detection** | Set prediction (100 queries) | Dense anchor-based/anchor-free |
| **Matching** | Hungarian algorithm (one-to-one) | Many-to-one (multiple anchors per object) |
| **NMS Required** | No (one-to-one matching) | Yes (multiple predictions per object) |
| **Loss Function** | Cross-entropy + L1 + GIoU | Focal loss + GIoU + centerness |
| **Normalization** | Manual (in transform) | Built-in (GeneralizedRCNNTransform) |
| **Batch Size** | 8 (with grad accum = 4) | 16 |
| **Learning Rates** | 3 separate groups | 2 groups (backbone + head) |

---

## Expected Behavior
- **Epochs 1-10**: Loss should decrease, mAP should rise above 0
- **Epochs 10-30**: mAP plateaus, learning rate continues decreasing
- **Epochs 30-50**: Marginal gains, early stopping may trigger
- **Final mAP**: Expect 10-25 mAP (limited by dataset size of ~600 images)
