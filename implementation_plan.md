# FCOS Training Stagnation Fix

The FCOS model is failing to converge (mAP remains 0.000) and producing low-confidence predictions. After investigating the logs and codebase, I've identified three critical issues causing this stagnation:

## User Review Required
Please review the proposed architectural and training fixes. These changes will significantly alter the learning dynamics and should allow the model to properly converge on the TBX11K dataset.

## Proposed Changes

### 1. Stable Focal Loss Implementation
**Problem**: The custom `_per_class_focal_loss` uses a naive PyTorch implementation of focal loss (`ce_loss * (1 - p_t)^gamma`). When backpropagating through `(1 - p_t)^gamma`, numerical instability and vanishing gradients often occur because `p_t` contains the sigmoid computation in the autograd graph.
**Solution**: Replace the naive implementation with `torchvision.ops.sigmoid_focal_loss`, which is written in C++ and uses an analytic backward pass for guaranteed numerical stability. We will apply it per-channel to support per-class alphas.

### 2. Pretrained Weights Destruction (Optimizer Groups)
**Problem**: The model initializes `cls_logits` randomly but keeps the highly optimized COCO pretrained weights for the rest of the classification head and the entire regression head. The optimizer applies a high learning rate (`1e-4`) to the *entire* head, which blasts the delicate pretrained regression weights, destroying their ability to localize objects (causing the erratic bounding boxes seen in the logs).
**Solution**: Split the optimizer parameter groups into three tiers:
- `backbone_params` (ResNet): `1e-5`
- `pretrained_head_params` (Pretrained convolutions & regression): `1e-5`
- `new_head_params` (Randomly initialized `cls_logits`): `1e-4`

### 3. Class Priors Initialization
**Problem**: The `class_priors` logic dynamically scales the focal loss initialization bias `pi` based on class frequency. This sets `pi=0.01` for the rare class (ActiveTB) but `pi=0.0028` for the frequent class (ObsoleteTB). Focal loss expects all foreground classes to start at `pi=0.01` to prevent the background loss from dominating early training. Starting at `0.0028` creates a massive initial gradient that destabilizes the early epochs.
**Solution**: Hardcode the initial prior `pi = 0.01` for all foreground classes, which is the standard practice for Focal Loss.

#### [MODIFY] train_fcos.py
- Update `_patch_fcos_loss` to loop over channels and use `torchvision.ops.sigmoid_focal_loss`.
- Update the optimizer parameter grouping to separate `pretrained_head_params` and `new_head_params`.
- Simplify the `class_priors` loop to use `pi = 0.01` consistently.

## Verification Plan

### Automated Tests
- The user can resume or restart training using `python train_fcos.py`. We expect the `Avg Loss` to drop steadily below 0.7 and the `mAP` to rise above 0.000 within the first 5-10 epochs.
- We will monitor the evaluation logs to ensure that `pred_scores` increase beyond the `0.11` floor, indicating that the model is successfully distinguishing foreground objects.
