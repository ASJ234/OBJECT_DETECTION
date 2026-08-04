# FCOS Training Stagnation Fix Walkthrough

I have successfully applied the fixes to the FCOS object detection pipeline to resolve the 0 mAP stagnation.

## Changes Made

### 1. Stable Focal Loss Implementation
- Replaced the custom naive PyTorch implementation of focal loss (`_per_class_focal_loss`) with the numerically stable, C++-backed `torchvision.ops.sigmoid_focal_loss`.
- To support class frequency weights (`alphas`), we apply the `sigmoid_focal_loss` separately per channel (one for ActiveTB, one for ObsoleteTB) and sum them up. This avoids vanishing gradients caused by backpropagating through the `(1 - p_t)^gamma` factor.

### 2. Protected Pretrained Weights in Optimizer
- Split the model parameters into three distinct groups:
  1. **`backbone_params`**: ResNet backbone (Learning Rate: `scaled_lr * 0.1`).
  2. **`pretrained_head_params`**: The convolution layers in the classification head and the entire class-agnostic regression head. These layers have COCO pretrained weights (Learning Rate: `scaled_lr * 0.1`).
  3. **`new_head_params`**: The randomly initialized `cls_logits` layer (Learning Rate: `scaled_lr`).
- Previously, the `regression_head` was lumped into `head_params` and trained with a high learning rate (`1e-4`), which destroyed its highly optimized pretrained weights in early epochs, causing the model to output garbage bounding boxes.

### 3. Focal Loss Prior Bias Fix
- Hardcoded the initial focal loss prior (`pi`) to `0.01` for both ActiveTB and ObsoleteTB, which aligns with standard best practices.
- Previously, the prior was dynamically scaled by the inverse class frequency, pushing the initial probability for ObsoleteTB down to `0.0028`. Because ObsoleteTB is highly frequent, this small prior generated massive gradient spikes that destabilized early training.

## Next Steps
You can now resume or restart your training run using:
```bash
python train_fcos.py --config config.yaml
```
You should notice the `Avg Loss` steadily decreasing, and the model should start outputting valid, localized predictions resulting in non-zero mAP.

> [!TIP]
> If you notice that one class is still struggling compared to the other, you can further fine-tune the `alpha` parameters or increase the `min_size` image transform if TB lesions are extremely small.

## Fix Round 2: Score Calibration & Head Learning (RetinaNet + FCOS)

Post-training analysis (`tools/analyze_predictions.py`) revealed both models localize well
(FCOS 88.7%, RetinaNet 79.6% label-agnostic recall) but output pathologically low scores:
RetinaNet scores were pinned at 0.050–0.058 (the bias init), FCOS saturated at 0.29, and
nothing cleared 0.3. Root causes and fixes:

### 1. Negative-image dominance (main culprit)
The dataset is 91% negative (6001/6600 train images). The RetinaNet loss patch normalized
each image's classification loss by its own foreground count, so a negative image's loss
(summed over ~300k background anchors, divided by 1) outweighed a positive image by ~500x,
crushing the classification logits toward the background prior. FCOS normalizes batch-wide so
it was milder, but still suffered the same pull.

- `train_retinanet.py`: negative-only images now contribute `loss * 0.1` (`neg_loss_scale` in
  `_patch_retinanet_loss`) instead of the full sum.

### 2. RetinaNet head was randomly initialized
RetinaNet replaces its whole classification head; with ~200 FG anchors vs ~300k BG anchors a
random head never escapes the background prior. FCOS keeps pretrained head convs (why it
learned more).

- `train_retinanet.py`: the class-agnostic `conv` stacks of the classification (and, for custom
  anchors, regression) head are now warm-started from the pretrained COCO head
  (`model.head.classification_head.conv.load_state_dict(...)`), built with matching
  `norm_layer=partial(nn.GroupNorm, 32)`. Only the final `cls_logits`/`bbox_pred` convs are
  random. Verified: `std` of the warm-started convs ≈ 0.026 vs 0.01 random init.

### 3. Focal alpha derived from SAMPLED class balance (not raw counts)
Raw counts are 724 ActiveTB vs 178 ObsoleteTB (4:1), but the balanced sampler
already caps training at ~227 vs 178 — so weighting alpha by raw counts would
double-count the imbalance. `get_class_frequency_sampler` now returns
`effective_counts` and the alphas are computed from those:
`per_class_focal_alphas(effective_counts, n, base_alpha=0.35, max_alpha=0.75)`
→ `[0.35, 0.446]` (a modest 1.28x minority boost, consistent with the sampler's
balance) instead of `[0.35, 0.75]`.

### 4. Linear warmup
`warmup_epochs` default changed 0 → 3 in both pipelines to stabilize the early epochs when
the random cls head is most fragile.

### 5. FCOS background-centerness supervision
FCOS final score = `sqrt(cls * centerness)`, but centerness was only supervised on
foreground cells. Background centerness was therefore unconstrained and multiplied tiny cls
probabilities into the ~50k low-confidence detections (P@0.1 ≈ 0.007). `_patch_fcos_loss` now
also trains background centerness toward 0 (mean-reduced, weighted 0.25), which collapses
background scores to ~0 and removes the flood.

### 6. Misc
- Fixed `scaled_lr` NameError in the FCOS SGD optimizer branch.
- Fixed the IoU-histogram binning bug in `tools/analyze_predictions.py` (all matches were
  landing in one bucket).

Retrain with the same commands (`python run_all.py`, or `python train_retinanet.py` /
`python train_fcos.py`), then re-run:
```bash
python tools/analyze_predictions.py --gt dataset/coco/val.json \
    --pred results/fcos/val_preds.json results/retinanet/val_preds.json
```
Success looks like: TP scores well above 0.3, fewer total predictions at the 0.05 floor, and a
rising ObsoleteTB AP.

### Fix Round 2 outcome (RetinaNet v2, batch 4 to fit 11 GB)

RetinaNet best mAP@0.5:0.95 rose 0.0477 → **0.0658** (epoch 75 = last epoch, still improving),
AP@0.5 0.123 → 0.174, ActiveTB AP 0.065 → 0.085, AR@100 0.187 → 0.237; scores moved off the
0.050 floor (top ~0.08) and localization stayed strong (both GT matched at IoU ≥ 0.58).
Remaining bottleneck: ObsoleteTB AP fell 0.030 → 0.0025 and TP confidence is still ~0.08 —
the inter-class discrimination problem, not geometry. Full numbers in `docs/retinanet_results.md`.
