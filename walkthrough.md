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
