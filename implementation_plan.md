# Fix Bounding Box Regression Stagnation

The `mAP=0.0000` issue in FCOS is now purely a bounding box regression problem. The classification head is successfully learning (scores are rising, labels match correctly), but the regression head is failing to produce boxes with IoU >= 0.50 (max IoU is stuck around 0.36).

## Root Cause Analysis
In the previous optimizer update, we split the parameters into three groups:
1. `backbone_params` (LR: 1e-5)
2. `pretrained_head_params` (LR: 1e-5)
3. `new_head_params` (LR: 1e-4)

The bounding box regression layer (`bbox_pred`) was placed into `pretrained_head_params` and restricted to an extremely low learning rate (`1e-5`). Because it was pretrained on COCO (which has mostly large objects), it naturally outputs large boxes. The `1e-5` learning rate is too slow to allow the regression head to adapt to the small, localized TB lesions within 10 epochs, resulting in massive false positives and 0 mAP.

## Proposed Changes

We will unify all head parameters into a single `head_params` group with the base learning rate (`1e-4`), allowing the regression layers to aggressively adapt to the TB dataset sizes while keeping the backbone frozen/slow at `1e-5`.

### [train_fcos.py](file:///home/asj/Internship/OBJECT_DETECTION/train_fcos.py)
- **[MODIFY]** Remove the three-tier optimizer grouping.
- **[MODIFY]** Group all parameters not containing "backbone", "body", or "fpn" into a single `head_params` list.
- **[MODIFY]** Apply `lr: scaled_lr` to `head_params`, and `lr: scaled_lr * 0.1` to `backbone_params`.

### [train_retinanet.py](file:///home/asj/Internship/OBJECT_DETECTION/train_retinanet.py)
- **[MODIFY]** Apply the exact same two-tier optimizer grouping (`backbone_params` vs `head_params`) to ensure the RetinaNet regression head can also learn effectively.

## Verification Plan
1. Apply the code edits to both scripts.
2. The user will restart `train_fcos.py` from scratch.
3. We expect the `max_iou_per_pred` to rapidly climb above 0.50 within the first 5-10 epochs, breaking the 0 mAP deadlock.

> [!NOTE]
> EfficientDet and DETR already use this two-tier approach (backbone vs. head), so they do not require this fix.
