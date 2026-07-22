"""
Dry-run diagnostic script for FCOS pipeline.
Tests: data loading, training loss, eval predictions, label/box formats.
Run: python dry_run_debug.py
"""
import os, sys, torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fcos_resnet50_fpn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.coco_dataset import CocoDetection, collate_fn

DATA_DIR = "dataset/coco"
TRAIN_ANN = os.path.join(DATA_DIR, "train.json")
VAL_ANN = os.path.join(DATA_DIR, "val.json")
TRAIN_IMGS = os.path.join(DATA_DIR, "train")
VAL_IMGS = os.path.join(DATA_DIR, "val")
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_data():
    """Step 1: Verify that data loads correctly and labels/boxes are sane."""
    print("=" * 60)
    print("STEP 1: DATA LOADING CHECK")
    print("=" * 60)

    # Simple transform: just to_tensor, no augmentation
    import torchvision.transforms.functional as TF
    def simple_transform(image, target):
        image = TF.to_tensor(image)
        return image, target

    dataset = CocoDetection(TRAIN_IMGS, TRAIN_ANN, transforms=simple_transform)
    print(f"  Train dataset size: {len(dataset)}")

    # Check a few samples
    label_counts = {}
    empty_count = 0
    for i in range(min(50, len(dataset))):
        img, tgt = dataset[i]
        if len(tgt['boxes']) == 0:
            empty_count += 1
            continue
        for lbl in tgt['labels'].tolist():
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # Sanity check box format (should be x1,y1,x2,y2)
        boxes = tgt['boxes']
        assert (boxes[:, 2] > boxes[:, 0]).all(), f"Sample {i}: x2 <= x1!"
        assert (boxes[:, 3] > boxes[:, 1]).all(), f"Sample {i}: y2 <= y1!"

    print(f"  Label distribution (first 50): {label_counts}")
    print(f"  Empty samples (no boxes): {empty_count}")

    # Check first annotated sample in detail
    for i in range(len(dataset)):
        img, tgt = dataset[i]
        if len(tgt['boxes']) > 0:
            print(f"\n  Sample {i} details:")
            print(f"    Image shape: {img.shape}")
            print(f"    Image pixel range: [{img.min():.3f}, {img.max():.3f}]")
            print(f"    Boxes: {tgt['boxes']}")
            print(f"    Labels: {tgt['labels']}")
            print(f"    Image ID: {tgt['image_id']}")
            break

    return dataset


def check_model_forward():
    """Step 2: Verify model forward pass in train and eval mode."""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL FORWARD PASS CHECK")
    print("=" * 60)

    model = fcos_resnet50_fpn(
        weights_backbone="DEFAULT", num_classes=NUM_CLASSES
    ).to(DEVICE)

    print(f"  Model score_thresh: {model.score_thresh}")
    print(f"  Model nms_thresh: {model.nms_thresh}")
    print(f"  Model detections_per_img: {model.detections_per_img}")
    print(f"  Classification head out_channels: {model.head.classification_head.cls_logits.out_channels}")
    print(f"  Classification head num_classes: {model.head.classification_head.num_classes}")

    # Test training forward
    img = torch.rand(3, 512, 512).to(DEVICE)
    target = {
        'boxes': torch.tensor([[100., 100., 300., 300.]]).to(DEVICE),
        'labels': torch.tensor([1]).to(DEVICE),
    }
    model.train()
    losses = model([img], [target])
    print(f"\n  Training losses: { {k: f'{v.item():.4f}' for k, v in losses.items()} }")

    # Test eval forward
    model.eval()
    with torch.no_grad():
        out = model([img])
    print(f"  Eval detections (random img): {len(out[0]['boxes'])}")

    return model


def dry_run_train(model, dataset):
    """Step 3: Train on a tiny subset and check if eval produces detections."""
    print("\n" + "=" * 60)
    print("STEP 3: DRY-RUN TRAINING (20 steps)")
    print("=" * 60)

    # Get a small subset with annotations
    annotated_indices = []
    for i in range(len(dataset)):
        _, tgt = dataset[i]
        if len(tgt['boxes']) > 0:
            annotated_indices.append(i)
        if len(annotated_indices) >= 4:
            break

    subset = Subset(dataset, annotated_indices)
    loader = DataLoader(subset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model.to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    model.train()
    for step in range(20):
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 5 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    # Now evaluate on the same images
    print("\n  Evaluating after training...")
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            for i, (out, tgt) in enumerate(zip(outputs, targets)):
                n_det = len(out['boxes'])
                print(f"\n  Image {tgt['image_id'].item()}:")
                print(f"    GT boxes: {tgt['boxes'][:3].tolist()}")
                print(f"    GT labels: {tgt['labels'][:3].tolist()}")
                print(f"    Pred count: {n_det}")
                if n_det > 0:
                    print(f"    Pred boxes: {out['boxes'][:3].cpu().tolist()}")
                    print(f"    Pred labels: {out['labels'][:3].cpu().tolist()}")
                    print(f"    Pred scores: {out['scores'][:3].cpu().tolist()}")
                else:
                    print(f"    *** NO DETECTIONS! Model score_thresh={model.score_thresh}")

                all_preds.append((tgt, out))

    return all_preds


def check_coco_eval_format(all_preds):
    """Step 4: Check if predictions match the COCO eval format correctly."""
    print("\n" + "=" * 60)
    print("STEP 4: COCO EVAL FORMAT CHECK")
    print("=" * 60)

    coco_gt = COCO(VAL_ANN)
    gt_cat_ids = coco_gt.getCatIds()
    print(f"  GT category IDs: {gt_cat_ids}")
    print(f"  GT categories: {[c['name'] for c in coco_gt.loadCats(gt_cat_ids)]}")

    # Check if any predictions have labels that don't match GT category IDs
    pred_labels = set()
    for tgt, out in all_preds:
        if len(out['labels']) > 0:
            pred_labels.update(out['labels'].cpu().tolist())
    print(f"  Predicted label set: {pred_labels}")
    print(f"  GT label set: {set(gt_cat_ids)}")

    if pred_labels and not pred_labels.issubset(set(gt_cat_ids)):
        print("  *** WARNING: Predicted labels DON'T match GT categories!")
        print("  *** This would cause COCO eval to ignore all predictions!")
    elif not pred_labels:
        print("  *** WARNING: No predictions at all — nothing to evaluate!")
    else:
        print("  OK: Predicted labels match GT categories")


def check_score_distribution():
    """Step 5: Check raw score distribution before thresholding."""
    print("\n" + "=" * 60)
    print("STEP 5: RAW SCORE DISTRIBUTION (bypassing score_thresh)")
    print("=" * 60)

    import torchvision.transforms.functional as TF
    def simple_transform(image, target):
        image = TF.to_tensor(image)
        return image, target

    val_dataset = CocoDetection(VAL_IMGS, VAL_ANN, transforms=simple_transform)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = fcos_resnet50_fpn(
        weights_backbone="DEFAULT", num_classes=NUM_CLASSES
    ).to(DEVICE)

    # LOWER the score threshold to see ALL detections
    model.score_thresh = 0.001  # Very low threshold
    model.detections_per_img = 300

    model.eval()
    total_dets = 0
    score_ranges = []
    label_dist = {}
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= 5:  # Just check 5 batches
                break
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)
            for out in outputs:
                n = len(out['boxes'])
                total_dets += n
                if n > 0:
                    score_ranges.append((out['scores'].min().item(), out['scores'].max().item()))
                    for l in out['labels'].cpu().tolist():
                        label_dist[l] = label_dist.get(l, 0) + 1

    print(f"  Total detections (with score_thresh=0.001): {total_dets}")
    if score_ranges:
        min_s = min(s[0] for s in score_ranges)
        max_s = max(s[1] for s in score_ranges)
        print(f"  Score range: [{min_s:.6f}, {max_s:.6f}]")
        print(f"  Label distribution: {label_dist}")
        if max_s < 0.05:
            print("  *** ROOT CAUSE: All scores are below 0.05!")
            print("  *** The evaluation threshold of 0.05 filters everything out!")
        elif max_s < 0.2:
            print("  *** ROOT CAUSE: All scores are below 0.2 (model.score_thresh)!")
            print("  *** The model's internal threshold filters everything out!")
    else:
        print("  *** No detections even with score_thresh=0.001!")
        print("  *** The model produces zero foreground proposals")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print()

    dataset = check_data()
    model = check_model_forward()
    preds = dry_run_train(model, dataset)
    check_coco_eval_format(preds)
    check_score_distribution()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
