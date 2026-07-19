#!/usr/bin/env python3
"""RetinaNet Training Pipeline — Research-Grade Implementation for TBX11K.

Usage:
    python train_retinanet.py
    python train_retinanet.py --epochs 50 --lr 0.001
    python train_retinanet.py --resume results/retinanet/weights/last_checkpoint.pth
    python train_retinanet.py --optimizer AdamW --batch-size 4
"""

import os
import sys
import json
import copy
import argparse

import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision.transforms as T
import torchvision.transforms.functional as TF

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
from utils.coco_dataset import CocoDetection, collate_fn
from utils.engine import (
    set_seed, MetricTracker, save_checkpoint, load_checkpoint,
    train_one_epoch, evaluate, evaluate_test,
    compute_confusion_matrix, save_confusion_matrix_plot,
    plot_training_curves,
)
from utils.ema import ModelEMA
from explain.gradcam import GradCAM, get_target_layer
from explain.visualize import overlay_heatmap, draw_detections

CLASS_NAMES = ["Background", "ActiveTuberculosis", "ObsoletePulmonaryTuberculosis"]

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_CONFIG = {
    "model": {
        "name": "RetinaNet-ResNet50-FPN-V2",
        "num_classes": 3,
        "use_custom_anchors": False,
        "anchor_sizes": ((16,), (32,), (64,), (128,), (256,)),
        "aspect_ratios": ((0.5, 1.0, 2.0),) * 5,
    },
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "lr": 0.01,
        "optimizer": "SGD",
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "clip_norm": 1.0,
        "warmup_epochs": 3,
        "ema_decay": 0.99,
        "early_stop_patience": 30,
        "save_every": 10,
        "seed": 42,
        "resume": None,
    },
    "augmentation": {
        "hflip_prob": 0.5,
        "brightness": 0.3,
        "contrast": 0.3,
        "gamma": 0.2,
        "noise_std": 0.05,
        "normalize": False,
    },
    "data": {
        "train_images": "dataset/coco/train",
        "val_images": "dataset/coco/val",
        "train_ann": "dataset/coco/train.json",
        "val_ann": "dataset/coco/val.json",
        "test_images": "dataset/coco/test",
        "test_ann": "dataset/coco/test.json",
        "num_workers": 2,
    },
    "results_dir": "results/retinanet",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# CLI Argument Parser
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="RetinaNet TB Detection Training")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--optimizer", type=str, default=None, choices=["SGD", "AdamW"])
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--clip-norm", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--no-custom-anchors", action="store_true",
                   help="Use default RetinaNet anchors instead of small-lesion anchors")
    args, _ = p.parse_known_args()
    return args


def get_config():
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    t = cfg["training"]
    m = cfg["model"]
    if args.epochs is not None: t["epochs"] = args.epochs
    if args.batch_size is not None: t["batch_size"] = args.batch_size
    if args.lr is not None: t["lr"] = args.lr
    if args.optimizer is not None: t["optimizer"] = args.optimizer
    if args.weight_decay is not None: t["weight_decay"] = args.weight_decay
    if args.momentum is not None: t["momentum"] = args.momentum
    if args.clip_norm is not None: t["clip_norm"] = args.clip_norm
    if args.warmup_epochs is not None: t["warmup_epochs"] = args.warmup_epochs
    if args.ema_decay is not None: t["ema_decay"] = args.ema_decay
    if args.early_stop_patience is not None:
        t["early_stop_patience"] = args.early_stop_patience
    if args.save_every is not None: t["save_every"] = args.save_every
    if args.seed is not None: t["seed"] = args.seed
    if args.resume is not None: t["resume"] = args.resume
    if args.results_dir is not None: cfg["results_dir"] = args.results_dir
    if args.no_custom_anchors: m["use_custom_anchors"] = False
    return cfg


# =============================================================================
# Data Augmentation Transforms
# =============================================================================
class AugmentedTransform:
    def __init__(self, train=True, cfg=None):
        self.train = train
        aug = (cfg or {}).get("augmentation", {})
        self.hflip_prob = aug.get("hflip_prob", 0.5)
        self.brightness = aug.get("brightness", 0.3)
        self.contrast = aug.get("contrast", 0.3)
        self.gamma_range = aug.get("gamma", 0.2)
        self.noise_std = aug.get("noise_std", 0.05)
        self.normalize = aug.get("normalize", True)

    def __call__(self, image, target):
        image = TF.to_tensor(image)

        if self.train:
            if torch.rand(1).item() < self.hflip_prob:
                image = TF.hflip(image)
                if len(target["boxes"]) > 0:
                    boxes = target["boxes"].clone()
                    w = image.shape[-1]
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    target["boxes"] = boxes

            if torch.rand(1).item() < 0.5:
                factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.brightness
                image = TF.adjust_brightness(image, factor)

            if torch.rand(1).item() < 0.5:
                factor = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.contrast
                image = TF.adjust_contrast(image, factor)

            if torch.rand(1).item() < 0.3:
                g = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.gamma_range
                image = image.clamp(min=1e-6).pow(g)

            if torch.rand(1).item() < 0.3:
                image = (image + torch.randn_like(image) * self.noise_std).clamp(0, 1)

        image = image.clamp(0, 1)

        if self.normalize:
            image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)

        return image, target


# =============================================================================
# Class-Frequency Weighted Sampler
# =============================================================================
def get_class_frequency_sampler(dataset):
    class_counts = {1: 0, 2: 0}
    image_labels = []
    
    for idx in range(len(dataset)):
        img_id = dataset.ids[idx]
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        labels = [ann['category_id'] for ann in anns]
        image_labels.append(labels)
        for lbl in labels:
            if lbl in class_counts:
                class_counts[lbl] += 1

    total = sum(class_counts.values()) or 1
    class_w = {k: total / (v + 1) for k, v in class_counts.items()}

    weights = []
    for labels in image_labels:
        if len(labels) == 0:
            weights.append(1.0)
        else:
            w = max(class_w.get(l, 1.0) for l in labels)
            weights.append(w)

    print(f"  Class counts: {class_counts}")
    print(f"  Class weights: { {k: f'{v:.2f}' for k, v in class_w.items()} }")
    return WeightedRandomSampler(weights, len(weights), replacement=True)


# =============================================================================
# Learning-Rate Scaling
# =============================================================================
def scale_lr(base_lr, batch_size, reference_bs=16):
    return base_lr * batch_size / reference_bs


# =============================================================================
# Build RetinaNet with optional custom anchors for small lesions
# =============================================================================
def build_retinanet(cfg):
    num_classes = cfg["model"]["num_classes"]
    use_custom = cfg["model"].get("use_custom_anchors", True)
    use_pretrained = False

    try:
        model = retinanet_resnet50_fpn_v2(weights="DEFAULT", score_thresh=0.05)
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
        in_features = model.head.classification_head.conv[0][0].in_channels
        
        if use_custom:
            anchor_sizes = cfg["model"]["anchor_sizes"]
            aspect_ratios = cfg["model"]["aspect_ratios"]
            model.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
            print(f"  Custom anchors: sizes={anchor_sizes}, ratios={aspect_ratios[0]}")
            n_anchors = len(anchor_sizes[0]) * len(aspect_ratios[0])
            print(f"  Total anchors per location: {n_anchors}")
        else:
            n_anchors = model.head.classification_head.num_anchors
            
        model.head.classification_head = RetinaNetClassificationHead(
            in_features, n_anchors, num_classes
        )
        if use_custom:
            model.head.regression_head = RetinaNetRegressionHead(
                in_features, n_anchors
            )
            
        use_pretrained = True
        print("  Loaded COCO-pretrained RetinaNet-V2 (backbone + neck) and replaced head")
    except Exception as e:
        print(f"  Failed to load full weights: {e}")
        try:
            model = retinanet_resnet50_fpn_v2(
                weights_backbone="DEFAULT", num_classes=num_classes, score_thresh=0.05,
            )
            if use_custom:
                anchor_sizes = cfg["model"]["anchor_sizes"]
                aspect_ratios = cfg["model"]["aspect_ratios"]
                model.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
            use_pretrained = True
            print("  Loaded backbone-only pretrained weights")
        except Exception:
            import warnings
            warnings.warn("Could not load pretrained weights, using random init.")
            model = retinanet_resnet50_fpn_v2(
                weights_backbone=None, num_classes=num_classes, score_thresh=0.05,
            )
            if use_custom:
                anchor_sizes = cfg["model"]["anchor_sizes"]
                aspect_ratios = cfg["model"]["aspect_ratios"]
                model.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    return model, use_pretrained


# =============================================================================
# Training
# =============================================================================
def train(cfg):
    results_dir = cfg["results_dir"]
    os.makedirs(f"{results_dir}/weights", exist_ok=True)
    os.makedirs(f"{results_dir}/curves", exist_ok=True)
    os.makedirs(f"{results_dir}/explain", exist_ok=True)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RetinaNet] Device: {device}")

    wandb.init(
        mode=os.environ.get("WANDB_MODE", "online"), project="tbx11k", name="retinanet",
        config=cfg,
    )

    train_dataset = CocoDetection(
        cfg["data"]["train_images"], cfg["data"]["train_ann"],
        transforms=AugmentedTransform(train=True, cfg=cfg),
    )
    val_dataset = CocoDetection(
        cfg["data"]["val_images"], cfg["data"]["val_ann"],
        transforms=AugmentedTransform(train=False, cfg=cfg),
    )

    batch_size = cfg["training"]["batch_size"]

    print("[RetinaNet] Building weighted sampler for class-imbalanced dataset...")
    sampler = get_class_frequency_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=collate_fn, num_workers=cfg["data"]["num_workers"],
        drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    model, use_pretrained = build_retinanet(cfg)
    model.to(device)

    ema = ModelEMA(model, decay=cfg["training"]["ema_decay"])

    scaled_lr = scale_lr(cfg["training"]["lr"], batch_size)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name or "body" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    if cfg["training"]["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": scaled_lr * 0.1},
                {"params": head_params, "lr": scaled_lr},
            ],
            weight_decay=cfg["training"]["weight_decay"],
        )
    else:
        optimizer = torch.optim.SGD(
            [
                {"params": backbone_params, "lr": scaled_lr * 0.1},
                {"params": head_params, "lr": scaled_lr},
            ],
            momentum=cfg["training"]["momentum"],
            weight_decay=cfg["training"]["weight_decay"],
        )

    epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-7,
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    scaler = None
    metric_tracker = MetricTracker()

    start_epoch = 1
    best_map = 0.0
    best_epoch = -1
    patience_counter = 0

    if cfg["training"]["resume"] and os.path.exists(cfg["training"]["resume"]):
        ckpt = load_checkpoint(
            cfg["training"]["resume"], model, optimizer, scaler, lr_scheduler,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        best_map = ckpt.get("metrics", {}).get("best_map", 0.0)
        best_epoch = ckpt.get("metrics", {}).get("best_epoch", -1)
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        print(
            f"[RetinaNet] Resumed from epoch {start_epoch}, "
            f"best mAP={best_map:.4f}"
        )

    print(
        f"[RetinaNet] Training {epochs} epochs, LR={scaled_lr:.6f}, "
        f"warmup={warmup_epochs}, optimizer={cfg['training']['optimizer']}"
    )

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            scaler=scaler, clip_norm=cfg["training"]["clip_norm"],
            metric_tracker=metric_tracker,
        )
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        ema.update(model)

        print(f"\n[RetinaNet] Validation after epoch {epoch}...")
        coco_eval = evaluate(ema.model, val_loader, device)

        current_map = 0.0
        if coco_eval is not None:
            current_map = coco_eval.stats[0]
            val_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "mAP@0.5:0.95": coco_eval.stats[0],
                "mAP@0.5": coco_eval.stats[1],
                "mAP@0.75": coco_eval.stats[2],
                "AR@10": coco_eval.stats[7],
                "learning_rate": current_lr,
            }
            wandb.log(val_metrics)
            metric_tracker.update(**val_metrics)

            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, epoch,
                    {"best_map": best_map, "best_epoch": best_epoch},
                    f"{results_dir}/weights/best_model.pth",
                    scaler=scaler, scheduler=lr_scheduler,
                    extra={"ema_state_dict": ema.state_dict()},
                )
                torch.save(
                    ema.state_dict(),
                    f"{results_dir}/weights/ema_model.pth",
                )
                print(
                    f"  New best model! mAP={best_map:.4f} (epoch {epoch})"
                )
            else:
                patience_counter += 1

        if epoch % cfg["training"]["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {"train_loss": train_loss, "mAP": current_map},
                f"{results_dir}/weights/epoch_{epoch:03d}.pth",
                scaler=scaler, scheduler=lr_scheduler,
            )

        save_checkpoint(
            model, optimizer, epoch,
            {"train_loss": train_loss, "mAP": current_map},
            f"{results_dir}/weights/last_checkpoint.pth",
            scaler=scaler, scheduler=lr_scheduler,
            extra={
                "best_map": best_map,
                "best_epoch": best_epoch,
                "ema_state_dict": ema.state_dict(),
            },
        )

        if patience_counter >= cfg["training"]["early_stop_patience"]:
            print(
                f"[RetinaNet] Early stopping at epoch {epoch} "
                f"(no improvement for {patience_counter} epochs)"
            )
            break

        print(
            f"  LR: {current_lr:.2e} | "
            f"Patience: {patience_counter}/{cfg['training']['early_stop_patience']}"
        )
        print()

    plot_training_curves(metric_tracker, f"{results_dir}/curves")
    with open(f"{results_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"\n[RetinaNet] Training complete. "
        f"Best mAP: {best_map:.4f} at epoch {best_epoch}"
    )
    return best_map, best_epoch


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(cfg):
    results_dir = cfg["results_dir"]
    print("\n[RetinaNet] Evaluating best model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = CocoDetection(
        cfg["data"]["val_images"], cfg["data"]["val_ann"],
        transforms=AugmentedTransform(train=False, cfg=cfg),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=cfg["data"]["num_workers"],
    )

    model, _ = build_retinanet(cfg)
    ema_path = f"{results_dir}/weights/ema_model.pth"
    best_path = f"{results_dir}/weights/best_model.pth"
    if os.path.exists(ema_path):
        model.load_state_dict(
            torch.load(ema_path, map_location=device, weights_only=True)
        )
        print("  Loaded EMA weights")
    elif os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
        print("  Loaded best model weights")
    model.to(device)
    model.eval()

    coco_eval = evaluate(
        model, val_loader, device,
        output_file=f"{results_dir}/val_preds.json",
    )

    if coco_eval is not None:
        metrics = {
            "mAP@0.5:0.95": coco_eval.stats[0],
            "mAP@0.5": coco_eval.stats[1],
            "mAP@0.75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
            "AR@1": coco_eval.stats[6],
            "AR@10": coco_eval.stats[7],
            "AR@100": coco_eval.stats[8],
        }
        with open(f"{results_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[RetinaNet] Metrics saved to {results_dir}/metrics.json")

    print("\n[RetinaNet] Running TTA evaluation...")
    coco_eval_tta = evaluate(
        model, val_loader, device,
        output_file=f"{results_dir}/val_preds_tta.json", tta=True,
    )
    if coco_eval_tta is not None:
        metrics_tta = {
            "mAP@0.5:0.95": coco_eval_tta.stats[0],
            "mAP@0.5": coco_eval_tta.stats[1],
            "mAP@0.75": coco_eval_tta.stats[2],
        }
        with open(f"{results_dir}/metrics_tta.json", "w") as f:
            json.dump(metrics_tta, f, indent=2)
        print(
            f"[RetinaNet] TTA metrics saved to {results_dir}/metrics_tta.json"
        )
        wandb.log({f'tta/{k}': v for k, v in metrics_tta.items()})

    confusion = compute_confusion_matrix(
        model, val_loader, device,
    )
    save_confusion_matrix_plot(
        confusion, f"{results_dir}/confusion_matrix.png",
    )

    test_ann = cfg["data"].get("test_ann", "dataset/coco/test.json")
    if os.path.isfile(test_ann):
        test_dataset = CocoDetection(
            cfg["data"].get("test_images", "dataset/coco/test"),
            test_ann,
            transforms=AugmentedTransform(train=False, cfg=cfg),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=cfg["training"]["batch_size"],
            shuffle=False, collate_fn=collate_fn,
            num_workers=cfg["data"]["num_workers"],
        )
        evaluate_test(
            model, test_loader, device,
            f"{results_dir}/test_preds.json",
            model_name=cfg.get("model", {}).get("name", "RetinaNet-ResNet50-FPN"),
        )
    else:
        print("[TEST] No annotations found. Running inference-only prediction.")


# =============================================================================
# XAI — Grad-CAM with TP / FP / FN Analysis
# =============================================================================
def run_xai(cfg):
    results_dir = cfg["results_dir"]
    print("\n[RetinaNet] Generating XAI explanations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_retinanet(cfg)
    ema_path = f"{results_dir}/weights/ema_model.pth"
    best_path = f"{results_dir}/weights/best_model.pth"
    if os.path.exists(ema_path):
        model.load_state_dict(
            torch.load(ema_path, map_location=device, weights_only=True)
        )
    elif os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    model.to(device)
    model.eval()

    xai_ann = cfg["data"].get("test_ann", "dataset/coco/test.json")
    xai_img_dir = cfg["data"].get("test_images", "dataset/coco/test")
    if not os.path.isfile(xai_ann):
        xai_ann = cfg["data"].get("val_ann", "dataset/coco/val.json")
        xai_img_dir = cfg["data"]["val_images"]
    if not os.path.exists(xai_ann):
        print("[RetinaNet] No dataset available for XAI.")
        return

    xai_dataset = CocoDetection(
        xai_img_dir, xai_ann,
        transforms=AugmentedTransform(train=False, cfg=cfg),
    )
    loader = DataLoader(
        xai_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    target_layer = get_target_layer(model, "retinanet")
    gradcam = GradCAM(model, target_layer)

    from torchvision.ops import box_iou as _box_iou

    categories = {1: "ActiveTB", 2: "ObsoleteTB"}
    xai_counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in categories}
    max_per_type = 3

    for images, targets in loader:
        if all(
            sum(v.values()) >= max_per_type * 3 for v in xai_counts.values()
        ):
            break

        image = images[0].unsqueeze(0).to(device)
        gt_boxes = targets[0]["boxes"].numpy()
        gt_labels = targets[0]["labels"].numpy()

        with torch.no_grad():
            outputs = model(image)
        pred_boxes = outputs[0]["boxes"].cpu().numpy()
        pred_scores = outputs[0]["scores"].cpu().numpy()
        pred_labels = outputs[0]["labels"].cpu().numpy()

        pred_type = ["FP"] * len(pred_boxes)
        matched_gt = set()
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            iou_mat = _box_iou(
                torch.tensor(gt_boxes), torch.tensor(pred_boxes)
            ).numpy()
            for gi in range(len(gt_boxes)):
                best_pi, best_iou = -1, 0.5
                for pi in range(len(pred_boxes)):
                    if iou_mat[gi, pi] > best_iou:
                        best_iou = iou_mat[gi, pi]
                        best_pi = pi
                if best_pi >= 0 and pred_labels[best_pi] == gt_labels[gi]:
                    pred_type[best_pi] = "TP"
                    matched_gt.add(gi)

        for pi, ptype in enumerate(pred_type):
            if pi >= len(pred_scores) or pred_scores[pi] < 0.3:
                continue
            cat = int(pred_labels[pi])
            if cat not in categories:
                continue
            if xai_counts[cat][ptype.lower()] >= max_per_type:
                continue

            with torch.set_grad_enabled(True):
                heatmap = gradcam.generate(image, target_class=cat)
            if heatmap is None:
                continue

            overlay = overlay_heatmap(image, heatmap)
            det = {
                "boxes": pred_boxes[pi : pi + 1],
                "scores": pred_scores[pi : pi + 1],
                "labels": pred_labels[pi : pi + 1],
            }
            overlay = draw_detections(overlay, det)
            name = categories[cat]
            n = xai_counts[cat][ptype.lower()]
            overlay.save(f"{results_dir}/explain/{name}_{ptype}_{n}.png")
            xai_counts[cat][ptype.lower()] += 1

        for gi in range(len(gt_boxes)):
            if gi in matched_gt:
                continue
            cat = int(gt_labels[gi])
            if cat not in categories:
                continue
            if xai_counts[cat]["fn"] >= max_per_type:
                continue

            with torch.set_grad_enabled(True):
                heatmap = gradcam.generate(image, target_class=cat)
            if heatmap is None:
                continue

            overlay = overlay_heatmap(image, heatmap)
            det = {
                "boxes": gt_boxes[gi : gi + 1],
                "scores": np.array([0.0]),
                "labels": gt_labels[gi : gi + 1],
            }
            overlay = draw_detections(overlay, det)
            name = categories[cat]
            n = xai_counts[cat]["fn"]
            overlay.save(f"{results_dir}/explain/{name}_FN_{n}.png")
            xai_counts[cat]["fn"] += 1

    gradcam.remove_hooks()
    print("[RetinaNet] XAI complete.")
    xai_wandb = {}
    for cat_id, counts in xai_counts.items():
        cat_name = categories[cat_id]
        print(
            f"  {cat_name}: "
            f"TP={counts['tp']} FP={counts['fp']} FN={counts['fn']}"
        )
        xai_wandb[f'xai/{cat_name}_TP'] = counts['tp']
        xai_wandb[f'xai/{cat_name}_FP'] = counts['fp']
        xai_wandb[f'xai/{cat_name}_FN'] = counts['fn']
    wandb.log(xai_wandb)
    for cat_id in categories:
        cat_name = categories[cat_id]
        for ptype in ['TP', 'FP', 'FN']:
            for n in range(max_per_type):
                img_path = f"{results_dir}/explain/{cat_name}_{ptype}_{n}.png"
                if os.path.exists(img_path):
                    wandb.log({f'xai_images/{cat_name}_{ptype}_{n}': wandb.Image(img_path)})


# =============================================================================
# Main
# =============================================================================
def main():
    cfg = get_config()
    train(cfg)
    evaluate_model(cfg)
    run_xai(cfg)


if __name__ == "__main__":
    main()
