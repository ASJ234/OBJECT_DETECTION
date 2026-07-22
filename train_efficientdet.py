#!/usr/bin/env python3
"""EfficientDet-D2 Training Pipeline — Research-Grade Implementation for TBX11K.

Usage:
    python train_efficientdet.py
    python train_efficientdet.py --epochs 50 --lr 1e-4
    python train_efficientdet.py --resume results/efficientdet/weights/last_checkpoint.pth
    python train_efficientdet.py --batch-size 4
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
import torchvision.transforms.functional as TF
import torch.nn.functional as F_interp

try:
    from effdet import create_model, DetBenchTrain
    HAS_EFFDET = True
except ImportError:
    HAS_EFFDET = False

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass
from utils.coco_dataset import CocoDetection, collate_fn
from utils.engine import (
    set_seed, MetricTracker, save_checkpoint, load_checkpoint,
    evaluate, evaluate_test,
    compute_confusion_matrix, save_confusion_matrix_plot,
    plot_training_curves,
)
from utils.ema import ModelEMA
from explain.gradcam import GradCAM
from explain.visualize import overlay_heatmap, draw_detections

CLASS_NAMES = {
    0: "Background", 1: "ActiveTuberculosis", 2: "ObsoletePulmonaryTuberculosis",
}
NUM_CLASSES = 3
IMG_SIZE = 768
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_CONFIG = {
    "model": {
        "name": "tf_efficientdet_d2",
        "num_classes": NUM_CLASSES,
        "img_size": IMG_SIZE,
    },
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "lr": 5e-3,
        "lr_backbone": 5e-4,
        "weight_decay": 1e-4,
        "clip_norm": 5.0,
        "warmup_epochs": 3,
        "ema_decay": 0.99,
        "early_stop_patience": 30,
        "save_every": 10,
        "seed": 42,
        "resume": None,
    },
    "augmentation": {
        "hflip_prob": 0.5,
        "brightness": 0.1,
        "contrast": 0.1,
        "gamma": 0.0,
        "noise_std": 0.0,
        "normalize": True,
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
    "results_dir": "results/efficientdet",
}


# =============================================================================
# CLI Argument Parser
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="EfficientDet-D2 TB Detection Training")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr-backbone", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--clip-norm", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    args, _ = p.parse_known_args()
    return args


def get_config():
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    t = cfg["training"]
    if args.epochs is not None: t["epochs"] = args.epochs
    if args.batch_size is not None: t["batch_size"] = args.batch_size
    if args.lr is not None: t["lr"] = args.lr
    if args.lr_backbone is not None: t["lr_backbone"] = args.lr_backbone
    if args.weight_decay is not None: t["weight_decay"] = args.weight_decay
    if args.clip_norm is not None: t["clip_norm"] = args.clip_norm
    if args.warmup_epochs is not None: t["warmup_epochs"] = args.warmup_epochs
    if args.ema_decay is not None: t["ema_decay"] = args.ema_decay
    if args.early_stop_patience is not None:
        t["early_stop_patience"] = args.early_stop_patience
    if args.save_every is not None: t["save_every"] = args.save_every
    if args.seed is not None: t["seed"] = args.seed
    if args.resume is not None: t["resume"] = args.resume
    if args.results_dir is not None: cfg["results_dir"] = args.results_dir
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
# EfficientDet Collate — resizes images AND scales bounding boxes
# =============================================================================
def make_collate_fn(img_size=IMG_SIZE):
    def collate_effdet(batch):
        images, targets = zip(*batch)
        orig_h, orig_w = images[0].shape[1], images[0].shape[2]
        scale_x = img_size / orig_w
        scale_y = img_size / orig_h

        resized = []
        scaled_targets = []
        for img, target in zip(images, targets):
            resized.append(
                F_interp.interpolate(
                    img.unsqueeze(0), size=(img_size, img_size),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            )
            new_target = {}
            for k, v in target.items():
                if k == "boxes" and len(v) > 0:
                    boxes = v.clone()
                    boxes[:, [0, 2]] *= scale_x
                    boxes[:, [1, 3]] *= scale_y
                    new_target[k] = boxes
                else:
                    new_target[k] = v
            scaled_targets.append(new_target)

        images = torch.stack(resized)

        cls_list = [t["labels"] for t in scaled_targets]
        box_list = [t["boxes"] for t in scaled_targets]
        device = box_list[0].device if len(box_list) > 0 else torch.device("cpu")
        batch_size = len(cls_list)
        max_boxes = max((len(c) for c in cls_list), default=1)
        max_boxes = max(max_boxes, 1)

        cls_pad = torch.zeros(batch_size, max_boxes, dtype=torch.int64, device=device)
        box_pad = torch.zeros(batch_size, max_boxes, 4, dtype=torch.float32, device=device)

        for i in range(batch_size):
            n = min(len(cls_list[i]), max_boxes)
            if n > 0:
                cls_pad[i, :n] = cls_list[i][:n]
                box_pad[i, :n] = box_list[i][:n]

        return images, {
            "cls": cls_pad,
            "bbox": box_pad,
            "img_scale": torch.ones(batch_size, 1, device=device),
            "img_size": torch.full(
                (batch_size, 2), img_size, dtype=torch.float32, device=device,
            ),
        }

    return collate_effdet


# =============================================================================
# EfficientDet → TorchVision output adapter
# =============================================================================
class EfficientDetWrapper(nn.Module):
    """Wraps raw EfficientDet to return TorchVision-format list of dicts."""

    def __init__(self, model, score_thresh=0.05):
        super().__init__()
        self.model = model
        self.score_thresh = score_thresh

    def forward(self, images):
        if isinstance(images, (list, tuple)):
            images = torch.stack(list(images))
        with torch.no_grad():
            detections = self.model(images)
        results = []
        for i in range(detections.shape[0]):
            dets = detections[i]
            valid = dets[:, 4] > self.score_thresh
            dets = dets[valid]
            results.append({
                "boxes": dets[:, :4],
                "scores": dets[:, 4],
                "labels": dets[:, 5].int(),
            })
        return results


# =============================================================================
# Build EfficientDet
# =============================================================================
def build_efficientdet(cfg):
    if not HAS_EFFDET:
        raise ImportError(
            "effdet not installed. Install with: pip install effdet"
        )
    num_classes = cfg["model"]["num_classes"]
    use_pretrained = False
    try:
        raw_model = create_model(
            "tf_efficientdet_d2", pretrained=True, num_classes=num_classes,
        )
        use_pretrained = True
        print("  Loaded COCO-pretrained EfficientDet-D2")
    except Exception:
        import warnings
        warnings.warn("Could not load pretrained weights, using random init. Disabling AMP.")
        raw_model = create_model(
            "tf_efficientdet_d2", pretrained=False, num_classes=num_classes,
        )
    return raw_model, use_pretrained


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
# Training
# =============================================================================
def train(cfg):
    results_dir = cfg["results_dir"]
    os.makedirs(f"{results_dir}/weights", exist_ok=True)
    os.makedirs(f"{results_dir}/curves", exist_ok=True)
    os.makedirs(f"{results_dir}/explain", exist_ok=True)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EfficientDet] Device: {device}")

    wandb.init(
        mode=os.environ.get("WANDB_MODE", "online"), project="tbx11k", name="efficientdet",
        config=cfg,
    )

    img_size = cfg["model"]["img_size"]
    collate_train = make_collate_fn(img_size)
    collate_val = make_collate_fn(img_size)

    train_dataset = CocoDetection(
        cfg["data"]["train_images"], cfg["data"]["train_ann"],
        transforms=AugmentedTransform(train=True, cfg=cfg),
    )
    val_dataset = CocoDetection(
        cfg["data"]["val_images"], cfg["data"]["val_ann"],
        transforms=AugmentedTransform(train=False, cfg=cfg),
    )

    batch_size = cfg["training"]["batch_size"]

    print("[EfficientDet] Building weighted sampler for class-imbalanced dataset...")
    sampler = get_class_frequency_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        collate_fn=collate_train, num_workers=cfg["data"]["num_workers"],
        drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_val, num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    raw_model, use_pretrained = build_efficientdet(cfg)
    raw_model.to(device)
    bench_train = DetBenchTrain(raw_model).to(device)

    ema = ModelEMA(raw_model, decay=cfg["training"]["ema_decay"])

    params = [p for p in raw_model.parameters() if p.requires_grad]
    backbone_params = [p for n, p in raw_model.named_parameters()
                       if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in raw_model.named_parameters()
                   if "backbone" not in n and p.requires_grad]

    lr_main = cfg["training"]["lr"]
    lr_bb = cfg["training"]["lr_backbone"]
    wd = cfg["training"]["weight_decay"]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_bb, "name": "backbone"},
        {"params": head_params, "lr": lr_main, "name": "head"},
    ], weight_decay=wd)

    print(f"  Optimizer: backbone lr={lr_bb:.1e}, head lr={lr_main:.1e}, wd={wd}")

    epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-8,
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
            cfg["training"]["resume"], raw_model, optimizer, scaler, lr_scheduler,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        best_map = ckpt.get("metrics", {}).get("best_map", 0.0)
        best_epoch = ckpt.get("metrics", {}).get("best_epoch", -1)
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        print(
            f"[EfficientDet] Resumed from epoch {start_epoch}, "
            f"best mAP={best_map:.4f}"
        )

    print(
        f"[EfficientDet] Training {epochs} epochs, batch={batch_size}, "
        f"img_size={img_size}, warmup={warmup_epochs}"
    )

    for epoch in range(start_epoch, epochs + 1):
        raw_model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = __import__("time").time()

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            for k in targets:
                if isinstance(targets[k], torch.Tensor):
                    targets[k] = targets[k].to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    out = bench_train(images, targets)
                    loss = out["loss"]
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), cfg["training"]["clip_norm"],
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                out = bench_train(images, targets)
                loss = out["loss"]
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    raw_model.parameters(), cfg["training"]["clip_norm"],
                )
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if i % 50 == 0 and num_batches > 0:
                lr = optimizer.param_groups[1]["lr"]
                elapsed = __import__("time").time() - start_time
                parts = [
                    f"Epoch [{epoch}] Step [{i}/{len(train_loader)}]",
                    f"Loss: {loss.item():.4f}",
                    f"LR: {lr:.2e}",
                    f"Time: {elapsed:.1f}s",
                ]
                print(" | ".join(parts))
                wandb.log({
                    "step_loss": loss.item(),
                    "step": i + epoch * len(train_loader),
                    "learning_rate": lr,
                })

        lr_scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        current_lr_bb = optimizer.param_groups[0]["lr"]
        current_lr_main = optimizer.param_groups[1]["lr"]

        ema.update(raw_model)

        print(f"\n[Evaluate] Validation after epoch {epoch}...")
        wrapper = EfficientDetWrapper(raw_model)
        wrapper_ema = EfficientDetWrapper(ema.model)
        coco_eval = evaluate(wrapper_ema, val_loader, device)

        current_map = 0.0
        if coco_eval is not None:
            current_map = coco_eval.stats[0]
            val_metrics = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "mAP@0.5:0.95": coco_eval.stats[0],
                "mAP@0.5": coco_eval.stats[1],
                "mAP@0.75": coco_eval.stats[2],
                "AR@10": coco_eval.stats[7],
                "lr_backbone": current_lr_bb,
                "lr_head": current_lr_main,
            }
            wandb.log(val_metrics)
            metric_tracker.update(**val_metrics)

            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    raw_model, optimizer, epoch,
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
                raw_model, optimizer, epoch,
                {"train_loss": avg_loss, "mAP": current_map},
                f"{results_dir}/weights/epoch_{epoch:03d}.pth",
                scaler=scaler, scheduler=lr_scheduler,
            )

        save_checkpoint(
            raw_model, optimizer, epoch,
            {"train_loss": avg_loss, "mAP": current_map},
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
                f"[EfficientDet] Early stopping at epoch {epoch} "
                f"(no improvement for {patience_counter} epochs)"
            )
            break

        print(
            f"  LR backbone: {current_lr_bb:.2e} | "
            f"LR head: {current_lr_main:.2e} | "
            f"Patience: {patience_counter}/{cfg['training']['early_stop_patience']}"
        )
        print()

    plot_training_curves(metric_tracker, f"{results_dir}/curves")
    with open(f"{results_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"\n[EfficientDet] Training complete. "
        f"Best mAP: {best_map:.4f} at epoch {best_epoch}"
    )
    return best_map, best_epoch


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(cfg):
    results_dir = cfg["results_dir"]
    print("\n[EfficientDet] Evaluating best model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = cfg["model"]["img_size"]
    collate_val = make_collate_fn(img_size)

    val_dataset = CocoDetection(
        cfg["data"]["val_images"], cfg["data"]["val_ann"],
        transforms=AugmentedTransform(train=False, cfg=cfg),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, collate_fn=collate_fn,
        num_workers=cfg["data"]["num_workers"],
    )

    raw_model, _ = build_efficientdet(cfg)
    ema_path = f"{results_dir}/weights/ema_model.pth"
    best_path = f"{results_dir}/weights/best_model.pth"
    if os.path.exists(ema_path):
        state = torch.load(ema_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            raw_model.load_state_dict(state["model_state_dict"])
        else:
            raw_model.load_state_dict(state)
        print("  Loaded EMA weights")
    elif os.path.exists(best_path):
        raw_model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
        print("  Loaded best model weights")
    raw_model.to(device)
    raw_model.eval()

    wrapper = EfficientDetWrapper(raw_model)

    coco_eval = evaluate(
        wrapper, val_loader, device,
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
        print(f"[EfficientDet] Metrics saved to {results_dir}/metrics.json")

    print("\n[EfficientDet] Running TTA evaluation...")
    coco_eval_tta = evaluate(
        wrapper, val_loader, device,
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
            f"[EfficientDet] TTA metrics saved to {results_dir}/metrics_tta.json"
        )
        wandb.log({f'tta/{k}': v for k, v in metrics_tta.items()})

    confusion = compute_confusion_matrix(
        wrapper, val_loader, device,
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
            wrapper, test_loader, device,
            f"{results_dir}/test_preds.json",
            model_name=cfg.get("model", {}).get("name", "EfficientDet-D2"),
        )
    else:
        print("[TEST] No annotations found. Running inference-only prediction.")


# =============================================================================
# XAI — Grad-CAM
# =============================================================================
def run_xai(cfg):
    results_dir = cfg["results_dir"]
    print("\n[EfficientDet] Generating XAI explanations...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_model, _ = build_efficientdet(cfg)
    ema_path = f"{results_dir}/weights/ema_model.pth"
    best_path = f"{results_dir}/weights/best_model.pth"
    if os.path.exists(ema_path):
        state = torch.load(ema_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            raw_model.load_state_dict(state["model_state_dict"])
        else:
            raw_model.load_state_dict(state)
    elif os.path.exists(best_path):
        raw_model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    raw_model.to(device)
    raw_model.eval()

    xai_ann = cfg["data"].get("test_ann", "dataset/coco/test.json")
    xai_img_dir = cfg["data"].get("test_images", "dataset/coco/test")
    if not os.path.isfile(xai_ann):
        xai_ann = cfg["data"].get("val_ann", "dataset/coco/val.json")
        xai_img_dir = cfg["data"]["val_images"]
    if not os.path.exists(xai_ann):
        print("[EfficientDet] No dataset available for XAI.")
        return

    img_size = cfg["model"]["img_size"]
    collate_val = make_collate_fn(img_size)
    xai_dataset = CocoDetection(
        xai_img_dir, xai_ann,
        transforms=AugmentedTransform(train=False, cfg=cfg),
    )
    loader = DataLoader(
        xai_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_val, num_workers=2,
    )

    wrapper = EfficientDetWrapper(raw_model, score_thresh=0.3)

    target_layer = raw_model.backbone.conv_head
    gradcam = GradCAM(raw_model, target_layer)

    count = 0
    max_samples = 5

    for images, targets in loader:
        if count >= max_samples:
            break
        image = images[0].unsqueeze(0).to(device)

        with torch.set_grad_enabled(True):
            heatmap = gradcam.generate(image)

        if heatmap is not None:
            overlay = overlay_heatmap(image, heatmap)
            outputs = wrapper(image)
            if outputs and len(outputs[0]["boxes"]) > 0:
                detections = {
                    "boxes": outputs[0]["boxes"].cpu().numpy(),
                    "scores": outputs[0]["scores"].cpu().numpy(),
                    "labels": outputs[0]["labels"].cpu().numpy(),
                }
                overlay = draw_detections(overlay, detections)
                overlay.save(f"{results_dir}/explain/sample_{count}.png")
                print(f"  Saved {results_dir}/explain/sample_{count}.png")
                count += 1

    gradcam.remove_hooks()
    print("[EfficientDet] XAI complete.")
    for n in range(max_samples):
        img_path = f"{results_dir}/explain/sample_{n}.png"
        if os.path.exists(img_path):
            wandb.log({f'xai_images/sample_{n}': wandb.Image(img_path)})


# =============================================================================
# Main
# =============================================================================
def main():
    if not HAS_EFFDET:
        print("[SKIP] effdet not installed — install with: pip install effdet")
        return
    cfg = get_config()
    train(cfg)
    evaluate_model(cfg)
    run_xai(cfg)


if __name__ == "__main__":
    main()
