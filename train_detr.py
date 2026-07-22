#!/usr/bin/env python3
"""DETR Training Pipeline — Research-Grade Implementation for TBX11K.

Usage:
    python train_detr.py
    python train_detr.py --epochs 200 --lr 1e-4
    python train_detr.py --resume results/detr/weights/last_checkpoint.pth
    python train_detr.py --grad-accum 4
"""

import os
import sys
import json
import copy
import argparse

import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DetrForObjectDetection
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
from explain.detr_attention import DETRAttentionExtractor
CLASS_NAMES = {0: "Background", 1: "ActiveTuberculosis", 2: "ObsoletePulmonaryTuberculosis"}
CLASS_COLORS = {1: (255, 0, 0), 2: (0, 200, 255)}
NUM_CLASSES = 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_CONFIG = {
    "model": {
        "name": "DETR-ResNet50",
        "num_classes": NUM_CLASSES,
    },
    "training": {
        "epochs": 150,
        "batch_size": 8,
        "lr_backbone": 1e-5,
        "lr_transformer": 1e-4,
        "lr_head": 1e-4,
        "weight_decay": 1e-4,
        "clip_norm": 5.0,
        "warmup_epochs": 10,
        "ema_decay": 0.99,
        "early_stop_patience": 30,
        "save_every": 10,
        "seed": 42,
        "resume": None,
        "gradient_accumulation_steps": 4,
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
    "results_dir": "results/detr",
}


# =============================================================================
# CLI Argument Parser
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="DETR TB Detection Training")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr-backbone", type=float, default=None)
    p.add_argument("--lr-transformer", type=float, default=None)
    p.add_argument("--lr-head", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--clip-norm", type=float, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--save-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Gradient accumulation steps (default: 4, effective batch=16)")
    args, _ = p.parse_known_args()
    return args


def get_config():
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    t = cfg["training"]
    if args.epochs is not None: t["epochs"] = args.epochs
    if args.batch_size is not None: t["batch_size"] = args.batch_size
    if args.lr_backbone is not None: t["lr_backbone"] = args.lr_backbone
    if args.lr_transformer is not None: t["lr_transformer"] = args.lr_transformer
    if args.lr_head is not None: t["lr_head"] = args.lr_head
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
    if args.grad_accum is not None: t["gradient_accumulation_steps"] = args.grad_accum
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
# HuggingFace DETR Wrapper — adapts HF DETR to torchvision-style API
# =============================================================================
class DetrHFWrapper(torch.nn.Module):
    """Wraps HuggingFace DetrForObjectDetection to match torchvision detection API.

    Training: model(images, targets) -> dict of losses
    Eval:     model(images) -> list of {"boxes", "scores", "labels"}
    """

    def __init__(self, hf_model, num_classes, id2label):
        super().__init__()
        self.hf_model = hf_model
        self.num_classes = num_classes
        self.id2label = id2label
        self._training = False

    def train(self, mode=True):
        self._training = mode
        self.hf_model.train(mode)
        return self

    def eval(self):
        self._training = False
        self.hf_model.eval()
        return self

    def parameters(self, recurse=True):
        return self.hf_model.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.hf_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.hf_model.load_state_dict(state_dict, *args, **kwargs)

    def forward(self, images, targets=None):
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = [images]
            elif images.dim() == 4:
                images = list(images)
        if self._training and targets is not None:
            return self._forward_train(images, targets)
        else:
            return self._forward_eval(images)

    def _prepare_pixel_values(self, images):
        """Convert list of CxHxW tensors to pixel_values batch for HF DETR."""
        pixel_values = torch.stack(images, dim=0)
        return pixel_values

    def _forward_train(self, images, targets):
        pixel_values = self._prepare_pixel_values(images)
        labels = []
        for t in targets:
            h, w = images[0].shape[1], images[0].shape[2]
            box_norm = t["boxes"].clone()
            # Convert xyxy absolute -> cxcywh normalized
            x1, y1, x2, y2 = box_norm[:, 0], box_norm[:, 1], box_norm[:, 2], box_norm[:, 3]
            box_norm[:, 0] = (x1 + x2) / (2.0 * w)   # cx
            box_norm[:, 1] = (y1 + y2) / (2.0 * h)   # cy
            box_norm[:, 2] = (x2 - x1) / w            # w
            box_norm[:, 3] = (y2 - y1) / h            # h
            labels.append({
                "class_labels": t["labels"] - 1,
                "boxes": box_norm,
                "image_id": t["image_id"],
                "area": t["area"],
                "iscrowd": t["iscrowd"],
            })
        outputs = self.hf_model(pixel_values=pixel_values, labels=labels)
        loss_dict = outputs.loss_dict
        return dict(loss_dict)

    @torch.no_grad()
    def _forward_eval(self, images):
        pixel_values = self._prepare_pixel_values(images)
        outputs = self.hf_model(pixel_values=pixel_values)
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        h, w = images[0].shape[1], images[0].shape[2]
        results = []
        for i in range(logits.shape[0]):
            probs = torch.softmax(logits[i], dim=-1)
            scores, labels = probs[:, :-1].max(dim=-1)
            labels = labels + 1
            boxes = pred_boxes[i]
            boxes_abs = boxes.clone()
            boxes_abs[:, 0] *= w
            boxes_abs[:, 2] *= w
            boxes_abs[:, 1] *= h
            boxes_abs[:, 3] *= h
            x1y1x2y2 = torch.zeros_like(boxes_abs)
            x1y1x2y2[:, 0] = boxes_abs[:, 0] - boxes_abs[:, 2] / 2
            x1y1x2y2[:, 1] = boxes_abs[:, 1] - boxes_abs[:, 3] / 2
            x1y1x2y2[:, 2] = boxes_abs[:, 0] + boxes_abs[:, 2] / 2
            x1y1x2y2[:, 3] = boxes_abs[:, 1] + boxes_abs[:, 3] / 2
            results.append({
                "boxes": x1y1x2y2,
                "scores": scores,
                "labels": labels,
            })
        return results


# =============================================================================
# Build DETR with COCO-pretrained weights
# =============================================================================
def build_detr(cfg):
    num_classes = cfg["model"]["num_classes"]
    id2label = {0: "ActiveTuberculosis", 1: "ObsoletePulmonaryTuberculosis"}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    hf_model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    print("  Loaded HuggingFace DETR-ResNet50 (COCO-pretrained)")
    wrapper = DetrHFWrapper(hf_model, num_classes, id2label)
    return wrapper, True


# =============================================================================
# Separate parameter groups: backbone vs transformer vs head
# =============================================================================
def get_param_groups(model, cfg):
    backbone_params = []
    transformer_params = []
    head_params = []

    raw = model.hf_model if hasattr(model, "hf_model") else model
    for name, param in raw.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        elif "encoder" in name or "decoder" in name or "query_position" in name or "input_projection" in name:
            transformer_params.append(param)
        else:
            head_params.append(param)

    lr_bb = cfg["training"]["lr_backbone"]
    lr_tr = cfg["training"]["lr_transformer"]
    lr_hd = cfg["training"]["lr_head"]
    wd = cfg["training"]["weight_decay"]

    param_groups = [
        {"params": backbone_params, "lr": lr_bb, "name": "backbone"},
        {"params": transformer_params, "lr": lr_tr, "name": "transformer"},
        {"params": head_params, "lr": lr_hd, "name": "head"},
    ]

    print(f"  Param groups: backbone={len(backbone_params)} ({lr_bb:.1e}), "
          f"transformer={len(transformer_params)} ({lr_tr:.1e}), "
          f"head={len(head_params)} ({lr_hd:.1e})")

    return param_groups


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
    print(f"[DETR] Device: {device}")

    wandb.init(
        mode=os.environ.get("WANDB_MODE", "online"), project="tbx11k", name="detr",
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

    print("[DETR] Building weighted sampler for class-imbalanced dataset...")
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

    model, use_pretrained = build_detr(cfg)
    model.to(device)

    ema = ModelEMA(model, decay=cfg["training"]["ema_decay"])

    param_groups = get_param_groups(model, cfg)
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"]["warmup_epochs"]
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs,
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
    grad_accum = cfg["training"]["gradient_accumulation_steps"]

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
            f"[DETR] Resumed from epoch {start_epoch}, "
            f"best mAP={best_map:.4f}"
        )

    effective_bs = batch_size * grad_accum
    print(
        f"[DETR] Training {epochs} epochs, batch={batch_size}, "
        f"grad_accum={grad_accum}, effective_bs={effective_bs}, "
        f"warmup={warmup_epochs}"
    )

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            scaler=scaler, clip_norm=cfg["training"]["clip_norm"],
            metric_tracker=metric_tracker,
            gradient_accumulation_steps=grad_accum,
        )
        lr_scheduler.step()
        current_lr_bb = optimizer.param_groups[0]["lr"]
        current_lr_tr = optimizer.param_groups[1]["lr"]

        ema.update(model)

        print(f"\n[DETR] Validation after epoch {epoch}...")
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
                "lr_backbone": current_lr_bb,
                "lr_transformer": current_lr_tr,
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
                f"[DETR] Early stopping at epoch {epoch} "
                f"(no improvement for {patience_counter} epochs)"
            )
            break

        print(
            f"  LR backbone: {current_lr_bb:.2e} | "
            f"LR transformer: {current_lr_tr:.2e} | "
            f"Patience: {patience_counter}/{cfg['training']['early_stop_patience']}"
        )
        print()

    plot_training_curves(metric_tracker, f"{results_dir}/curves")
    with open(f"{results_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(
        f"\n[DETR] Training complete. "
        f"Best mAP: {best_map:.4f} at epoch {best_epoch}"
    )
    return best_map, best_epoch


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(cfg):
    results_dir = cfg["results_dir"]
    print("\n[DETR] Evaluating best model...")
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

    model, _ = build_detr(cfg)
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
        print(f"[DETR] Metrics saved to {results_dir}/metrics.json")

    print("\n[DETR] Running TTA evaluation...")
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
        print(f"[DETR] TTA metrics saved to {results_dir}/metrics_tta.json")
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
            model_name=cfg.get("model", {}).get("name", "DETR-ResNet50"),
        )
    else:
        print("[TEST] No annotations found. Running inference-only prediction.")


# =============================================================================
# XAI — DETR Attention with Query Visualization
# =============================================================================
def run_xai(cfg):
    results_dir = cfg["results_dir"]
    print("\n[DETR] Generating XAI attention maps...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = build_detr(cfg)
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
        print("[DETR] No dataset available for XAI.")
        return

    xai_dataset = CocoDetection(
        xai_img_dir, xai_ann,
        transforms=AugmentedTransform(train=False, cfg=cfg),
    )
    loader = DataLoader(
        xai_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage, ImageDraw

    count = 0
    max_samples = 5

    for images, targets in loader:
        if count >= max_samples:
            break
        image = images[0].unsqueeze(0).to(device)

        extractor = DETRAttentionExtractor(model)
        heatmaps, meta = extractor.extract(image)
        extractor.remove_hook()

        if not heatmaps or len(heatmaps) == 0:
            continue

        img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN))
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

        model.eval()
        with torch.no_grad():
            outputs = model(image)
        boxes = outputs[0]["boxes"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        gt_boxes = targets[0]["boxes"].numpy()
        gt_labels = targets[0]["labels"].numpy()

        n_queries_show = min(len(heatmaps), 6)
        n_cols = 2 + n_queries_show
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        axes[0].imshow(img_np)
        draw_img = PILImage.fromarray(img_np.copy())
        draw = ImageDraw.Draw(draw_img)
        for box, sc, lb in zip(boxes, scores, labels):
            if sc < 0.3:
                continue
            x1, y1, x2, y2 = box[:4]
            color = CLASS_COLORS.get(int(lb), (255, 255, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text(
                (x1 + 2, y1 - 16),
                f'{CLASS_NAMES.get(int(lb), str(lb))}: {sc:.2f}',
                fill=color,
            )
        axes[0].imshow(np.array(draw_img))
        axes[0].set_title("DETR — Detections", fontsize=10)
        axes[0].axis("off")

        gt_img = PILImage.fromarray(img_np.copy())
        gt_draw = ImageDraw.Draw(gt_img)
        for box, lb in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box[:4]
            color = CLASS_COLORS.get(int(lb), (255, 255, 0))
            gt_draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            gt_draw.text(
                (x1 + 2, y1 - 14),
                f'GT: {CLASS_NAMES.get(int(lb), str(lb))}',
                fill=color,
            )
        axes[1].imshow(np.array(gt_img))
        axes[1].set_title("Ground Truth", fontsize=10)
        axes[1].axis("off")

        for qi in range(n_queries_show):
            attn = heatmaps[qi]
            attn_resized = np.array(
                PILImage.fromarray((attn * 255).astype(np.uint8)).resize(
                    (img_np.shape[1], img_np.shape[0]), PILImage.BILINEAR
                )
            ) / 255.0
            cmap = plt.get_cmap("jet")
            attn_colored = cmap(attn_resized)[:, :, :3]
            combined = (1 - 0.5) * img_np / 255 + 0.5 * attn_colored
            axes[2 + qi].imshow(np.clip(combined, 0, 1))

            m = meta[qi] if qi < len(meta) else {}
            label_name = CLASS_NAMES.get(m.get("label", -1), "bg")
            score_val = m.get("score", 0.0)
            axes[2 + qi].set_title(
                f'Q{qi}: {label_name}\n{score_val:.2f}', fontsize=9,
            )
            axes[2 + qi].axis("off")

        plt.suptitle(
            f"DETR Attention Analysis (epoch={cfg['training']['epochs']})",
            fontsize=12,
        )
        plt.tight_layout()
        out_path = f"{results_dir}/explain/sample_{count}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")

        query_info = []
        for qi in range(min(len(heatmaps), len(meta))):
            m = meta[qi]
            query_info.append({
                "query": qi,
                "label": CLASS_NAMES.get(m["label"], "unknown"),
                "score": m["score"],
                "box": m["box"],
            })
        with open(f"{results_dir}/explain/queries_{count}.json", "w") as f:
            json.dump(query_info, f, indent=2)

        count += 1

    print(f"[DETR] XAI complete. {count} samples analyzed.")
    for n in range(max_samples):
        img_path = f"{results_dir}/explain/sample_{n}.png"
        if os.path.exists(img_path):
            wandb.log({f'xai_images/sample_{n}': wandb.Image(img_path)})


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
