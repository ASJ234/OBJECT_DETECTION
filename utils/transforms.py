"""Shared augmentation transforms for all detection models."""

import math
import random

import torch
import torchvision.transforms.functional as TF


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _rotate_boxes(boxes, angle, h, w):
    """Rotate boxes by angle (degrees) around center, return xyxy."""
    if len(boxes) == 0:
        return boxes
    cx = w / 2.0
    cy = h / 2.0
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners_x = torch.stack([x1, x2, x1, x2], dim=1)
    corners_y = torch.stack([y1, y2, y2, y1], dim=1)
    dx = corners_x - cx
    dy = corners_y - cy
    new_x = cx + dx * cos_a - dy * sin_a
    new_y = cy + dx * sin_a + dy * cos_a
    new_x1 = new_x.min(dim=1).values
    new_y1 = new_y.min(dim=1).values
    new_x2 = new_x.max(dim=1).values
    new_y2 = new_y.max(dim=1).values
    boxes_out = torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)
    boxes_out[:, 0] = boxes_out[:, 0].clamp(0, w)
    boxes_out[:, 1] = boxes_out[:, 1].clamp(0, h)
    boxes_out[:, 2] = boxes_out[:, 2].clamp(0, w)
    boxes_out[:, 3] = boxes_out[:, 3].clamp(0, h)
    return boxes_out


def _translate_boxes(boxes, tx, ty, w, h):
    """Translate boxes by (tx, ty) pixels."""
    if len(boxes) == 0:
        return boxes
    boxes_out = boxes.clone()
    boxes_out[:, [0, 2]] += tx
    boxes_out[:, [1, 3]] += ty
    boxes_out[:, 0] = boxes_out[:, 0].clamp(0, w)
    boxes_out[:, 1] = boxes_out[:, 1].clamp(0, h)
    boxes_out[:, 2] = boxes_out[:, 2].clamp(0, w)
    boxes_out[:, 3] = boxes_out[:, 3].clamp(0, h)
    return boxes_out


def _scale_boxes(boxes, scale_x, scale_y, w, h):
    """Scale boxes around center by (scale_x, scale_y)."""
    if len(boxes) == 0:
        return boxes
    cx = w / 2.0
    cy = h / 2.0
    boxes_out = boxes.clone()
    boxes_out[:, 0] = cx + (boxes[:, 0] - cx) * scale_x
    boxes_out[:, 1] = cy + (boxes[:, 1] - cy) * scale_y
    boxes_out[:, 2] = cx + (boxes[:, 2] - cx) * scale_x
    boxes_out[:, 3] = cy + (boxes[:, 3] - cy) * scale_y
    boxes_out[:, 0] = boxes_out[:, 0].clamp(0, w)
    boxes_out[:, 1] = boxes_out[:, 1].clamp(0, h)
    boxes_out[:, 2] = boxes_out[:, 2].clamp(0, w)
    boxes_out[:, 3] = boxes_out[:, 3].clamp(0, h)
    return boxes_out


class SharedAugmentedTransform:
    """Train/test transform with medical-imaging-appropriate augmentations.

    Includes spatial augmentations (rotation, translation, scale) to
    improve generalization, especially for rare classes.
    """

    def __init__(self, train=True, cfg=None):
        self.train = train
        aug = (cfg or {}).get("augmentation", {})
        self.hflip_prob = aug.get("hflip_prob", 0.5)
        self.brightness = aug.get("brightness", 0.3)
        self.contrast = aug.get("contrast", 0.3)
        self.gamma_range = aug.get("gamma", 0.2)
        self.noise_std = aug.get("noise_std", 0.05)
        self.normalize = aug.get("normalize", True)

        self.rotation_deg = aug.get("rotation_deg", 10)
        self.rotation_prob = aug.get("rotation_prob", 0.3)
        self.translate_ratio = aug.get("translate_ratio", 0.1)
        self.translate_prob = aug.get("translate_prob", 0.3)
        self.scale_range = aug.get("scale_range", 0.3)
        self.scale_prob = aug.get("scale_prob", 0.3)

    def __call__(self, image, target):
        image = TF.to_tensor(image)
        _, h, w = image.shape

        if self.train:
            if torch.rand(1).item() < self.hflip_prob:
                image = TF.hflip(image)
                if len(target["boxes"]) > 0:
                    boxes = target["boxes"].clone()
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                    target["boxes"] = boxes

            if torch.rand(1).item() < self.rotation_prob:
                angle = (torch.rand(1).item() - 0.5) * 2 * self.rotation_deg
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                target["boxes"] = _rotate_boxes(target["boxes"], angle, h, w)

            if torch.rand(1).item() < self.translate_prob:
                tx = (torch.rand(1).item() - 0.5) * 2 * self.translate_ratio * w
                ty = (torch.rand(1).item() - 0.5) * 2 * self.translate_ratio * h
                image = TF.affine(image, angle=0, translate=(tx, ty), scale=1.0, shear=0, fill=0)
                target["boxes"] = _translate_boxes(target["boxes"], tx, ty, w, h)

            if torch.rand(1).item() < self.scale_prob:
                s = 1.0 + (torch.rand(1).item() - 0.5) * 2 * self.scale_range
                image = TF.affine(image, angle=0, translate=(0, 0), scale=s, shear=0, fill=0)
                target["boxes"] = _scale_boxes(target["boxes"], s, s, w, h)

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
