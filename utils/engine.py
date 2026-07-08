import os
import json
import time
from collections import defaultdict

import torch
import torchvision
import numpy as np
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ['Background', 'ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler=None, print_freq=50):
    model.train()
    total_loss = 0
    loss_keys = None
    start_time = time.time()

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        optimizer.zero_grad()
        total_loss += losses.item()

        if loss_keys is None:
            loss_keys = list(loss_dict.keys())

        if i % print_freq == 0:
            elapsed = time.time() - start_time
            log = f'  Epoch [{epoch}] Step [{i}/{len(data_loader)}] Loss: {losses.item():.4f} | Time: {elapsed:.1f}s'
            print(log)

    avg_loss = total_loss / len(data_loader)
    print(f'  Epoch [{epoch}] Avg Loss: {avg_loss:.4f}')
    return avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device, output_file=None):
    model.eval()
    coco_gt = data_loader.dataset.coco

    results = []
    img_ids = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            img_id = target['image_id'].item()
            img_ids.append(img_id)

            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                results.append({
                    'image_id': img_id,
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score),
                    'category_id': int(label),
                })

    if not results:
        print('[EVAL] No predictions, skipping COCOeval.')
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = sorted(set(img_ids))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats.tolist()

    metrics = {
        'mAP@0.5:0.95': stats[0],
        'mAP@0.5': stats[1],
        'mAP@0.75': stats[2],
        'mAP_small': stats[3],
        'mAP_medium': stats[4],
        'mAP_large': stats[5],
        'AR@1': stats[6],
        'AR@10': stats[7],
        'AR@100': stats[8],
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({'metrics': metrics, 'predictions': results}, f, indent=2)
        print(f'[EVAL] Predictions saved to {output_file}')

    return coco_eval


@torch.no_grad()
def evaluate_test(model, data_loader, device, output_file):
    model.eval()
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            img_id = target['image_id'].item()
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                results.append({
                    'image_id': img_id,
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score),
                    'category_id': int(label),
                })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[TEST] {len(results)} predictions saved to {output_file}')


@torch.no_grad()
def compute_confusion_matrix(model, data_loader, device, iou_thresh=0.5, score_thresh=0.05):
    model.eval()
    confusion = np.zeros((3, 3), dtype=int)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)

        for pred, target in zip(outputs, targets):
            gt_boxes = target['boxes'].cpu()
            gt_labels = target['labels'].cpu()

            keep = pred['scores'] >= score_thresh
            pred_boxes = pred['boxes'][keep].cpu()
            pred_labels = pred['labels'][keep].cpu()

            matched_gt = set()
            matched_pred = set()

            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                iou_matrix = box_iou(gt_boxes, pred_boxes)
                for gt_idx in range(len(gt_boxes)):
                    best_pred = -1
                    best_iou = iou_thresh
                    for pred_idx in range(len(pred_boxes)):
                        if pred_idx not in matched_pred:
                            iou = iou_matrix[gt_idx, pred_idx].item()
                            if iou > best_iou:
                                best_iou = iou
                                best_pred = pred_idx

                    if best_pred >= 0:
                        gt_cls = gt_labels[gt_idx].item()
                        pred_cls = pred_labels[best_pred].item()
                        confusion[gt_cls, pred_cls] += 1
                        matched_gt.add(gt_idx)
                        matched_pred.add(best_pred)
                    else:
                        confusion[gt_labels[gt_idx].item(), 0] += 1
                        matched_gt.add(gt_idx)

            for gt_idx in range(len(gt_boxes)):
                if gt_idx not in matched_gt:
                    confusion[gt_labels[gt_idx].item(), 0] += 1

            for pred_idx in range(len(pred_boxes)):
                if pred_idx not in matched_pred:
                    confusion[0, pred_labels[pred_idx].item()] += 1

    return confusion


def save_confusion_matrix_plot(confusion, output_path, class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(7, 6))

    total_per_row = confusion.sum(axis=1, keepdims=True)
    total_per_row = np.where(total_per_row == 0, 1, total_per_row)
    confusion_norm = confusion / total_per_row

    sns.heatmap(confusion_norm, annot=confusion, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Fraction'})

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title(f'Confusion Matrix (IoU ≥ 0.5, score ≥ 0.05)\n'
                 f'TP=$\\Sigma$diag | FP=col0 | FN=row0')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'[CONFUSION] Saved to {output_path}')

    tp = confusion[1, 1] + confusion[2, 2]
    fp_bg = confusion[0, 1] + confusion[0, 2]
    fn_bg = confusion[1, 0] + confusion[2, 0]

    print(f'  TP (correct detections): {tp}')
    print(f'  FP (false positives):    {fp_bg}')
    print(f'  FN (false negatives):    {fn_bg}')

    for cls_idx in [1, 2]:
        denom_p = confusion[:, cls_idx].sum()
        denom_r = confusion[cls_idx, :].sum()
        prec = confusion[cls_idx, cls_idx] / denom_p if denom_p > 0 else 0
        rec = confusion[cls_idx, cls_idx] / denom_r if denom_r > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f'  {CLASS_NAMES[cls_idx]}: P={prec:.3f} R={rec:.3f} F1={f1:.3f}')

    return confusion
