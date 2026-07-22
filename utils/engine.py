# =============================================================================
# Section 1: Imports, utility classes, reproducibility, metric trackers, checkpoints
# =============================================================================
import os
import json
import csv
import time
import gc
import random
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import box_iou, nms
from scipy.optimize import linear_sum_assignment
from pycocotools.cocoeval import COCOeval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

CLASS_NAMES = ['Background', 'ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']
NUM_CLASSES = 3

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def gpu_cleanup(*models):
    for m in models:
        del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MetricTracker:
    def __init__(self):
        self.history = defaultdict(list)

    def update(self, **metrics):
        for k, v in metrics.items():
            self.history[k].append(v)

    def average(self, last_n=None):
        avgs = {}
        for k, vals in self.history.items():
            subset = vals[-last_n:] if last_n else vals
            avgs[k] = sum(subset) / len(subset) if subset else 0.0
        return avgs

    def best(self, metric, mode='max'):
        vals = self.history.get(metric, [])
        if not vals:
            return None, None
        idx = int(np.argmax(vals)) if mode == 'max' else int(np.argmin(vals))
        return vals[idx], idx

    def state_dict(self):
        return dict(self.history)

    def load_state_dict(self, d):
        self.history = defaultdict(list, {k: list(v) for k, v in d.items()})


def save_checkpoint(model, optimizer, epoch, metrics, path,
                    scaler=None, scheduler=None, extra=None):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    if scaler is not None:
        state['scaler_state_dict'] = scaler.state_dict()
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    if extra is not None:
        state.update(extra)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)
    logger.info(f'Checkpoint saved: {path}')


def load_checkpoint(path, model, optimizer=None, scaler=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scaler is not None and 'scaler_state_dict' in ckpt:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    logger.info(f'Checkpoint loaded: {path} (epoch {ckpt.get("epoch", "?")})')
    return ckpt


# =============================================================================
# Section 2: Production-grade train_one_epoch
# =============================================================================
def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    scaler=None, print_freq=50, clip_norm=None,
                    metric_tracker=None, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0.0
    loss_keys = None
    loss_accum = defaultdict(float)
    num_batches = 0
    start_time = time.time()
    grad_norms = []
    accum_count = 0

    optimizer.zero_grad()

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values()) / gradient_accumulation_steps
            if not torch.isfinite(losses):
                logger.warning(f'NaN/Inf loss at step {i}, skipping batch')
                for k, v in loss_dict.items():
                    logger.warning(f'  {k}: {v.item():.6f}')
                optimizer.zero_grad()
                continue
            scaler.scale(losses).backward()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values()) / gradient_accumulation_steps
            if not torch.isfinite(losses):
                logger.warning(f'NaN/Inf loss at step {i}, skipping batch')
                for k, v in loss_dict.items():
                    logger.warning(f'  {k}: {v.item():.6f}')
                optimizer.zero_grad()
                continue
            losses.backward()

        accum_count += 1
        total_loss += losses.item() * gradient_accumulation_steps
        num_batches += 1

        if loss_keys is None:
            loss_keys = list(loss_dict.keys())
        for k, v in loss_dict.items():
            loss_accum[k] += v.item()

        if accum_count % gradient_accumulation_steps == 0:
            if clip_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), clip_norm)
                grad_norms.append(grad_norm.item())
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if i % print_freq == 0 and num_batches > 0:
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            parts = [f'Epoch [{epoch}] Step [{i}/{len(data_loader)}]',
                     f'Loss: {losses.item() * gradient_accumulation_steps:.4f}',
                     f'LR: {lr:.2e}',
                     f'Time: {elapsed:.1f}s']
            if gradient_accumulation_steps > 1:
                parts.append(f'Accum: {accum_count}/{gradient_accumulation_steps}')
            print(' | '.join(parts))
            wandb.log({
                'step_loss': losses.item() * gradient_accumulation_steps,
                'step': i + epoch * len(data_loader),
                'learning_rate': lr,
            })

    avg_loss = total_loss / max(num_batches, 1)
    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

    epoch_metrics = {'epoch_loss': avg_loss, 'avg_grad_norm': avg_grad_norm}
    if loss_keys:
        for k in loss_keys:
            epoch_metrics[f'loss_{k}'] = loss_accum[k] / max(num_batches, 1)

    wandb.log(epoch_metrics)
    if metric_tracker is not None:
        metric_tracker.update(**epoch_metrics)

    print(f'  Epoch [{epoch}] Avg Loss: {avg_loss:.4f} | '
          f'Grad Norm: {avg_grad_norm:.4f}')
    return avg_loss


# =============================================================================
# Section 3: COCO evaluation with per-class metrics, CSV, optional TTA
# =============================================================================
def _tta_forward(model, images):
    outputs = model(images)
    flipped = [torch.flip(img, dims=[-1]) for img in images]
    flipped_outputs = model(flipped)
    merged = []
    for out, f_out in zip(outputs, flipped_outputs):
        if len(f_out['boxes']) == 0:
            merged.append(out)
            continue
        img_w = float(out['boxes'][:, 2].max()) + 1.0 if len(out['boxes']) > 0 else 1.0
        f_boxes = f_out['boxes'].clone()
        f_boxes[:, [0, 2]] = img_w - f_boxes[:, [2, 0]]
        all_boxes = torch.cat([out['boxes'], f_boxes])
        all_scores = torch.cat([out['scores'], f_out['scores']])
        all_labels = torch.cat([out['labels'], f_out['labels']])
        keep = nms(all_boxes, all_scores, iou_threshold=0.5)
        merged.append({
            'boxes': all_boxes[keep], 'scores': all_scores[keep],
            'labels': all_labels[keep],
        })
    return merged


def _compute_per_class_ap(coco_gt, coco_dt, img_ids):
    per_class = {}
    cat_ids = coco_gt.getCatIds()
    for cat_id in cat_ids:
        ce = COCOeval(coco_gt, coco_dt, iouType='bbox')
        ce.params.imgIds = img_ids
        ce.params.catIds = [cat_id]
        ce.evaluate()
        ce.accumulate()
        ce.summarize()
        cat_name = coco_gt.loadCats(cat_id)[0]['name']
        per_class[f'AP_{cat_name}'] = float(ce.stats[0]) if len(ce.stats) > 0 else 0.0
    return per_class


def _append_csv(path, **kwargs):
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(kwargs.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(kwargs)


@torch.no_grad()
def evaluate(model, data_loader, device, output_file=None, tta=False):
    model.eval()
    coco_gt = data_loader.dataset.coco
    results = []
    img_ids = []
    _debug_done = False

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = _tta_forward(model, images) if tta else model(images)

        if not _debug_done:
            for bi in range(len(outputs)):
                out = outputs[bi]
                tgt = targets[bi]
                n_pred = len(out['boxes'])
                n_gt = len(tgt['boxes'])
                if n_pred > 0 and n_gt > 0:
                    print(f'[DEBUG] batch_img={bi} n_pred={n_pred} n_gt={n_gt}')
                    print(f'[DEBUG] pred_boxes_range: x[{out["boxes"][:,0].min():.1f}-{out["boxes"][:,0].max():.1f}] '
                          f'y[{out["boxes"][:,1].min():.1f}-{out["boxes"][:,1].max():.1f}]')
                    print(f'[DEBUG] gt_boxes_range:  x[{tgt["boxes"][:,0].min():.1f}-{tgt["boxes"][:,0].max():.1f}] '
                          f'y[{tgt["boxes"][:,1].min():.1f}-{tgt["boxes"][:,1].max():.1f}]')
                    print(f'[DEBUG] pred_labels: {out["labels"][:10].cpu().numpy()}')
                    print(f'[DEBUG] gt_labels:  {tgt["labels"][:10].numpy()}')
                    print(f'[DEBUG] pred_scores: {out["scores"][:5].cpu().numpy()}')
                    print(f'[DEBUG] pred_boxes[:3]: {out["boxes"][:3].cpu().numpy()}')
                    print(f'[DEBUG] gt_boxes[:3]: {tgt["boxes"][:3].numpy()}')
                    ious = box_iou(tgt['boxes'].to(out['boxes'].device), out['boxes'][:20]).cpu().numpy()
                    print(f'[DEBUG] max_iou_per_pred: {ious.max(axis=0)[:5]}')
                    print(f'[DEBUG] max_iou_per_gt:  {ious.max(axis=1)[:5]}')
                    _debug_done = True
                    break

        for target, output in zip(targets, outputs):
            img_id = target['image_id'].item()
            img_ids.append(img_id)
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            keep = scores >= 0.05
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if len(boxes) > 0:
                keep_idx = nms(torch.tensor(boxes), torch.tensor(scores),
                               iou_threshold=0.5)
                boxes, scores, labels = (boxes[keep_idx], scores[keep_idx],
                                         labels[keep_idx])
                boxes = np.atleast_2d(boxes)
                scores = np.atleast_1d(scores)
                labels = np.atleast_1d(labels)

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                results.append({
                    'image_id': img_id,
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score),
                    'category_id': int(label) + 1,
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
        'mAP@0.5:0.95': stats[0], 'mAP@0.5': stats[1], 'mAP@0.75': stats[2],
        'mAP_small': stats[3], 'mAP_medium': stats[4], 'mAP_large': stats[5],
        'AR@1': stats[6], 'AR@10': stats[7], 'AR@100': stats[8],
    }

    per_class = _compute_per_class_ap(coco_gt, coco_dt, sorted(set(img_ids)))
    metrics.update(per_class)

    wandb_log = {f'val/{k}': v for k, v in metrics.items() if isinstance(v, (int, float))}
    wandb.log(wandb_log)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({'metrics': metrics, 'per_class': per_class,
                       'predictions': results}, f, indent=2)
        csv_path = output_file.replace('.json', '_log.csv')
        _append_csv(csv_path, epoch=0, **metrics)
        print(f'[EVAL] Saved {output_file} and {csv_path}')

    return coco_eval


@torch.no_grad()
def evaluate_test(model, data_loader, device, output_file, model_name="unknown"):
    import datetime
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

            keep = scores >= 0.05
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            if len(boxes) > 0:
                keep_idx = nms(torch.tensor(boxes), torch.tensor(scores), iou_threshold=0.5)
                boxes, scores, labels = (boxes[keep_idx], scores[keep_idx], labels[keep_idx])
                boxes = np.atleast_2d(boxes)
                scores = np.atleast_1d(scores)
                labels = np.atleast_1d(labels)

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                results.append({
                    'image_id': img_id,
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score),
                    'category_id': int(label) + 1,
                })

    output = {
        'model': model_name,
        'dataset': 'TBX11K',
        'num_predictions': len(results),
        'date': datetime.datetime.now().isoformat(),
        'predictions': results,
    }

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'[TEST] {len(results)} predictions saved to {output_file}')


# =============================================================================
# Section 4: Hungarian-matching confusion matrix with per-class metrics
# =============================================================================
@torch.no_grad()
def compute_confusion_matrix(model, data_loader, device,
                             iou_thresh=0.5, score_thresh=0.05):
    model.eval()
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

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
                iou_matrix = box_iou(gt_boxes, pred_boxes).numpy()
                cost_matrix = np.where(iou_matrix >= iou_thresh,
                                       iou_matrix, 0.0)

                if cost_matrix.sum() > 0:
                    gt_idx, pred_idx = linear_sum_assignment(
                        cost_matrix, maximize=True)
                    for gi, pi in zip(gt_idx, pred_idx):
                        if iou_matrix[gi, pi] >= iou_thresh:
                            gt_cls = gt_labels[gi].item()
                            pred_cls = pred_labels[pi].item() + 1
                            confusion[gt_cls, pred_cls] += 1
                            matched_gt.add(gi)
                            matched_pred.add(pi)

            for gi in range(len(gt_boxes)):
                if gi not in matched_gt:
                    confusion[gt_labels[gi].item(), 0] += 1

            for pi in range(len(pred_boxes)):
                if pi not in matched_pred:
                    confusion[0, pred_labels[pi].item() + 1] += 1

    return confusion


def save_confusion_matrix_plot(confusion, output_path, class_names=None):
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(8, 7))
    total_per_row = confusion.sum(axis=1, keepdims=True)
    total_per_row = np.where(total_per_row == 0, 1, total_per_row)
    confusion_norm = confusion / total_per_row

    sns.heatmap(confusion_norm, annot=confusion, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Fraction'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix (Hungarian Match, IoU >= 0.5)')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f'[CONFUSION] Saved to {output_path}')

    wandb.log({'confusion_matrix': wandb.Image(fig)})
    plt.close()

    print('\n--- Per-Class Metrics (Hungarian) ---')
    results = {}
    for cls_idx in range(1, NUM_CLASSES):
        tp = int(confusion[cls_idx, cls_idx])
        fp = int(confusion[:, cls_idx].sum() - tp)
        fn = int(confusion[cls_idx, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        name = class_names[cls_idx] if cls_idx < len(class_names) else f'Class{cls_idx}'
        print(f'  {name}: P={precision:.3f} R={recall:.3f} F1={f1:.3f}')
        results[name] = {
            'precision': precision, 'recall': recall, 'f1': f1,
        }

    wandb_log = {}
    for cls_name, cls_metrics in results.items():
        short = cls_name.replace('Tuberculosis', 'TB').replace('Pulmonary', 'P')
        for metric_name, val in cls_metrics.items():
            wandb_log[f'confusion/{short}_{metric_name}'] = val
    if wandb_log:
        wandb.log(wandb_log)

    return confusion, results


# =============================================================================
# Section 5: Visualization utilities and reporting helpers
# =============================================================================
def plot_training_curves(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if 'epoch_loss' in metrics.history:
        axes[0].plot(metrics.history['epoch_loss'], marker='o', color='#4C72B0')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)

    if 'learning_rate' in metrics.history:
        axes[1].plot(metrics.history['learning_rate'], marker='s', color='#DD8452')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=150)
    wandb.log({'training_curves': wandb.Image(fig)})
    plt.close()
    print(f'[VIZ] Training curves saved to {output_dir}/training_curves.png')


def generate_summary_report(all_metrics, output_path):
    lines = ['=' * 70,
             'TBX11K Object Detection - Model Comparison Report',
             '=' * 70, '']
    for model_name, m in all_metrics.items():
        lines.append(f'--- {model_name.upper()} ---')
        for k, v in m.items():
            if isinstance(v, float):
                lines.append(f'  {k}: {v:.4f}')
            else:
                lines.append(f'  {k}: {v}')
        lines.append('')

    report = '\n'.join(lines)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    print(f'[REPORT] Saved to {output_path}')
    return report
