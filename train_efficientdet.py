import os
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils.coco_dataset import CocoDetection, get_transform, collate_fn
from utils.engine import (
    evaluate, evaluate_test,
    compute_confusion_matrix, save_confusion_matrix_plot,
)
from explain.gradcam import GradCAM, get_target_layer
from explain.visualize import overlay_heatmap, draw_detections

RESULTS_DIR = 'results/efficientdet'
os.makedirs(f'{RESULTS_DIR}/explain', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/weights', exist_ok=True)


def collate_effdet(batch):
    images, targets = zip(*batch)
    gt_cls = []
    gt_box = []
    img_ids = []
    for t in targets:
        gt_cls.append(t['labels'])
        gt_box.append(t['boxes'])
        img_ids.append(t['image_id'])
    images = torch.stack(list(images))
    return images, {'cls': gt_cls, 'box': gt_box, 'img_id': img_ids}


def pad_targets(cls_list, box_list, batch_size, max_boxes=50):
    device = box_list[0].device
    cls_pad = torch.zeros(batch_size, max_boxes, dtype=torch.int64, device=device)
    box_pad = torch.zeros(batch_size, max_boxes, 4, dtype=torch.float32, device=device)
    for i in range(batch_size):
        n = min(len(cls_list[i]), max_boxes)
        cls_pad[i, :n] = cls_list[i][:n]
        box_pad[i, :n] = box_list[i][:n]
    return cls_pad, box_pad


def get_weighted_sampler(dataset, pos_weight=4.0):
    weights = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        if len(target['boxes']) > 0:
            weights.append(pos_weight)
        else:
            weights.append(1.0)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[EfficientDet-D2] Device: {device}')

    from effdet import create_model

    train_dataset = CocoDetection(
        'dataset/coco/train', 'dataset/coco/train.json',
        transforms=get_transform(train=True),
    )
    val_dataset = CocoDetection(
        'dataset/coco/val', 'dataset/coco/val.json',
        transforms=get_transform(train=False),
    )

    sampler = get_weighted_sampler(train_dataset, pos_weight=4.0)
    train_loader = DataLoader(
        train_dataset, batch_size=8, sampler=sampler,
        collate_fn=collate_effdet, num_workers=4, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    model = create_model('tf_efficientdet_d2', pretrained=True, num_classes=3)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_map = 0.0
    best_epoch = -1

    for epoch in range(1, 101):
        model.train()
        total_loss = 0

        for i, (images, batch_targets) in enumerate(train_loader):
            cls_list = [c.to(device) for c in batch_targets['cls']]
            box_list = [b.to(device) for b in batch_targets['box']]
            images = images.to(device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    cls_targets, box_targets = pad_targets(cls_list, box_list, images.shape[0])
                    loss_dict, _ = model(images, cls_targets, box_targets)
                loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                cls_targets, box_targets = pad_targets(cls_list, box_list, images.shape[0])
                loss_dict, _ = model(images, cls_targets, box_targets)
                loss = sum(loss_dict.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(f'  Epoch [{epoch}] Step [{i}/{len(train_loader)}] Loss: {loss.item():.4f}')

        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'  Epoch [{epoch}] Avg Loss: {avg_loss:.4f}')

        print(f'\n[EfficientDet-D2] Validation after epoch {epoch}...')
        model.eval()
        coco_eval = evaluate(model, val_loader, device)

        if coco_eval is not None:
            current_map = coco_eval.stats[0]
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                torch.save(model.state_dict(), f'{RESULTS_DIR}/weights/best_model.pth')
                print(f'  New best model! mAP={best_map:.4f} (epoch {epoch})')
        print()

    print(f'[EfficientDet-D2] Training complete. Best mAP: {best_map:.4f} at epoch {best_epoch}')


def evaluate_model():
    print('\n[EfficientDet-D2] Evaluating best model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from effdet import create_model

    val_dataset = CocoDetection(
        'dataset/coco/val', 'dataset/coco/val.json',
        transforms=get_transform(train=False),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    model = create_model('tf_efficientdet_d2', pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(
        f'{RESULTS_DIR}/weights/best_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    coco_eval = evaluate(model, val_loader, device,
                         output_file=f'{RESULTS_DIR}/val_preds.json')

    if coco_eval is not None:
        metrics = {
            'mAP@0.5:0.95': coco_eval.stats[0],
            'mAP@0.5': coco_eval.stats[1],
            'mAP@0.75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5],
            'AR@1': coco_eval.stats[6],
            'AR@10': coco_eval.stats[7],
            'AR@100': coco_eval.stats[8],
        }
        with open(f'{RESULTS_DIR}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'[EfficientDet-D2] Metrics saved to {RESULTS_DIR}/metrics.json')

    confusion = compute_confusion_matrix(model, val_loader, device)
    save_confusion_matrix_plot(confusion, f'{RESULTS_DIR}/confusion_matrix.png')

    test_dataset = CocoDetection(
        'dataset/coco/test', 'dataset/coco/test.json',
        transforms=get_transform(train=False),
    ) if os.path.exists('dataset/coco/test.json') else None

    if test_dataset:
        test_loader = DataLoader(
            test_dataset, batch_size=8, shuffle=False,
            collate_fn=collate_fn, num_workers=4,
        )
        evaluate_test(model, test_loader, device, f'{RESULTS_DIR}/test_preds.json')


def run_xai():
    print('\n[EfficientDet-D2] Generating XAI explanations...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from effdet import create_model

    model = create_model('tf_efficientdet_d2', pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(
        f'{RESULTS_DIR}/weights/best_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_dataset = CocoDetection(
        'dataset/coco/test', 'dataset/coco/test.json',
        transforms=get_transform(train=False),
    ) if os.path.exists('dataset/coco/test.json') else None

    if not test_dataset:
        print('[EfficientDet-D2] No test dataset for XAI.')
        return

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )

    target_layer = get_target_layer(model, 'efficientdet')
    gradcam = GradCAM(model, target_layer)

    count = 0
    for images, targets in test_loader:
        if count >= 5:
            break
        image = images[0].unsqueeze(0).to(device)

        with torch.set_grad_enabled(True):
            heatmap = gradcam.generate(image)

        if heatmap is not None:
            overlay = overlay_heatmap(image, heatmap)
            model.eval()
            with torch.no_grad():
                outputs = model(image)

            if isinstance(outputs, list) and len(outputs) > 0:
                detections = {
                    'boxes': outputs[0]['boxes'].cpu().numpy(),
                    'scores': outputs[0]['scores'].cpu().numpy(),
                    'labels': outputs[0]['labels'].cpu().numpy(),
                }
                overlay = draw_detections(overlay, detections)
                out_path = f'{RESULTS_DIR}/explain/sample_{count}.png'
                overlay.save(out_path)
                print(f'  Saved {out_path}')
                count += 1

    gradcam.remove_hooks()
    print('[EfficientDet-D2] XAI complete.')


def main():
    train()
    evaluate_model()
    run_xai()


if __name__ == '__main__':
    main()
