import os
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2

sys.path.insert(0, os.path.dirname(__file__))
from utils.coco_dataset import CocoDetection, get_transform, collate_fn
from utils.engine import (
    train_one_epoch, evaluate, evaluate_test,
    compute_confusion_matrix, save_confusion_matrix_plot,
)
from explain.gradcam import GradCAM, get_target_layer
from explain.visualize import overlay_heatmap, draw_detections

RESULTS_DIR = 'results/retinanet'
os.makedirs(f'{RESULTS_DIR}/explain', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/weights', exist_ok=True)


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
    print(f'[RetinaNet] Device: {device}')

    train_dataset = CocoDetection(
        'dataset/coco/train',
        'dataset/coco/train.json',
        transforms=get_transform(train=True),
    )
    val_dataset = CocoDetection(
        'dataset/coco/val',
        'dataset/coco/val.json',
        transforms=get_transform(train=False),
    )

    sampler = get_weighted_sampler(train_dataset, pos_weight=4.0)
    train_loader = DataLoader(
        train_dataset, batch_size=8, sampler=sampler,
        collate_fn=collate_fn, num_workers=4, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    model = retinanet_resnet50_fpn_v2(weights_backbone="DEFAULT", num_classes=3)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_map = 0.0
    best_epoch = -1

    for epoch in range(1, 101):
        train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        lr_scheduler.step()

        print(f'\n[RetinaNet] Validation after epoch {epoch}...')
        coco_eval = evaluate(model, val_loader, device)

        if coco_eval is not None:
            current_map = coco_eval.stats[0]
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                torch.save(model.state_dict(), f'{RESULTS_DIR}/weights/best_model.pth')
                print(f'  New best model! mAP={best_map:.4f} (epoch {epoch})')

        print()

    print(f'[RetinaNet] Training complete. Best mAP: {best_map:.4f} at epoch {best_epoch}')


def evaluate_model():
    print('\n[RetinaNet] Evaluating best model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_dataset = CocoDetection(
        'dataset/coco/val', 'dataset/coco/val.json',
        transforms=get_transform(train=False),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    model = retinanet_resnet50_fpn_v2(weights_backbone="DEFAULT", num_classes=3)
    model.load_state_dict(torch.load(f'{RESULTS_DIR}/weights/best_model.pth',
                                     map_location=device, weights_only=True))
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
        print(f'[RetinaNet] Metrics saved to {RESULTS_DIR}/metrics.json')

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
    print('\n[RetinaNet] Generating XAI explanations...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = retinanet_resnet50_fpn_v2(weights_backbone="DEFAULT", num_classes=3)
    model.load_state_dict(torch.load(f'{RESULTS_DIR}/weights/best_model.pth',
                                     map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_dataset = CocoDetection(
        'dataset/coco/test', 'dataset/coco/test.json',
        transforms=get_transform(train=False),
    ) if os.path.exists('dataset/coco/test.json') else None

    if not test_dataset:
        print('[RetinaNet] No test dataset for XAI.')
        return

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )

    target_layer = get_target_layer(model, 'retinanet')
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
    print('[RetinaNet] XAI complete.')


def main():
    train()
    evaluate_model()
    run_xai()


if __name__ == '__main__':
    main()
