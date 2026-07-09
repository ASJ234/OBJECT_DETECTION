import os
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
from torchvision.models.detection import detr_resnet50

sys.path.insert(0, os.path.dirname(__file__))
from utils.coco_dataset import CocoDetection, get_transform, collate_fn
from utils.engine import (
    train_one_epoch, evaluate, evaluate_test,
    compute_confusion_matrix, save_confusion_matrix_plot,
)
from explain.detr_attention import DETRAttentionExtractor
from explain.visualize import overlay_heatmap, draw_detections, generate_xai_report

RESULTS_DIR = 'results/detr'
os.makedirs(f'{RESULTS_DIR}/explain', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/weights', exist_ok=True)

CLASS_NAMES = {0: 'Background', 1: 'ActiveTuberculosis', 2: 'ObsoletePulmonaryTuberculosis'}
CLASS_COLORS = {1: (255, 0, 0), 2: (0, 200, 255)}


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
    print(f'[DETR] Device: {device}')

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
        train_dataset, batch_size=4, sampler=sampler,
        collate_fn=collate_fn, num_workers=4, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    model = detr_resnet50(weights_backbone="DEFAULT", num_classes=3)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_map = 0.0
    best_epoch = -1

    for epoch in range(1, 101):
        train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler, clip_norm=1.0)
        lr_scheduler.step()

        print(f'\n[DETR] Validation after epoch {epoch}...')
        coco_eval = evaluate(model, val_loader, device)

        if coco_eval is not None:
            current_map = coco_eval.stats[0]
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                torch.save(model.state_dict(), f'{RESULTS_DIR}/weights/best_model.pth')
                print(f'  New best model! mAP={best_map:.4f} (epoch {epoch})')

        print()

    print(f'[DETR] Training complete. Best mAP: {best_map:.4f} at epoch {best_epoch}')


def evaluate_model():
    print('\n[DETR] Evaluating best model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_dataset = CocoDetection(
        'dataset/coco/val', 'dataset/coco/val.json',
        transforms=get_transform(train=False),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )

    model = detr_resnet50(weights_backbone="DEFAULT", num_classes=3)
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
        print(f'[DETR] Metrics saved to {RESULTS_DIR}/metrics.json')

    confusion = compute_confusion_matrix(model, val_loader, device)
    save_confusion_matrix_plot(confusion, f'{RESULTS_DIR}/confusion_matrix.png')

    test_dataset = CocoDetection(
        'dataset/coco/test', 'dataset/coco/test.json',
        transforms=get_transform(train=False),
    ) if os.path.exists('dataset/coco/test.json') else None

    if test_dataset:
        test_loader = DataLoader(
            test_dataset, batch_size=4, shuffle=False,
            collate_fn=collate_fn, num_workers=4,
        )
        evaluate_test(model, test_loader, device, f'{RESULTS_DIR}/test_preds.json')


def run_xai():
    print('\n[DETR] Generating XAI attention maps...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = detr_resnet50(weights_backbone="DEFAULT", num_classes=3)
    model.load_state_dict(torch.load(f'{RESULTS_DIR}/weights/best_model.pth',
                                     map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    test_dataset = CocoDetection(
        'dataset/coco/test', 'dataset/coco/test.json',
        transforms=get_transform(train=False),
    ) if os.path.exists('dataset/coco/test.json') else None

    if not test_dataset:
        print('[DETR] No test dataset for XAI.')
        return

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )

    count = 0
    for images, targets in test_loader:
        if count >= 5:
            break
        image = images[0].unsqueeze(0).to(device)

        extractor = DETRAttentionExtractor(model)
        heatmaps, meta = extractor.extract(image)
        extractor.remove_hook()

        if heatmaps and len(heatmaps) > 0:
            best_idx = 0
            best_score = 0
            for i, m in enumerate(meta):
                if m['score'] > best_score:
                    best_score = m['score']
                    best_idx = i

            if best_idx < len(heatmaps):
                import numpy as np
                from PIL import Image as PILImage

                img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
                          np.array([0.485, 0.456, 0.406]))
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                axes[0].imshow(img_np)
                model.eval()
                with torch.no_grad():
                    outputs = model(image)
                boxes = outputs[0]['boxes'].cpu().numpy()
                scores = outputs[0]['scores'].cpu().numpy()
                labels = outputs[0]['labels'].cpu().numpy()
                from PIL import ImageDraw
                draw_img = PILImage.fromarray(img_np.copy())
                draw = ImageDraw.Draw(draw_img)
                for box, sc, lb in zip(boxes, scores, labels):
                    if sc < 0.3:
                        continue
                    x1, y1, x2, y2 = box[:4]
                    color = CLASS_COLORS.get(int(lb), (255, 255, 0))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1 + 2, y1 - 16),
                              f'{CLASS_NAMES.get(int(lb), str(lb))}: {sc:.2f}',
                              fill=color)
                axes[0].imshow(np.array(draw_img))
                axes[0].set_title('DETR — Detections', fontsize=11)

                attn = heatmaps[best_idx]
                attn_resized = np.array(
                    PILImage.fromarray((attn * 255).astype(np.uint8)).resize(
                        (img_np.shape[1], img_np.shape[0]), PILImage.BILINEAR
                    )) / 255.0
                cmap = plt.get_cmap('jet')
                attn_colored = cmap(attn_resized)[:, :, :3]
                combined = (1 - 0.5) * img_np / 255 + 0.5 * attn_colored
                axes[1].imshow(np.clip(combined, 0, 1))
                axes[1].set_title(f'DETR — Cross-Attention (score={best_score:.2f})',
                                  fontsize=11)

                for ax in axes:
                    ax.axis('off')
                plt.tight_layout()
                out_path = f'{RESULTS_DIR}/explain/sample_{count}.png'
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f'  Saved {out_path}')
                count += 1

    print('[DETR] XAI complete.')


def main():
    train()
    evaluate_model()
    run_xai()


if __name__ == '__main__':
    main()
