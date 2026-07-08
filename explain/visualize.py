import os
import numpy as np
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASS_NAMES = {0: 'Background', 1: 'ActiveTuberculosis', 2: 'ObsoletePulmonaryTuberculosis'}
CLASS_COLORS = {1: (255, 0, 0), 2: (0, 200, 255)}


def overlay_heatmap(image_tensor, heatmap, alpha=0.6):
    if heatmap is None:
        img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)

    img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    cmap = plt.get_cmap('jet')
    heatmap_colored = cmap(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    blended = (1 - alpha) * img_np + alpha * heatmap_colored
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def draw_detections(pil_image, predictions, class_names=None, score_thresh=0.3):
    if class_names is None:
        class_names = CLASS_NAMES
    draw = ImageDraw.Draw(pil_image)

    boxes = predictions.get('boxes', [])
    scores = predictions.get('scores', [])
    labels = predictions.get('labels', [])

    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box[:4]
        color = CLASS_COLORS.get(int(label), (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f'{class_names.get(int(label), str(label))}: {score:.2f}'
        draw.text((x1 + 2, y1 - 16), label_text, fill=color)

    return pil_image


def generate_xai_report(model_outputs, image_tensor, ground_truth, output_path,
                        arch_name, attention_data=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    base_img = Image.fromarray(img_np)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_np)
    axes[0].set_title(f'{arch_name} — Original + Detections', fontsize=11)
    if model_outputs:
        draw = ImageDraw.Draw(base_img)
        boxes = model_outputs[0]['boxes'].cpu().numpy()
        scores = model_outputs[0]['scores'].cpu().numpy()
        labels = model_outputs[0]['labels'].cpu().numpy()
        for box, sc, lb in zip(boxes, scores, labels):
            if sc < 0.3:
                continue
            x1, y1, x2, y2 = box
            color = CLASS_COLORS.get(int(lb), (255, 255, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        axes[0].imshow(np.array(base_img))

    if attention_data is not None:
        heatmaps, meta = attention_data
        if heatmaps and len(heatmaps) > 0:
            best_idx = 0
            best_score = 0
            for i, m in enumerate(meta):
                if m['score'] > best_score:
                    best_score = m['score']
                    best_idx = i

            if best_idx < len(heatmaps):
                attn_resized = np.array(Image.fromarray(heatmaps[best_idx]).resize(
                    (img_np.shape[1], img_np.shape[0]), Image.BILINEAR))
                cmap = plt.get_cmap('jet')
                attn_colored = cmap(attn_resized)[:, :, :3]
                blended = (1 - 0.5) * img_np / 255 + 0.5 * attn_colored
                axes[1].imshow(np.clip(blended, 0, 1))
                axes[1].set_title(f'{arch_name} — Attention (score={best_score:.2f})', fontsize=11)
            else:
                axes[1].imshow(img_np)
                axes[1].set_title(f'{arch_name} — No attention', fontsize=11)
        else:
            axes[1].imshow(img_np)
            axes[1].set_title(f'{arch_name} — No attention', fontsize=11)
    else:
        axes[1].imshow(img_np)
        axes[1].set_title(f'{arch_name} — No attention', fontsize=11)

    axes[2].imshow(img_np)
    ax2_title = 'Ground Truth'
    if ground_truth is not None and len(ground_truth.get('boxes', [])) > 0:
        draw_gt = ImageDraw.Draw(Image.fromarray(img_np.copy()))
        for box, label in zip(ground_truth['boxes'], ground_truth['labels']):
            x1, y1, x2, y2 = box
            color = CLASS_COLORS.get(int(label), (255, 255, 0))
            draw_gt.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw_gt.text((x1 + 2, y1 - 16), CLASS_NAMES.get(int(label), str(label)), fill=color)
    axes[2].imshow(img_np)
    axes[2].set_title(ax2_title, fontsize=11)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_comparison_grid(model_results, output_path, class_names=None):
    if class_names is None:
        class_names = ['FCOS', 'EfficientDet-D2', 'RetinaNet', 'DETR']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for idx, (name, img_path) in enumerate(model_results.items()):
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[idx].imshow(np.array(img))
        axes[idx].set_title(name, fontsize=12)
        axes[idx].axis('off')

    plt.suptitle('Model Comparison — XAI Overlays', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[XAI] Comparison grid saved to {output_path}')
