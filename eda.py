import os
import json
import glob
from collections import Counter, defaultdict
from statistics import mean, median

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw

sns.set_style('whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

DATA_ROOT = 'Images'
OUTPUT_DIR = 'results/eda'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_MAP = {
    'ActiveTuberculosis': 1,
    'ObsoletePulmonaryTuberculosis': 2,
}
CLASS_NAMES = ['Background', 'ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']


def load_annotations(splits=None):
    if splits is None:
        splits = ['train', 'val', 'test']
    records = []
    for split in splits:
        ann_dir = os.path.join(DATA_ROOT, split, 'ann')
        img_dir = os.path.join(DATA_ROOT, split, 'img')
        for f in sorted(glob.glob(os.path.join(ann_dir, '*.json'))):
            with open(f) as fh:
                d = json.load(fh)
            basename = os.path.basename(f).replace('.png.json', '')
            tags = [t['name'] for t in d.get('tags', [])]
            objs = []
            for o in d.get('objects', []):
                ext = o['points']['exterior']
                x1, y1 = ext[0]
                x2, y2 = ext[1]
                w, h = x2 - x1, y2 - y1
                objs.append({
                    'class': o['classTitle'],
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'w': w, 'h': h,
                    'cx': (x1 + x2) / 2,
                    'cy': (y1 + y2) / 2,
                    'area': w * h,
                })
            records.append({
                'split': split,
                'img_id': basename,
                'img_path': os.path.join(img_dir, f'{basename}.png'),
                'tags': tags,
                'objs': objs,
                'width': d['size']['width'],
                'height': d['size']['height'],
            })
    return records


def plot_tag_distribution(records):
    splits = ['train', 'val', 'test']
    tag_order = ['healthy', 'sick_but_non-tb', 'active_tb', 'latent_tb', 'active&latent_tb']
    data = {}
    for sp in splits:
        counter = Counter()
        for r in records:
            if r['split'] == sp:
                for t in r['tags']:
                    counter[t] += 1
        data[sp] = [counter.get(t, 0) for t in tag_order]

    x = np.arange(len(tag_order))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for i, sp in enumerate(splits):
        bars = ax.bar(x + i * w, data[sp], w, label=sp.upper(), color=colors[i])
        for bar, val in zip(bars, data[sp]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                        str(val), ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + w)
    ax.set_xticklabels([t.replace('_', '\n') for t in tag_order])
    ax.set_ylabel('Image count')
    ax.set_title('Image-level Tag Distribution per Split')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'tag_distribution.png'))
    plt.close()
    print('[EDA] Saved tag_distribution.png')


def plot_bbox_class_distribution(records):
    splits = ['train', 'val']
    classes = ['ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']
    data = {}
    for sp in splits:
        counter = Counter()
        for r in records:
            if r['split'] == sp:
                for o in r['objs']:
                    counter[o['class']] += 1
        data[sp] = [counter.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4C72B0', '#DD8452']
    for i, sp in enumerate(splits):
        bars = ax.bar(x + i * w, data[sp], w, label=sp.upper(), color=colors[i])
        for bar, val in zip(bars, data[sp]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    str(val), ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(classes, rotation=15, ha='right')
    ax.set_ylabel('Bounding box count')
    ax.set_title('Object Class Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_class_distribution.png'))
    plt.close()
    print('[EDA] Saved bbox_class_distribution.png')


def plot_bbox_spatial_heatmap(records):
    all_cx = []
    all_cy = []
    for r in records:
        if r['split'] == 'test':
            continue
        for o in r['objs']:
            all_cx.append(o['cx'] / r['width'])
            all_cy.append(o['cy'] / r['height'])

    if not all_cx:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap, xedges, yedges = np.histogram2d(all_cx, all_cy, bins=32,
                                              range=[[0, 1], [0, 1]])
    ax.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='bilinear',
              extent=[0, 1, 0, 1])
    ax.set_xlabel('Normalized X (left → right)')
    ax.set_ylabel('Normalized Y (top → bottom)')
    ax.set_title('Bounding Box Center Spatial Distribution\n(Train + Val)')
    plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_spatial_heatmap.png'))
    plt.close()
    print('[EDA] Saved bbox_spatial_heatmap.png')

    print(f'  Box centers: {len(all_cx)} total')
    print(f'  Mean center: ({mean(all_cx):.3f}, {mean(all_cy):.3f})')
    print(f'  Top half: {sum(1 for cy in all_cy if cy < 0.5)} / {len(all_cy)} ({sum(1 for cy in all_cy if cy < 0.5)/len(all_cy)*100:.1f}%)')


def plot_bbox_size_distribution(records):
    widths = []
    heights = []
    for r in records:
        if r['split'] == 'test':
            continue
        for o in r['objs']:
            widths.append(o['w'])
            heights.append(o['h'])

    if not widths:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(widths, heights, alpha=0.4, s=15, c='#4C72B0', edgecolors='none')
    ax.set_xlabel('Width (px)')
    ax.set_ylabel('Height (px)')
    ax.set_title(f'Box Size Scatter (n={len(widths)})')
    ax.axhline(mean(heights), color='red', linestyle='--', alpha=0.6, label=f'Mean H={mean(heights):.0f}')
    ax.axvline(mean(widths), color='green', linestyle='--', alpha=0.6, label=f'Mean W={mean(widths):.0f}')
    ax.legend(fontsize=9)

    ax = axes[1]
    rel_areas = [(w * h) / (512 * 512) * 100 for w, h in zip(widths, heights)]
    ax.hist(rel_areas, bins=40, color='#DD8452', edgecolor='white', alpha=0.8)
    ax.axvline(mean(rel_areas), color='red', linestyle='--', alpha=0.7,
               label=f'Mean={mean(rel_areas):.2f}%')
    ax.set_xlabel('Relative area (% of image)')
    ax.set_ylabel('Count')
    ax.set_title('Box Area Distribution')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_size_distribution.png'))
    plt.close()
    print('[EDA] Saved bbox_size_distribution.png')


def plot_bbox_aspect_ratio(records):
    ratios = []
    for r in records:
        if r['split'] == 'test':
            continue
        for o in r['objs']:
            if o['h'] > 0:
                ratios.append(o['w'] / o['h'])

    if not ratios:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=30, color='#55A868', edgecolor='white', alpha=0.8)
    ax.axvline(mean(ratios), color='red', linestyle='--', alpha=0.7,
               label=f'Mean={mean(ratios):.2f}')
    ax.axvline(median(ratios), color='blue', linestyle=':', alpha=0.7,
               label=f'Median={median(ratios):.2f}')
    ax.axvline(1.0, color='gray', linestyle='-', alpha=0.4, label='Square (1:1)')
    ax.set_xlabel('Aspect Ratio (width / height)')
    ax.set_ylabel('Count')
    ax.set_title('Bounding Box Aspect Ratio Distribution')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bbox_aspect_ratio.png'))
    plt.close()
    print('[EDA] Saved bbox_aspect_ratio.png')

    print(f'  Aspect ratio: min={min(ratios):.2f}, max={max(ratios):.2f}, mean={mean(ratios):.2f}, median={median(ratios):.2f}')


def plot_boxes_per_image(records):
    splits = ['train', 'val']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, sp in enumerate(splits):
        counter = Counter()
        for r in records:
            if r['split'] == sp:
                counter[len(r['objs'])] += 1
        max_bins = max(counter.keys()) + 1
        bins = list(range(max_bins + 1))
        counts = [counter.get(i, 0) for i in range(max_bins + 1)]

        ax = axes[idx]
        ax.bar(bins, counts, color='#4C72B0', edgecolor='white', width=0.8)
        ax.set_xlabel('Number of bounding boxes')
        ax.set_ylabel('Image count')
        ax.set_title(f'{sp.upper()} — Boxes per Image')
        ax.set_xticks(bins)
        for i, c in enumerate(counts):
            if c > 0:
                ax.text(i, c + 2, str(c), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'boxes_per_image.png'))
    plt.close()
    print('[EDA] Saved boxes_per_image.png')


def plot_sample_grid(records):
    pos_samples = [r for r in records if r['objs'] and r['split'] != 'test']
    if len(pos_samples) > 16:
        import random
        random.seed(42)
        pos_samples = random.sample(pos_samples, 16)

    n = len(pos_samples)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if n > 1 else [axes]

    for i, r in enumerate(pos_samples):
        img = Image.open(r['img_path']).convert('RGB')
        draw = ImageDraw.Draw(img, 'RGBA')
        for o in r['objs']:
            color = (255, 0, 0, 120) if o['class'] == 'ActiveTuberculosis' else (0, 255, 170, 120)
            draw.rectangle([o['x1'], o['y1'], o['x2'], o['y2']], outline=color[:3], width=3)
            label = 'ATB' if o['class'] == 'ActiveTuberculosis' else 'OTB'
            draw.text((o['x1'] + 2, o['y1'] - 14), label, fill=color[:3])
        axes[i].imshow(np.array(img))
        axes[i].set_title(f'{r["split"]}: {", ".join(r["tags"])}', fontsize=9)
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Sample Images with Bounding Boxes\n(Red=ActiveTB, Cyan=ObsoleteTB)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_grid.png'))
    plt.close()
    print('[EDA] Saved sample_grid.png')


def print_dataset_summary(records):
    total = len(records)
    splits = Counter(r['split'] for r in records)
    total_with_boxes = sum(1 for r in records if r['objs'])
    total_boxes = sum(len(r['objs']) for r in records)

    tag_counter = Counter()
    box_class_counter = Counter()
    for r in records:
        for t in r['tags']:
            tag_counter[t] += 1
        for o in r['objs']:
            box_class_counter[o['class']] += 1

    lines = [
        '=' * 60,
        'DATASET SUMMARY — TBX11K',
        '=' * 60,
        f'Total images:         {total}',
        f'  Train:              {splits.get("train", 0)}',
        f'  Val:                {splits.get("val", 0)}',
        f'  Test:               {splits.get("test", 0)}',
        f'Image size:           512 × 512',
        '',
        f'Images with boxes:    {total_with_boxes} ({total_with_boxes/total*100:.1f}%)',
        f'Total bounding boxes: {total_boxes}',
        '',
        'Image-level tag distribution:',
    ]
    for tag, count in tag_counter.most_common():
        lines.append(f'  {tag}: {count} ({count/total*100:.1f}%)')

    lines.extend([
        '',
        'Bounding box class distribution:',
    ])
    for cls_name, count in box_class_counter.most_common():
        lines.append(f'  {cls_name}: {count} ({count/total_boxes*100:.1f}%)')

    lines.extend([
        '',
        'Tag → Box mapping (train+val):',
        '  active_tb         → 100% ActiveTuberculosis',
        '  latent_tb         → 100% ObsoletePulmonaryTuberculosis',
        '  active&latent_tb  → Both classes (~50/50)',
        '  healthy           → No boxes',
        '  sick_but_non-tb   → No boxes',
        '',
        'Test set: 3302 images with no tags and no boxes — inference only.',
        '=' * 60,
    ])

    summary = '\n'.join(lines)
    print(summary)
    with open(os.path.join(OUTPUT_DIR, 'dataset_summary.txt'), 'w') as f:
        f.write(summary)
    print('[EDA] Saved dataset_summary.txt')


def main():
    print('Loading annotations...')
    records = load_annotations()
    print(f'Loaded {len(records)} annotation files.\n')

    print_dataset_summary(records)
    print()
    plot_tag_distribution(records)
    plot_bbox_class_distribution(records)
    plot_bbox_spatial_heatmap(records)
    plot_bbox_size_distribution(records)
    plot_bbox_aspect_ratio(records)
    plot_boxes_per_image(records)
    plot_sample_grid(records)

    print(f'\nAll EDA outputs saved to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
