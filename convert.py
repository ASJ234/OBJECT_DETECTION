import os
import json
import glob
import shutil
import argparse

from PIL import Image

DATA_ROOT = 'Images'
OUTPUT_DIR = 'dataset'

CLASS_MAP = {
    'ActiveTuberculosis': 1,
    'ObsoletePulmonaryTuberculosis': 2,
}
CLASS_NAMES = ['Background', 'ActiveTuberculosis', 'ObsoletePulmonaryTuberculosis']


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def convert_to_coco(splits):
    for split in splits:
        images = []
        annotations = []
        ann_id = 1

        ann_dir = os.path.join(DATA_ROOT, split, 'ann')
        img_dir = os.path.join(DATA_ROOT, split, 'img')
        out_img_dir = os.path.join(OUTPUT_DIR, 'coco', split)
        make_dir(out_img_dir)

        for f in sorted(glob.glob(os.path.join(ann_dir, '*.json'))):
            with open(f) as fh:
                d = json.load(fh)

            basename = os.path.basename(f).replace('.json', '')
            img_id = len(images) + 1

            src_path = os.path.join(img_dir, basename)
            if not os.path.exists(src_path):
                print(f'[WARN] Image not found: {src_path}, skipping')
                continue

            dst_path = os.path.join(out_img_dir, basename)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)

            w_img = d['size']['width']
            h_img = d['size']['height']

            images.append({
                'id': img_id,
                'file_name': basename,
                'width': w_img,
                'height': h_img,
            })

            for obj in d.get('objects', []):
                if obj.get('classTitle') not in CLASS_MAP:
                    continue
                ext = obj['points']['exterior']
                if len(ext) < 2:
                    continue
                x1, y1 = ext[0]
                x2, y2 = ext[1]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                w = x_max - x_min
                h = y_max - y_min
                if w <= 0 or h <= 0:
                    continue
                x_min = max(0, min(x_min, w_img))
                y_min = max(0, min(y_min, h_img))
                w = min(w, w_img - x_min)
                h = min(h, h_img - y_min)
                if w <= 0 or h <= 0:
                    continue

                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'bbox': [float(x_min), float(y_min), float(w), float(h)],
                    'area': float(w * h),
                    'category_id': CLASS_MAP[obj['classTitle']],
                    'iscrowd': 0,
                })
                ann_id += 1

        out = {
            'images': images,
            'annotations': annotations,
            'categories': [
                {'id': 1, 'name': 'ActiveTuberculosis'},
                {'id': 2, 'name': 'ObsoletePulmonaryTuberculosis'},
            ],
        }

        out_path = os.path.join(OUTPUT_DIR, 'coco', f'{split}.json')
        with open(out_path, 'w') as f:
            json.dump(out, f)
        print(f'[COCO] {split}: {len(images)} images, {len(annotations)} annotations → {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Convert TBX11K annotations to COCO format')
    parser.add_argument('--data-root', type=str, default='Images', help='Path to Images directory')
    parser.add_argument('--output-dir', type=str, default='dataset', help='Output directory')
    args = parser.parse_args()

    global DATA_ROOT, OUTPUT_DIR
    DATA_ROOT = args.data_root
    OUTPUT_DIR = args.output_dir

    make_dir(OUTPUT_DIR)

    convert_to_coco(['train', 'val'])

    print('\nConversion complete.')
    print(f'  COCO: {os.path.join(OUTPUT_DIR, "coco")}/')


if __name__ == '__main__':
    main()
