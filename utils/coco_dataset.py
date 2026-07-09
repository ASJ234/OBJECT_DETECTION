import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import torchvision.transforms as T


class CocoDetection(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(ann_file)
        self.ids = sorted(self.coco.imgs.keys())

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(path).convert('RGB')

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for ann in anns:
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(ann['category_id'])
            area.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = torch.as_tensor(area, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id]),
            'area': area,
            'iscrowd': iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.ids)


def get_transform(train):
    transforms_list = [T.ToTensor()]
    if train:
        transforms_list.append(T.RandomHorizontalFlip(0.5))
    transform = T.Compose(transforms_list)
    def _apply(image, target):
        return transform(image), target
    return _apply


def collate_fn(batch):
    return tuple(zip(*batch))
