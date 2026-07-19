import torch
from torchvision.models.detection import fcos_resnet50_fpn

model = fcos_resnet50_fpn(weights="DEFAULT")
print("Original num classes:", model.head.classification_head.num_classes)
