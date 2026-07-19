import torch
import torch.nn as nn
import math
from torchvision.models.detection import fcos_resnet50_fpn

model = fcos_resnet50_fpn(weights="DEFAULT")
in_channels = model.head.classification_head.cls_logits.in_channels
num_anchors = model.head.classification_head.num_anchors
num_classes = 3

model.head.classification_head.num_classes = num_classes
model.head.classification_head.cls_logits = nn.Conv2d(
    in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
)
torch.nn.init.constant_(model.head.classification_head.cls_logits.bias, -math.log((1 - 0.01) / 0.01))
torch.nn.init.normal_(model.head.classification_head.cls_logits.weight, std=0.01)

model.eval()
x = [torch.rand(3, 300, 400)]
out = model(x)
print(out)
print("SUCCESS")
