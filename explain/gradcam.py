import torch
import torch.nn.functional as F
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._activations = None
        self._gradients = None

        self._forward_handle = target_layer.register_forward_hook(self._hook_forward)
        self._backward_handle = target_layer.register_full_backward_hook(self._hook_backward)

    def _hook_forward(self, module, input, output):
        self._activations = output.detach()

    def _hook_backward(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        self._forward_handle.remove()
        self._backward_handle.remove()

    def generate(self, image, target_class=None):
        self.model.zero_grad()

        if image.grad is not None:
            image.grad.zero_()

        predictions = self.model(image)

        if target_class is None:
            if len(predictions[0]['scores']) == 0:
                return None
            best_idx = predictions[0]['scores'].argmax()
            target_class = predictions[0]['labels'][best_idx].item()

        loss = torch.tensor(0.0, device=image.device)
        for pred in predictions:
            for label, score in zip(pred['labels'], pred['scores']):
                if label.item() == target_class:
                    loss = loss + score

        if loss == 0:
            return None

        loss.backward()

        if self._gradients is None or self._activations is None:
            return None

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def get_target_layer(model, architecture):
    if architecture == 'fcos':
        return model.backbone.body.layer4
    elif architecture == 'retinanet':
        return model.backbone.body.layer4
    elif architecture == 'efficientdet':
        return model.backbone.conv_head
    else:
        raise ValueError(f'Unsupported architecture: {architecture}')
