import copy
import torch
import torch.nn as nn


class ModelEMA:
    """Exponential Moving Average of model weights.

    During training, call `ema.update(model)` after each optimizer step.
    At evaluation, use `ema.model` instead of the raw model.
    """

    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        decay = self.decay
        with torch.no_grad():
            for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
                ema_p.mul_(decay).add_(model_p, alpha=1.0 - decay)
            for ema_b, model_b in zip(self.model.buffers(), model.buffers()):
                ema_b.copy_(model_b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
