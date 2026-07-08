import torch
import torch.nn.functional as F
import numpy as np


class DETRAttentionExtractor:
    def __init__(self, model):
        self.model = model
        self._attention_weights = None

        decoder_layer = model.transformer.decoder.layers[-1]
        self._handle = decoder_layer.multihead_attn.register_forward_hook(
            self._hook_attention
        )

    def _hook_attention(self, module, input, output):
        q = input[0]
        k = input[1]
        attn_weights = torch.bmm(q.transpose(0, 1), k.transpose(0, 1).transpose(1, 2))
        d_k = q.size(-1) ** 0.5
        attn_weights = F.softmax(attn_weights / d_k, dim=-1)
        self._attention_weights = attn_weights.detach()

    def remove_hook(self):
        self._handle.remove()

    def extract(self, image):
        self.model.zero_grad()

        with torch.no_grad():
            output = self.model(image)

        if self._attention_weights is None:
            return None, output

        attn = self._attention_weights

        attn = attn.mean(dim=0)

        num_queries = attn.shape[0]

        pred_boxes = output[0]['boxes'].cpu().numpy()
        pred_scores = output[0]['scores'].cpu().numpy()
        pred_labels = output[0]['labels'].cpu().numpy()

        heatmaps = []
        meta = []
        for q_idx in range(min(num_queries, 100)):
            if q_idx >= len(pred_scores):
                continue
            if pred_scores[q_idx] < 0.1:
                continue

            attn_map = attn[q_idx].reshape(-1).cpu().numpy()
            h = w = int(np.sqrt(len(attn_map)))

            if h * w != len(attn_map):
                h = w = 32

            attn_map = attn_map[:h * w].reshape(h, w)
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

            heatmaps.append(attn_map)
            meta.append({
                'query_idx': q_idx,
                'score': float(pred_scores[q_idx]),
                'label': int(pred_labels[q_idx]),
                'box': pred_boxes[q_idx].tolist(),
            })

        return heatmaps, meta


def get_attention_map(detr_model, image, output_size=(512, 512)):
    extractor = DETRAttentionExtractor(detr_model)
    heatmaps, meta = extractor.extract(image)
    extractor.remove_hook()

    if heatmaps is None or len(heatmaps) == 0:
        return None, None

    return heatmaps, meta
