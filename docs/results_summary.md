# TBX11K Detection — Combined Results (FCOS + RetinaNet)

**Dataset**: TBX11K chest X-rays — train 6,600 imgs / 902 boxes, val 1,800 / 309, test 3,302 / 0
**Classes**: Active Tuberculosis (1, 724 train boxes), Obsolete Pulmonary TB (2, 178 train boxes) — ~4:1 imbalance
**Models**: FCOS (ResNet-50 + FPN) and RetinaNet (ResNet-50 + FPN v2), both torchvision, COCO-pretrained, heads replaced for 2 classes
**Input resolution**: 1024 / 1536 · **Sampling**: WeightedRandomSampler — ActiveTB capped at 200 boxes/epoch
(**201 ActiveTB annotations sampled, 227 effective incl. mixed images**) vs **178 ObsoleteTB**,
negatives weight 0.05. Both models trained on this same ~1.1:1 per-epoch class balance.

Two runs per model:
- **v1** — baseline (raw-count alphas `[0.25, 0.35]`, RetinaNet head randomly initialized, no negative-image loss damping)
- **v2** — Fix Round 2 (sampled-count alphas `[0.35, 0.446]`, warmup 3, RetinaNet head warm-started + `neg_loss_scale=0.1`, FCOS background-centerness supervision, RetinaNet batch 4 to fit the 11 GB card)

## v2 — headline comparison

| Metric | FCOS v2 | FCOS v1 | RetinaNet v2 | RetinaNet v1 |
|--------|---------|---------|--------------|--------------|
| mAP@0.5:0.95 | **0.0805** | 0.0496 | 0.0658 | 0.0477 |
| mAP@0.5 | **0.2090** | 0.1522 | 0.1739 | 0.1228 |
| mAP@0.75 | **0.0403** | 0.0142 | 0.0264 | 0.0255 |
| AR@1 | 0.137 | 0.105 | 0.072 | 0.107 |
| AR@10 | 0.236 | 0.216 | 0.195 | 0.179 |
| AR@100 | 0.248 | 0.229 | 0.237 | 0.187 |

Sources: `results/fcos/metrics.json`, `results/retinanet/metrics.json` (v2 best-model COCO eval);
v1 numbers from `docs/fcos_results.md`, `docs/retinanet_results.md`.

Per-class AP (v2; RetinaNet per-class from the val/EMA evaluation — FCOS per-class not yet extracted):

| Class | FCOS v2 | RetinaNet v2 | RetinaNet v1 |
|-------|---------|--------------|--------------|
| ActiveTB | — | **0.0848** | 0.065 |
| ObsoleteTB | — | 0.0025 | 0.030 |

## Test-time augmentation (TTA)

| Model | mAP@0.5:0.95 no TTA | with TTA |
|-------|--------------------:|---------:|
| FCOS v2 | **0.0805** | 0.0477 |
| RetinaNet v2 (EMA eval) | 0.0437 | 0.0437 |

TTA does not help on this task (hurts FCOS, neutral on RetinaNet). **Non-TTA is the reported configuration.**

## Reading the numbers

1. **Fix Round 2 helped both models**: FCOS mAP 0.0496 → 0.0805 (+62%), RetinaNet 0.0477 → 0.0658 (+38%). Scores moved off the 0.050 floor (RetinaNet top ~0.08; FCOS climbs higher, still ≤ ~0.3) and localization stayed strong.
2. **FCOS now leads on every headline metric** — it already had the geometry advantage (label-agnostic recall 88.7% vs 79.6%) and benefits most from the score-calibration fixes; its boxes also land tighter at mAP@0.75 (0.0403 vs 0.0264).
3. **The remaining bottleneck is class discrimination, not geometry.** RetinaNet's minority-class AP collapsed on v2 (ObsoleteTB 0.030 → 0.0025) while ActiveTB improved (0.065 → 0.085). Both GT boxes are localized at IoU ≥ 0.5–0.64, but correct detections still score ~0.08 — far below a usable 0.3 threshold — so precision is crushed by background false positives at low scores.
4. **Best epoch hit the last epoch** (RetinaNet E75, LR 1e-7): both models were still improving when training stopped — a longer run (150 epochs / lower LR) is the obvious next lever.

## Artifacts & references

- `docs/fcos_results.md` — FCOS v1 deep-dive
- `docs/retinanet_results.md` — RetinaNet v1 deep-dive + v2 section
- `walkthrough.md` — Fix Round 2 (changes + outcome)
- Weights/metrics/preds: `results/{fcos,retinanet}/` (training artifacts, gitignored)
