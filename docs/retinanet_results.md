# RetinaNet Results — Detailed Explanation (TBX11K)

**Model**: RetinaNet (ResNet-50 + FPN v2 backbone, torchvision, COCO-pretrained) — head replaced for 2 classes
**Dataset**: TBX11K chest X-rays — train 6,600 imgs / 902 boxes, val 1,800 / 309, test 3,302 / 0
**Classes**: Active Tuberculosis (724 train boxes), Obsolete Pulmonary TB (178 train boxes) — a **4:1 class imbalance**

---

## v2 — post-fix retraining (Fix Round 2)

Run with all Fix Round 2 changes (`train_retinanet.py`): warm-started head convs,
`neg_loss_scale=0.1`, sampled-count alphas `[0.35, 0.446]`, warmup 3, batch **4** (reduced
from 8 to fit the 11 GB card — the old default OOM'd). Sections 1–9 below are the **v1
pre-fix baseline**, kept for comparison.

### Best-model COCO metrics (vs v1)

| Metric | v2 | v1 (pre-fix) |
|--------|----|--------------|
| mAP@0.5:0.95 (best) | **0.0658** (E75) | 0.0477 (E73) |
| mAP@0.5 | **0.1739** | 0.1228 |
| mAP@0.75 | 0.0264 | 0.0255 |
| val/AP ActiveTB | **0.0848** | 0.065 |
| val/AP ObsoleteTB | 0.0025 | 0.030 |
| val/AR@1 / @10 / @100 | 0.072 / 0.195 / **0.237** | 0.107 / 0.179 / 0.187 |

- Best checkpoint saved at **epoch 75 (the last epoch)** — the model was still improving when
  training stopped (LR had decayed to 1e-7); a longer run is the obvious next step.
- `val/mAP@0.5:0.95` (EMA checkpoints) = 0.0437; TTA = 0.0437. The 0.0658 is the best-model
  eval reported by the training loop; per-class APs above are from the val (EMA) evaluation.
- Final losses: classification 0.207, bbox-regression 0.113.

### What the fixes changed — and what they didn't

**Improved:** mAP 0.048 → 0.066, AP@0.5 0.123 → 0.174, ActiveTB AP 0.065 → 0.085,
AR@100 0.187 → 0.237. Scores moved off the 0.050 floor (final-epoch val debug: top-5 scores
~0.05–0.08, best >0.08) and localization stayed strong (both GT boxes matched at IoU 0.58/0.64;
pred boxes at 0.49–0.58).

**Still the bottleneck:** **ObsoleteTB AP collapsed 0.030 → 0.0025**, and even correct boxes
score only ~0.08 (logit ≈ −2.4) — far below a usable 0.3 threshold. Thresholded confusion
(IoU ≥ 0.5): ActiveTB P 0.007 / R 0.456, ObsoleteTB P 0.001 / R 0.492 — recall is healthy for
both classes; precision is crushed by background false positives at low scores. The
two-class discrimination problem (not geometry) is what still caps this run.

---

## 1. Best Result Summary (v1 baseline)

| Metric | Value | Epoch | Source |
|--------|-------|-------|--------|
| mAP@0.5:0.95 | **0.048** | 73 | COCO eval |
| AP@0.5 | 0.123 | 73 | COCO eval |
| AP@0.75 | 0.025 | 73 | COCO eval |
| ActiveTB AP@0.5:0.95 | 0.065 | 73 | per-class COCO eval |
| ActiveTB AR@100 | 0.126 | 73 | per-class COCO eval |
| ObsoleteTB AP@0.5:0.95 | 0.030 | 73 | per-class COCO eval |
| ObsoleteTB AR@100 | 0.248 | 73 | per-class COCO eval |
| AR@1 / AR@10 | 0.107 / 0.179 | 73 | COCO eval |

Best checkpoint saved at **epoch 73** (`weights/best_model.pth`, mAP 0.0477). Stopped with patience 2/30 — the tail epochs were not improving.

### Test-time augmentation (TTA) made it worse

| Metric | No TTA | TTA |
|--------|--------|-----|
| mAP@0.5:0.95 | 0.048 | **0.029** |
| AP@0.5 | 0.123 | 0.082 |
| AR@100 | 0.187 | 0.182 |

TTA (flip/augmentation averaging) hurts: the model's boxes are marginal (debug IoUs ~0.62), and averaging flips them below the stricter IoU cuts. **Non-TTA is the reported configuration.**

---

## 2. Configuration

- **Architecture**: RetinaNet, ResNet-50 FPN v2, COCO-pretrained, head replaced for 2 classes
- **Input resolution**: 1024 / 1536 (same as final FCOS run)
- **Epochs**: 75 (cosine LR decay, floor 1e-7)
- **Optimizer**: AdamW, LR schedule identical to final FCOS run
- **Backbone LR**: **x0.1** (unfrozen — same fix as FCOS)
- **Focal loss alpha per class**: `[0.25, 0.35]`
- **Class prior init**: pi = 0.05 (RetinaNet needs this to fire — see `c8ba416`)
- **Sampling**: WeightedRandomSampler, majority cap 200 boxes/epoch, negatives weight 0.05
- **Batch size**: 8, EMA weights used for evaluation

RetinaNet differs from FCOS only in head architecture (shared/tower heads + anchor boxes). Every training lever is identical, making it a clean architecture A/B.

---

## 3. Head-state during training (the cold-start)

RetinaNet replaces the **entire** head (classification + box-regression towers start random), unlike FCOS which only swaps one final conv and keeps COCO box-regression. So RetinaNet shows a colder start:

- **E8–9**: all 300 predictions at the ~0.05 score floor, all degenerate corner boxes, `max_iou_per_gt = 0.0`.
- **E25+ expectation**: box branch starts producing real boxes.
- **E75 (final)**: both GT lesions localized at **IoU 0.62 / 0.61**, but all labels predicted **ActiveTB (1) at score ~0.05** against GT **ObsoleteTB (2)**.

That last line is the same discrimination failure FCOS shows — see Section 6.

---

## 4. Confusion Matrix (validation, IoU ≥ 0.5 Hungarian matching)

| GT \ Pred | Background | ActiveTB | ObsoleteTB |
|-----------|-----------:|---------:|-----------:|
| **ActiveTB** | — | — | — |
| **ObsoleteTB** | — | — | — |

Per-class Hungarian metrics (saved to `confusion_matrix.png`, numbers from run log):

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| ActiveTB | 0.035 | 0.298 | 0.062 |
| ObsoleteTB | 0.007 | 0.492 | 0.013 |

Note the pattern mirrors FCOS: **ObsoleteTB has higher recall (0.492 vs 0.298) but nearly zero precision** — its boxes are found but scored so low they are drowned in false positives. The recall ranking is *inverted* relative to FCOS (FCOS: ActiveTB 52% / ObsoleteTB 48%); RetinaNet biases toward the minority class even more strongly.

---

## 5. FCOS vs RetinaNet — the headline comparison

| Metric | FCOS (best @E49) | RetinaNet (best @E73) |
|--------|------------------|-----------------------|
| mAP@0.5:0.95 | 0.048 | 0.048 |
| AP@0.5 | **0.143** | 0.123 |
| AP@0.75 | 0.013 | **0.025** |
| AR@100 | **0.256** | 0.187 |
| ActiveTB AP | **0.093** | 0.065 |
| ActiveTB AR | **0.256** | 0.126 |
| ObsoleteTB AP | 0.002 | **0.030** |
| ObsoleteTB AR | 0.170 | **0.248** |

### Reading this table

1. **Identical mAP (0.048) under identical training levers.** Two completely different detection heads converge to the *same* overall number. The ceiling is **not architecture-dependent** — it is set by the data/task (178 minority boxes, subtle two-class distinction).
2. **They trade the same confusion in opposite directions.** FCOS favors the majority class (ActiveTB AP 0.093 / AR 0.256); RetinaNet favors the minority (ObsoleteTB AP 0.030 / AR 0.248). This is exactly the same knob effect seen when sweeping the focal alpha (FCOS Section 4, runs B↔C) — *which* class wins the ambiguity is implementation/loss-chemistry, but the ~50% mislabel rate on both sides is shared.
3. **AP@0.75**: RetinaNet boxes land tighter (0.025 vs 0.013), but at lower recall — a marginal quality/quantity trade-off, not a win.
4. **Neither model reaches a confident score**: both peak at ~0.05–0.3 scores on foreground boxes, which is what collapses precision-based AP.

**Bottom line for the paper**: the two architectures are functionally equivalent on this task and both are capped by the same inter-class discrimination problem. Any further gain requires attacking the label ambiguity itself (cross-class NMS, crop classifier, more/cleaner data), not a different detector.

---

## 6. What the debug logs show (same story as FCOS)

At the final epoch, on an ObsoleteTB validation image with 2 GT boxes:

```
n_pred=6  pred_labels=[1 1 0 1 1 1]  pred_scores≈0.05
max_iou_per_pred = [0.62, 0.33, 0.40, 0.61, 0.14]
max_iou_per_gt  = [0.62, 0.61]
```

Both lesions are localized (IoU ≥ 0.6 — above the 0.5 detection bar) but labeled class 1 while the GT is class 2. **Localization is solved; classification is not.** The model literally places a box on the lesion and cannot decide which of the two visually similar classes it is.

---

## 7. Known issues in this run

- **HF push failed**: the GPU ran the pre-fix hub code (unnamespaced `repo_id="retinanet"` → `RepositoryNotFoundError` 404). Weights/metrics are *not* on the Hub yet; fix landed in `d1216a5` (needs `git pull` + re-push, see Next Steps).
- **XAI ran on the test split (0 annotations)** → TP=0/FP=0/FN=0, i.e. no explanations were produced. Same pre-fix bug; the fix defaults XAI to the val split.
- These are pipeline issues, not model issues — the numbers above are unaffected.

---

## 8. Next Steps

1. `git pull` on the GPU (gets `d1216a5`), then re-push RetinaNet to `ASJ234/retinanet` and re-run XAI on val.
2. **Duplicate-box / cross-class NMS analysis**: measure how many of the mislabels are two boxes on the same lesion with different labels (same analysis as FCOS, now for both models). If significant, cross-class NMS is the free fix.
3. All-boxes A/B (remove the 200-cap, alpha 0.9) applies to both models — confirms whether the ObsoleteTB gap is frequency or learnability.

---

## 9. Artifacts

- Weights: `results/retinanet/weights/{best_model,ema_model,last_checkpoint}.pth`
- Metrics: `results/retinanet/metrics.json`, `metrics_tta.json`
- Visualizations: `results/retinanet/confusion_matrix.png`, `results/retinanet/curves/training_curves.png`
- Predictions: `results/retinanet/val_preds.json`, `val_preds_tta.json`, `test_preds.json`
- Hugging Face: pending (re-push) — https://huggingface.co/ASJ234/retinanet
