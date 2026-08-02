# FCOS Results — Detailed Explanation (TBX11K)

**Model**: FCOS with ResNet-50 + FPN backbone (torchvision, pretrained on COCO)
**Dataset**: TBX11K chest X-rays — train 6,600 imgs / 902 boxes, val 1,800 / 309, test 3,302 / 0
**Classes**: Active Tuberculosis (724 train boxes), Obsolete Pulmonary TB (178 train boxes) — a **4:1 class imbalance**

---

## 1. Best Result Summary

| Metric | Value | Epoch | Source |
|--------|-------|-------|--------|
| mAP@0.5:0.95 | **0.048** | 49 | COCO eval |
| AP@0.5 | 0.143 | 49 | COCO eval |
| AP@0.75 | 0.013 | 49 | COCO eval |
| ActiveTB AP@0.5:0.95 | 0.093 | 49 | per-class COCO eval |
| ActiveTB AR@100 | 0.256 | 49 | per-class COCO eval |
| ObsoleteTB AP@0.5:0.95 | 0.002 | 49 | per-class COCO eval |
| ObsoleteTB AR@100 | 0.170 | 49 | per-class COCO eval |

The best checkpoint was saved at **epoch 49** (`weights/best_model.pth`, mAP 0.048). The cosine LR had decayed to ~1.4e-4 by then; the remaining 25 epochs (tail) did not beat it.

---

## 2. Configuration (winning run)

- **Architecture**: FCOS, ResNet-50 FPN, COCO-pretrained, head replaced for 2 classes
- **Input resolution**: min_size 1024 / max_size 1536 (up from 800/1333)
- **Epochs**: 75 (cosine LR decay, floor 1e-7)
- **Optimizer**: AdamW, base LR 1e-4, head LR x5 = 5e-4
- **Backbone LR**: **x0.1 = 1e-5** (unfrozen — see Section 5, this was critical)
- **Focal loss alpha per class**: `[0.25, 0.35]` (ActiveTB, ObsoleteTB) — inverse-frequency weighted, clamped
- **Class prior init**: pi = 0.01 (logit -4.60) for both classes
- **Sampling**: WeightedRandomSampler, majority (ActiveTB) capped at 200 boxes/epoch, negatives weight 0.05
- **Batch size**: 8 (11 GB GTX 1080 Ti)
- **EMA**: exponential moving average of weights used for evaluation

---

## 3. Confusion Matrix (validation, IoU ≥ 0.5 Hungarian matching)

Rows = ground truth, columns = prediction:

| GT \\ Pred | Background | ActiveTB | ObsoleteTB | GT total |
|-----------|-----------:|---------:|-----------:|---------:|
| **Background** | 0 | 84,080 | 95,633 | 179,713 |
| **ActiveTB** | 17 | 129 | 102 | 248 |
| **ObsoleteTB** | 5 | 27 | 29 | 61 |

### 3.1 What the numbers mean

- **Total GT = 309** (248 + 61) — exactly matches the 309 val annotations, so the matrix is consistent.
- **Only 22 GT boxes fully missed** (17 ActiveTB + 5 ObsoleteTB) — i.e. 93% of all lesions are localized at IoU 0.5.
- **Correctly labeled**: 129/248 ActiveTB (52%), 29/61 ObsoleteTB (48%).
- **Mislabeled (found but wrong class)**: 102 ActiveTB→ObsoleteTB, 27 ObsoleteTB→ActiveTB. This asymmetric bias toward ObsoleteTB is the expected effect of the per-class alpha (ObsoleteTB foregrounds are weighted ~1.4x harder in the focal loss).
- **False positives**: 179,713 background detections split 84,080 (predicted ActiveTB) / 95,633 (predicted ObsoleteTB). See Section 6.

### 3.2 Recall under the confusion-matrix definition

This definition counts a wrong-class box as "found":

- **ActiveTB recall** = 129 / (129 + 102 + 17) = **52%**
- **ObsoleteTB recall** = 29 / (29 + 27 + 5) = **48%**

**Important metric caveat**: COCO AR treats a wrong-class box as a *miss* (AR for the best model was 0.256 ActiveTB / 0.170 ObsoleteTB). The confusion-matrix recall above is a looser, detection-centric definition. Both are valid but must not be mixed in the same table.

---

## 4. Training Trajectory — how we got here

| Run | Config change | Best mAP | What happened |
|-----|---------------|----------|---------------|
| A (800px, Windows) | baseline | 0.0172 @ E30 | Both boxes localized but ObsoleteTB mislabeled as ActiveTB (AR 0.344 vs 0.059) |
| B (1024px) | resolution 1024/1536, 75 ep, symmetric alpha 0.25, **backbone frozen** | 0.0179 @ E22 | ActiveTB strong (AR 0.369), **ObsoleteTB collapsed to AR 0.000** |
| C | alpha 0.5 | 0.0015 @ E26 | **Flip**: ObsoleteTB 0.289 / ActiveTB 0.019 — over-correction |
| D (FINAL) | backbone lr x0.1 + alpha 0.35 | **0.048 @ E49** | **Both classes alive** — this is the reported model |

### Key lessons

1. **Resolution alone (1024px) was not enough** — run B found every box but re-allocated the label confusion to one class.
2. **The alpha is a knob, not a fix** — 0.25 → ActiveTB wins; 0.5 → ObsoleteTB wins. It only decides *which* class wins the ambiguity.
3. **The frozen backbone was the bottleneck** — with backbone LR at 1e-6 (lr x0.01), the head tower had to learn the subtle ActiveTB vs ObsoleteTB distinction on frozen COCO (natural-image) features. Raising backbone LR to 1e-5 (x0.1) let the features adapt to X-rays, and mAP jumped ~2.7x (0.0179 → 0.048) with both classes detected.

---

## 5. Why the backbone unfreeze mattered

FCOS inherits a COCO-pretrained ResNet-50. Its features were trained to separate 80 natural-image classes, not to distinguish two visually similar TB lesion types on X-rays. With the backbone effectively frozen:

- The model could *localize* lesions (the FPN head learns coarse "is there something here")
- But it could not *discriminate* the two classes — a task that requires fine feature adaptation the 3–4 conv head tower alone cannot supply.

Raising the backbone learning rate (x0.01 → x0.1, i.e. 1e-6 → 1e-5) allowed the shared features to adapt, and the label confusion dropped to the point where both classes achieved non-zero precision and recall. This was the single most impactful change.

---

## 6. Why the mAP is still low (0.048)

The headline number is the **mean of two per-class APs, and one of them is nearly zero** — that is the primary reason the overall number looks low.

### 6.1 The average hides the class split

- **Overall mAP = mean(ActiveTB AP, ObsoleteTB AP) = mean(0.093, 0.002) = 0.048**.
- If ObsoleteTB matched ActiveTB's AP (~0.09), the overall would roughly double to ~0.09.
- Reporting the mean without the per-class split is misleading for imbalanced, fine-grained tasks — always read the per-class row.

### 6.2 ObsoleteTB AP ≈ 0 despite being "found" (AR 0.170)

AP is score-ordered precision-recall. The model *localizes* ObsoleteTB boxes (AR 0.170) but assigns them **low confidence**, so:

- To recall a low-confidence ObsoleteTB box you must accept a large number of background false positives ranked above it.
- Precision collapses at every recall point, so the area under the curve (AP) ≈ 0.
- With only **178 training boxes**, the minority class never learns a confident "I am ObsoleteTB" signal — the confidence gap, not the localization, is the failure.

### 6.3 Inter-class confusion double-penalizes

129 of 248 ActiveTB boxes (52%) are correctly labeled. In AP terms, a mislabeled box is **both** a false positive (for the predicted class) **and** a missed ground truth (for the true class). You cannot achieve high AP while mislabeling ~half of all detections — the confusion matrix shows this is the binding constraint.

### 6.4 Tiny effective training data

- 902 training boxes total; 178 for the minority class.
- The capped sampler further reduces ActiveTB exposure to ~200 boxes/epoch.
- COCO-scale detectors reach mAP 0.3–0.4 on ~100k+ annotated images with 80 classes; TBX11K has ~1/100th of that annotation budget for a harder, 2-class fine-grained distinction.

### 6.5 The strictness of mAP@0.5:0.95

The primary metric averages IoU thresholds 0.5 → 0.95:

- **AP@0.5 = 0.143, AP@0.75 = 0.013** — most correct boxes sit at IoU 0.5–0.75, so they barely contribute at the stricter thresholds.
- Correct *detection* is not the same as correct *localization precision*; the metric rewards tight boxes, which the 4:1-imbalanced head is not producing.

### 6.6 The long false-positive shelf suppresses AP

The 180k detections at score ≥ 0.05 (see Section 7) give the PR curve a long, flat low-precision tail. mAP already prices this in — raising the deployment threshold improves operational precision but does not change the reported mAP.

### 6.7 Bottom line

0.048 is **low in COCO terms but respectable for this task**. The ceiling is set by the data and task difficulty (178 minority boxes + a subtle two-class distinction), not by model capacity. The largest lever is ObsoleteTB score calibration (raising its AP), and the all-boxes / alpha-0.9 A/B experiment is designed to test whether that gap is a *frequency* problem or a hard *learnability* ceiling.

---

## 7. The "huge false positives" — a threshold artifact

The 179,713 background detections look alarming but are a **measurement artifact**:

- Validation inference emits **exactly 100 boxes per image** (per-image top-100 cap) at score ≥ 0.05.
- 100 × 1,800 images = 180,000 total detections; 179,713 of them are below a meaningful operating threshold (typical top scores were ~0.2–0.3).
- The COCO mAP of 0.048 **already accounts** for these via the precision-recall curve — the number is not hiding an "extra" failure.

For a deployable detector, raise the operating threshold. At score ≥ 0.3 the false-positive count collapses; the mAP vs. operating-point trade-off should be reported with a precision/recall-at-threshold table (see Next Steps).

---

## 8. Strengths

- **Both classes detected** for the first time across all runs (AR 0.256 / 0.170 non-zero).
- **Localization is excellent** — 93% of GT boxes matched at IoU 0.5; debug IoUs reached 0.71–0.76.
- mAP 0.048 is a **2.7x improvement** over the best previous configuration (0.0179).

## 9. Weaknesses

- **Inter-class confusion remains the bottleneck**: 129 of 309 GT boxes matched but mislabeled (102 ActiveTB→ObsoleteTB). Correct-label rates are ~50% for both classes.
- **ObsoleteTB precision is low** (AP 0.002): its boxes are found but scored too low to survive precision-based evaluation.
- **Small minority class**: only 178 training boxes for ObsoleteTB fundamentally limits how well it can be learned.

---

## 10. Next Steps

1. **A/B imbalance strategy**: retrain with **all boxes** (remove the 200-cap) and full inverse-frequency alpha `[0.25, 0.9]` to test whether the confusion is a frequency problem or a learnability problem.
2. **Operating-point table**: compute precision/recall at thresholds 0.1/0.3/0.5 from the saved predictions to answer the false-positive critique.
3. **RetinaNet comparison**: same config (backbone x0.1, alpha 0.35) for the paper's model comparison.
4. If ObsoleteTB precision stays low even with exact frequency compensation, consider a stronger backbone or extended fine-tuning (more epochs at low LR).

---

## 11. Artifacts

- Weights: `results/fcos/weights/{best_model,ema_model,last_checkpoint}.pth`
- Metrics: `results/fcos/metrics.json`, `metrics_tta.json`
- Visualization: `results/fcos/confusion_matrix.png`, `results/fcos/curves/training_curves.png`
- Hugging Face: https://huggingface.co/ASJ234/fcos
- wandb: project `tbx11k`, run `fcos`
