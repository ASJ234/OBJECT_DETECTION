# v1 — Baseline Results (pre calibration fixes)

This folder archives the **v1 baseline** results: the first training run
before the classification-collapse fixes (see `walkthrough.md` Fix Round 2).

- Commit of the training run: `3836bb6`
- Docs of the fixes applied afterward: `b48a07e`
- Models archived: `fcos/`, `retinanet/`
- Trained with the sampler/alpha scheme of that run (per-class focal alpha,
  majority-until-cap sampling, negative weight 0.05).

> Note: `config.json` files still point at `results/{fcos,retinanet}` (the
> original output paths); files were relocated here to freeze the v1 baseline
> before retraining.

## Validation metrics (COCO eval, val split)

| Metric | FCOS | RetinaNet |
|--------|------|-----------|
| mAP@0.5:0.95 | 0.0496 | 0.0477 |
| mAP@0.5 | 0.1522 | 0.1228 |
| mAP@0.75 | 0.0142 | 0.0255 |
| AR@1 | 0.1052 | 0.1070 |
| AR@10 | 0.2158 | 0.1795 |
| AR@100 | 0.2288 | 0.1869 |

## Known issues (addressed in v2)

- Detection scores pathologically low: FCOS max ~0.29, RetinaNet pinned at
  ~0.050–0.058; classification dominated by negative-only images.
- RetinaNet classification head randomly re-initialized (not warm-started).
- FCOS background centerness unconstrained -> tens of thousands of low-score
  detections.
- Per-class alpha computed from raw (unsampled) counts.
