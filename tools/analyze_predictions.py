#!/usr/bin/env python
"""Consolidated analysis of saved validation predictions vs COCO ground truth.

Reproduces the confusion-matrix matching (greedy, IoU >= 0.5, one pred per GT)
and additionally reports:
  1. Label-agnostic localization recall  (how good geometry is on its own)
  2. Mislabel structure + cross-class-NMS recoverability (same-lesion dups)
  3. Score statistics for TP / mislabeled / FP detections per class
  4. IoU histogram of matched boxes
  5. Precision / recall at operating thresholds 0.1 / 0.3 / 0.5
  6. Ensemble complementarity (union of two models' localizations)

Usage:
  python tools/analyze_predictions.py \
      --gt dataset/coco/val.json \
      --pred results/fcos/val_preds.json results/retinanet/val_preds.json
"""
import argparse
import json
from collections import defaultdict

NAMES = {1: "ActiveTB", 2: "ObsoleteTB"}


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def group_by_image(anns):
    out = defaultdict(list)
    for a in anns:
        out[a["image_id"]].append(a)
    return out


def match(gt_by_img, preds_by_img, score_min=0.0, thr=0.5):
    """Greedy per-image matching. Returns per-GT match info and per-pred usage.

    match_info: list of (gt_label, matched_pred_label or None, matched_iou or 0,
                          matched_score or 0)
    """
    results = []
    for img_id, gts in gt_by_img.items():
        ps = preds_by_img.get(img_id, [])
        used = set()
        for g in gts:
            label = g["category_id"]
            best, bi = None, thr
            for pi, p in enumerate(ps):
                if pi in used or p["score"] < score_min:
                    continue
                v = iou_xywh(g["bbox"], p["bbox"])
                if v > bi:
                    bi, best = v, pi
            if best is None:
                results.append((img_id, label, g["bbox"], None, 0.0, 0.0))
            else:
                used.add(best)
                p = ps[best]
                results.append(
                    (img_id, label, g["bbox"],
                     int(p["category_id"]), bi, float(p["score"]))
                )
    return results


def analyze(model, gt_by_img, preds_by_img):
    print("=" * 78)
    print(f"MODEL: {model}   preds={sum(len(v) for v in preds_by_img.values())}")
    print("=" * 78)

    total_gt = {1: 0, 2: 0}
    for gts in gt_by_img.values():
        for g in gts:
            total_gt[g["category_id"]] += 1

    m = match(gt_by_img, preds_by_img)
    correct = {1: 0, 2: 0}
    mis = {1: 0, 2: 0}
    missed = {1: 0, 2: 0}
    dup_ok = {1: 0, 2: 0}     # mislabels with correct-class box on same lesion
    loc = {1: 0, 2: 0}        # localized at IoU>=0.5, any label
    iou_hist = [0] * 6
    score_tp = defaultdict(list)
    score_mis = defaultdict(list)
    score_best = defaultdict(list)  # best pred score on each GT (any label)

    for img_id, gt_label, gt_bbox, p_label, iou, score in m:
        if p_label is None:
            missed[gt_label] += 1
            continue
        loc[gt_label] += 1
        iou_hist[min(int((iou - 0.5) // 0.1), 5)] += 1
        if p_label == gt_label:
            correct[gt_label] += 1
            score_tp[gt_label].append(score)
        else:
            mis[gt_label] += 1
            score_mis[gt_label].append(score)
            ps = preds_by_img[img_id]
            if any(
                p["category_id"] == gt_label
                and iou_xywh(gt_bbox, p["bbox"]) >= 0.5
                for p in ps
            ):
                dup_ok[gt_label] += 1

    # best-scoring pred per GT (label-agnostic) for score stats
    for img_id, gts in gt_by_img.items():
        ps = preds_by_img.get(img_id, [])
        for g in gts:
            if not ps:
                continue
            best = max(
                (iou_xywh(g["bbox"], p["bbox"]), p["score"]) for p in ps
            )
            score_best[g["category_id"]].append(best[1])

    print("\n--- 1. Confusion reproduction + label-agnostic headroom ---")
    for lab in (1, 2):
        c, ms, miss = correct[lab], mis[lab], missed[lab]
        print(f"  {NAMES[lab]:10s} GT={total_gt[lab]:3d}  correct={c:3d} "
              f"mislabeled={ms:3d}  missed={miss:3d}  "
              f"localized(any label)={loc[lab]:3d} ({100.0*loc[lab]/total_gt[lab]:.0f}%)")
    print("  -> label-agnostic localization recall (the geometry ceiling): "
          f"{100.0*sum(loc.values())/sum(total_gt.values()):.1f}%")

    print("\n--- 2. Cross-class NMS recoverability ---")
    for lab in (1, 2):
        if mis[lab]:
            print(f"  {NAMES[lab]:10s} mislabels={mis[lab]:3d}  "
                  f"with correct-class box on same lesion: {dup_ok[lab]:3d} "
                  f"({100.0*dup_ok[lab]/mis[lab]:.0f}%)")

    print("\n--- 3. Score percentiles (p50 / p90 / max) ---")
    for lab in (1, 2):
        row = []
        for name, arr in (("TP     ", score_tp[lab]),
                          ("mislab ", score_mis[lab])):
            if not arr:
                row.append(f"{name} -")
                continue
            a = sorted(arr)
            p50 = a[len(a) // 2]
            p90 = a[int(0.9 * len(a))]
            row.append(f"{name} {p50:.3f}/{p90:.3f}/{a[-1]:.3f}")
        print(f"  {NAMES[lab]:10s} " + "  ".join(row))
        b = sorted(score_best[lab])
        print(f"             best-on-GT   p50={b[len(b)//2]:.3f} "
              f"p90={b[int(0.9*len(b))]:.3f} max={b[-1]:.3f}")

    print("\n--- 4. IoU histogram of matched boxes (any label) ---")
    total = sum(iou_hist)
    for i, n in enumerate(iou_hist):
        lo, hi = 0.5 + 0.1 * i, 0.6 + 0.1 * i
        print(f"  IoU {lo:.1f}-{hi:.1f}: {n:4d} ({100.0*n/total:5.1f}%)")

    print("\n--- 5. Precision/Recall at operating thresholds (per class) ---")
    for t in (0.1, 0.3, 0.5):
        mt = match(gt_by_img, preds_by_img, score_min=t)
        row = []
        for lab in (1, 2):
            matched = sum(1 for (_, gl, _, pl, _, _) in mt if gl == lab and pl == lab)
            n_pred = sum(
                1 for ps in preds_by_img.values()
                for p in ps if p["category_id"] == lab and p["score"] >= t
            )
            p = matched / n_pred if n_pred else 0.0
            r = matched / total_gt[lab]
            row.append(f"{NAMES[lab]}: P={p:.3f} R={r:.3f} (n_pred={n_pred})")
        print(f"  thr>={t}: " + "  ".join(row))
    return m, total_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", nargs="+", required=True,
                    help="paths to val_preds.json files (name:path pairs)")
    ap.add_argument("--name", nargs="*", default=None)
    args = ap.parse_args()

    gt = json.load(open(args.gt))
    gt_by_img = group_by_image(gt["annotations"])
    gt_total = sum(len(v) for v in gt_by_img.values())
    print(f"GT boxes: {gt_total}")

    all_matches = {}
    preds_by_model = {}
    for path in args.pred:
        name = path.split("/")[-2]
        data = json.load(open(path))
        preds = defaultdict(list)
        for p in data.get("predictions", []):
            preds[p["image_id"]].append(p)
        preds_by_model[name] = preds
        m, _ = analyze(name, gt_by_img, preds)
        all_matches[name] = m

    if len(preds_by_model) == 2:
        a, b = list(preds_by_model.keys())
        print("\n" + "=" * 78)
        print(f"ENSEMBLE COMPLEMENTARITY ({a} + {b}, label-agnostic IoU>=0.5)")
        print("=" * 78)
        got_a = {(i, gl) for (i, gl, _, pl, _, _) in all_matches[a] if pl is not None}
        got_b = {(i, gl) for (i, gl, _, pl, _, _) in all_matches[b] if pl is not None}
        both = got_a & got_b
        only_a = got_a - got_b
        only_b = got_b - got_a
        union = got_a | got_b
        print(f"  localized by both : {len(both):3d}")
        print(f"  only {a:10s}      : {len(only_a):3d}")
        print(f"  only {b:10s}      : {len(only_b):3d}")
        print(f"  union localizes   : {len(union):3d} / {gt_total} "
              f"({100.0*len(union)/gt_total:.0f}%)  "
              f"(vs {a}={len(got_a)} / {b}={len(got_b)})")


if __name__ == "__main__":
    main()
