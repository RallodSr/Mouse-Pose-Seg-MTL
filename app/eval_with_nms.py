"""
Instance-level precision / recall / mIoU with and without mask-overlap
suppression (reviewer-requested false-positive analysis).

HybridMTLNet emits a fixed 2-channel mask. On single-mouse images the surplus
slot usually replicates the instance, which a confidence/precision-aware metric
should count as a false positive. We therefore match predicted instances to
ground truth (greedy, IoU > 0.5 = true positive) and report precision, recall,
and matched-instance mIoU:

  * RAW : every non-empty channel is a predicted instance.
  * NMS : a channel is dropped if its mask overlaps a higher-confidence
          channel with IoU > 0.5 (simple overlap suppression).

This shows whether the duplicate slot inflates the metrics and whether a trivial
suppression removes it without hurting recall or segmentation quality.

Usage: python app/eval_with_nms.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

from config import DATA_CFG, MODEL_CFG
from src.models.mtl_net import HybridMTLNet

W, H = DATA_CFG.target_size
MIN_PX = int(0.0008 * W * H)          # ~52 px: same threshold used to render an instance
TP_IOU = 0.5                          # IoU for a predicted instance to count as a true positive
NMS_IOU = 0.5                         # overlap above which the lower-confidence channel is dropped


def _iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


def gt_masks(item):
    out = []
    for mp in item.get("mask_paths", []):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            out.append(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127)
    return [m for m in out if m.sum() > 0]


@torch.no_grad()
def pred_instances(model, item, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = cv2.resize(cv2.imread(item["image_path"]), (W, H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = ((torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0 - mean) / std).unsqueeze(0).to(device)
    seg = torch.sigmoid(model(t)[0])[0].cpu().numpy()
    insts = []
    for n in range(MODEL_CFG.num_instances):
        m = seg[n] > 0.5
        if m.sum() >= MIN_PX:
            conf = float(seg[n][m].mean())          # mean foreground probability = confidence proxy
            insts.append((conf, m))
    return insts


def nms(insts):
    kept = []
    for conf, m in sorted(insts, key=lambda x: -x[0]):
        if all(_iou(m, k) <= NMS_IOU for _, k in kept):
            kept.append((conf, m))
    return kept


def score(all_preds, all_gts):
    TP = FP = FN = 0
    matched_iou = []
    for preds, gts in zip(all_preds, all_gts):
        used = set()
        masks = [m for _, m in sorted(preds, key=lambda x: -x[0])]
        for pm in masks:
            best_j, best_iou = -1, 0.0
            for j, gm in enumerate(gts):
                if j in used:
                    continue
                iou = _iou(pm, gm)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= TP_IOU:
                TP += 1; used.add(best_j); matched_iou.append(best_iou)
            else:
                FP += 1
        FN += len(gts) - len(used)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    miou = float(np.mean(matched_iou)) if matched_iou else 0.0
    return P, R, miou, TP, FP, FN


def main():
    device = torch.device("cpu")
    model = HybridMTLNet(MODEL_CFG.num_instances, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load("models/checkpoints/1_500/model_best.pth", map_location=device))
    model.eval()

    items = json.load(open(DATA_CFG.json_path))["test"]
    raw, nmsed, gts = [], [], []
    for it in items:
        g = gt_masks(it)
        if not g:
            continue
        p = pred_instances(model, it, device)
        raw.append(p); nmsed.append(nms(p)); gts.append(g)

    for name, preds in (("RAW (no suppression)", raw), ("NMS (overlap > 0.5)", nmsed)):
        P, R, m, TP, FP, FN = score(preds, gts)
        print(f"{name:22s}  P={P:.4f}  R={R:.4f}  mIoU={m:.4f}  (TP={TP} FP={FP} FN={FN})")


if __name__ == "__main__":
    main()
