"""
Failure-case montage for HybridMTLNet (reviewer-requested limitation figure).

Runs the trained joint 1:500 model over the test set, scores each image by
per-instance mIoU and PCK, and renders the worst cases (one per behavioral test
for diversity) with the predicted mask + skeleton overlaid and the ground-truth
keypoints drawn as white crosses, so the failure (missed mask, swapped/locked
keypoint, low-contrast miss) is visible. Runs on CPU by default so it does not
contend with a GPU training job.

Usage:
    python app/plot_failure_cases.py --n 4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from config import DATA_CFG, MODEL_CFG
from src.models.mtl_net import HybridMTLNet

NUM_KP = 3
SKELETON = [(0, 1), (1, 2)]
MASK_COLORS = [np.array([0, 255, 0]), np.array([255, 255, 0])]   # green / cyan (BGR)
KP_COLORS = [[(0, 0, 255), (0, 165, 255), (0, 255, 0)],
             [(255, 0, 255), (0, 255, 255), (200, 100, 0)]]
PCK_RADIUS = 0.05 * 256                                          # 12.8 px


def _brighten(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _load_gt(item, W, H):
    """Return (gt_masks list of bool HxW, gt_kps list of [3,2] in 256-space)."""
    img = cv2.imread(item["image_path"])
    h0, w0 = img.shape[:2]
    sx, sy = W / w0, H / h0
    masks, kps = [], []
    for mp, kp in zip(item.get("mask_paths", []), item.get("all_keypoints", [])):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        masks.append(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127)
        kps.append([[p[0] * sx, p[1] * sy] for p in kp])
    return masks, kps


@torch.no_grad()
def _infer(model, item, device, W, H):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_bgr = cv2.resize(cv2.imread(item["image_path"]), (W, H))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = ((torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0 - mean) / std).unsqueeze(0).to(device)
    seg, pose = model(t)
    return img_bgr, torch.sigmoid(seg)[0].cpu().numpy(), pose[0].cpu().numpy()


def _score(seg, pose, gt_masks, gt_kps):
    """Per-image mIoU + PCK with Hungarian mask matching."""
    pred_bin = [(seg[n] > 0.5) for n in range(MODEL_CFG.num_instances)]
    n_gt = len(gt_masks)
    if n_gt == 0:
        return 1.0, 1.0, {}
    n_pred = len(pred_bin)
    iou = np.zeros((n_pred, n_gt))
    for i in range(n_pred):
        for j in range(n_gt):
            inter = (pred_bin[i] & gt_masks[j]).sum()
            union = (pred_bin[i] | gt_masks[j]).sum()
            iou[i, j] = inter / union if union > 0 else 0.0
    row, col = linear_sum_assignment(1.0 - iou)
    match = {c: r for r, c in zip(row, col)}                 # gt_j -> pred_i
    ious = [iou[match[j], j] if j in match else 0.0 for j in range(n_gt)]

    hits = total = 0
    for j in range(n_gt):
        pi = match.get(j)
        for k in range(NUM_KP):
            gx, gy = gt_kps[j][k]
            if gx <= 0 or gy <= 0:
                continue
            total += 1
            if pi is not None:
                _, _, _, loc = cv2.minMaxLoc(pose[pi * NUM_KP + k])
                if np.hypot(loc[0] - gx, loc[1] - gy) <= PCK_RADIUS:
                    hits += 1
    pck = hits / total if total else 1.0
    return float(np.mean(ious)), float(pck), match


def _render(img_bgr, seg, pose, gt_kps, match, miou, pck, label, T=384):
    W = H = 256
    out = cv2.resize(_brighten(img_bgr), (T, T), interpolation=cv2.INTER_CUBIC)
    s = T / W
    for n in range(MODEL_CFG.num_instances):
        m = (seg[n] > 0.5).astype(np.uint8)
        if m.sum() < int(0.0008 * W * H):
            continue
        mb = cv2.resize(m, (T, T), interpolation=cv2.INTER_NEAREST)
        fill = out.copy(); fill[mb == 1] = MASK_COLORS[n]
        out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)
        cnts, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, [int(c) for c in MASK_COLORS[n]], 2)
        pts = []
        for k in range(NUM_KP):
            _, _, _, loc = cv2.minMaxLoc(pose[n * NUM_KP + k])
            pts.append((int(loc[0] * s), int(loc[1] * s)))
        for a, b in SKELETON:
            cv2.line(out, pts[a], pts[b], (255, 255, 255), 2)
        for k, p in enumerate(pts):
            cv2.circle(out, p, 6, KP_COLORS[n][k], -1)
            cv2.circle(out, p, 6, (255, 255, 255), 1)
    # GT keypoints as white crosses
    for kps in gt_kps:
        for (gx, gy) in kps:
            if gx > 0 and gy > 0:
                x, y = int(gx * s), int(gy * s)
                cv2.drawMarker(out, (x, y), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    cv2.rectangle(out, (0, T - 24), (T, T), (0, 0, 0), -1)
    cv2.putText(out, f"{label}  mIoU {miou:.2f}  PCK {pck:.2f}", (5, T - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/checkpoints/1_500/model_best.pth")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default="paper/figures/failure_cases.png")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = HybridMTLNet(MODEL_CFG.num_instances, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    W, H = DATA_CFG.target_size
    items = json.load(open(DATA_CFG.json_path))["test"]

    scored = []
    for it in items:
        gt_masks, gt_kps = _load_gt(it, W, H)
        if not gt_masks:
            continue
        img, seg, pose = _infer(model, it, device, W, H)
        miou, pck, match = _score(seg, pose, gt_masks, gt_kps)
        scored.append((miou + pck, miou, pck, it, gt_kps, match))   # lower sum = worse

    scored.sort(key=lambda r: r[0])
    # diversity: at most one per behavioral test prefix among the worst
    picked, seen = [], set()
    for tot, miou, pck, it, gt_kps, match in scored:
        pre = Path(it["image_path"]).name[:3]
        if pre in seen:
            continue
        seen.add(pre)
        img, seg, pose = _infer(model, it, device, W, H)
        tile = _render(img, seg, pose, gt_kps, match, miou, pck, pre)
        picked.append(tile)
        print(f"  worst[{len(picked)}] {Path(it['image_path']).name}  mIoU={miou:.3f} PCK={pck:.3f}")
        if len(picked) >= args.n:
            break

    grid = np.hstack(picked)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"Failure montage saved -> {out}  ({len(picked)} tiles)")


if __name__ == "__main__":
    main()
