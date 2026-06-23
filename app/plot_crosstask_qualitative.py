"""
Qualitative evidence for the cross-task mechanism: side-by-side predictions of
the plain joint model vs. the bidirectional (joint_bi) model on the SAME test
images. We rank images by how far the predicted keypoints move between the two
models (largest = where mask-gating most strongly suppresses a spurious
background peak) and render the top-K as a contact sheet to curate from.

Run AFTER training, when the GPU is free:
    python app/plot_crosstask_qualitative.py \
        --baseline models/checkpoints/seed42_joint/model_best.pth \
        --bi       models/checkpoints/seed42_joint_bi/model_best.pth
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from src.models.mtl_net import HybridMTLNet
from app.visualize_mtl import _brighten, NUM_KP, SKELETON, MASK_COLORS, KP_COLORS, MIN_MASK_FRAC

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load(weights, mask_guided, pose_guided, device):
    m = HybridMTLNet(MODEL_CFG.num_instances, MODEL_CFG.num_keypoints,
                     mask_guided=mask_guided, pose_guided=pose_guided).to(device)
    m.load_state_dict(torch.load(weights, map_location=device))
    m.eval()
    return m


def _predict(model, item, device):
    """Return (seg sigmoid [N,H,W], keypoints[N][NUM_KP] in 256-space)."""
    tw, th = DATA_CFG.target_size
    img_bgr = cv2.resize(cv2.imread(item["image_path"]), (tw, th))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = ((torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0 - _MEAN) / _STD).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_seg, pred_pose = model(tensor)
    seg = torch.sigmoid(pred_seg).squeeze(0).cpu().numpy()
    pose = pred_pose.squeeze(0).cpu().numpy()
    kpts = []
    for n in range(MODEL_CFG.num_instances):
        pts = []
        for k in range(NUM_KP):
            _, _, _, loc = cv2.minMaxLoc(pose[n * NUM_KP + k])
            pts.append((int(loc[0]), int(loc[1])))
        kpts.append(pts)
    return seg, kpts, img_bgr


def _render(seg, kpts, img_bgr, render=384):
    tw, th = DATA_CFG.target_size
    out = cv2.resize(_brighten(img_bgr), (render, render), interpolation=cv2.INTER_CUBIC)
    sx, sy = render / tw, render / th
    min_px = int(MIN_MASK_FRAC * tw * th)
    for n in range(MODEL_CFG.num_instances):
        m = (seg[n] > 0.5).astype(np.uint8)
        if m.sum() < min_px:
            continue
        m_big = cv2.resize(m, (render, render), interpolation=cv2.INTER_NEAREST)
        fill = out.copy()
        fill[m_big == 1] = MASK_COLORS[n]
        out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)
        contours, _ = cv2.findContours(m_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, [int(c) for c in MASK_COLORS[n]], 2)
        pts = [(int(x * sx), int(y * sy)) for (x, y) in kpts[n]]
        for a, b in SKELETON:
            cv2.line(out, pts[a], pts[b], (255, 255, 255), 2)
        for k, p in enumerate(pts):
            cv2.circle(out, p, 7, KP_COLORS[n][k], -1)
            cv2.circle(out, p, 7, (255, 255, 255), 2)
    return out


def _label(tile, text):
    cv2.rectangle(tile, (0, 0), (8 + 12 * len(text), 26), (0, 0, 0), -1)
    cv2.putText(tile, text, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="models/checkpoints/seed42_joint/model_best.pth")
    ap.add_argument("--bi", default="models/checkpoints/seed42_joint_bi/model_best.pth")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--output", default="paper/figures/crosstask_qualitative_contact.png")
    args = ap.parse_args()

    device = torch.device(TRAIN_CFG.device)
    base = _load(args.baseline, False, False, device)
    bi = _load(args.bi, True, True, device)

    with open(DATA_CFG.json_path) as f:
        test_items = json.load(f)["test"]

    ranked = []
    for item in test_items:
        seg_b, kp_b, img = _predict(base, item, device)
        seg_i, kp_i, _ = _predict(bi, item, device)
        disp = np.mean([np.hypot(kp_b[n][k][0] - kp_i[n][k][0], kp_b[n][k][1] - kp_i[n][k][1])
                        for n in range(MODEL_CFG.num_instances) for k in range(NUM_KP)])
        ranked.append((disp, item, seg_b, kp_b, seg_i, kp_i, img))

    ranked.sort(key=lambda r: -r[0])
    rows = []
    print("Top keypoint-shift images (baseline -> joint_bi):")
    for disp, item, seg_b, kp_b, seg_i, kp_i, img in ranked[:args.topk]:
        name = Path(item["image_path"]).name
        print(f"  {name}: mean kp shift = {disp:.1f}px")
        tb = _label(_render(seg_b, kp_b, img), "joint")
        ti = _label(_render(seg_i, kp_i, img), "joint_bi")
        rows.append(np.hstack([tb, ti]))

    grid = np.vstack(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"Contact sheet -> {out}  ({args.topk} rows: left=joint, right=joint_bi)")


if __name__ == "__main__":
    main()
