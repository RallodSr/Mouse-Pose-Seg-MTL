"""
Generate a qualitative-results grid for HybridMTLNet: per-instance segmentation
mask + keypoint skeleton overlaid on one test image from each behavioral paradigm.

Usage:
    python app/visualize_mtl.py --weights models/checkpoints/1_500/model_best.pth
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

NUM_KP = 3
SKELETON = [(0, 1), (1, 2)]  # nose-shoulder, shoulder-tail

MASK_COLORS = [np.array([0, 255, 0]), np.array([255, 255, 0])]   # green / cyan (BGR)
KP_COLORS = [
    [(0, 0, 255), (0, 165, 255), (0, 255, 0)],      # instance 0
    [(255, 0, 255), (0, 255, 255), (200, 100, 0)],  # instance 1
]


def _mask_area(item: dict) -> int:
    """Total GT foreground pixels — proxy for how clearly the mouse is visible."""
    total = 0
    for mp in item.get("mask_paths", []):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            total += int((m > 127).sum())
    return total


def _select_one_per_paradigm(test_items: list) -> list:
    """Pick, per paradigm, the test image with the largest mouse (clearest)."""
    groups: dict[str, list] = {}
    for s in test_items:
        groups.setdefault(Path(s["image_path"]).name[:3], []).append(s)
    chosen = []
    for prefix, items in groups.items():
        chosen.append(max(items, key=_mask_area))
    return chosen


def _overlay(model, item, device) -> np.ndarray:
    tw, th = DATA_CFG.target_size
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_bgr = cv2.imread(item["image_path"])
    img_bgr = cv2.resize(img_bgr, (tw, th))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = ((torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0 - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_seg, pred_pose = model(tensor)
    seg = torch.sigmoid(pred_seg).squeeze(0).cpu().numpy()   # (2, H, W)
    pose = pred_pose.squeeze(0).cpu().numpy()                # (6, H, W)

    out = img_bgr.copy()
    min_px = int(0.005 * tw * th)
    for n in range(MODEL_CFG.num_instances):
        m = (seg[n] > 0.5).astype(np.uint8)
        if m.sum() < min_px:
            continue
        out[m == 1] = (out[m == 1] * 0.55 + MASK_COLORS[n] * 0.45).astype(np.uint8)

        pts = []
        for k in range(NUM_KP):
            _, _, _, loc = cv2.minMaxLoc(pose[n * NUM_KP + k])
            pts.append((int(loc[0]), int(loc[1])))
        for a, b in SKELETON:
            cv2.line(out, pts[a], pts[b], KP_COLORS[n][0], 2)
        for k, p in enumerate(pts):
            cv2.circle(out, p, 5, KP_COLORS[n][k], -1)
            cv2.circle(out, p, 5, (255, 255, 255), 1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/checkpoints/1_500/model_best.pth")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--output", default="paper/figures/qualitative_results.png")
    args = parser.parse_args()

    device = torch.device(TRAIN_CFG.device)
    model = HybridMTLNet(MODEL_CFG.num_instances, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    with open(DATA_CFG.json_path) as f:
        test_items = json.load(f)["test"]
    samples = _select_one_per_paradigm(test_items)
    print(f"Rendering {len(samples)} paradigms...")

    tiles = [cv2.resize(_overlay(model, s, device), (256, 256)) for s in samples]
    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((256, 256, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    print(f"Grid saved -> {out}")


if __name__ == "__main__":
    main()
