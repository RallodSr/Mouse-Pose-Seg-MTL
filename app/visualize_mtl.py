"""
Qualitative-results grid for HybridMTLNet: per-instance segmentation mask +
keypoint skeleton on one test image from each behavioral paradigm.

Workflow: random-sample one image per paradigm, keep the good tiles, and
re-roll only the bad ones. The chosen images are persisted to a JSON so the
figure is reproducible and locked tiles stay fixed across re-rolls.

Usage:
    python app/visualize_mtl.py                      # sample all paradigms
    python app/visualize_mtl.py --reroll NOT mwm     # re-randomize only these
"""
import argparse
import json
import random
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
MIN_MASK_FRAC = 0.0008       # min predicted-mask fraction to draw an instance (~52px @256)

MASK_COLORS = [np.array([0, 255, 0]), np.array([255, 255, 0])]   # green / cyan (BGR)
KP_COLORS = [
    [(0, 0, 255), (0, 165, 255), (0, 255, 0)],      # instance 0
    [(255, 0, 255), (0, 255, 255), (200, 100, 0)],  # instance 1
]

SELECTION_JSON = Path("paper/figures/qual_selection.json")


def _brighten(img_bgr: np.ndarray) -> np.ndarray:
    """Boost visibility of dark night-vision footage via CLAHE on luminance."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _overlay(model, item, device, render=384) -> tuple[np.ndarray, int]:
    tw, th = DATA_CFG.target_size
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_bgr = cv2.resize(cv2.imread(item["image_path"]), (tw, th))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = ((torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0 - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_seg, pred_pose = model(tensor)
    seg = torch.sigmoid(pred_seg).squeeze(0).cpu().numpy()   # (2, H, W)
    pose = pred_pose.squeeze(0).cpu().numpy()                # (6, H, W)

    out = cv2.resize(_brighten(img_bgr), (render, render), interpolation=cv2.INTER_CUBIC)
    sx, sy = render / tw, render / th
    min_px = int(MIN_MASK_FRAC * tw * th)

    n_valid = 0
    for n in range(MODEL_CFG.num_instances):
        m = (seg[n] > 0.5).astype(np.uint8)
        if m.sum() < min_px:
            continue
        n_valid += 1
        m_big = cv2.resize(m, (render, render), interpolation=cv2.INTER_NEAREST)
        fill = out.copy()
        fill[m_big == 1] = MASK_COLORS[n]
        out = cv2.addWeighted(fill, 0.35, out, 0.65, 0)
        contours, _ = cv2.findContours(m_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, [int(c) for c in MASK_COLORS[n]], 2)

        pts = []
        for k in range(NUM_KP):
            _, _, _, loc = cv2.minMaxLoc(pose[n * NUM_KP + k])
            pts.append((int(loc[0] * sx), int(loc[1] * sy)))
        for a, b in SKELETON:
            cv2.line(out, pts[a], pts[b], (255, 255, 255), 2)
        for k, p in enumerate(pts):
            cv2.circle(out, p, 7, KP_COLORS[n][k], -1)
            cv2.circle(out, p, 7, (255, 255, 255), 2)
    return out, n_valid


def _select(test_items, reroll, locked):
    """Return {prefix: item}. Locked prefixes (not in reroll) reuse the saved
    image; everything else is randomly sampled."""
    groups: dict[str, list] = {}
    for s in test_items:
        groups.setdefault(Path(s["image_path"]).name[:3], []).append(s)

    chosen = {}
    for prefix, items in sorted(groups.items()):
        saved = locked.get(prefix)
        if saved and prefix not in reroll:
            match = [it for it in items if Path(it["image_path"]).name == saved]
            chosen[prefix] = match[0] if match else random.choice(items)
        else:
            chosen[prefix] = random.choice(items)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/checkpoints/1_500/model_best.pth")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--reroll", nargs="*", default=[],
                        help="Paradigm prefixes to re-randomize (others stay locked)")
    parser.add_argument("--output", default="paper/figures/qualitative_results.png")
    args = parser.parse_args()

    device = torch.device(TRAIN_CFG.device)
    model = HybridMTLNet(MODEL_CFG.num_instances, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    with open(DATA_CFG.json_path) as f:
        test_items = json.load(f)["test"]

    locked = {}
    if SELECTION_JSON.exists():
        locked = json.loads(SELECTION_JSON.read_text())

    chosen = _select(test_items, set(args.reroll), locked)

    T = 384
    tiles, selection = [], {}
    for prefix, item in chosen.items():
        tile, n_inst = _overlay(model, item, device, render=T)
        name = Path(item["image_path"]).name
        selection[prefix] = name
        print(f"  {prefix}: {name}  ({n_inst} instance(s) seg+pose)")
        cv2.rectangle(tile, (0, 0), (78, 26), (0, 0, 0), -1)
        cv2.putText(tile, prefix, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)

    cols = args.cols
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros((T, T, 3), dtype=np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    SELECTION_JSON.write_text(json.dumps(selection, indent=2))
    print(f"Grid saved -> {out}\nSelection saved -> {SELECTION_JSON}")


if __name__ == "__main__":
    main()
