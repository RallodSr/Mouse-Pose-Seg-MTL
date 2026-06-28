"""
Empty-slot false-positive check (reviewer-requested).

HybridMTLNet emits a fixed 2-channel mask, so on a single-mouse image one slot
should be left empty. This measures how often the model instead hallucinates a
second instance: for every single-mouse test image we count predicted mask
channels whose foreground area exceeds the same threshold used to render an
instance; >=2 non-empty channels is an empty-slot false positive.

Usage: python app/measure_empty_slot_fp.py
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

MIN_FRAC = 0.0008          # same threshold the qualitative figure uses to draw an instance
W, H = DATA_CFG.target_size


@torch.no_grad()
def pred_instance_count(model, item, device, min_px):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = cv2.resize(cv2.imread(item["image_path"]), (W, H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = ((torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0 - mean) / std).unsqueeze(0).to(device)
    seg, _ = model(t)
    seg = torch.sigmoid(seg)[0].cpu().numpy()
    return sum(int((seg[n] > 0.5).sum()) >= min_px for n in range(MODEL_CFG.num_instances))


def main():
    device = torch.device("cpu")
    model = HybridMTLNet(MODEL_CFG.num_instances, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load("models/checkpoints/1_500/model_best.pth", map_location=device))
    model.eval()
    min_px = int(MIN_FRAC * W * H)

    items = json.load(open(DATA_CFG.json_path))["test"]
    n_single = n_fp = 0
    for it in items:
        n_gt = len(it.get("mask_paths", []))
        if n_gt != 1:                       # only single-mouse images have an empty slot
            continue
        n_single += 1
        if pred_instance_count(model, it, device, min_px) >= 2:
            n_fp += 1

    rate = n_fp / n_single if n_single else 0.0
    print(f"single-mouse test images : {n_single}")
    print(f"empty-slot false positives: {n_fp}")
    print(f"FP rate                   : {rate:.4f}  ({rate*100:.2f}%)")
    print(f"correctly leaves slot empty: {(1-rate)*100:.2f}%")


if __name__ == "__main__":
    main()
