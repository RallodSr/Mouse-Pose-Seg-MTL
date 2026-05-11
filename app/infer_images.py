"""
Run MTL inference on a folder of images and overlay segmentation mask + keypoints.

Usage:
    python app/infer_images.py --input data/images --weights models/checkpoints/model_final.pth
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import cv2
import numpy as np
import torch

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from src.models.mtl_net import HybridMTLNet

NUM_KP = 3  # keypoints per mouse (nose, shoulder, tail)

# BGR colors — one row per instance
MASK_COLORS = [
    np.array([0, 255, 0]),    # instance 0 → green
    np.array([255, 255, 0]),  # instance 1 → cyan
]
KP_COLORS = [
    (0, 0, 255), (0, 165, 255), (255, 0, 0),    # instance 0: nose(red) shoulder(orange) tail(blue)
    (255, 0, 255), (0, 255, 255), (128, 0, 128), # instance 1: nose(magenta) shoulder(yellow) tail(purple)
]


def run(input_dir: str, weights: str, output_dir: str) -> None:
    device = torch.device(TRAIN_CFG.device)
    model = HybridMTLNet(MODEL_CFG.num_instances, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tw, th = DATA_CFG.target_size

    image_files = list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpg"))
    print(f"Running inference on {len(image_files)} images...")

    for img_file in image_files:
        img_bgr = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        resized = cv2.resize(img_rgb, (tw, th))
        tensor = ((torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0 - mean) / std)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_seg, pred_pose = model(tensor)

        overlay = img_bgr.copy()

        seg_np  = torch.sigmoid(pred_seg).squeeze().cpu().numpy()  # (2, H, W)
        pose_np = pred_pose.squeeze().cpu().numpy()                 # (6, H, W)
        min_mask_pixels = int(0.005 * tw * th)  # instance valid if >0.5% of resized image

        for n in range(MODEL_CFG.num_instances):
            mask_small = (seg_np[n] > 0.5).astype(np.uint8)
            if mask_small.sum() < min_mask_pixels:
                continue  # no mouse in this instance slot

            mask = cv2.resize(mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            overlay[mask == 1] = (
                overlay[mask == 1] * 0.6 + MASK_COLORS[n] * 0.4
            ).astype(np.uint8)

            for k in range(NUM_KP):
                ch = n * NUM_KP + k
                _, _, _, loc = cv2.minMaxLoc(pose_np[ch])
                px = int(loc[0] / tw * w_orig)
                py = int(loc[1] / th * h_orig)
                cv2.circle(overlay, (px, py), 5, KP_COLORS[ch], -1)

        cv2.imwrite(str(out_path / img_file.name), overlay)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   default="data/images")
    parser.add_argument("--weights", default="models/checkpoints/model_final.pth")
    parser.add_argument("--output",  default="models/inference_output")
    args = parser.parse_args()
    run(args.input, args.weights, args.output)
