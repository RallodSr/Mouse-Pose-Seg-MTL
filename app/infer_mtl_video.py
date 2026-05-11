"""
Run MTL inference on a video file.

Usage:
    python app/infer_mtl_video.py --input video.mp4 --weights models/checkpoints/model_final.pth
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

# BGR colors — one row per instance
MASK_COLORS = [
    np.array([0, 255, 0]),    # instance 0 → green
    np.array([255, 255, 0]),  # instance 1 → cyan
]
KP_COLORS = [
    (0, 0, 255), (0, 165, 255), (255, 0, 0),    # instance 0: nose(red) shoulder(orange) tail(blue)
    (255, 0, 255), (0, 255, 255), (128, 0, 128), # instance 1: nose(magenta) shoulder(yellow) tail(purple)
]


def run(input_path: str, weights: str, output_path: str) -> None:
    device = torch.device(TRAIN_CFG.device)
    model = HybridMTLNet(MODEL_CFG.num_instances, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tw, th = DATA_CFG.target_size

    print(f"Processing video: {input_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (tw, th))
        tensor = ((torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0 - mean) / std)
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_seg, pred_pose = model(tensor)

        seg_np  = torch.sigmoid(pred_seg).squeeze().cpu().numpy()  # (2, H, W)
        pose_np = pred_pose.squeeze().cpu().numpy()                 # (6, H, W)
        num_kp  = MODEL_CFG.num_keypoints // MODEL_CFG.num_instances
        min_mask_pixels = int(0.005 * tw * th)

        for n in range(MODEL_CFG.num_instances):
            mask_small = (seg_np[n] > 0.5).astype(np.uint8)
            if mask_small.sum() < min_mask_pixels:
                continue  # no mouse in this slot

            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[mask == 1] = (
                frame[mask == 1] * 0.6 + MASK_COLORS[n] * 0.4
            ).astype(np.uint8)

            for k in range(num_kp):
                ch = n * num_kp + k
                _, _, _, loc = cv2.minMaxLoc(pose_np[ch])
                px, py = int(loc[0] / tw * w), int(loc[1] / th * h)
                cv2.circle(frame, (px, py), 5, KP_COLORS[ch], -1)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   required=True, help="Input video path")
    parser.add_argument("--weights", default="models/checkpoints/model_final.pth")
    parser.add_argument("--output",  default="models/inference_output/output.mp4")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run(args.input, args.weights, args.output)
