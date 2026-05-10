"""
Run MTL inference on a video file.

Usage:
    python app/infer_mtl_video.py --input video.mp4 --weights models/checkpoints/model_final.pth
"""
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from src.models.mtl_net import HybridMTLNet

KP_COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255)]


def run(input_path: str, weights: str, output_path: str) -> None:
    device = torch.device(TRAIN_CFG.device)
    model = HybridMTLNet(MODEL_CFG.num_classes, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
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

        seg_mask = torch.argmax(pred_seg, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[seg_mask == 1] = (frame[seg_mask == 1] * 0.6 + np.array([0, 200, 0]) * 0.4).astype(np.uint8)

        pose_np = pred_pose.squeeze().cpu().numpy()
        for k in range(MODEL_CFG.num_keypoints):
            _, max_val, _, loc = cv2.minMaxLoc(pose_np[k])
            if max_val < 0.1:
                continue
            px, py = int(loc[0] / tw * w), int(loc[1] / th * h)
            cv2.circle(frame, (px, py), 5, KP_COLORS[k], -1)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--weights", default="models/checkpoints/model_final.pth")
    parser.add_argument("--output", default="models/inference_output/output.mp4")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run(args.input, args.weights, args.output)
