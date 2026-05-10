"""
Run MTL inference on a folder of images and overlay segmentation mask + keypoints.

Usage:
    python app/infer_images.py --input data/images --weights models/checkpoints/model_final.pth
"""
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from src.models.mtl_net import HybridMTLNet

KP_COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255)]   # nose, shoulder, tail


def run(input_dir: str, weights: str, output_dir: str) -> None:
    device = torch.device(TRAIN_CFG.device)
    model = HybridMTLNet(MODEL_CFG.num_classes, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    image_files = list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpg"))
    print(f"Running inference on {len(image_files)} images...")

    for img_file in image_files:
        img_bgr = cv2.imread(str(img_file))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        # Preprocess
        img_resized = cv2.resize(img_rgb, DATA_CFG.target_size)
        tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_seg, pred_pose = model(tensor)

        # Segmentation overlay
        seg_mask = torch.argmax(pred_seg, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        seg_mask = cv2.resize(seg_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        overlay = img_bgr.copy()
        overlay[seg_mask == 1] = (overlay[seg_mask == 1] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)

        # Keypoint dots
        pose_np = pred_pose.squeeze().cpu().numpy()
        tw, th = DATA_CFG.target_size
        for k in range(MODEL_CFG.num_keypoints):
            _, max_val, _, loc = cv2.minMaxLoc(pose_np[k])
            if max_val < 0.1:
                continue
            px = int(loc[0] / tw * w_orig)
            py = int(loc[1] / th * h_orig)
            cv2.circle(overlay, (px, py), 5, KP_COLORS[k], -1)

        cv2.imwrite(str(out_path / img_file.name), overlay)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/images")
    parser.add_argument("--weights", default="models/checkpoints/model_final.pth")
    parser.add_argument("--output", default="models/inference_output")
    args = parser.parse_args()
    run(args.input, args.weights, args.output)
