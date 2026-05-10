"""YOLO baseline evaluation — mIoU (seg) and PCK (pose)."""
import json

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def evaluate_yolo_miou(
    json_path: str,
    model_path: str,
    target_size: tuple = (256, 256),
) -> float:
    model = YOLO(model_path)
    with open(json_path) as f:
        test_items = json.load(f).get("test", [])

    total_iou = 0.0
    for item in tqdm(test_items, desc="YOLO-Seg mIoU"):
        img = cv2.imread(item["image_path"])
        if img is None:
            continue
        h, w = img.shape[:2]

        true_mask = np.zeros((h, w), dtype=np.uint8)
        for m_path in item["mask_paths"]:
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                true_mask = np.maximum(true_mask, np.squeeze(m))
        true_bin = (cv2.resize(true_mask, target_size, interpolation=cv2.INTER_NEAREST) > 0).astype(int)

        results = model(cv2.resize(img, target_size), verbose=False, device="cpu")
        pred_bin = np.zeros(target_size, dtype=int)
        if results[0].masks is not None:
            for m in results[0].masks.data.cpu().numpy():
                m_r = cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST)
                pred_bin = np.maximum(pred_bin, (m_r > 0.5).astype(int))

        inter = (pred_bin & true_bin).sum()
        union = (pred_bin | true_bin).sum()
        total_iou += inter / union if union > 0 else 1.0

    miou = total_iou / len(test_items)
    print(f"YOLO-Seg mIoU: {miou:.4f} ({miou*100:.2f}%)")
    return miou


def evaluate_yolo_pck(
    json_path: str,
    model_path: str,
    target_size: tuple = (256, 256),
    threshold: float = 0.05,
) -> float:
    model = YOLO(model_path)
    with open(json_path) as f:
        test_items = json.load(f).get("test", [])

    allowed_dist = threshold * target_size[0]
    total_hits = total_kps = 0

    for item in tqdm(test_items, desc="YOLO-Pose PCK"):
        img = cv2.imread(item["image_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        sx, sy = target_size[0] / w, target_size[1] / h

        gt_list = [[[kp[0] * sx, kp[1] * sy] for kp in mouse_kps]
                   for mouse_kps in item["all_keypoints"]]

        results = model.predict(cv2.resize(img, target_size), device="cpu", verbose=False)
        pred_list = []
        if results[0].keypoints is not None and len(results[0].keypoints) > 0:
            pred_list = results[0].keypoints.xy.cpu().numpy()

        for gt_kps in gt_list:
            best_pred = None
            best_dist = float("inf")
            for pred_kps in pred_list:
                if len(pred_kps) == 0:
                    continue
                d = sum(
                    np.hypot(g[0] - p[0], g[1] - p[1])
                    for g, p in zip(gt_kps, pred_kps)
                    if g[0] > 0 and g[1] > 0 and p[0] > 0 and p[1] > 0
                )
                if d < best_dist:
                    best_dist, best_pred = d, pred_kps

            for i, (gx, gy) in enumerate(gt_kps):
                if gx <= 0 or gy <= 0:
                    continue
                total_kps += 1
                if best_pred is not None:
                    px, py = best_pred[i][0], best_pred[i][1]
                    if px > 0 and py > 0 and np.hypot(gx - px, gy - py) <= allowed_dist:
                        total_hits += 1

    pck = total_hits / total_kps if total_kps > 0 else 0.0
    print(f"YOLO-Pose PCK: {pck:.4f} ({pck*100:.2f}%)")
    return pck
