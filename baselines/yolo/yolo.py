"""YOLO baseline evaluation — mIoU (seg) and PCK (pose)."""
import json

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def _iou_np(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return inter / union if union > 0 else 0.0


def evaluate_yolo_miou(
    json_path: str,
    model_path: str,
    target_size: tuple = (256, 256),
) -> float:
    """Per-instance mIoU: Hungarian matching between YOLO predictions and GT instances."""
    from scipy.optimize import linear_sum_assignment

    model = YOLO(model_path)
    with open(json_path) as f:
        test_items = json.load(f).get("test", [])

    all_ious = []
    for item in tqdm(test_items, desc="YOLO-Seg mIoU"):
        img = cv2.imread(item["image_path"])
        if img is None:
            continue

        # GT per-instance masks
        gt_masks = []
        for m_path in item["mask_paths"]:
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                gt_masks.append(
                    (cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST) > 0)
                )
        if not gt_masks:
            continue

        # YOLO predicted per-instance masks
        results = model(cv2.resize(img, target_size), verbose=False, device="cpu")
        pred_masks = []
        if results[0].masks is not None:
            for m in results[0].masks.data.cpu().numpy():
                pred_masks.append(
                    (cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST) > 0.5)
                )

        if not pred_masks:
            # no predictions → all GT instances are missed (IoU = 0)
            all_ious.extend([0.0] * len(gt_masks))
            continue

        # Hungarian matching: minimize cost = 1 - IoU
        n_gt, n_pred = len(gt_masks), len(pred_masks)
        cost = np.ones((n_gt, n_pred), dtype=np.float32)
        for i, g in enumerate(gt_masks):
            for j, p in enumerate(pred_masks):
                cost[i, j] = 1.0 - _iou_np(g, p)

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            all_ious.append(_iou_np(gt_masks[r], pred_masks[c]))

        # unmatched GT instances (if fewer preds than GT)
        matched_gt = set(row_ind)
        for i in range(n_gt):
            if i not in matched_gt:
                all_ious.append(0.0)

    miou = float(np.mean(all_ious)) if all_ious else 0.0
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


def main():
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import DATA_CFG, TRAIN_CFG

    parser = argparse.ArgumentParser(description="YOLO baseline evaluation")
    parser.add_argument("--task", choices=["seg", "pose"], required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    if args.task == "seg":
        evaluate_yolo_miou(DATA_CFG.json_path, args.weights, DATA_CFG.target_size)
    else:
        evaluate_yolo_pck(DATA_CFG.json_path, args.weights,
                          DATA_CFG.target_size, TRAIN_CFG.pck_threshold)


if __name__ == "__main__":
    main()
