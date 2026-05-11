import cv2
import numpy as np
import torch


def calculate_miou(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> float:
    pred = torch.argmax(pred_logits, dim=1).view(-1).cpu().numpy()
    gt = target.view(-1).cpu().numpy()
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (gt == cls)).sum()
        union = ((pred == cls) | (gt == cls)).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def calculate_pck(pred_hms: torch.Tensor, target_hms: torch.Tensor, threshold: float = 0.05) -> float:
    """PCK (Yang & Ramanan, CVPR 2013): fraction of keypoints within threshold * max(H,W) pixels.
    Keypoints annotated as invisible (x=0, y=0) produce all-zero GT heatmaps and are skipped,
    equivalent to the standard visibility-flag convention used in coordinate-based evaluation.
    """
    _, _, h, w = pred_hms.shape
    allowed_dist = threshold * max(h, w)

    pred_np = pred_hms.detach().cpu().numpy()
    gt_np = target_hms.detach().cpu().numpy()

    hits = []
    for b in range(pred_np.shape[0]):
        for k in range(pred_np.shape[1]):
            _, gt_peak, _, t_loc = cv2.minMaxLoc(gt_np[b, k])
            if gt_peak < 1e-6:  # all-zero heatmap → keypoint was invisible in annotation
                continue
            _, _, _, p_loc = cv2.minMaxLoc(pred_np[b, k])
            dist = np.hypot(p_loc[0] - t_loc[0], p_loc[1] - t_loc[1])
            hits.append(1 if dist <= allowed_dist else 0)

    return float(np.mean(hits)) if hits else 0.0
