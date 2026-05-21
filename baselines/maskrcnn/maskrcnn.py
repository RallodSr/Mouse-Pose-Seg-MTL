"""
Mask R-CNN instance segmentation baseline (torchvision built-in).

Reuses data/dataset.json and reports mIoU with the SAME Hungarian-matched
per-instance IoU as HybridMTLNet (256x256, threshold 0.5), so the number is
directly comparable to the MTL result in the paper.

Usage (base env with torch cu128):
    python baselines/maskrcnn.py train --epochs 50
    python baselines/maskrcnn.py eval  --weights baselines/maskrcnn_best.pth
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from config import DATA_CFG

NUM_CLASSES   = 2          # background + mouse
MAX_INSTANCES = 2
SCORE_THRESH  = 0.3        # ปรับเป็น 0.3 แล้วลอง eval ใหม่
MASK_THRESH   = 0.5
WEIGHTS_PATH  = Path(__file__).parent / "maskrcnn_best.pth"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MouseDetDataset(Dataset):
    """Yields (image_tensor[0..1], target_dict) for torchvision detection models."""

    def __init__(self, data_list, target_size=(256, 256)):
        self.data = data_list
        self.W, self.H = target_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = cv2.cvtColor(cv2.imread(item["image_path"]), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.W, self.H))

        masks, boxes = [], []
        for mp in item.get("mask_paths", []):
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            m_bin = (m > 127).astype(np.uint8)
            if m_bin.sum() < 1:
                continue
            x, y, w, h = cv2.boundingRect(m_bin)
            if w < 1 or h < 1:
                continue
            masks.append(m_bin)
            boxes.append([x, y, x + w, y + h])

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if boxes:
            target = {
                "boxes":  torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.ones((len(boxes),), dtype=torch.int64),
                "masks":  torch.as_tensor(np.stack(masks), dtype=torch.uint8),
            }
        else:
            target = {
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks":  torch.zeros((0, self.H, self.W), dtype=torch.uint8),
            }
        return img_t, target


def _collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model():
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, NUM_CLASSES)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, NUM_CLASSES)
    return model


# ---------------------------------------------------------------------------
# mIoU (matches src/evaluation/metrics.calculate_miou semantics)
# ---------------------------------------------------------------------------

def _instance_ious(pred_masks, gt_masks):
    """Hungarian-match predicted to GT instances; return per-GT IoU list.
    Unmatched GT instances contribute IoU = 0."""
    n_gt = len(gt_masks)
    if n_gt == 0:
        return []
    if len(pred_masks) == 0:
        return [0.0] * n_gt

    n_pred = len(pred_masks)
    iou = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i in range(n_pred):
        for j in range(n_gt):
            inter = np.logical_and(pred_masks[i], gt_masks[j]).sum()
            union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
            iou[i, j] = inter / union if union > 0 else 0.0

    row, col = linear_sum_assignment(1.0 - iou)
    matched = {c: r for r, c in zip(row, col)}
    return [iou[matched[j], j] if j in matched else 0.0 for j in range(n_gt)]


@torch.no_grad()
def compute_miou(model, loader, device) -> float:
    model.eval()
    all_ious = []
    for imgs, targets in loader:
        imgs = [im.to(device) for im in imgs]
        outputs = model(imgs)
        for out, tgt in zip(outputs, targets):
            keep = out["scores"] >= SCORE_THRESH
            pred_masks = (out["masks"][keep, 0] > MASK_THRESH).cpu().numpy()
            pred_masks = list(pred_masks[:MAX_INSTANCES])
            gt_masks = [m.numpy().astype(bool) for m in tgt["masks"]]
            all_ious.extend(_instance_ious(pred_masks, gt_masks))
    return float(np.mean(all_ious)) if all_ious else 0.0


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def _loaders(splits, batch_size):
    with open(DATA_CFG.json_path) as f:
        data = json.load(f)
    out = {}
    for s in splits:
        ds = MouseDetDataset(data[s], DATA_CFG.target_size)
        out[s] = DataLoader(ds, batch_size=batch_size, shuffle=(s == "train"),
                            collate_fn=_collate, num_workers=2)
    return out


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = _loaders(["train", "val"], args.batch_size)

    model = build_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_miou = 0.0
    print(f"Device: {device} | Epochs: {args.epochs} | Batch: {args.batch_size}")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, targets in dl["train"]:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        scheduler.step()
        val_miou = compute_miou(model, dl["val"], device)
        print(f"Epoch {epoch+1:03d}/{args.epochs} | loss {running/len(dl['train']):.4f} | val mIoU {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> best saved (val mIoU {best_miou:.4f})")

    print(f"Done. Best val mIoU {best_miou:.4f} -> {WEIGHTS_PATH}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl = _loaders(["test"], args.batch_size)
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    miou = compute_miou(model, dl["test"], device)
    print(f"\nMask R-CNN Test mIoU : {miou:.4f}")
    return miou


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN segmentation baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--batch-size", type=int, default=4)
    pt.add_argument("--lr", type=float, default=0.005)

    pe = sub.add_parser("eval")
    pe.add_argument("--weights", default=str(WEIGHTS_PATH))
    pe.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()
    {"train": train, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
