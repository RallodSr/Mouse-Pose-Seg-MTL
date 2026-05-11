import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.data.dataset import MouseMTLDataset
from src.evaluation.metrics import calculate_miou, calculate_pck
from src.models.mtl_net import HybridMTLNet

# CSV columns (order matters — header written once at start)
_LOG_FIELDS = [
    "epoch",
    "lr",
    "train_total_loss", "train_loss_seg", "train_loss_pose", "train_miou", "train_pck",
    "val_total_loss",   "val_loss_seg",   "val_loss_pose",   "val_miou",   "val_pck",
]


def _dice_loss(pred_logits: torch.Tensor, gt_masks: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss for per-instance binary segmentation (Milletari et al., V-Net 2016).
    Handles class imbalance between background and foreground naturally.
    Empty GT slots (padding for 1-mouse images) are skipped.

    pred_logits : (B, N, H, W) raw logits
    gt_masks    : (B, N, H, W) binary float
    """
    pred = torch.sigmoid(pred_logits)
    B, N = pred.shape[:2]

    losses = []
    for b in range(B):
        for n in range(N):
            if gt_masks[b, n].sum() < 1:   # empty slot — skip
                continue
            p = pred[b, n].reshape(-1)
            g = gt_masks[b, n].reshape(-1)
            inter = (p * g).sum()
            losses.append(1.0 - (2.0 * inter + smooth) / (p.sum() + g.sum() + smooth))

    if not losses:
        return pred_logits.sum() * 0.0   # zero with gradient attached
    return torch.stack(losses).mean()


def _match_instances(
    pred_seg_logits: torch.Tensor,
    gt_masks:        torch.Tensor,
    gt_hms:          torch.Tensor,
    n_kp:            int,
) -> tuple:
    """Hungarian matching: reorder GT instances to best align with predictions.

    Uses mask IoU as similarity. Matching decisions are detached from the graph
    so gradients flow only through the loss, not through the assignment.

    Args:
        pred_seg_logits : (B, N, H, W) raw logits
        gt_masks        : (B, N, H, W) binary float masks
        gt_hms          : (B, N*n_kp, H, W) GT heatmaps
        n_kp            : keypoints per instance

    Returns:
        matched_gt_masks, matched_gt_hms — same shape as inputs, GT permuted per sample
    """
    B, N = pred_seg_logits.shape[:2]
    pred_bin = (torch.sigmoid(pred_seg_logits) > 0.5).detach().cpu().numpy().astype(np.float32)
    gt_np    = gt_masks.detach().cpu().numpy().astype(np.float32)

    matched_masks = gt_masks.clone()
    matched_hms   = gt_hms.clone()

    for b in range(B):
        # N×N cost matrix: 1 - IoU(pred_i, gt_j)
        cost = np.ones((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                inter = (pred_bin[b, i] * gt_np[b, j]).sum()
                union = pred_bin[b, i].sum() + gt_np[b, j].sum() - inter
                cost[i, j] = 1.0 - inter / (union + 1e-6)

        _, col_ind = linear_sum_assignment(cost)
        # col_ind[i] = GT slot matched to pred slot i
        matched_masks[b] = gt_masks[b][col_ind]
        matched_hms[b]   = torch.cat([
            gt_hms[b, col_ind[i] * n_kp:(col_ind[i] + 1) * n_kp]
            for i in range(N)
        ])

    return matched_masks, matched_hms


class Trainer:
    def __init__(self, data_cfg, model_cfg, train_cfg):
        self.dcfg = data_cfg
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.out_dir = Path(train_cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n_kp = model_cfg.num_keypoints // model_cfg.num_instances  # kp per instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        train_dl, val_dl = self._build_loaders()
        model = HybridMTLNet(self.mcfg.num_instances, self.mcfg.num_keypoints).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.tcfg.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        crit_seg  = nn.BCEWithLogitsLoss()
        crit_pose = nn.MSELoss()

        history = {k: [] for k in [
            "total_loss", "loss_seg", "loss_pose", "miou", "pck",
            "val_total_loss", "val_loss_seg", "val_loss_pose", "val_miou", "val_pck",
        ]}

        best_val_pck = 0.0
        log_path = self._init_log()

        print(
            f"Output  : {self.out_dir}\n"
            f"Device  : {self.device} | Epochs: {self.tcfg.epochs} | Batch: {self.tcfg.batch_size}\n"
            f"Weights : seg={self.tcfg.loss_seg_weight}  pose={self.tcfg.loss_pose_weight}"
        )

        for epoch in range(self.tcfg.epochs):
            train_m = self._train_epoch(model, train_dl, optimizer, crit_seg, crit_pose)
            val_m   = self._val_epoch(model, val_dl, crit_seg, crit_pose)
            lr      = optimizer.param_groups[0]["lr"]

            for k, v in train_m.items():
                history[k].append(v)
            for k, v in val_m.items():
                history[f"val_{k}"].append(v)

            scheduler.step(val_m["total_loss"])

            # ── console ──────────────────────────────────────────────────
            print(
                f"Epoch {epoch+1:03d}/{self.tcfg.epochs} | lr={lr:.2e} | "
                f"Loss {train_m['total_loss']:.4f} "
                f"[Seg {train_m['loss_seg']:.4f}  Pose {train_m['loss_pose']:.4f}] | "
                f"mIoU {train_m['miou']:.4f} | PCK {train_m['pck']:.4f} || "
                f"Val Loss {val_m['total_loss']:.4f} | "
                f"Val mIoU {val_m['miou']:.4f} | Val PCK {val_m['pck']:.4f}"
            )

            # ── CSV log ───────────────────────────────────────────────────
            self._append_log(log_path, epoch + 1, lr, train_m, val_m)

            # ── checkpoint ────────────────────────────────────────────────
            if val_m["pck"] > best_val_pck:
                best_val_pck = val_m["pck"]
                torch.save(model.state_dict(), self.out_dir / "model_best.pth")
                print(f"  → Best saved  (Val PCK: {best_val_pck:.4f})")

            if (epoch + 1) % self.tcfg.checkpoint_interval == 0:
                torch.save(model.state_dict(), self.out_dir / f"checkpoint_epoch_{epoch+1:03d}.pth")

        torch.save(model.state_dict(), self.out_dir / "model_final.pth")
        self._plot_history(history)
        print(f"Log saved → {log_path}")
        return model, history

    # ------------------------------------------------------------------
    # Epoch loops
    # ------------------------------------------------------------------

    def _train_epoch(self, model, dataloader, optimizer, crit_seg, crit_pose) -> dict:
        model.train()
        totals = {k: 0.0 for k in ["total_loss", "loss_seg", "loss_pose", "miou", "pck"]}

        for imgs, masks, hms in dataloader:
            imgs, masks, hms = imgs.to(self.device), masks.to(self.device), hms.to(self.device)

            optimizer.zero_grad()
            pred_seg, pred_pose = model(imgs)

            masks_m, hms_m = _match_instances(pred_seg, masks, hms, self.n_kp)

            l_seg  = crit_seg(pred_seg, masks_m) + _dice_loss(pred_seg, masks_m)
            l_pose = crit_pose(pred_pose, hms_m)
            loss   = self.tcfg.loss_seg_weight * l_seg + self.tcfg.loss_pose_weight * l_pose

            loss.backward()
            optimizer.step()

            totals["total_loss"] += loss.item()
            totals["loss_seg"]   += l_seg.item()
            totals["loss_pose"]  += l_pose.item()
            totals["miou"]       += calculate_miou(pred_seg, masks_m)
            totals["pck"]        += calculate_pck(pred_pose, hms_m, self.tcfg.pck_threshold)

        n = len(dataloader)
        return {k: v / n for k, v in totals.items()}

    def _val_epoch(self, model, dataloader, crit_seg, crit_pose) -> dict:
        model.eval()
        totals = {k: 0.0 for k in ["total_loss", "loss_seg", "loss_pose", "miou", "pck"]}

        with torch.no_grad():
            for imgs, masks, hms in dataloader:
                imgs, masks, hms = imgs.to(self.device), masks.to(self.device), hms.to(self.device)

                pred_seg, pred_pose = model(imgs)

                masks_m, hms_m = _match_instances(pred_seg, masks, hms, self.n_kp)

                l_seg  = crit_seg(pred_seg, masks_m) + _dice_loss(pred_seg, masks_m)
                l_pose = crit_pose(pred_pose, hms_m)
                loss   = self.tcfg.loss_seg_weight * l_seg + self.tcfg.loss_pose_weight * l_pose

                totals["total_loss"] += loss.item()
                totals["loss_seg"]   += l_seg.item()
                totals["loss_pose"]  += l_pose.item()
                totals["miou"]       += calculate_miou(pred_seg, masks_m)
                totals["pck"]        += calculate_pck(pred_pose, hms_m, self.tcfg.pck_threshold)

        n = len(dataloader)
        return {k: v / n for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_log(self) -> Path:
        log_path = self.out_dir / "train_log.csv"
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_LOG_FIELDS).writeheader()
        return log_path

    def _append_log(self, log_path: Path, epoch: int, lr: float,
                    train_m: dict, val_m: dict) -> None:
        row = {
            "epoch": epoch,
            "lr":    f"{lr:.6f}",
            "train_total_loss": f"{train_m['total_loss']:.6f}",
            "train_loss_seg":   f"{train_m['loss_seg']:.6f}",
            "train_loss_pose":  f"{train_m['loss_pose']:.6f}",
            "train_miou":       f"{train_m['miou']:.6f}",
            "train_pck":        f"{train_m['pck']:.6f}",
            "val_total_loss":   f"{val_m['total_loss']:.6f}",
            "val_loss_seg":     f"{val_m['loss_seg']:.6f}",
            "val_loss_pose":    f"{val_m['loss_pose']:.6f}",
            "val_miou":         f"{val_m['miou']:.6f}",
            "val_pck":          f"{val_m['pck']:.6f}",
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_LOG_FIELDS).writerow(row)

    # ------------------------------------------------------------------
    # Data & Plotting
    # ------------------------------------------------------------------

    def _build_loaders(self):
        with open(self.dcfg.json_path) as f:
            data = json.load(f)

        def make_loader(split_key: str, shuffle: bool):
            ds = MouseMTLDataset(data[split_key], self.dcfg.target_size, self.dcfg.sigma,
                                 augment=(split_key == "train"))
            return DataLoader(ds, batch_size=self.tcfg.batch_size, shuffle=shuffle,
                              num_workers=self.tcfg.num_workers, pin_memory=True)

        return make_loader("train", shuffle=True), \
               make_loader("val",   shuffle=False) if "val" in data else None

    def _plot_history(self, history: dict) -> None:
        import matplotlib.pyplot as plt
        epochs = range(1, len(history["total_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs, history["loss_seg"],     color="tab:orange", label="Train")
        axes[0].plot(epochs, history["val_loss_seg"], color="tab:orange", linestyle="--", label="Val")
        axes[0].set(title="Segmentation Loss (BCE)", xlabel="Epoch", ylabel="Loss")
        axes[0].legend()

        axes[1].plot(epochs, history["loss_pose"],     color="tab:purple", label="Train")
        axes[1].plot(epochs, history["val_loss_pose"], color="tab:purple", linestyle="--", label="Val")
        axes[1].set(title="Pose Loss (MSE)", xlabel="Epoch", ylabel="Loss")
        axes[1].legend()

        axes[2].plot(epochs, history["miou"],     color="blue",  label="Train mIoU")
        axes[2].plot(epochs, history["val_miou"], color="blue",  linestyle="--", label="Val mIoU")
        axes[2].plot(epochs, history["pck"],      color="green", label="Train PCK")
        axes[2].plot(epochs, history["val_pck"],  color="green", linestyle="--", label="Val PCK")
        axes[2].axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="target 0.8")
        axes[2].set(title="Accuracy over Epochs", xlabel="Epoch", ylabel="Score")
        axes[2].legend()

        plt.tight_layout()
        out_path = self.out_dir / "training_curves.png"
        plt.savefig(out_path, dpi=150)
        print(f"Curves saved → {out_path}")
