import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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


class Trainer:
    def __init__(self, data_cfg, model_cfg, train_cfg):
        self.dcfg = data_cfg
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.out_dir = Path(train_cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self):
        train_dl, val_dl = self._build_loaders()
        model = HybridMTLNet(self.mcfg.num_classes, self.mcfg.num_keypoints).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.tcfg.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        crit_seg  = nn.CrossEntropyLoss()
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

            l_seg  = crit_seg(pred_seg, masks)
            l_pose = crit_pose(pred_pose, hms)
            loss   = self.tcfg.loss_seg_weight * l_seg + self.tcfg.loss_pose_weight * l_pose

            loss.backward()
            optimizer.step()

            totals["total_loss"] += loss.item()
            totals["loss_seg"]   += l_seg.item()
            totals["loss_pose"]  += l_pose.item()
            totals["miou"]       += calculate_miou(pred_seg, masks)
            totals["pck"]        += calculate_pck(pred_pose, hms, self.tcfg.pck_threshold)

        n = len(dataloader)
        return {k: v / n for k, v in totals.items()}

    def _val_epoch(self, model, dataloader, crit_seg, crit_pose) -> dict:
        model.eval()
        totals = {k: 0.0 for k in ["total_loss", "loss_seg", "loss_pose", "miou", "pck"]}

        with torch.no_grad():
            for imgs, masks, hms in dataloader:
                imgs, masks, hms = imgs.to(self.device), masks.to(self.device), hms.to(self.device)

                pred_seg, pred_pose = model(imgs)

                l_seg  = crit_seg(pred_seg, masks)
                l_pose = crit_pose(pred_pose, hms)
                loss   = self.tcfg.loss_seg_weight * l_seg + self.tcfg.loss_pose_weight * l_pose

                totals["total_loss"] += loss.item()
                totals["loss_seg"]   += l_seg.item()
                totals["loss_pose"]  += l_pose.item()
                totals["miou"]       += calculate_miou(pred_seg, masks)
                totals["pck"]        += calculate_pck(pred_pose, hms, self.tcfg.pck_threshold)

        n = len(dataloader)
        return {k: v / n for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_log(self) -> Path:
        """Create train_log.csv with header. Returns path."""
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
    # Plotting
    # ------------------------------------------------------------------

    def _build_loaders(self):
        with open(self.dcfg.json_path) as f:
            data = json.load(f)

        def make_loader(split_key: str, shuffle: bool):
            ds = MouseMTLDataset(data[split_key], self.dcfg.target_size, self.dcfg.sigma)
            return DataLoader(ds, batch_size=self.tcfg.batch_size, shuffle=shuffle,
                              num_workers=self.tcfg.num_workers, pin_memory=True)

        return make_loader("train", shuffle=True), \
               make_loader("val",   shuffle=False) if "val" in data else None

    def _plot_history(self, history: dict) -> None:
        epochs = range(1, len(history["total_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(epochs, history["loss_seg"],     color="tab:orange", label="Train")
        axes[0].plot(epochs, history["val_loss_seg"], color="tab:orange", linestyle="--", label="Val")
        axes[0].set(title="Segmentation Loss (CE)", xlabel="Epoch", ylabel="Loss")
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
