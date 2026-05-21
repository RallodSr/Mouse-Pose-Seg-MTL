"""Unified test-set evaluation for HybridMTLNet.

Replaces the duplicated eval loops previously in main.py, run_experiments.py
and run_ablation.py with a single reusable class.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MouseMTLDataset
from src.models.mtl_net import HybridMTLNet
from src.evaluation.metrics import calculate_miou, calculate_pck
from src.training.trainer import _match_instances, _match_instances_by_pose


class Evaluator:
    """Loads a trained HybridMTLNet checkpoint and reports test mIoU / PCK."""

    def __init__(self, data_cfg, model_cfg, train_cfg):
        self.dcfg = data_cfg
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.device = torch.device(train_cfg.device)
        self.n_kp = model_cfg.num_keypoints // model_cfg.num_instances

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_weights(output_dir: str) -> Path | None:
        """Prefer model_best.pth, fall back to model_final.pth."""
        for name in ("model_best.pth", "model_final.pth"):
            p = Path(output_dir) / name
            if p.exists():
                return p
        return None

    def _test_loader(self) -> DataLoader:
        with open(self.dcfg.json_path) as f:
            data = json.load(f)
        ds = MouseMTLDataset(data["test"], self.dcfg.target_size, self.dcfg.sigma)
        return DataLoader(ds, batch_size=self.tcfg.batch_size, shuffle=False,
                          num_workers=self.tcfg.num_workers)

    def _load_model(self, weights_path: str) -> HybridMTLNet:
        model = HybridMTLNet(self.mcfg.num_instances, self.mcfg.num_keypoints).to(self.device)
        model.load_state_dict(torch.load(weights_path, map_location=self.device))
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, weights_path: str, task: str = "joint") -> tuple[float, float]:
        """Return (mIoU, PCK) on the test set.

        task = "pose" uses heatmap-based instance matching (segmentation head
        untrained); otherwise matching is done on mask IoU.
        """
        model = self._load_model(weights_path)
        test_dl = self._test_loader()

        total_miou = total_pck = 0.0
        for imgs, masks, hms in test_dl:
            imgs, masks, hms = imgs.to(self.device), masks.to(self.device), hms.to(self.device)
            pred_seg, pred_pose = model(imgs)
            if task == "pose":
                masks_m, hms_m = _match_instances_by_pose(pred_pose, masks, hms, self.n_kp)
            else:
                masks_m, hms_m = _match_instances(pred_seg, masks, hms, self.n_kp)
            total_miou += calculate_miou(pred_seg, masks_m)
            total_pck += calculate_pck(pred_pose, hms_m, self.tcfg.pck_threshold)

        n = len(test_dl)
        return total_miou / n, total_pck / n
