"""
Loss Weight Ratio Experiment Runner
====================================
รัน 4 config ต่อเนื่อง แล้วเปรียบเทียบผลบน test set อัตโนมัติ

Experiments
-----------
  Ours      : seg_w=1, pose_w=1
  Baseline A: seg_w=1, pose_w=40
  Baseline B: seg_w=1, pose_w=80
  Baseline C: seg_w=1, pose_w=100

Usage
-----
  python run_experiments.py              # รันทุก experiment
  python run_experiments.py --eval-only  # เปรียบเทียบผลที่เทรนไว้แล้ว
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from src.data.dataset import MouseMTLDataset
from src.evaluation.metrics import calculate_miou, calculate_pck
from src.models.mtl_net import HybridMTLNet
from src.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {"name": "1_1",   "label": "1:1",   "seg_weight": 1, "pose_weight": 1},
    {"name": "1_100", "label": "1:100", "seg_weight": 1, "pose_weight": 100},
    {"name": "1_500", "label": "1:500", "seg_weight": 1, "pose_weight": 500},
    {"name": "1_1000", "label": "1:1000", "seg_weight": 1, "pose_weight": 1000},
]

RESULTS_JSON = Path("models/experiment_results.json")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def run_all() -> dict:
    """Train all experiments sequentially and return evaluation results."""
    results = _load_existing_results()

    for exp in EXPERIMENTS:
        if exp["name"] in results:
            print(f"\nSkipping {exp['name']} — already trained. Use --eval-only to re-evaluate.")
            continue

        print(f"\n{'='*65}")
        print(f"  Experiment : {exp['label']}")
        print(f"  seg_weight={exp['seg_weight']}  pose_weight={exp['pose_weight']}")
        print(f"{'='*65}")

        cfg = _make_cfg(exp)
        Trainer(DATA_CFG, MODEL_CFG, cfg).run()

        miou, pck = _evaluate_test(cfg)
        results[exp["name"]] = {
            "label":     exp["label"],
            "seg_weight":  exp["seg_weight"],
            "pose_weight": exp["pose_weight"],
            "test_miou": round(miou, 4),
            "test_pck":  round(pck, 4),
        }
        _save_results(results)   # บันทึกระหว่างทางเผื่อ crash

    return results


def eval_only() -> dict:
    """Re-evaluate all experiments using saved best weights (no re-training)."""
    results = {}
    for exp in EXPERIMENTS:
        cfg = _make_cfg(exp)
        weights = Path(cfg.output_dir) / "model_best.pth"
        if not weights.exists():
            weights = Path(cfg.output_dir) / "model_final.pth"
        if not weights.exists():
            print(f"No weights found for {exp['name']} — skipping.")
            continue

        print(f"Evaluating {exp['label']} from {weights} ...")
        miou, pck = _evaluate_test(cfg)
        results[exp["name"]] = {
            "label":      exp["label"],
            "seg_weight":  exp["seg_weight"],
            "pose_weight": exp["pose_weight"],
            "test_miou":  round(miou, 4),
            "test_pck":   round(pck, 4),
        }

    _save_results(results)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(exp: dict):
    cfg = copy.copy(TRAIN_CFG)
    cfg.loss_seg_weight  = exp["seg_weight"]
    cfg.loss_pose_weight = exp["pose_weight"]
    cfg.output_dir       = f"models/checkpoints/{exp['name']}"
    return cfg


def _evaluate_test(train_cfg) -> tuple[float, float]:
    device = torch.device(train_cfg.device)

    with open(DATA_CFG.json_path) as f:
        data = json.load(f)

    test_ds = MouseMTLDataset(data["test"], DATA_CFG.target_size, DATA_CFG.sigma)
    test_dl = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
    )

    weights = Path(train_cfg.output_dir) / "model_best.pth"
    if not weights.exists():
        weights = Path(train_cfg.output_dir) / "model_final.pth"

    model = HybridMTLNet(MODEL_CFG.num_instances, MODEL_CFG.num_keypoints).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    from src.training.trainer import _match_instances
    n_kp = MODEL_CFG.num_keypoints // MODEL_CFG.num_instances

    total_miou = total_pck = 0.0
    with torch.no_grad():
        for imgs, masks, hms in test_dl:
            imgs, masks, hms = imgs.to(device), masks.to(device), hms.to(device)
            pred_seg, pred_pose = model(imgs)
            masks_m, hms_m = _match_instances(pred_seg, masks, hms, n_kp)
            total_miou += calculate_miou(pred_seg, masks_m)
            total_pck  += calculate_pck(pred_pose, hms_m, train_cfg.pck_threshold)

    n = len(test_dl)
    return total_miou / n, total_pck / n


def _load_existing_results() -> dict:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f)
    return {}


def _save_results(results: dict) -> None:
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved → {RESULTS_JSON}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_comparison(results: dict) -> None:
    import matplotlib.pyplot as plt
    labels = [r["label"]     for r in results.values()]
    mious  = [r["test_miou"] for r in results.values()]
    pcks   = [r["test_pck"]  for r in results.values()]
    colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    def _bar(ax, values, title, ylabel):
        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
        ax.set(title=title, xlabel="Loss Weight Ratio (seg : pose)", ylabel=ylabel)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="target 0.8")
        ax.legend(fontsize=9)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.015,
                f"{v:.4f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

    _bar(axes[0], mious, "Test mIoU  (Segmentation)", "mIoU")
    _bar(axes[1], pcks,  "Test PCK   (Keypoint)",     "PCK")

    plt.suptitle("MTL Loss Weight Ratio Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = Path("models/experiment_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved → {out_path}")
    plt.show()


def print_table(results: dict) -> None:
    header = f"{'Experiment':<22} {'Ratio (seg:pose)':>18} {'mIoU':>10} {'PCK':>10}"
    line   = "─" * len(header)
    print(f"\n{line}\n{header}\n{line}")

    best_miou = max(r["test_miou"] for r in results.values())
    best_pck  = max(r["test_pck"]  for r in results.values())

    for name, r in results.items():
        miou_str = f"{r['test_miou']:.4f}" + (" ◀ best" if r["test_miou"] == best_miou else "")
        pck_str  = f"{r['test_pck']:.4f}"  + (" ◀ best" if r["test_pck"]  == best_pck  else "")
        print(f"{name:<22} {r['label']:>18} {miou_str:>10} {pck_str:>10}")

    print(line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run & compare weight ratio experiments")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training — only re-evaluate saved checkpoints",
    )
    args = parser.parse_args()

    results = eval_only() if args.eval_only else run_all()

    if results:
        print_table(results)
        plot_comparison(results)
    else:
        print("No results to display.")
