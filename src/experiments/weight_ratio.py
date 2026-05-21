"""Loss-weight ratio experiment: compare seg:pose weightings on the joint model."""
from pathlib import Path

from .base import Experiment

RUNS = [
    {"name": "1_1",    "label": "1:1",    "seg_weight": 1, "pose_weight": 1},
    {"name": "1_100",  "label": "1:100",  "seg_weight": 1, "pose_weight": 100},
    {"name": "1_400",  "label": "1:400",  "seg_weight": 1, "pose_weight": 400},  # chosen value
    {"name": "1_500",  "label": "1:500",  "seg_weight": 1, "pose_weight": 500},
    {"name": "1_1000", "label": "1:1000", "seg_weight": 1, "pose_weight": 1000},
]


class WeightRatioExperiment(Experiment):
    results_filename = "experiment_results.json"

    def configs(self) -> list[dict]:
        return RUNS

    def make_cfg(self, run: dict):
        cfg = self._base_cfg(run)
        cfg.task = "joint"
        cfg.loss_seg_weight = run["seg_weight"]
        cfg.loss_pose_weight = run["pose_weight"]
        cfg.output_dir = f"models/checkpoints/{run['name']}"
        return cfg

    def record(self, run: dict, miou: float, pck: float) -> dict:
        return {
            "label":       run["label"],
            "seg_weight":  run["seg_weight"],
            "pose_weight": run["pose_weight"],
            "test_miou":   round(miou, 4),
            "test_pck":    round(pck, 4),
        }

    def print_table(self, results: dict) -> None:
        header = f"{'Experiment':<22} {'Ratio (seg:pose)':>18} {'mIoU':>10} {'PCK':>10}"
        line = "-" * len(header)
        print(f"\n{line}\n{header}\n{line}")
        best_miou = max(r["test_miou"] for r in results.values())
        best_pck = max(r["test_pck"] for r in results.values())
        for name, r in results.items():
            miou = f"{r['test_miou']:.4f}" + (" *" if r["test_miou"] == best_miou else "")
            pck = f"{r['test_pck']:.4f}" + (" *" if r["test_pck"] == best_pck else "")
            print(f"{name:<22} {r['label']:>18} {miou:>10} {pck:>10}")
        print(line)

    def plot(self, results: dict) -> None:
        import matplotlib.pyplot as plt
        labels = [r["label"] for r in results.values()]
        mious = [r["test_miou"] for r in results.values()]
        pcks = [r["test_pck"] for r in results.values()]
        colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        def _bar(ax, values, title, ylabel):
            bars = ax.bar(labels, values, color=colors[:len(values)], edgecolor="white", width=0.5)
            ax.set(title=title, xlabel="Loss Weight Ratio (seg : pose)", ylabel=ylabel)
            ax.set_ylim(0, 1.05)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.4f}",
                        ha="center", fontsize=10, fontweight="bold")

        _bar(axes[0], mious, "Test mIoU (Segmentation)", "mIoU")
        _bar(axes[1], pcks, "Test PCK (Keypoint)", "PCK")
        plt.suptitle("MTL Loss Weight Ratio Comparison", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = Path("models/experiment_comparison.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Chart saved -> {out}")
