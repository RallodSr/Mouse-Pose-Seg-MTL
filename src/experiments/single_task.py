"""Single-task ablation: same network as seg-only / pose-only / joint.

Isolates whether multi-task learning helps, hurts, or is neutral relative to
training each task alone on the identical architecture.
"""
from .base import Experiment

RUNS = [
    {"name": "seg_only",  "label": "Seg-only",  "task": "seg",   "seg_weight": 1, "pose_weight": 0},
    {"name": "pose_only", "label": "Pose-only", "task": "pose",  "seg_weight": 0, "pose_weight": 500},
    {"name": "joint",     "label": "Joint MTL", "task": "joint", "seg_weight": 1, "pose_weight": 500},
]


class SingleTaskAblation(Experiment):
    results_filename = "ablation_results.json"

    def configs(self) -> list[dict]:
        return RUNS

    def make_cfg(self, run: dict):
        cfg = self._base_cfg(run)
        cfg.task = run["task"]
        cfg.loss_seg_weight = run["seg_weight"]
        cfg.loss_pose_weight = run["pose_weight"]
        cfg.output_dir = f"models/checkpoints/ablation_{run['name']}"
        return cfg

    def record(self, run: dict, miou: float, pck: float) -> dict:
        return {
            "label":     run["label"],
            "task":      run["task"],
            "test_miou": round(miou, 4) if run["task"] != "pose" else None,
            "test_pck":  round(pck, 4) if run["task"] != "seg" else None,
        }

    def print_table(self, results: dict) -> None:
        header = f"{'Configuration':<14} {'Task':<8} {'mIoU':>10} {'PCK':>10}"
        line = "-" * len(header)
        print(f"\n{line}\n{header}\n{line}")
        for r in results.values():
            miou = f"{r['test_miou']:.4f}" if r["test_miou"] is not None else "-"
            pck = f"{r['test_pck']:.4f}" if r["test_pck"] is not None else "-"
            print(f"{r['label']:<14} {r['task']:<8} {miou:>10} {pck:>10}")
        print(line)
