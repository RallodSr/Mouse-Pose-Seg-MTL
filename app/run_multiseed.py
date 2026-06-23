"""
Multi-seed runs for rigor: train seg-only / pose-only / joint across several
seeds, evaluate each on the test set, and report mean +/- std. Writes results
incrementally so a partial run survives interruption.

Usage:
    python app/run_multiseed.py --seeds 42 43 44 --epochs 100
    python app/run_multiseed.py --seeds 42 --epochs 2          # quick timing
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

CONFIGS = [
    {"name": "seg_only",  "task": "seg",   "seg_weight": 1, "pose_weight": 0},
    {"name": "pose_only", "task": "pose",  "seg_weight": 0, "pose_weight": 500},
    {"name": "joint",     "task": "joint", "seg_weight": 1, "pose_weight": 500},
    {"name": "joint_mg",  "task": "joint", "seg_weight": 1, "pose_weight": 500, "mask_guided": True},
    {"name": "joint_pg",  "task": "joint", "seg_weight": 1, "pose_weight": 500, "pose_guided": True},
    {"name": "joint_bi",  "task": "joint", "seg_weight": 1, "pose_weight": 500, "mask_guided": True, "pose_guided": True},
    {"name": "joint_uw",  "task": "joint", "seg_weight": 1, "pose_weight": 500, "uncertainty": True},
]

OUT_JSON = Path("models/multiseed_results.json")


def _load() -> dict:
    return json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}


def _save(d: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(d, indent=2))


def _summary(results: dict) -> None:
    import statistics as st
    print("\n================ MULTI-SEED SUMMARY ================")
    for name, runs in results.items():
        mious = [v["miou"] for v in runs.values() if v.get("miou") is not None]
        pcks = [v["pck"] for v in runs.values() if v.get("pck") is not None]
        def fmt(xs):
            if not xs:
                return "-"
            m = st.mean(xs)
            s = st.pstdev(xs) if len(xs) > 1 else 0.0
            return f"{m:.4f} +/- {s:.4f}  (n={len(xs)})"
        print(f"  {name:<10}  mIoU {fmt(mious):<26}  PCK {fmt(pcks)}")
    print("====================================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=TRAIN_CFG.epochs)
    ap.add_argument("--configs", nargs="+", default=[c["name"] for c in CONFIGS])
    args = ap.parse_args()

    results = _load()
    for seed in args.seeds:
        for c in CONFIGS:
            if c["name"] not in args.configs:
                continue
            tag = f"seed{seed}_{c['name']}"
            results.setdefault(c["name"], {})
            if str(seed) in results[c["name"]]:
                print(f"[skip] {tag} already done")
                continue

            cfg = copy.copy(TRAIN_CFG)
            cfg.seed = seed
            cfg.task = c["task"]
            cfg.loss_seg_weight = c["seg_weight"]
            cfg.loss_pose_weight = c["pose_weight"]
            cfg.mask_guided = c.get("mask_guided", False)
            cfg.pose_guided = c.get("pose_guided", False)
            cfg.uncertainty_weighting = c.get("uncertainty", False)
            cfg.epochs = args.epochs
            cfg.output_dir = f"models/checkpoints/{tag}"

            print(f"\n######## TRAIN {tag} (epochs={args.epochs}) ########")
            t0 = time.time()
            Trainer(DATA_CFG, MODEL_CFG, cfg).run()
            train_min = (time.time() - t0) / 60.0

            best = f"{cfg.output_dir}/model_best.pth"
            miou, pck = Evaluator(DATA_CFG, MODEL_CFG, cfg).evaluate(best, task=c["task"])
            rec = {
                "miou": round(miou, 4) if c["task"] != "pose" else None,
                "pck":  round(pck, 4) if c["task"] != "seg" else None,
                "train_min": round(train_min, 1),
            }
            results[c["name"]][str(seed)] = rec
            _save(results)
            print(f"[done] {tag}: mIoU={rec['miou']} PCK={rec['pck']} ({train_min:.1f} min)")

    _summary(results)
    print(f"\nresults -> {OUT_JSON}")


if __name__ == "__main__":
    main()
