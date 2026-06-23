"""
Loss-weight (seg:pose) sweep at fixed seed 42 with the CURRENT pipeline, so
Table 2 is consistent with the 3-seed numbers in Tables 1/3 (which use the same
trainer/seed). Trains the plain joint model (no cross-task guidance) at several
lambda_pose values, evaluates each on the test set, and writes
models/sweep_results.json incrementally.

1:500 is included to confirm it reproduces the multi-seed joint seed-42 value
(mIoU 0.8092, PCK 0.9583).
"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATA_CFG, MODEL_CFG, TRAIN_CFG
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator

SEED = 42
POSE_WEIGHTS = [1, 100, 400, 500, 1000]
OUT = Path("models/sweep_results.json")


def _load() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def _save(d: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2))


def main():
    res = _load()
    for w in POSE_WEIGHTS:
        key = f"1:{w}"
        if key in res:
            print(f"[skip] {key} already done", flush=True)
            continue
        cfg = copy.copy(TRAIN_CFG)
        cfg.seed = SEED
        cfg.task = "joint"
        cfg.loss_seg_weight = 1
        cfg.loss_pose_weight = w
        cfg.mask_guided = False
        cfg.pose_guided = False
        cfg.uncertainty_weighting = False
        cfg.epochs = 100
        cfg.output_dir = f"models/checkpoints/sweep_1_{w}"

        print(f"\n######## SWEEP {key} (seed {SEED}) ########", flush=True)
        t0 = time.time()
        Trainer(DATA_CFG, MODEL_CFG, cfg).run()
        miou, pck = Evaluator(DATA_CFG, MODEL_CFG, cfg).evaluate(
            f"{cfg.output_dir}/model_best.pth", task="joint")
        res[key] = {"miou": round(miou, 4), "pck": round(pck, 4),
                    "train_min": round((time.time() - t0) / 60, 1)}
        _save(res)
        print(f"[done] {key}: mIoU={res[key]['miou']} PCK={res[key]['pck']}", flush=True)

    print("\n=== LOSS-WEIGHT SWEEP (seg:pose), seed 42, current pipeline ===")
    for w in POSE_WEIGHTS:
        k = f"1:{w}"
        if k in res:
            print(f"  {k:8s}  mIoU {res[k]['miou']}  PCK {res[k]['pck']}")
    print(f"results -> {OUT}")


if __name__ == "__main__":
    main()
