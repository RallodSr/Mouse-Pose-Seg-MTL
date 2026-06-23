"""
Phase 2 orchestrator: wait for the cross-task variants (joint_mg/pg/bi x3 seeds)
to finish, then re-train the Mask R-CNN segmentation baseline WITH augmentation
(matching HybridMTLNet's recipe) for a fair headline comparison. Self-contained.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
RES = Path("models/multiseed_results.json")
CROSS = {"joint_mg", "joint_pg", "joint_bi"}
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def cross_done() -> bool:
    if not RES.exists():
        return False
    try:
        d = json.loads(RES.read_text())
    except Exception:
        return False
    return all(len(d.get(c, {})) >= 3 for c in CROSS)


def main():
    t0 = time.time()
    while not cross_done() and time.time() - t0 < 24 * 3600:
        time.sleep(300)
    print(f"[phase2] cross_done={cross_done()} after {(time.time()-t0)/3600:.1f}h "
          "-- re-training Mask R-CNN (augmented, 100 epochs)", flush=True)

    # keep the original (no-aug) checkpoint for reference before overwriting
    old = Path("baselines/maskrcnn/maskrcnn_best.pth")
    bak = old.with_name("maskrcnn_noaug.pth")
    if old.exists() and not bak.exists():
        shutil.copy(old, bak)

    subprocess.run([PY, "baselines/maskrcnn/maskrcnn.py", "train", "--epochs", "100"], env=ENV)
    subprocess.run([PY, "baselines/maskrcnn/maskrcnn.py", "eval"], env=ENV)
    print("[phase2] MASK R-CNN AUGMENTED RETRAIN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
