"""
Phase 3 orchestrator (self-contained, runs unattended).

Waits for the standalone joint_bi run (seeds 43, 44) to finish, then runs the
remaining paper experiments on the freed GPU, in order:
  1. Uncertainty-weighting MTL baseline (joint_uw) x3 seeds   -- defends "is the
     gain from cross-task structure or just better loss balancing?"
  2. Mask R-CNN augmented x3 seeds                            -- fair, multi-seed
     segmentation baseline (replaces the old single no-aug number).
  3. Multi-seed / cross-task summary table.
  4. Cross-task qualitative contact sheet (GPU now free).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
os.chdir(ROOT)
RES = Path("models/multiseed_results.json")
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def bi_done() -> bool:
    if not RES.exists():
        return False
    try:
        d = json.loads(RES.read_text())
    except Exception:
        return False
    return len(d.get("joint_bi", {})) >= 3


def main():
    t0 = time.time()
    while not bi_done() and time.time() - t0 < 12 * 3600:
        time.sleep(300)
    print(f"[phase3] joint_bi done={bi_done()} after {(time.time()-t0)/3600:.1f}h "
          "-- starting uncertainty baseline + Mask R-CNN", flush=True)

    # 1) uncertainty-weighting MTL baseline x3 seeds
    subprocess.run([PY, "app/run_multiseed.py", "--configs", "joint_uw",
                    "--seeds", "42", "43", "44"], env=ENV)

    # 2) Mask R-CNN augmented x3 seeds (100 epochs)
    subprocess.run([PY, "baselines/maskrcnn/maskrcnn.py", "multiseed",
                    "--seeds", "42", "43", "44", "--epochs", "100"], env=ENV)

    # 3) multi-seed / cross-task summary
    subprocess.run([PY, "app/summarize_results.py"], env=ENV)

    # 4) cross-task qualitative contact sheet (GPU free now)
    subprocess.run([PY, "app/plot_crosstask_qualitative.py"], env=ENV)

    print("[phase3] PHASE 3 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
