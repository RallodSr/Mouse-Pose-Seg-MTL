"""
Mask R-CNN phase (cross-task investigation stopped at joint_bi n=2/3-finishing;
uncertainty-weighting baseline cancelled per decision). Waits for the in-flight
joint_bi run to finish (so the GPU is free), then runs the must-have experiment:

  1. Mask R-CNN augmented x3 seeds  -- fair, multi-seed Table-1 segmentation
     baseline (replaces the old single no-aug number).
  2. Multi-seed / cross-task summary table.
  3. Cross-task qualitative contact sheet (joint vs joint_bi, seed42).
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


def bi_running() -> bool:
    """True while the standalone joint_bi run still holds the GPU (seed44 not yet
    recorded). We poll the results file rather than process tables."""
    if not RES.exists():
        return False
    try:
        d = json.loads(RES.read_text())
    except Exception:
        return True
    return len(d.get("joint_bi", {})) < 3


def main():
    t0 = time.time()
    # wait for joint_bi seed44 to finish (max 1h safety) so we don't contend for GPU
    while bi_running() and time.time() - t0 < 3600:
        time.sleep(120)
    print(f"[maskrcnn-phase] joint_bi finished; starting Mask R-CNN augmented x3 "
          f"after {(time.time()-t0)/60:.1f} min wait", flush=True)

    subprocess.run([PY, "baselines/maskrcnn/maskrcnn.py", "multiseed",
                    "--seeds", "42", "43", "44", "--epochs", "100"], env=ENV)

    subprocess.run([PY, "app/summarize_results.py"], env=ENV)
    subprocess.run([PY, "app/plot_crosstask_qualitative.py"], env=ENV)

    print("[maskrcnn-phase] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
