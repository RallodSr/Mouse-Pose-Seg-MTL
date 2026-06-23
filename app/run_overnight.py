"""
Overnight orchestrator: wait for the base multi-seed runs (seg/pose/joint x3
seeds) to finish, then train the cross-task variants (mask-guided, pose-guided,
bidirectional) x3 seeds, then write the comparison summary. Self-contained so it
runs unattended even without re-prompting.
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
BASE = {"seg_only", "pose_only", "joint"}
PY = sys.executable
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def base_done() -> bool:
    if not RES.exists():
        return False
    try:
        d = json.loads(RES.read_text())
    except Exception:
        return False
    return all(len(d.get(c, {})) >= 3 for c in BASE)


def main():
    # 1) wait for the base 9 runs (max 15h safety), polling every 5 min.
    t0 = time.time()
    while not base_done() and time.time() - t0 < 15 * 3600:
        time.sleep(300)
    print(f"[overnight] base_done={base_done()} after {(time.time()-t0)/3600:.1f}h "
          "-- starting cross-task variants", flush=True)

    # 2) cross-task variants x3 seeds (runner skips anything already done).
    subprocess.run([PY, "app/run_multiseed.py", "--configs",
                    "joint_mg", "joint_pg", "joint_bi",
                    "--seeds", "42", "43", "44"], env=ENV)

    # 3) comparison summary.
    subprocess.run([PY, "app/summarize_results.py"], env=ENV)
    print("[overnight] OVERNIGHT PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
