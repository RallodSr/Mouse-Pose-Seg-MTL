"""
Train + evaluate the YOLO26 seg & pose baselines across THREE seeds, so the
Table 1 baseline numbers can be reported as mean +/- std (matching HybridMTLNet's
3-seed protocol). This closes the "baselines are single-run" fairness gap raised
in review.

Pipeline (per task = seg | pose):
  1. export  : dataset.json  ->  YOLO format (images/ + labels/ + data.yaml)   [once]
  2. run     : for seed in {42,43,44}: Ultralytics .train(seed=seed, imgsz=256,
               epochs=100) then evaluate best.pt with the SAME metric code the
               paper uses (baselines/yolo/yolo.py) on the dataset.json *test* split
  3. results : mean +/- std  ->  models/yolo_<task>_multiseed.json

The metric is computed by the existing evaluate_yolo_miou / evaluate_yolo_pck so
it is identical to the single-run numbers already in the paper (Hungarian-matched
per-instance IoU at 256x256; PCK@0.05 with the same 12.8 px radius).

REQUIREMENTS (NOT present on the paper-writing machine — run where YOLO lives):
    pip install ultralytics            # a build that ships YOLO26 weights
    # base weights auto-download: yolo26m-seg.pt / yolo26m-pose.pt

Usage:
    # inspect the converted data first (no training):
    python baselines/yolo/yolo_multiseed.py export --task seg
    python baselines/yolo/yolo_multiseed.py export --task pose
    # then the real 3-seed runs:
    python baselines/yolo/yolo_multiseed.py run --task seg  --seeds 42 43 44 --epochs 100
    python baselines/yolo/yolo_multiseed.py run --task pose --seeds 42 43 44 --epochs 100
"""
import argparse
import json
import shutil
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import DATA_CFG, TRAIN_CFG  # noqa: E402
from baselines.yolo.yolo import evaluate_yolo_miou, evaluate_yolo_pck  # noqa: E402

KPT_NAMES = ("nose", "shoulder", "tail")          # 3 keypoints / mouse
YOLO_ROOT = ROOT / "data" / "yolo"                # generated YOLO datasets live here
BASE_WEIGHTS = {"seg": "yolo26m-seg.pt", "pose": "yolo26m-pose.pt"}


# ---------------------------------------------------------------------------
# dataset.json  ->  YOLO format
# ---------------------------------------------------------------------------

def _img_size(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def _seg_label_lines(item, w, h):
    """One polygon line per instance: '0 x1 y1 x2 y2 ...' (normalised)."""
    lines = []
    for mp in item["mask_paths"]:
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 127).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 10 or len(c) < 3:
            continue
        eps = 0.002 * cv2.arcLength(c, True)         # light simplification
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        pts = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in c)
        lines.append(f"0 {pts}")
    return lines


def _pose_label_lines(item, w, h):
    """One line per instance: '0 cx cy bw bh  kx ky v  kx ky v  kx ky v' (normalised)."""
    lines = []
    for mp, kps in zip(item["mask_paths"], item["all_keypoints"]):
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 127).astype(np.uint8)
        if m.sum() < 1:
            continue
        x, y, bw, bh = cv2.boundingRect(m)
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        nbw, nbh = bw / w, bh / h
        kp_str = []
        for (kx, ky) in kps:                          # 3 keypoints in original px
            if kx > 0 and ky > 0:
                kp_str.append(f"{kx / w:.6f} {ky / h:.6f} 2")
            else:
                kp_str.append("0 0 0")                 # missing / occluded -> v=0
        lines.append(f"0 {cx:.6f} {cy:.6f} {nbw:.6f} {nbh:.6f} " + " ".join(kp_str))
    return lines


def export(task: str) -> Path:
    """Write data/yolo/<task>/{images,labels}/{train,val} + data.yaml. Returns yaml path."""
    with open(DATA_CFG.json_path) as f:
        data = json.load(f)

    out = YOLO_ROOT / task
    if out.exists():
        shutil.rmtree(out)
    label_fn = _seg_label_lines if task == "seg" else _pose_label_lines

    counts = {}
    for split in ("train", "val"):
        img_dir = out / "images" / split
        lbl_dir = out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for i, item in enumerate(data[split]):
            sz = _img_size(item["image_path"])
            if sz is None:
                continue
            w, h = sz
            lines = label_fn(item, w, h)
            if not lines:
                continue
            stem = f"{i:05d}_{Path(item['image_path']).stem}"
            shutil.copy(item["image_path"], img_dir / f"{stem}.jpg")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n += 1
        counts[split] = n

    yaml_path = out / "data.yaml"
    yaml = [
        f"path: {out.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: mouse",
    ]
    if task == "pose":
        yaml.insert(3, "kpt_shape: [3, 3]")           # 3 keypoints, (x, y, visibility)
        yaml.insert(4, "flip_idx: [0, 1, 2]")         # nose/shoulder/tail: no L-R pairs
    yaml_path.write_text("\n".join(yaml) + "\n")

    print(f"[export {task}] train={counts['train']} val={counts['val']}  ->  {yaml_path}")
    print(f"   (inspect a label: {out / 'labels' / 'train'})")
    return yaml_path


# ---------------------------------------------------------------------------
# 3-seed train + test eval
# ---------------------------------------------------------------------------

def _evaluate(task: str, weights: str) -> float:
    if task == "seg":
        return evaluate_yolo_miou(DATA_CFG.json_path, weights, DATA_CFG.target_size)
    return evaluate_yolo_pck(DATA_CFG.json_path, weights,
                             DATA_CFG.target_size, TRAIN_CFG.pck_threshold)


def run(task: str, seeds, epochs: int, imgsz: int):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. `pip install ultralytics` on the YOLO machine.")

    yaml_path = export(task)                            # always refresh the export
    metric = "mIoU" if task == "seg" else "PCK@0.05"

    res_path = ROOT / "models" / f"yolo_{task}_multiseed.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    for seed in seeds:
        if str(seed) in results:
            print(f"[skip] yolo-{task} seed{seed} already done")
            continue
        model = YOLO(BASE_WEIGHTS[task])
        model.train(data=str(yaml_path), epochs=epochs, imgsz=imgsz, seed=seed,
                    deterministic=True, project=str((ROOT / "models" / "yolo_runs").resolve()),
                    name=f"{task}_seed{seed}", exist_ok=True, verbose=False, plots=False)
        # read the real save dir (ultralytics may nest under runs/<task>/...)
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
        score = _evaluate(task, str(best))
        results[str(seed)] = round(float(score), 4)
        res_path.write_text(json.dumps(results, indent=2))
        print(f"[done] yolo-{task} seed{seed}: {metric}={results[str(seed)]}")

    xs = list(results.values())
    if xs:
        m = st.mean(xs)
        s = st.pstdev(xs) if len(xs) > 1 else 0.0
        print(f"\nYOLO26m-{task} TEST {metric}: {m:.4f} +/- {s:.4f}  "
              f"(n={len(xs)}, seeds={list(results.keys())})")
    print(f"results -> {res_path}")


def main():
    p = argparse.ArgumentParser(description="YOLO26 seg/pose 3-seed baseline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="only convert dataset.json -> YOLO format")
    pe.add_argument("--task", choices=["seg", "pose"], required=True)

    pr = sub.add_parser("run", help="export + train 3 seeds + test eval")
    pr.add_argument("--task", choices=["seg", "pose"], required=True)
    pr.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    pr.add_argument("--epochs", type=int, default=100)
    pr.add_argument("--imgsz", type=int, default=256)

    a = p.parse_args()
    if a.cmd == "export":
        export(a.task)
    else:
        run(a.task, a.seeds, a.epochs, a.imgsz)


if __name__ == "__main__":
    main()
