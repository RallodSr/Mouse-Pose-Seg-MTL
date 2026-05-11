"""
Dataset preparation utilities:
  - prepare_dataset   : convert Label Studio export → dataset.json
  - convert_yolo_pose : dataset.json → YOLO pose format
  - convert_yolo_seg  : dataset.json → YOLO seg format
"""
import json
import os
import re
import shutil

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

STANDARD_BODY_PARTS = ["nose", "shoulder", "tail"]
CLASS_ID = 0


# ---------------------------------------------------------------------------
# 1. Label Studio → dataset.json
# ---------------------------------------------------------------------------

def prepare_dataset(
    json_path: str = "data/label_studio_export.json",
    images_dir: str = "data/images",
    masks_dir: str = "data/masks",
    output_json: str = "data/dataset.json",
) -> None:
    print("Loading Label Studio export...")
    with open(json_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    task_data = {}
    for item in raw_data:
        if "data" not in item:
            continue
        fname = os.path.basename(item["data"]["url"])
        if not item.get("annotations"):
            continue

        res_list = item["annotations"][0].get("result", [])
        if not res_list:
            continue

        orig_w = res_list[0].get("original_width")
        orig_h = res_list[0].get("original_height")
        if not orig_w:
            continue

        kps = []
        for res in res_list:
            if res["type"] == "keypointlabels":
                x = res["value"]["x"] * orig_w / 100
                y = res["value"]["y"] * orig_h / 100
                label = re.sub(r"\d+", "", res["value"]["keypointlabels"][0])
                kps.append({"label": label, "x": x, "y": y})

        if kps:
            task_data[fname] = {"kps": kps, "w": orig_w, "h": orig_h}

    if not os.path.exists(masks_dir):
        print(f"Masks directory not found: {masks_dir}")
        return

    merged_data: dict = {}
    for mask_file in sorted(f for f in os.listdir(masks_dir) if f.endswith(".png")):
        base_name = mask_file.rsplit("_", 1)[0] + ".png"
        if base_name not in task_data:
            base_name = mask_file.rsplit("_", 1)[0] + ".jpg"
            if base_name not in task_data:
                continue

        img_path = os.path.join(images_dir, base_name)
        mask = cv2.imread(os.path.join(masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        all_kps = task_data[base_name]["kps"]
        matched: dict = {}
        for kp in all_kps:
            ix, iy = int(kp["x"]), int(kp["y"])
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix] > 128:
                matched[kp["label"]] = [kp["x"], kp["y"]]

        kp_list = []
        has_kps = False
        for part in STANDARD_BODY_PARTS:
            if part in matched:
                kp_list.append(matched[part])
                has_kps = True
            else:
                kp_list.append([0.0, 0.0])

        if not has_kps:
            continue

        if img_path not in merged_data:
            merged_data[img_path] = {"image_path": img_path, "mask_paths": [], "all_keypoints": []}
        merged_data[img_path]["mask_paths"].append(os.path.join(masks_dir, mask_file))
        merged_data[img_path]["all_keypoints"].append(kp_list)

    all_items = list(merged_data.values())
    if not all_items:
        print("No matched items found. Check image/mask filenames.")
        return

    train_items, test_items = train_test_split(all_items, test_size=0.2, random_state=42)
    val_items, test_items = train_test_split(test_items, test_size=0.5, random_state=42)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(
            {"info": "Merged Multi-Mice Dataset", "keypoint_names": STANDARD_BODY_PARTS,
             "train": train_items, "val": val_items, "test": test_items},
            f, indent=4,
        )

    print(f"Dataset saved to {output_json}")
    print(f"Split → Train: {len(train_items)} | Val: {len(val_items)} | Test: {len(test_items)}")


# ---------------------------------------------------------------------------
# 2. dataset.json → YOLO Pose format
# ---------------------------------------------------------------------------

def convert_yolo_pose(
    json_path: str = "data/dataset.json",
    output_dir: str = "data/yolo_pose",
) -> None:
    print("Converting to YOLO Pose format...")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    for split_name, items in data.items():
        if split_name not in ["train", "val", "test"]:
            continue
        print(f"  [{split_name}]")
        for item in tqdm(items):
            img_path = item["image_path"]
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            base_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(output_dir, "images", split_name, base_name))

            lines = []
            for idx, kps in enumerate(item.get("all_keypoints", [])):
                masks = item.get("mask_paths", [])
                if idx < len(masks) and os.path.exists(masks[idx]):
                    mask_img = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
                    x, y, w, h = cv2.boundingRect(mask_img)
                else:
                    arr = np.array(kps)
                    pad = 20
                    x, y = max(0, arr[:, 0].min() - pad), max(0, arr[:, 1].min() - pad)
                    w = min(img_w - x, arr[:, 0].ptp() + pad * 2)
                    h = min(img_h - y, arr[:, 1].ptp() + pad * 2)

                if w == 0 or h == 0:
                    continue

                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw, nh = w / img_w, h / img_h
                row = [CLASS_ID, cx, cy, nw, nh]
                for kp in kps:
                    kx, ky = kp[0], kp[1]
                    if kx <= 0 or ky <= 0:
                        row.extend([0.0, 0.0, 0])
                    else:
                        row.extend([kx / img_w, ky / img_h, 2])
                lines.append(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row))

            txt_name = os.path.splitext(base_name)[0] + ".txt"
            with open(os.path.join(output_dir, "labels", split_name, txt_name), "w") as f:
                f.write("\n".join(lines))

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {os.path.abspath(output_dir)}\n"
            "train: images/train\nval: images/val\n\n"
            "names:\n  0: mouse\n\nkpt_shape: [3, 3]\n"
        )
    print(f"YOLO Pose dataset ready at {output_dir}")


# ---------------------------------------------------------------------------
# 3. dataset.json → YOLO Seg format
# ---------------------------------------------------------------------------

def convert_yolo_seg(
    json_path: str = "data/dataset.json",
    output_dir: str = "data/yolo_seg",
) -> None:
    print("Converting to YOLO Seg format...")
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    for split_name, items in data.items():
        if split_name not in ["train", "val", "test"]:
            continue
        print(f"  [{split_name}]")
        for item in tqdm(items):
            img_path = item["image_path"]
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            base_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(output_dir, "images", split_name, base_name))

            lines = []
            for mask_path in item.get("mask_paths", []):
                if not os.path.exists(mask_path):
                    continue
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                _, thresh = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < 50:
                        continue
                    pts = contour.flatten().tolist()
                    poly = [pts[i] / img_w if i % 2 == 0 else pts[i] / img_h for i in range(len(pts))]
                    lines.append(f"{CLASS_ID} " + " ".join(f"{v:.6f}" for v in poly))

            txt_name = os.path.splitext(base_name)[0] + ".txt"
            with open(os.path.join(output_dir, "labels", split_name, txt_name), "w") as f:
                f.write("\n".join(lines))

    yaml_path = os.path.join(output_dir, "dataset_seg.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {os.path.abspath(output_dir)}\n"
            "train: images/train\nval: images/val\n\n"
            "names:\n  0: mouse\n"
        )
    print(f"YOLO Seg dataset ready at {output_dir}")
