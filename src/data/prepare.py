"""
Dataset preparation utilities:
  - prepare_dataset : convert Label Studio export → dataset.json
"""
import json
import os
import re

import cv2
from sklearn.model_selection import train_test_split

STANDARD_BODY_PARTS = ["nose", "shoulder", "tail"]


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
