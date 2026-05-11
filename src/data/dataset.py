import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

MAX_INSTANCES = 2   # maximum mice per image
NUM_KEYPOINTS = 3   # nose, shoulder, tail


class MouseMTLDataset(Dataset):
    def __init__(self, data_list: list, target_size: tuple = (256, 256),
                 sigma: float = 3.0, augment: bool = False):
        self.data        = data_list
        self.target_size = target_size
        self.sigma       = sigma
        self.augment     = augment
        self.transform   = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        W, H = self.target_size

        image = cv2.cvtColor(cv2.imread(item["image_path"]), cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]
        scale_x = W / w_orig
        scale_y = H / h_orig

        img_resized = cv2.resize(image, (W, H))

        # Build per-instance lists (resize mask + scale keypoints)
        masks_list, kps_list = [], []
        for mask_path, kps in zip(item.get("mask_paths", []), item.get("all_keypoints", [])):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            masks_list.append(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST))
            kps_list.append([[p[0] * scale_x, p[1] * scale_y] for p in kps])

        # Augmentation (train only)
        if self.augment:
            img_resized, masks_list, kps_list = self._augment(img_resized, masks_list, kps_list)

        # Sort instances left → right by mask centroid x
        instances = []
        for mask, kps in zip(masks_list, kps_list):
            m_bin = (mask > 0).astype(np.float32)
            xs    = np.where(m_bin > 0)[1]
            cx    = float(xs.mean()) if len(xs) > 0 else 0.0
            instances.append((cx, m_bin, self._generate_heatmap(kps, H, W)))

        instances.sort(key=lambda t: t[0])

        # Pack into fixed-size tensors, zero-pad if fewer than MAX_INSTANCES
        inst_masks = np.zeros((MAX_INSTANCES, H, W), dtype=np.float32)
        inst_hms   = np.zeros((MAX_INSTANCES * NUM_KEYPOINTS, H, W), dtype=np.float32)
        for i, (_, mask, hm) in enumerate(instances[:MAX_INSTANCES]):
            inst_masks[i] = mask
            inst_hms[i * NUM_KEYPOINTS:(i + 1) * NUM_KEYPOINTS] = hm

        t_img   = self.transform(img_resized)
        t_masks = torch.from_numpy(inst_masks)   # (2, H, W) float — BCEWithLogitsLoss
        t_hms   = torch.from_numpy(inst_hms)     # (6, H, W) float — MSELoss

        return t_img, t_masks, t_hms

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(self, image: np.ndarray, masks_list: list, kps_list: list):
        H, W = image.shape[:2]

        # Random horizontal flip
        if random.random() < 0.5:
            image      = image[:, ::-1, :].copy()
            masks_list = [m[:, ::-1].copy() for m in masks_list]
            kps_list   = [
                [[W - kp[0], kp[1]] if kp[0] > 0 else [0.0, 0.0] for kp in kps]
                for kps in kps_list
            ]

        # Random vertical flip
        if random.random() < 0.5:
            image      = image[::-1, :, :].copy()
            masks_list = [m[::-1, :].copy() for m in masks_list]
            kps_list   = [
                [[kp[0], H - kp[1]] if kp[1] > 0 else [0.0, 0.0] for kp in kps]
                for kps in kps_list
            ]

        # Random rotation ±30°
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            M     = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
            image      = cv2.warpAffine(image, M, (W, H))
            masks_list = [cv2.warpAffine(m.astype(np.float32), M, (W, H)) for m in masks_list]
            kps_list   = [self._rotate_kps(kps, M, W, H) for kps in kps_list]

        # Color jitter (image only — masks and keypoints unchanged)
        if random.random() < 0.5:
            image = self._color_jitter(image)

        return image, masks_list, kps_list

    @staticmethod
    def _rotate_kps(kps: list, M: np.ndarray, W: int, H: int) -> list:
        result = []
        for kp in kps:
            if kp[0] <= 0 or kp[1] <= 0:
                result.append([0.0, 0.0])
                continue
            pt  = M @ np.array([kp[0], kp[1], 1.0])
            x, y = float(pt[0]), float(pt[1])
            result.append([x, y] if 0 < x < W and 0 < y < H else [0.0, 0.0])
        return result

    @staticmethod
    def _color_jitter(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        image *= random.uniform(0.7, 1.3)          # brightness
        image  = (image - image.mean()) * random.uniform(0.8, 1.2) + image.mean()  # contrast
        return np.clip(image, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Heatmap generation
    # ------------------------------------------------------------------

    def _generate_heatmap(self, kps: list, h: int, w: int) -> np.ndarray:
        heatmaps = np.zeros((NUM_KEYPOINTS, h, w), dtype=np.float32)
        xx, yy   = np.meshgrid(np.arange(w), np.arange(h))
        for i, kp in enumerate(kps):
            cx, cy = int(kp[0]), int(kp[1])
            if cx <= 0 or cy <= 0 or cx >= w or cy >= h:
                continue
            dist        = (xx - cx) ** 2 + (yy - cy) ** 2
            heatmaps[i] = np.exp(-dist / (2 * self.sigma ** 2))
        return heatmaps
