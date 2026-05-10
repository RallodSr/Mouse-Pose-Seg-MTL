import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MouseMTLDataset(Dataset):
    def __init__(self, data_list: list, target_size: tuple = (256, 256), sigma: float = 3.0):
        self.data = data_list
        self.target_size = target_size
        self.sigma = sigma
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]

        image = cv2.cvtColor(cv2.imread(item["image_path"]), cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        final_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        for m_path in item.get("mask_paths", []):
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                final_mask = np.maximum(final_mask, m)

        img_resized = cv2.resize(image, self.target_size)
        mask_resized = cv2.resize(final_mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        scale_x = self.target_size[0] / w_orig
        scale_y = self.target_size[1] / h_orig
        combined_hms = np.zeros((3, self.target_size[1], self.target_size[0]), dtype=np.float32)

        for mouse_kps in item["all_keypoints"]:
            scaled_kps = [[p[0] * scale_x, p[1] * scale_y] for p in mouse_kps]
            mouse_hm = self._generate_heatmap(scaled_kps, self.target_size[1], self.target_size[0])
            combined_hms = np.maximum(combined_hms, mouse_hm)

        t_img = self.transform(img_resized)
        t_mask = torch.from_numpy(mask_resized).long()
        t_mask[t_mask > 0] = 1
        t_hm = torch.from_numpy(combined_hms).float()
        return t_img, t_mask, t_hm

    def _generate_heatmap(self, kps: list, h: int, w: int) -> np.ndarray:
        heatmaps = np.zeros((3, h, w), dtype=np.float32)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        for i, kp in enumerate(kps):
            cx, cy = int(kp[0]), int(kp[1])
            if cx <= 0 or cy <= 0 or cx >= w or cy >= h:
                continue
            dist = (xx - cx) ** 2 + (yy - cy) ** 2
            # np.maximum prevents overblown peaks when multiple mice overlap
            heatmaps[i] = np.maximum(heatmaps[i], np.exp(-dist / (2 * self.sigma ** 2)))
        return heatmaps
