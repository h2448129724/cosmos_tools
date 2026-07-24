import json
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .utils import _imread
except ImportError:
    from utils import _imread


class KeypointDataset(Dataset):
    def __init__(self, samples, augment=False, aug_multiplier=6, sigma=2,
                 img_size=256):
        self.samples = samples
        self.augment = augment
        self.aug_multiplier = aug_multiplier
        self.sigma = sigma
        self.img_size = img_size

    def __len__(self):
        if self.augment:
            return len(self.samples) * self.aug_multiplier
        return len(self.samples)

    def __getitem__(self, idx):
        real_idx = idx % len(self.samples)
        bmp_path, json_path = self.samples[real_idx]

        img = _imread(bmp_path)
        h, w = img.shape[:2]

        with open(json_path) as f:
            ann = json.load(f)
        points = []
        for shape in ann["shapes"]:
            if shape["shape_type"] == "point":
                x, y = shape["points"][0]
                points.append((x, y))

        # Ensure points is always shaped (N, 2). When no points exist, numpy would
        # otherwise create a 1D array with shape (0,), which breaks later indexing.
        if len(points) == 0:
            points = np.zeros((0, 2), dtype=np.float32)
        else:
            points = np.asarray(points, dtype=np.float32).reshape(-1, 2)

        if self.augment:
            img, points = self._augment(img, points, w, h)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        heatmap = self._make_heatmap(points, w, h)

        return torch.from_numpy(img), torch.from_numpy(heatmap)

    def _augment(self, img, points, w, h):
        # 1. Random edge padding
        if random.random() > 0.5:
            pad = random.randint(10, 50)
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
            points[:, 0] += pad
            points[:, 1] += pad
            new_h, new_w = img.shape[:2]
            x0 = random.randint(0, new_w - w)
            y0 = random.randint(0, new_h - h)
            img = img[y0:y0 + h, x0:x0 + w]
            points[:, 0] -= x0
            points[:, 1] -= y0

        # 2. Random rotation 0~360
        angle = random.uniform(0, 360)
        cx, cy = w / 2, h / 2
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        ones = np.ones((len(points), 1), dtype=np.float32)
        pts_hom = np.hstack([points, ones])
        points = (M_rot @ pts_hom.T).T

        # 3. Random horizontal flip
        if random.random() > 0.5:
            img = img[:, ::-1, :].copy()
            points[:, 0] = w - 1 - points[:, 0]

        # 4. Random vertical flip
        if random.random() > 0.5:
            img = img[::-1, :, :].copy()
            points[:, 1] = h - 1 - points[:, 1]

        # 5. Random translation (+/-20%)
        if random.random() > 0.3:
            tx = random.uniform(-0.2, 0.2) * w
            ty = random.uniform(-0.2, 0.2) * h
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M_trans, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            points[:, 0] += tx
            points[:, 1] += ty

        # 6. Random scale (0.8~1.2)
        if random.random() > 0.3:
            scale = random.uniform(0.8, 1.2)
            M_scale = cv2.getRotationMatrix2D((cx, cy), 0, scale)
            img = cv2.warpAffine(img, M_scale, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            points = ((points - np.array([cx, cy])) * scale + np.array([cx, cy])).astype(np.float32)

        # 7. Random crop and resize back
        if random.random() > 0.3:
            crop_ratio = random.uniform(0.7, 1.0)
            cw, ch = int(w * crop_ratio), int(h * crop_ratio)
            x0 = random.randint(0, w - cw)
            y0 = random.randint(0, h - ch)
            img = img[y0:y0 + ch, x0:x0 + cw]
            points[:, 0] = (points[:, 0] - x0) * (w / cw)
            points[:, 1] = (points[:, 1] - y0) * (h / ch)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # 8. Brightness / contrast
        if random.random() > 0.3:
            alpha = random.uniform(0.5, 1.5)
            beta = random.uniform(-50, 50)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 9. Gamma
        if random.random() > 0.3:
            gamma = random.uniform(0.5, 2.0)
            table = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
            table = np.clip(table, 0, 255).astype(np.uint8)
            img = table[img]

        # 10. Gaussian noise
        if random.random() > 0.3:
            noise_std = random.uniform(10, 40)
            noise = np.random.randn(*img.shape).astype(np.float32) * noise_std
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 11. Salt and pepper noise
        if random.random() > 0.5:
            ratio = random.uniform(0.01, 0.05)
            n_pixels = int(img.shape[0] * img.shape[1] * ratio)
            ys_s = np.random.randint(0, img.shape[0], n_pixels)
            xs_s = np.random.randint(0, img.shape[1], n_pixels)
            img[ys_s, xs_s] = 255
            ys_p = np.random.randint(0, img.shape[0], n_pixels)
            xs_p = np.random.randint(0, img.shape[1], n_pixels)
            img[ys_p, xs_p] = 0

        # 12. Color jitter (per-channel)
        if random.random() > 0.5:
            for c in range(3):
                shift = random.uniform(-30, 30)
                img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) + shift, 0, 255).astype(np.uint8)

        # 13. Motion blur
        if random.random() > 0.7:
            ksize = random.choice([3, 5, 7])
            angle = random.uniform(0, 180)
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            kernel[ksize // 2, :] = 1.0 / ksize
            M_blur = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1.0)
            kernel = cv2.warpAffine(kernel, M_blur, (ksize, ksize))
            kernel = kernel / (kernel.sum() + 1e-6)
            img = cv2.filter2D(img, -1, kernel)

        # 14. Gaussian blur
        if random.random() > 0.5:
            ksize = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # Filter out points outside image
        mask = (points[:, 0] >= 0) & (points[:, 0] < w) & \
               (points[:, 1] >= 0) & (points[:, 1] < h)
        points = points[mask]

        return img, points

    def _make_heatmap(self, points, w, h):
        sigma = self.sigma
        heatmap = np.zeros((h, w), dtype=np.float32)
        size = int(6 * sigma + 1)
        x_grid = np.arange(0, size, dtype=np.float32) - size // 2
        y_grid = x_grid.copy()
        xx, yy = np.meshgrid(x_grid, y_grid)
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

        for x, y in points:
            xi, yi = int(round(x)), int(round(y))
            x1 = max(0, xi - size // 2)
            x2 = min(w, xi + size // 2 + 1)
            y1 = max(0, yi - size // 2)
            y2 = min(h, yi + size // 2 + 1)

            gx1 = x1 - (xi - size // 2)
            gx2 = gx1 + (x2 - x1)
            gy1 = y1 - (yi - size // 2)
            gy2 = gy1 + (y2 - y1)

            heatmap[y1:y2, x1:x2] = np.maximum(
                heatmap[y1:y2, x1:x2], gaussian[gy1:gy2, gx1:gx2]
            )

        return heatmap[np.newaxis, ...]
