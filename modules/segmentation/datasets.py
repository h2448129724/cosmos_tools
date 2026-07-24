import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2

# These must be set before importing albumentations; setting them afterwards
# still allows its import-time online version check to run.
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset


class BinarySegmentationDataset(Dataset):
    def __init__(self, img_paths, ann_paths, augment: bool = False,
                 image_size: int = 1024,
                 target_label: str = "glue",
                 target_labels: list[str] | tuple[str, ...] | None = None,
                 task_type: str = "binary",
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225)):
        self.img_paths = [p for p in img_paths if os.path.exists(p)]
        self.ann_paths = [
            p if p and os.path.exists(p) and os.path.isfile(p) else None
            for p in ann_paths
        ]
        self.augment = augment
        self.task_type = str(task_type).lower()
        if self.task_type not in {"binary", "multilabel"}:
            raise ValueError(f"Unsupported task_type: {task_type}")
        self.target_label = str(target_label)
        if target_labels is None:
            target_labels = [self.target_label]
        self.target_labels = [str(label) for label in target_labels]
        if self.task_type == "multilabel" and not self.target_labels:
            raise ValueError("target_labels must not be empty for multilabel segmentation")

        self.aug = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
            A.ElasticTransform(p=0.3),
            A.GridDistortion(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

        self.no_aug = A.Compose([
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask'})

        if len(self.img_paths) != len(self.ann_paths):
            print(f'Loaded {len(self.img_paths)} images and {len(self.ann_paths)} annotations.')
            raise ValueError("Image and annotation counts do not match!")

    def __len__(self):
        return len(self.img_paths)

    def _json_to_binary_mask(self, ann_path: str, height: int, width: int) -> np.ndarray:
        mask_img = Image.new('L', (width, height), 0)
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        draw = ImageDraw.Draw(mask_img)
        for shape in data.get('shapes', []):
            if shape.get('label') == self.target_label:
                pts = shape.get('points', [])
                if not pts:
                    continue
                poly = [tuple(pt) for pt in pts]
                draw.polygon(poly, outline=1, fill=1)
        return np.array(mask_img, dtype=np.uint8)

    def _json_to_multilabel_mask(self, ann_path: str, height: int, width: int) -> np.ndarray:
        masks = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for label in self.target_labels:
            mask_img = Image.new('L', (width, height), 0)
            draw = ImageDraw.Draw(mask_img)
            for shape in data.get('shapes', []):
                if shape.get('label') == label:
                    pts = shape.get('points', [])
                    if not pts:
                        continue
                    poly = [tuple(pt) for pt in pts]
                    draw.polygon(poly, outline=1, fill=1)
            masks.append(np.array(mask_img, dtype=np.uint8))
        return np.stack(masks, axis=-1)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert('RGB'))
        ann_path = self.ann_paths[idx]
        if ann_path is None:
            if self.task_type == "multilabel":
                mask = np.zeros((img.shape[0], img.shape[1], len(self.target_labels)), dtype=np.uint8)
            else:
                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            ext = os.path.splitext(ann_path.lower())[1]
            if ext == '.json':
                if self.task_type == "multilabel":
                    mask = self._json_to_multilabel_mask(ann_path, img.shape[0], img.shape[1])
                else:
                    mask = self._json_to_binary_mask(ann_path, img.shape[0], img.shape[1])
            elif ext == '.png':
                # 支持直接读取二值mask（0/1 或 0/255）
                m = np.array(Image.open(ann_path).convert('L'), dtype=np.uint8)
                mask = (m > 0).astype(np.uint8)
                if self.task_type == "multilabel":
                    if len(self.target_labels) != 1:
                        raise ValueError('Multilabel PNG annotations only support one target label per mask file.')
                    mask = mask[:, :, None]
            else:
                raise ValueError(f'Unsupported annotation format: {ann_path}')

        aug = self.aug if self.augment else self.no_aug
        augmented = aug(image=img, mask=mask)
        img_t = augmented['image'].float()
        mask_t = augmented['mask']
        if self.task_type == "multilabel":
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)
            elif mask_t.ndim == 3 and mask_t.shape[-1] == len(self.target_labels):
                mask_t = mask_t.permute(2, 0, 1)
            mask_t = mask_t.float()
        else:
            mask_t = mask_t.long()
        return img_t, mask_t
