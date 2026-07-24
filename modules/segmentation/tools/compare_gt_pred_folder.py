#!/usr/bin/env python3
"""Run multilabel segmentation and save one GT/prediction comparison per image."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _ensure_import_paths() -> None:
    toolbox_root = Path(__file__).resolve().parents[3]
    modules_root = toolbox_root / "modules"
    for path in (toolbox_root, modules_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


_ensure_import_paths()

from segmentation.model_registry import get_model  # noqa: E402


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量预测 LabelMe 数据集，并保存左 GT、右 Pred 的对比图。"
    )
    parser.add_argument("--image-dir", required=True, help="PNG 图片及同名 LabelMe JSON 所在目录")
    parser.add_argument("--checkpoint", required=True, help="PyTorch 权重路径")
    parser.add_argument("--output-dir", required=True, help="对比图输出目录")
    parser.add_argument("--model", default="microunet_gn", help="模型名称")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["ear", "knife", "circle"],
        help="训练时的通道标签顺序",
    )
    parser.add_argument("--target-label", default="ear", help="要对比的标签")
    parser.add_argument("--threshold", type=float, default=0.5, help="sigmoid 二值化阈值")
    parser.add_argument("--batch-size", type=int, default=4, help="推理批大小")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 张（用于检查）")
    return parser.parse_args()


def load_labelme_mask(json_path: Path, shape: tuple[int, int], label: str) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for item in data.get("shapes", []):
        if item.get("label") != label:
            continue
        points = np.asarray(item.get("points", []), dtype=np.float32)
        if len(points) < 3:
            continue
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


def image_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normalized.transpose(2, 0, 1)).float()


def make_comparison(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    height, width = gt.shape
    header_height = 40
    canvas = np.zeros((height + header_height, width * 2), dtype=np.uint8)
    canvas[header_height:, :width] = gt
    canvas[header_height:, width:] = pred
    cv2.putText(canvas, "GT", (width // 2 - 22, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    cv2.putText(
        canvas,
        "PRED",
        (width + width // 2 - 42, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        2,
    )
    return canvas


def load_model(
    checkpoint: Path,
    model_name: str,
    n_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    model = get_model(model_name, in_channels=3, n_classes=n_classes).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if args.target_label not in args.labels:
        raise ValueError(f"目标标签 {args.target_label!r} 不在标签顺序 {args.labels!r} 中")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"权重不存在: {checkpoint}")
    if args.batch_size < 1:
        raise ValueError("batch-size 必须大于 0")

    image_paths = sorted(image_dir.glob("*.png"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    missing_json = [path.with_suffix(".json") for path in image_paths if not path.with_suffix(".json").is_file()]
    if missing_json:
        raise FileNotFoundError(f"{len(missing_json)} 张图片缺少同名 JSON，例如: {missing_json[0]}")
    if not image_paths:
        raise ValueError(f"目录内没有 PNG 图片: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, args.model, len(args.labels), device)
    target_index = args.labels.index(args.target_label)
    print(f"device={device}, images={len(image_paths)}, target={args.target_label}[{target_index}]")

    dice_values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[start : start + args.batch_size]
            images: list[np.ndarray] = []
            tensors: list[torch.Tensor] = []
            for path in batch_paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"无法读取图片: {path}")
                images.append(image)
                tensors.append(image_to_tensor(image))

            logits = model(torch.stack(tensors).to(device))
            predictions = (
                torch.sigmoid(logits[:, target_index]) > float(args.threshold)
            ).to(torch.uint8).cpu().numpy()

            for path, image, pred01 in zip(batch_paths, images, predictions):
                pred = pred01 * 255
                gt = load_labelme_mask(
                    path.with_suffix(".json"),
                    image.shape[:2],
                    args.target_label,
                )
                intersection = np.logical_and(gt > 0, pred > 0).sum()
                denominator = (gt > 0).sum() + (pred > 0).sum()
                dice_values.append(1.0 if denominator == 0 else 2.0 * intersection / denominator)

                comparison = make_comparison(gt, pred)
                output_path = output_dir / f"{path.stem}_gt_pred.png"
                if not cv2.imwrite(str(output_path), comparison):
                    raise IOError(f"保存失败: {output_path}")

            done = min(start + len(batch_paths), len(image_paths))
            print(f"[{done:03d}/{len(image_paths):03d}] {batch_paths[-1].name}")

    print(f"完成: {len(image_paths)} 张，平均 Dice={np.mean(dice_values):.6f}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
