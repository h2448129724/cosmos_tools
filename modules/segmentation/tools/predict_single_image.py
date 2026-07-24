#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单张图片预测脚本（自包含实现）

用法示例：
python segmentation/tools/predict_single_image.py --image "D:\\project\\changrui\\CAB-F\\260312\\CAB-F_D01-L_20260312_072202704_bottom.png" --ckpt "checkpoints\\260428\\CAB-F_microunet_gn_best_model.pth" --model_name microunet_gn --save check --tile-size 1024 --batch-size 1 --thresh 0.9 --center-size 768

说明：
- 输入图片：任意尺寸的BGR/RGB图像（cv2会按BGR读入），脚本内部会处理。
- 模型：支持 PyTorch 权重（.pth/.pt/.ckpt）和 ONNX 模型（.onnx）。
- 输出结果：单通道二值mask，255为胶路，0为背景（与现有实现保持一致的反色处理）。
"""

import argparse
import os
from typing import Any, List, Optional, Tuple

# ruff: noqa: E402  -- direct-script bootstrap must run before project imports

import cv2
import numpy as np

# Bootstrap: allow direct execution of ``python segmentation/tools/predict_single_image.py``.
# No-op when launched via ``python -m segmentation.tools.predict_single_image``.
try:
    from ._bootstrap import ensure_module_paths
except ImportError:
    from _bootstrap import ensure_module_paths

ensure_module_paths(__file__)

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from segmentation.model_registry import get_model


PYTORCH_MODEL_SUFFIXES = {".pth", ".pt", ".ckpt"}


def _model_format(model_path: str) -> str:
    suffix = os.path.splitext(model_path)[1].lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in PYTORCH_MODEL_SUFFIXES:
        return "pytorch"
    supported = ", ".join(sorted(PYTORCH_MODEL_SUFFIXES | {".onnx"}))
    raise ValueError(f"不支持的模型格式 {suffix or '<无后缀>'}，支持格式: {supported}")


def _load_inference_backend(model_path: str, model_name: str) -> Tuple[str, Any, Optional[torch.device]]:
    model_format = _model_format(model_path)
    if model_format == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("使用 ONNX 模型需要安装 onnxruntime 或 onnxruntime-gpu") from exc

        # Windows 下让 ONNX Runtime 复用 PyTorch 环境内的 CUDA/cuDNN DLL。
        # 没有 GPU 依赖时 preload_dlls 也会安全返回，随后使用 CPU provider。
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers.insert(0, "CUDAExecutionProvider")

        session = ort.InferenceSession(model_path, providers=providers)
        print(f"Using backend: ONNX Runtime ({session.get_providers()[0]})")
        return model_format, session, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using backend: PyTorch ({device})")
    model = get_model(str(model_name), in_channels=3, n_classes=2).to(device)
    # 兼容不同 torch 版本（有的版本没有 weights_only 参数）
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model_format, model, device


def _onnx_foreground_probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 4 or logits.shape[1] < 2:
        raise ValueError(f"ONNX 输出应为 (B, C, H, W) 且 C >= 2，实际为 {logits.shape}")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits[:, 1] / np.sum(exp_logits, axis=1)


def _predict_foreground_probabilities(
    backend: str,
    runtime: Any,
    device: Optional[torch.device],
    batch: torch.Tensor,
    use_fp16: bool = False,
) -> np.ndarray:
    if backend == "onnx":
        input_meta = runtime.get_inputs()[0]
        output_meta = runtime.get_outputs()[0]
        input_array = batch.detach().cpu().numpy().astype(np.float32, copy=False)
        logits = runtime.run([output_meta.name], {input_meta.name: input_array})[0]
        return _onnx_foreground_probabilities(logits)

    assert device is not None
    batch = batch.to(device)
    with torch.no_grad():
        if use_fp16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = runtime(batch)
        else:
            logits = runtime(batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
    return probs.detach().float().cpu().numpy()


def get_transform() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def _iter_tiles(h: int, w: int, tile_size: int, stride: int) -> List[Tuple[int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    xs = list(range(0, max(1, w - tile_size + 1), stride))
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]
    if xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))
    if ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))

    coords: List[Tuple[int, int]] = []
    for y in ys:
        for x in xs:
            coords.append((x, y))
    return coords


def _make_center_slices(tile_size: int, center_size: int) -> Tuple[slice, slice]:
    if center_size <= 0 or center_size > tile_size:
        raise ValueError("center_size must be in (0, tile_size]")
    pad = (tile_size - center_size) // 2
    y0 = pad
    y1 = y0 + center_size
    x0 = pad
    x1 = x0 + center_size
    return slice(y0, y1), slice(x0, x1)


def run_inference_tiled(
    image_bgr: np.ndarray,
    model_path: str,
    model_name: str = "microunet_gn",
    tile_size: int = 256,
    batch_size: int = 8,
    thresh: float = 0.9,
    center_size: int = 196,
    use_fp16: bool = False,
) -> np.ndarray:
    """
    滑窗切块推理，返回整图二值mask（uint8，0/255）。
    说明：
    - 这才会真正用到 batch_size（每次送入网络的 tile 批大小）
    - center_size 用于只拼回 tile 的中心区域，减少边缘伪影
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("输入图像必须为 HxWx3 的BGR图像")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape

    backend, runtime, device = _load_inference_backend(model_path, model_name)

    transforms = get_transform()

    stride = int(center_size)
    coords = _iter_tiles(h, w, tile_size=tile_size, stride=stride)
    cy, cx = _make_center_slices(tile_size=tile_size, center_size=center_size)

    prob_acc = np.zeros((h, w), dtype=np.float32)
    w_acc = np.zeros((h, w), dtype=np.float32)

    def flush_batch(batch_imgs: List[np.ndarray], batch_xy: List[Tuple[int, int]]):
        if not batch_imgs:
            return
        xs = []
        for img_tile in batch_imgs:
            xs.append(transforms(image=img_tile)["image"])
        x = torch.stack(xs, dim=0)
        probs_np = _predict_foreground_probabilities(
            backend=backend,
            runtime=runtime,
            device=device,
            batch=x,
            use_fp16=use_fp16,
        )

        for i, (x0, y0) in enumerate(batch_xy):
            p_tile = probs_np[i]
            yy0 = y0 + cy.start
            yy1 = y0 + cy.stop
            xx0 = x0 + cx.start
            xx1 = x0 + cx.stop
            prob_acc[yy0:yy1, xx0:xx1] += p_tile[cy, cx]
            w_acc[yy0:yy1, xx0:xx1] += 1.0

        batch_imgs.clear()
        batch_xy.clear()

    batch_imgs: List[np.ndarray] = []
    batch_xy: List[Tuple[int, int]] = []

    total_tiles = len(coords)
    for tile_index, (x0, y0) in enumerate(coords, start=1):
        tile = image_rgb[y0 : y0 + tile_size, x0 : x0 + tile_size]
        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]
            tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
        batch_imgs.append(tile)
        batch_xy.append((x0, y0))
        print(f"推理切块 step {tile_index:03d}/{total_tiles:03d} x={x0} y={y0}")
        if len(batch_imgs) >= int(batch_size):
            flush_batch(batch_imgs, batch_xy)

    flush_batch(batch_imgs, batch_xy)

    w_acc[w_acc == 0] = 1.0
    prob_map = prob_acc / w_acc
    pred_mask = ((prob_map > float(thresh)) * 255).astype(np.uint8)
    return pred_mask


def run_inference(
    image_bgr: np.ndarray,
    model_path: str,
    model_name: str = "microunet_gn",
    thresh: float = 0.9,
) -> np.ndarray:
    """直接对输入图像进行推理，返回预测图像。"""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("输入图像必须为 HxWx3 的BGR图像")

    # 转为RGB以匹配训练预处理
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    backend, runtime, device = _load_inference_backend(model_path, model_name)

    # 原始图像尺寸
    h, w, _ = image_rgb.shape

    # 转换图像
    transforms = get_transform()
    image_tensor = transforms(image=image_rgb)["image"].unsqueeze(0)

    # 执行推理
    probs = _predict_foreground_probabilities(
        backend=backend,
        runtime=runtime,
        device=device,
        batch=image_tensor,
    )
    pred = probs > thresh

    # 将预测结果转回图像格式，确保是uint8类型
    pred_mask = (pred[0] * 255).astype(np.uint8)

    return pred_mask


def predict_single_image(
    image_path: str,
    model_path: str,
    model_name: str = "microunet_gn",
    save_path: Optional[str] = None,
    tile_size: int = 256,
    batch_size: int = 16,
    thresh: float = 0.9,
    center_size: int = 196,
    fp16: bool = False,
) -> str:
    """对单张图片进行胶路预测并保存结果。返回保存路径。"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到输入图片: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重: {model_path}")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    mask = run_inference_tiled(
        image_bgr=image,
        model_path=model_path,
        model_name=model_name,
        tile_size=tile_size,
        batch_size=batch_size,
        thresh=thresh,
        center_size=center_size,
        use_fp16=bool(fp16),
    )

    if save_path is None:
        save_dir = os.path.join(os.path.dirname(image_path), "pred")
        name = os.path.splitext(os.path.basename(image_path))[0] + "_glue.png"
        save_path = os.path.join(save_dir, name)
    else:
        # 兼容：
        # - save_path 是目录（可能还不存在，比如 "check"）
        # - save_path 是文件路径（比如 "out.png" 或 "D:\\xx\\out.png"）
        ext = os.path.splitext(save_path)[1].lower()
        looks_like_file = ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        if (not looks_like_file) or save_path.endswith(os.sep):
            name = os.path.splitext(os.path.basename(image_path))[0] + "_glue.png"
            save_path = os.path.join(save_path, name)

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f'save_path: {save_path}')
    ok = cv2.imwrite(save_path, mask)
    if not ok:
        raise IOError(f"保存结果失败: {save_path}")

    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='单张图片胶路预测')
    parser.add_argument('--image', required=True, help='输入图片路径')
    parser.add_argument(
        '--ckpt',
        default=r"checkpoints\260427\UGE_microunet_best_model.pth",
        help='模型路径（支持 .pth/.pt/.ckpt/.onnx）',
    )
    parser.add_argument('--model_name', default='microunet_gn', help='模型名（例如 microunet / microunet_gn）')
    parser.add_argument('--save', default="check", help='结果保存路径（默认与输入同目录，后缀 _glue.png）')
    parser.add_argument('--tile-size', type=int, default=1024, help='滑窗大小')
    parser.add_argument('--batch-size', type=int, default=1, help='批大小')
    parser.add_argument('--thresh', type=float, default=0.9, help='分类阈值')
    parser.add_argument('--center-size', type=int, default=768, help='有效中心区域尺寸')
    parser.add_argument('--fp16', action='store_true', help='使用半精度推理（CUDA）以减少显存')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = predict_single_image(
        image_path=args.image,
        model_path=args.ckpt,
        model_name=args.model_name,
        save_path=args.save,
        tile_size=args.tile_size,
        batch_size=args.batch_size,
        thresh=args.thresh,
        center_size=args.center_size,
        fp16=bool(args.fp16),
    )
    print(f"预测完成，结果已保存：{save_path}")


if __name__ == '__main__':
    main()
