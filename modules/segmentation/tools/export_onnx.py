"""Export a registered Segmentation checkpoint to a verified dynamic ONNX model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

# Support both ``python -m segmentation.tools.export_onnx`` and direct script
# execution from an arbitrary working directory.
try:
    from ._bootstrap import ensure_module_paths
except ImportError:
    from _bootstrap import ensure_module_paths

ensure_module_paths(__file__)

from cosmos_toolbox.training.model_registry import create_model  # noqa: E402


PYTORCH_CHECKPOINT_SUFFIXES = frozenset({".pth", ".pt", ".ckpt"})
MODEL_NAMES = ("microunet", "microunet_gn")
DEVICE_CHOICES = ("auto", "cpu", "cuda")


def _checkpoint_state(payload: object) -> dict[str, Any]:
    """Extract a state dict from raw and commonly wrapped checkpoints."""

    candidate = payload
    if isinstance(candidate, Mapping):
        for key in ("state_dict", "model_state", "model"):
            nested = candidate.get(key)
            if isinstance(nested, Mapping):
                candidate = nested
                break
    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError("checkpoint 中没有可加载的模型参数")

    state: dict[str, Any] = {}
    for key, value in candidate.items():
        if not isinstance(key, str):
            raise ValueError("checkpoint 参数名必须是字符串")
        state[key.removeprefix("module.")] = value
    return state


def _dynamic_probe_size(input_size: int) -> int:
    """Choose a bounded second shape that still proves dynamic H/W support."""

    return input_size + 32 if input_size <= 512 else 320


def _validate_paths_and_options(
    checkpoint: Path,
    onnx_out: Path,
    *,
    n_classes: int,
    input_size: int,
    opset: int,
    max_abs_diff: float,
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    if checkpoint.suffix.lower() not in PYTORCH_CHECKPOINT_SUFFIXES:
        supported = ", ".join(sorted(PYTORCH_CHECKPOINT_SUFFIXES))
        raise ValueError(f"checkpoint 必须是 {supported} 文件")
    if onnx_out.suffix.lower() != ".onnx":
        raise ValueError("ONNX 输出路径必须以 .onnx 结尾")
    if n_classes < 1:
        raise ValueError("n_classes 必须大于等于 1")
    if input_size < 16 or input_size % 4:
        raise ValueError("input_size 必须大于等于 16 且能被 4 整除")
    if opset < 11:
        raise ValueError("opset 必须大于等于 11")
    if max_abs_diff <= 0:
        raise ValueError("max_abs_diff 必须大于 0")


def _resolve_test_image(test_image: str) -> Path | None:
    """Resolve an optional test image without relying on OpenCV path handling."""

    if not test_image.strip():
        return None
    image_path = Path(test_image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"测试图片不存在: {image_path}")
    return image_path


def _torch_device(requested: str):
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前 PyTorch 环境不可用")
    selected = "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
    return torch.device(selected)


def _load_model(checkpoint: Path, model_name: str, n_classes: int, device):
    import torch

    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)

    model = create_model(
        "segmentation",
        mode="export",
        key=model_name,
        in_channels=3,
        n_classes=n_classes,
    ).to(device)
    try:
        model.load_state_dict(_checkpoint_state(payload), strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "checkpoint 与所选模型结构不匹配；请检查 model_name 和 n_classes"
        ) from exc
    return model.eval()


def _export_graph(model, output_path: Path, sample_input, opset: int) -> None:
    import torch

    torch.onnx.export(
        model,
        sample_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )


def _compare_input(model, session, probe) -> dict[str, object]:
    import numpy as np
    import torch

    with torch.inference_mode():
        torch_logits = model(probe).detach().cpu().numpy()
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: probe.detach().cpu().numpy()})[0]
    difference = np.abs(torch_logits - onnx_logits)
    return {
        "input_shape": list(probe.shape),
        "output_shape": list(onnx_logits.shape),
        "max_abs_diff": float(difference.max(initial=0.0)),
        "mean_abs_diff": float(difference.mean()),
    }


def _compare_probe(model, session, device, size: int) -> dict[str, object]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(size)
    probe = torch.rand((1, 3, size, size), generator=generator, dtype=torch.float32).to(device)
    report = _compare_input(model, session, probe)
    report["source"] = "random"
    return report


def _load_test_image_probe(image_path: Path, size: int, device):
    """Load and normalize a real image using the training-time preprocessing."""

    import numpy as np
    import torch
    from PIL import Image

    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            original_width, original_height = rgb.size
            resized = rgb.resize((size, size), resample=Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取测试图片: {image_path}") from exc

    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    normalized = (array - mean) / std
    chw = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    probe = torch.from_numpy(chw).unsqueeze(0).to(device)
    metadata = {
        "source": "test_image",
        "source_image": str(image_path),
        "source_image_size": [original_height, original_width],
    }
    return probe, metadata


def export_segmentation_onnx(
    checkpoint: str,
    onnx_out: str,
    test_image: str = "",
    model_name: str = "microunet",
    n_classes: int = 2,
    input_size: int = 256,
    opset: int = 17,
    device: str = "auto",
    max_abs_diff: float = 1e-3,
) -> dict[str, object]:
    """Export, validate and atomically publish one Segmentation ONNX model."""

    import onnx
    import onnxruntime as ort
    import torch

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(onnx_out).expanduser().resolve()
    test_image_path = _resolve_test_image(test_image)
    _validate_paths_and_options(
        checkpoint_path,
        output_path,
        n_classes=n_classes,
        input_size=input_size,
        opset=opset,
        max_abs_diff=max_abs_diff,
    )
    if model_name not in MODEL_NAMES:
        raise ValueError(f"不支持的 model_name: {model_name}")
    if device not in DEVICE_CHOICES:
        raise ValueError(f"不支持的 device: {device}")

    selected_device = _torch_device(device)
    model = _load_model(checkpoint_path, model_name, n_classes, selected_device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.{uuid4().hex}.tmp.onnx")
    try:
        sample = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32, device=selected_device)
        _export_graph(model, temporary_path, sample, opset)
        onnx.checker.check_model(onnx.load(str(temporary_path)))
        session = ort.InferenceSession(str(temporary_path), providers=["CPUExecutionProvider"])
        if test_image_path is None:
            static_probe = _compare_probe(model, session, selected_device, input_size)
        else:
            real_probe, test_metadata = _load_test_image_probe(
                test_image_path,
                input_size,
                selected_device,
            )
            static_probe = _compare_input(model, session, real_probe)
            static_probe.update(test_metadata)
        dynamic_probe = _compare_probe(model, session, selected_device, _dynamic_probe_size(input_size))
        observed_diff = max(
            float(static_probe["max_abs_diff"]),
            float(dynamic_probe["max_abs_diff"]),
        )
        if observed_diff > max_abs_diff:
            raise RuntimeError(
                f"PyTorch/ONNX 最大绝对误差 {observed_diff:.6g} 超过阈值 {max_abs_diff:.6g}"
            )
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "checkpoint": str(checkpoint_path),
        "onnx_out": str(output_path),
        "test_image": str(test_image_path) if test_image_path is not None else None,
        "model_name": model_name,
        "n_classes": n_classes,
        "device": str(selected_device),
        "opset": opset,
        "providers": session.get_providers(),
        "input_name": session.get_inputs()[0].name,
        "output_name": session.get_outputs()[0].name,
        "static_probe": static_probe,
        "dynamic_probe": dynamic_probe,
        "passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出并验证 Segmentation 动态 ONNX 模型")
    parser.add_argument("--checkpoint", required=True, help="训练产生的 .pth/.pt/.ckpt 权重")
    parser.add_argument("--onnx-out", required=True, help="ONNX 输出路径")
    parser.add_argument("--test-image", default="", help="可选测试图片；用于真实数据 PyTorch/ONNX 一致性验证")
    parser.add_argument("--model-name", choices=MODEL_NAMES, default="microunet", help="训练时使用的模型结构")
    parser.add_argument("--n-classes", type=int, default=2, help="模型输出通道数；多标签任务填写标签数量")
    parser.add_argument("--input-size", type=int, default=256, help="导出验证尺寸，必须能被 4 整除")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto", help="PyTorch 导出与对照验证设备")
    parser.add_argument("--max-abs-diff", type=float, default=1e-3, help="PyTorch/ONNX 最大允许绝对误差")
    return parser


def main() -> int:
    report = export_segmentation_onnx(**vars(build_parser().parse_args()))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
