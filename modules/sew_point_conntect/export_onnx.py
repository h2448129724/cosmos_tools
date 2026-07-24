"""Export a Sew Point Connect checkpoint as the production split-ONNX pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_connector_onnx(
    *,
    checkpoint: str,
    onnx_out: str,
    patch_onnx_out: str,
    annotation_json: str = "",
    image: str = "",
    opset: int = 17,
    device: str = "cpu",
    patch_batch_size: int = 128,
    max_abs_diff: float = 1e-4,
) -> dict:
    import onnxruntime as ort
    import torch

    from algo.models.ort_providers import get_default_ort_providers
    from train.models.sew_point_connector import EdgeGraphCore, EdgePatchEncoder
    from train.tools.export_sew_point_connector_onnx import (
        build_model,
        compare_split_graph,
        compare_torch_models,
        export_graph_core,
        export_patch_encoder,
        load_checkpoint,
        load_real_graph,
        make_probe_graph,
    )

    requested_device = str(device).lower()
    torch_device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint_data = load_checkpoint(checkpoint, torch_device)
    reference_model = build_model(checkpoint_data, torch_device, aggregation="index_add")
    export_model = build_model(checkpoint_data, torch_device, aggregation="matmul")
    patch_encoder = EdgePatchEncoder(export_model).to(torch_device).eval()
    graph_core = EdgeGraphCore(export_model).to(torch_device).eval()
    sample_inputs = make_probe_graph(checkpoint_data, num_nodes=6, num_edges=9, device=torch_device)

    core_path = Path(onnx_out).expanduser().resolve()
    patch_path = Path(patch_onnx_out).expanduser().resolve()
    if core_path == patch_path:
        raise ValueError("Connector ONNX 与 Patch ONNX 必须使用不同输出路径")

    with torch.inference_mode():
        sample_patch_features = patch_encoder(sample_inputs[3])
    export_patch_encoder(patch_encoder, patch_path, sample_inputs[3], opset=int(opset))
    export_graph_core(graph_core, core_path, sample_inputs, sample_patch_features, opset=int(opset))

    providers = (
        get_default_ort_providers()
        if requested_device == "cuda"
        else ["CPUExecutionProvider"]
    )
    patch_session = ort.InferenceSession(str(patch_path), providers=providers)
    core_session = ort.InferenceSession(str(core_path), providers=providers)
    metrics = {
        "core_model_path": str(core_path),
        "patch_model_path": str(patch_path),
        "providers": core_session.get_providers(),
        "patch_providers": patch_session.get_providers(),
        "torch_export_probe": compare_torch_models(reference_model, export_model, sample_inputs),
        "probe": compare_split_graph(export_model, patch_session, core_session, sample_inputs),
    }

    dynamic_inputs = make_probe_graph(checkpoint_data, num_nodes=11, num_edges=17, device=torch_device)
    metrics["torch_export_dynamic_probe"] = compare_torch_models(reference_model, export_model, dynamic_inputs)
    metrics["dynamic_probe"] = compare_split_graph(export_model, patch_session, core_session, dynamic_inputs)

    real_graph = load_real_graph(annotation_json or None, image or None, checkpoint_data, torch_device)
    if real_graph is not None:
        metrics["torch_export_real_sample"] = compare_torch_models(reference_model, export_model, real_graph)
        metrics["real_sample"] = compare_split_graph(export_model, patch_session, core_session, real_graph)

    metrics["patch_batch_size"] = int(patch_batch_size)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    worst = max(
        item["max_abs_diff"]
        for key, item in metrics.items()
        if key.endswith("probe") or key.endswith("sample")
    )
    if worst > float(max_abs_diff):
        raise RuntimeError(f"max_abs_diff {worst:.6g} exceeds threshold {float(max_abs_diff):.6g}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出 Sew Point Connect split ONNX")
    parser.add_argument("--checkpoint", required=True, help="输入 Connector .pth/.pt/.ckpt 权重")
    parser.add_argument("--onnx-out", required=True, help="输出 Connector 图网络 ONNX")
    parser.add_argument("--patch-onnx-out", required=True, help="输出 Connector Patch Encoder ONNX")
    parser.add_argument("--annotation-json", default="", help="可选真实标注 JSON，用于导出后数值验证")
    parser.add_argument("--image", default="", help="与 annotation-json 对应的可选图片")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="导出设备；默认 CPU 更稳定")
    parser.add_argument("--patch-batch-size", type=int, default=128, help="生产 Patch Encoder 推理批大小记录")
    parser.add_argument("--max-abs-diff", type=float, default=1e-4, help="PyTorch/ONNX 最大允许绝对误差")
    return parser


def main() -> None:
    export_connector_onnx(**vars(build_parser().parse_args()))


if __name__ == "__main__":
    main()
