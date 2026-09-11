"""
Export UNet keypoint detection model to ONNX format.

Usage:
    python export_onnx.py
    python export_onnx.py --model best.pth --output best.onnx
"""

import argparse
import os
import sys

import torch

try:
    from .model_registry import DEFAULT_MODEL, get_model, model_choices
    from .inference_core import decode_checkpoint
except ImportError:
    from model_registry import DEFAULT_MODEL, get_model, model_choices
    from inference_core import decode_checkpoint


def _configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


def export_onnx(model_path, output_path, input_size=256, opset_version=18, embed_weights=True,
                model_name=DEFAULT_MODEL):
    """Export PyTorch model to ONNX."""
    device = torch.device("cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    decoded = decode_checkpoint(checkpoint, model_name)
    model_name = decoded.model_name or model_name
    state = decoded.state
    model = get_model(model_name, in_ch=3, out_ch=1)
    model.load_state_dict(state, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, input_size, input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        dynamo=False,
        input_names=["input"],
        output_names=["heatmap"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "heatmap": {0: "batch", 2: "height", 3: "width"},
        },
    )
    print(f"[EXPORT] ONNX model saved to: {output_path}")

    # Some exporters may save large weights as external data (xxx.onnx.data).
    # If requested, force embedding weights back into a single ONNX file.
    if embed_weights:
        import onnx  # type: ignore

        # Load model and external tensors if any.
        model_proto = onnx.load_model(output_path, load_external_data=True)
        onnx.save_model(model_proto, output_path, save_as_external_data=False)

        data_path = output_path + ".data"
        if os.path.exists(data_path):
            try:
                os.remove(data_path)
                print(f"[CLEAN] Removed external data file: {data_path}")
            except OSError:
                # If deletion fails (e.g., permission), keep going; model is already embedded.
                print(f"[WARN] Failed to remove external data file: {data_path}")

    # Verify
    import onnx  # type: ignore
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[VERIFY] ONNX model is valid")


def main():
    _configure_stdio()
    parser = argparse.ArgumentParser(description="Export UNet to ONNX")
    parser.add_argument("--model", type=str, default="best.pth", help="PyTorch model path")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, choices=model_choices())
    parser.add_argument("--output", type=str, default="best.onnx", help="Output ONNX path")
    parser.add_argument("--input_size", type=int, default=256, help="Input size for export")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--embed_weights", action="store_true", help="Embed weights into single .onnx (no .onnx.data)")
    args = parser.parse_args()

    export_onnx(
        args.model,
        args.output,
        args.input_size,
        opset_version=args.opset,
        embed_weights=args.embed_weights,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
