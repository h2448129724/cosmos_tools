from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import torch

from cosmos_toolbox.paths import ensure_import_paths
from cosmos_toolbox.training.model_registry import create_model
from cosmos_toolbox.training.cab_f_project import project_entry

ensure_import_paths()

from sew_point.export_onnx import export_onnx as export_sew_point  # noqa: E402
from train.tools.export_microunet_onnx import compare_outputs, export_onnx as export_microunet  # noqa: E402

_cab_f = project_entry()
EdgeGraphCore = _cab_f.EdgeGraphCore
EdgePatchEncoder = _cab_f.EdgePatchEncoder
build_connector = _cab_f.build_model
compare_split_graph = _cab_f.compare_split_graph
export_graph_core = _cab_f.export_graph_core
export_patch_encoder = _cab_f.export_patch_encoder
make_probe_graph = _cab_f.make_probe_graph


def verify_contracts() -> dict:
    device = torch.device("cpu")
    report: dict = {}
    with TemporaryDirectory() as temporary:
        root = Path(temporary)

        segmentation = create_model("segmentation", key="microunet", in_channels=3, n_classes=2).eval()
        segmentation_path = root / "microunet.onnx"
        export_microunet(segmentation, segmentation_path, torch.randn(1, 3, 64, 64), 17)
        segmentation_metrics = compare_outputs(
            segmentation, segmentation_path, np.zeros((64, 64, 3), dtype=np.uint8), device
        )
        report["segmentation"] = segmentation_metrics

        point_model = create_model("sew_point", key="sew_point_unet").eval()
        point_checkpoint = root / "sew_point.pth"
        point_onnx = root / "sew_point.onnx"
        torch.save(
            {"schema_version": 1, "model_key": "sew_point_unet", "model_state": point_model.state_dict()},
            point_checkpoint,
        )
        export_sew_point(str(point_checkpoint), str(point_onnx), input_size=64, opset_version=18)
        point_session = ort.InferenceSession(str(point_onnx), providers=["CPUExecutionProvider"])
        point_output = point_session.run(None, {"input": np.zeros((1, 3, 64, 64), dtype=np.float32)})[0]
        point_dynamic_output = point_session.run(None, {"input": np.zeros((1, 3, 80, 96), dtype=np.float32)})[0]
        report["sew_point"] = {
            "input": point_session.get_inputs()[0].name,
            "output": point_session.get_outputs()[0].name,
            "shape": list(point_output.shape),
            "dynamic_shape": list(point_dynamic_output.shape),
        }

        connector = create_model(
            "sew_point_connect",
            key="edge_graph_net",
            node_dim=4,
            edge_dim=5,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
        )
        checkpoint = {
            "node_dim": 4,
            "edge_dim": 5,
            "model_state": connector.state_dict(),
            "args": {"hidden_dim": 16, "num_layers": 1, "dropout": 0.0, "patch_height": 24, "patch_width": 96},
        }
        export_model = build_connector(checkpoint, device, aggregation="matmul")
        graph = make_probe_graph(checkpoint, num_nodes=4, num_edges=6, device=device)
        patch_model = EdgePatchEncoder(export_model).eval()
        core_model = EdgeGraphCore(export_model).eval()
        patch_path = root / "connector_patch.onnx"
        core_path = root / "connector.onnx"
        with torch.inference_mode():
            patch_features = patch_model(graph[3])
        export_patch_encoder(patch_model, patch_path, graph[3], 17)
        export_graph_core(core_model, core_path, graph, patch_features, 17)
        patch_session = ort.InferenceSession(str(patch_path), providers=["CPUExecutionProvider"])
        core_session = ort.InferenceSession(str(core_path), providers=["CPUExecutionProvider"])
        connector_metrics = compare_split_graph(export_model, patch_session, core_session, graph)
        connector_metrics["inputs"] = [item.name for item in core_session.get_inputs()]
        report["sew_point_connect"] = connector_metrics

    report["passed"] = (
        report["segmentation"]["max_abs_diff"] < 1e-3
        and report["sew_point"]["shape"] == [1, 1, 64, 64]
        and report["sew_point"]["dynamic_shape"] == [1, 1, 80, 96]
        and report["sew_point_connect"]["max_abs_diff"] < 1e-3
    )
    return report


def main() -> int:
    report = verify_contracts()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
