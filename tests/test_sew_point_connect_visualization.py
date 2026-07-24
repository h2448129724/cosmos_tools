from __future__ import annotations

from pathlib import Path

import numpy as np

from sew_point_conntect.export_onnx import (
    build_parser as build_export_parser,
    export_connector_onnx,
)
from sew_point_conntect.visualize_onnx_pipeline import build_parser, draw_pipeline_visualization
from sew_point_conntect.visualize_onnx_pipeline import _drawing_sizes
from trainer_gui.argparse_parser import ArgparseSchemaParser
from trainer_gui.feature_scanner import FeatureScanner


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_visualization_draws_points_and_edges_without_ground_truth():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    annotation = {"points": [{"id": 0, "x": 10, "y": 10}, {"id": 1, "x": 50, "y": 50}]}

    visual = draw_pipeline_visualization(image, annotation, [{"src": 0, "dst": 1}])

    assert visual.shape == image.shape
    assert np.any(visual != image)
    assert np.any(visual[28:36, 28:36] != 0)


def test_pipeline_parser_exposes_small_and_large_modes():
    mode_action = next(action for action in build_parser()._actions if action.dest == "mode")

    assert mode_action.default == "small"
    assert tuple(mode_action.choices) == ("small", "large")


def test_large_visualization_uses_generic_stride_and_bounded_point_size():
    parser = build_parser()
    destinations = {action.dest for action in parser._actions}

    assert "stride" in destinations
    assert "center_size" not in destinations
    assert _drawing_sizes((8192, 26664, 3)) == (3, 5, 1.2)
    source = (REPO_ROOT / "modules" / "sew_point_conntect" / "visualize_onnx_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "sew_point_classifier" not in source
    assert "detect_large_image" in source
    assert "KeypointDetectorONNX" in source
    assert "algo.cab_f.sew_point_detector" not in source


def test_sew_point_onnx_preprocessing_preserves_training_bgr_order():
    from sew_point.inference_onnx import KeypointDetectorONNX

    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    tensor = KeypointDetectorONNX._preprocess(None, bgr)

    assert tensor.shape == (1, 3, 1, 1)
    assert np.allclose(tensor[0, :, 0, 0], np.array([10, 20, 30], dtype=np.float32) / 255.0)


def test_connector_export_parser_defaults_to_safe_cpu_validation():
    parser = build_export_parser()
    device_action = next(action for action in parser._actions if action.dest == "device")
    opset_action = next(action for action in parser._actions if action.dest == "opset")

    assert device_action.default == "cpu"
    assert tuple(device_action.choices) == ("cpu", "cuda")
    assert opset_action.default == 17


def test_connector_export_writes_and_validates_split_pair(tmp_path):
    import torch

    from cosmos_toolbox.training.model_registry import create_model

    model = create_model(
        "sew_point_connect",
        mode="train",
        key="edge_graph_net",
        node_dim=4,
        edge_dim=5,
        hidden_dim=16,
        num_layers=1,
        dropout=0.0,
    ).eval()
    checkpoint = tmp_path / "connector.pth"
    core = tmp_path / "connector.onnx"
    patch = tmp_path / "connector_patch.onnx"
    torch.save(
        {
            "node_dim": 4,
            "edge_dim": 5,
            "model_state": model.state_dict(),
            "args": {
                "hidden_dim": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "patch_height": 24,
                "patch_width": 96,
            },
        },
        checkpoint,
    )

    metrics = export_connector_onnx(
        checkpoint=str(checkpoint),
        onnx_out=str(core),
        patch_onnx_out=str(patch),
        device="cpu",
        max_abs_diff=1e-3,
    )

    assert core.is_file()
    assert patch.is_file()
    assert metrics["providers"][0] == "CPUExecutionProvider"
    assert metrics["dynamic_probe"]["max_abs_diff"] < 1e-3


def test_connect_module_registers_managed_visualization_outputs():
    feature = next(
        item
        for item in FeatureScanner(REPO_ROOT).scan()
        if item.feature_name == "sew_point_conntect"
    )
    action = next(item for item in feature.actions if item.action_name == "visualize_onnx_pipeline")

    assert action.entry.value == "sew_point_conntect.visualize_onnx_pipeline"
    assert action.output.arg_name == "output_image"
    assert action.output.default_file_name == "sew_point_connect_visualization.png"
    assert [(item.arg_name, item.default_file_name) for item in action.output.extra_outputs] == [
        ("output_json", "sew_point_connect_predictions.json")
    ]
    fields = ArgparseSchemaParser().parse_action(action)
    assert {field.name for field in fields} >= {
        "image_path",
        "point_model",
        "connector_model",
        "patch_model",
        "mode",
    }


def test_connect_module_registers_both_split_onnx_outputs():
    feature = next(
        item
        for item in FeatureScanner(REPO_ROOT).scan()
        if item.feature_name == "sew_point_conntect"
    )
    action = next(item for item in feature.actions if item.action_name == "export_onnx")
    fields = ArgparseSchemaParser().parse_action(action)

    assert action.entry.value == "sew_point_conntect.export_onnx"
    assert action.output.arg_name == "onnx_out"
    assert action.output.default_file_name == "connector.onnx"
    assert [(item.arg_name, item.default_file_name) for item in action.output.extra_outputs] == [
        ("patch_onnx_out", "connector_patch.onnx")
    ]
    assert {field.name for field in fields} >= {
        "checkpoint",
        "onnx_out",
        "patch_onnx_out",
        "device",
        "opset",
        "max_abs_diff",
    }


def test_sew_point_and_connect_distance_defaults_are_three():
    features = {item.feature_name: item for item in FeatureScanner(REPO_ROOT).scan()}
    parser = ArgparseSchemaParser()
    expected = {
        "sew_point": {
            "single_inference": "cluster_dist",
            "batch_infer": "cluster_dist",
            "large_image_infer": "cluster_dist",
        },
        "sew_point_conntect": {
            "train": "stage1_cluster_dist",
            "evaluate_pipeline": "stage1_cluster_dist",
            "visualize_onnx_pipeline": "cluster_dist",
        },
    }

    for feature_name, actions in expected.items():
        feature = features[feature_name]
        for action_name, field_name in actions.items():
            action = next(item for item in feature.actions if item.action_name == action_name)
            field = next(item for item in parser.parse_action(action) if item.name == field_name)
            assert float(field.default) == 3.0, f"{feature_name}/{action_name}/{field_name}"
