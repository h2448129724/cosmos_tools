from __future__ import annotations

import importlib

import torch

from cosmos_toolbox.paths import COSMOS_ROOT, TOOLBOX_ROOT
from cosmos_toolbox.training.model_registry import create_model, resolve_model


def test_paths_resolve_to_containing_cosmos_checkout():
    assert COSMOS_ROOT.name == "cosmos"
    assert (COSMOS_ROOT / "algo" / "cab_f").is_dir()
    assert TOOLBOX_ROOT == COSMOS_ROOT / "tools" / "cosmos_toolbox"


def test_training_resolution_prefers_cosmos_train_models():
    segmentation = resolve_model("segmentation", "train", "microunet")
    connector = resolve_model("sew_point_connect", "train", "edge_graph_net")
    sew_point = resolve_model("sew_point", "train", "sew_point_unet")

    assert segmentation.source == "cosmos_train"
    assert segmentation.factory == "train.models.microunet:MicroUNet"
    assert connector.source == "cosmos_train"
    assert connector.factory == "train.models.sew_point_connector:EdgeGraphNet"
    assert sew_point.source == "toolbox_local"


def test_inference_resolution_prefers_cabf_runtime():
    point_runtime = resolve_model("sew_point", "infer")
    connector_runtime = resolve_model("sew_point_connect", "infer")

    assert point_runtime.source == "algo_cab_f"
    assert connector_runtime.source == "algo_cab_f"


def test_created_models_come_from_expected_modules():
    segmentation = create_model("segmentation", key="microunet", in_channels=3, n_classes=2)
    connector = create_model("sew_point_connect", key="edge_graph_net", node_dim=4, edge_dim=5)
    sew_point = create_model("sew_point", key="sew_point_unet")

    assert segmentation.__class__.__module__ == "train.models.microunet"
    assert connector.__class__.__module__ == "train.models.sew_point_connector"
    assert sew_point.__class__.__module__ == "sew_point.model"


def test_old_connector_state_dict_loads_into_cosmos_model_strictly():
    local_module = importlib.import_module("sew_point_conntect.model")
    old_model = local_module.EdgeGraphNet(node_dim=4, edge_dim=5, hidden_dim=16, num_layers=1)
    cosmos_model = create_model(
        "sew_point_connect",
        key="edge_graph_net",
        node_dim=4,
        edge_dim=5,
        hidden_dim=16,
        num_layers=1,
    )

    result = cosmos_model.load_state_dict(old_model.state_dict(), strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    assert isinstance(cosmos_model, torch.nn.Module)
