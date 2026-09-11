from __future__ import annotations

import importlib
from typing import Any

from cosmos_toolbox.paths import ensure_import_paths
from cosmos_toolbox.training.model_resolution import (
    ModelSpec,
    list_model_specs,
    plan_model_construction,
    resolve_model_spec,
)


_SPECS = (
    ModelSpec(
        key="cabf.sew_point.runtime",
        task="sew_point",
        source="cab_f_project",
        factory="cosmos_toolbox.training.cab_f_project:create_sew_point_detector",
        modes=frozenset({"infer"}),
        priority=300,
        checkpoint_formats=(".onnx",),
        description="Cosmos CAB-F production ONNX point detector.",
    ),
    ModelSpec(
        key="cabf.sew_point_connector.runtime",
        task="sew_point_connect",
        source="cab_f_project",
        factory="cosmos_toolbox.training.cab_f_project:create_sew_point_connector",
        modes=frozenset({"infer"}),
        priority=300,
        checkpoint_formats=(".onnx",),
        description="Cosmos CAB-F production split-ONNX graph consumer.",
    ),
    ModelSpec(
        key="microunet",
        task="segmentation",
        source="cosmos_train",
        factory="train.models.microunet:MicroUNet",
        modes=frozenset({"train", "infer", "export"}),
        priority=200,
        checkpoint_formats=(".pth", ".pt", ".ckpt"),
        production_consumer="algo.models.unet_segmenter:OnnxUNetSegmenter",
    ),
    ModelSpec(
        key="microunet_gn",
        task="segmentation",
        source="cosmos_train",
        factory="train.models.microunet:MicroUNet",
        modes=frozenset({"train", "infer", "export"}),
        priority=200,
        checkpoint_formats=(".pth", ".pt", ".ckpt"),
        production_consumer="algo.models.unet_segmenter:OnnxUNetSegmenter",
    ),
    ModelSpec(
        key="edge_graph_net",
        task="sew_point_connect",
        source="cab_f_project",
        factory="cosmos_toolbox.training.cab_f_project:create_edge_graph_net",
        modes=frozenset({"train", "infer", "export"}),
        priority=200,
        checkpoint_formats=(".pth", ".pt", ".ckpt"),
        production_consumer="cosmos_toolbox.training.cab_f_project:create_sew_point_connector",
    ),
    ModelSpec(
        key="microunet",
        task="segmentation",
        source="toolbox_local",
        factory="segmentation.MicroUNet:MicroUNet",
        modes=frozenset({"train", "infer", "export"}),
        priority=100,
        checkpoint_formats=(".pth", ".pt", ".ckpt"),
    ),
    ModelSpec(
        key="microunet_gn",
        task="segmentation",
        source="toolbox_local",
        factory="segmentation.MicroUNet:MicroUNet",
        modes=frozenset({"train", "infer", "export"}),
        priority=100,
        checkpoint_formats=(".pth", ".pt", ".ckpt"),
    ),
    ModelSpec(
        key="edge_graph_net",
        task="sew_point_connect",
        source="toolbox_local",
        factory="sew_point_conntect.model:EdgeGraphNet",
        modes=frozenset({"train", "infer", "export"}),
        priority=100,
        checkpoint_formats=(".pth", ".pt", ".ckpt"),
    ),
    ModelSpec(
        key="sew_point_unet",
        task="sew_point",
        source="toolbox_local",
        factory="sew_point.model:UNet",
        modes=frozenset({"train", "infer", "export"}),
        priority=100,
        checkpoint_formats=(".pth", ".pt", ".ckpt"),
        production_consumer="cosmos_toolbox.training.cab_f_project:create_sew_point_detector",
    ),
    ModelSpec(
        key="ultralytics_yolo",
        task="yolo",
        source="external",
        factory="ultralytics:YOLO",
        modes=frozenset({"train", "infer", "export"}),
        priority=50,
        checkpoint_formats=(".pt", ".yaml"),
    ),
)


def _load_symbol(reference: str) -> Any:
    ensure_import_paths()
    module_name, separator, symbol_name = reference.partition(":")
    if not separator:
        raise ValueError(f"Invalid factory reference: {reference}")
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def _is_available(spec: ModelSpec) -> bool:
    try:
        _load_symbol(spec.factory)
        return True
    except (ImportError, AttributeError):
        return False


def list_models(*, task: str | None = None, mode: str | None = None, available_only: bool = True) -> list[ModelSpec]:
    available = None
    if available_only:
        available = {spec.factory for spec in _SPECS if _is_available(spec)}
    return list(list_model_specs(_SPECS, task=task, mode=mode, available_factory_refs=available))


def resolve_model(task: str, mode: str, key: str | None = None) -> ModelSpec:
    available = {spec.factory for spec in _SPECS if _is_available(spec)}
    return resolve_model_spec(
        _SPECS,
        task=task,
        mode=mode,
        key=key,
        available_factory_refs=available,
    )


def create_model(task: str, mode: str = "train", key: str | None = None, **kwargs: Any) -> Any:
    spec = resolve_model(task, mode, key)
    plan = plan_model_construction(spec, kwargs)
    factory = _load_symbol(spec.factory)
    return factory(**plan.keyword_arguments())
