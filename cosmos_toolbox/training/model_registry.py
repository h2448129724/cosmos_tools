from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Iterable

from cosmos_toolbox.paths import ensure_import_paths


@dataclass(frozen=True)
class ModelSpec:
    key: str
    task: str
    source: str
    factory: str
    modes: frozenset[str]
    priority: int
    checkpoint_formats: tuple[str, ...] = ()
    production_consumer: str | None = None
    description: str = ""

    @property
    def trainable(self) -> bool:
        return "train" in self.modes


_SPECS = (
    ModelSpec(
        key="cabf.sew_point.runtime",
        task="sew_point",
        source="algo_cab_f",
        factory="algo.cab_f.sew_point_detector:SewPointDetector",
        modes=frozenset({"infer"}),
        priority=300,
        checkpoint_formats=(".onnx",),
        description="Cosmos CAB-F production ONNX point detector.",
    ),
    ModelSpec(
        key="cabf.sew_point_connector.runtime",
        task="sew_point_connect",
        source="algo_cab_f",
        factory="algo.cab_f.sew_point_connector:SewPointConnector",
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
        source="cosmos_train",
        factory="train.models.sew_point_connector:EdgeGraphNet",
        modes=frozenset({"train", "infer", "export"}),
        priority=200,
        checkpoint_formats=(".pth", ".pt", ".ckpt"),
        production_consumer="algo.cab_f.sew_point_connector:SewPointConnector",
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
        production_consumer="algo.cab_f.sew_point_detector:SewPointDetector",
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
    specs: Iterable[ModelSpec] = _SPECS
    if task is not None:
        specs = (spec for spec in specs if spec.task == task)
    if mode is not None:
        specs = (spec for spec in specs if mode in spec.modes)
    result = list(specs)
    if available_only:
        result = [spec for spec in result if _is_available(spec)]
    return sorted(result, key=lambda spec: (-spec.priority, spec.key, spec.source))


def resolve_model(task: str, mode: str, key: str | None = None) -> ModelSpec:
    candidates = list_models(task=task, mode=mode, available_only=True)
    if key is not None:
        candidates = [spec for spec in candidates if spec.key == key]
    if not candidates:
        label = f" key={key!r}" if key else ""
        raise LookupError(f"No available model for task={task!r}, mode={mode!r}{label}")
    return candidates[0]


def create_model(task: str, mode: str = "train", key: str | None = None, **kwargs: Any) -> Any:
    spec = resolve_model(task, mode, key)
    factory = _load_symbol(spec.factory)
    if spec.key == "microunet_gn":
        kwargs.setdefault("norm", "gn")
        kwargs.setdefault("dropout", 0.0)
    return factory(**kwargs)
