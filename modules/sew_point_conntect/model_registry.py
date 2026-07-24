from __future__ import annotations

from typing import Any

from cosmos_toolbox.training.model_registry import create_model, list_models as list_registered_models


DEFAULT_MODEL = "edge_graph_net"


def get_model(name: str = DEFAULT_MODEL, **kwargs: Any):
    return create_model("sew_point_connect", mode="train", key=name, **kwargs)


def model_choices() -> tuple[str, ...]:
    return tuple(dict.fromkeys(spec.key for spec in list_registered_models(task="sew_point_connect", mode="train")))
