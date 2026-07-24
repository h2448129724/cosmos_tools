from __future__ import annotations

from typing import Any, Callable, Dict

from cosmos_toolbox.training.model_registry import create_model, list_models as list_registered_models


def get_model(name: str, **kwargs: Any):
    """Build a segmentation model using Cosmos-first provider resolution."""
    return create_model("segmentation", mode="train", key=name.lower(), **kwargs)


def list_models() -> Dict[str, Callable[..., Any]]:
    names = {spec.key for spec in list_registered_models(task="segmentation", mode="train")}
    return {name: (lambda model_name=name, **kwargs: get_model(model_name, **kwargs)) for name in sorted(names)}


if __name__ == "__main__":
    print(list_models())
