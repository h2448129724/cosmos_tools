"""Pure model selection and construction planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class ModelConstructionPlan:
    spec: ModelSpec
    kwargs: tuple[tuple[str, Any], ...]

    def keyword_arguments(self) -> dict[str, Any]:
        return dict(self.kwargs)


def list_model_specs(
    specs: Iterable[ModelSpec],
    *,
    task: str | None = None,
    mode: str | None = None,
    available_factory_refs: frozenset[str] | set[str] | None = None,
) -> tuple[ModelSpec, ...]:
    """Filter and deterministically order explicit registry facts."""

    result = [
        spec
        for spec in specs
        if (task is None or spec.task == task)
        and (mode is None or mode in spec.modes)
        and (available_factory_refs is None or spec.factory in available_factory_refs)
    ]
    return tuple(sorted(result, key=lambda spec: (-spec.priority, spec.key, spec.source)))


def resolve_model_spec(
    specs: Iterable[ModelSpec],
    *,
    task: str,
    mode: str,
    key: str | None = None,
    available_factory_refs: frozenset[str] | set[str] | None = None,
) -> ModelSpec:
    candidates = list_model_specs(
        specs,
        task=task,
        mode=mode,
        available_factory_refs=available_factory_refs,
    )
    if key is not None:
        candidates = tuple(spec for spec in candidates if spec.key == key)
    if not candidates:
        label = f" key={key!r}" if key else ""
        raise LookupError(f"No available model for task={task!r}, mode={mode!r}{label}")
    return candidates[0]


def plan_model_construction(
    spec: ModelSpec,
    kwargs: Mapping[str, Any],
) -> ModelConstructionPlan:
    """Apply model-key defaults without importing or instantiating a factory."""

    planned = dict(kwargs)
    if spec.key == "microunet_gn":
        planned.setdefault("norm", "gn")
        planned.setdefault("dropout", 0.0)
    return ModelConstructionPlan(spec, tuple(planned.items()))


__all__ = [
    "ModelConstructionPlan",
    "ModelSpec",
    "list_model_specs",
    "plan_model_construction",
    "resolve_model_spec",
]
