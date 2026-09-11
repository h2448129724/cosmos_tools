"""Pure request-scoped facts for one complete-inspection run.

The production Cosmos algorithms still read process globals while they are
being migrated.  This module defines the explicit interface used by the
toolbox runner; the shell may mirror these facts into the legacy globals, but
all runner decisions consume this request snapshot instead.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CosmosRuntimeSession:
    """Resolved product and backend facts captured for one request."""

    inspection: Mapping[str, Any]
    backend_key: str
    backend: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inspection", deepcopy(dict(self.inspection)))
        object.__setattr__(self, "backend_key", str(self.backend_key))
        object.__setattr__(self, "backend", deepcopy(dict(self.backend)))

    @property
    def backend_subtree(self) -> Mapping[str, Any]:
        return self.backend[self.backend_key]

    def face_config(self, face: str) -> Mapping[str, Any]:
        conf = self.inspection.get("conf") or {}
        if not isinstance(conf, Mapping):
            return {}
        selected = conf.get(face, conf)
        return selected if isinstance(selected, Mapping) else {}


def build_runtime_session(
    product_config: Mapping[str, Any],
    resolved_backend: Mapping[str, Any],
    backend_key: str,
) -> CosmosRuntimeSession:
    """Validate and snapshot already-loaded request configuration facts."""

    inspection = product_config.get("inspection")
    if not isinstance(inspection, Mapping):
        raise ValueError("产品配置缺少 inspection")
    subtree = resolved_backend.get(backend_key)
    if not isinstance(subtree, Mapping):
        raise ValueError(f"后端配置缺少项目节点：{backend_key}")
    return CosmosRuntimeSession(
        inspection=inspection,
        backend_key=backend_key,
        backend={backend_key: subtree},
    )


__all__ = ["CosmosRuntimeSession", "build_runtime_session"]
