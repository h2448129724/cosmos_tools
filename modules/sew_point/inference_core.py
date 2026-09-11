"""Pure rules shared by the sew-point inference adapters.

The PyTorch and ONNX modules intentionally keep all model/session, image I/O,
and visualisation concerns in their imperative shells.  This module contains
only deterministic checkpoint decoding, seven-path test-time augmentation,
and heatmap post-processing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

import numpy as np


class CheckpointParts(NamedTuple):
    """Decoded checkpoint while retaining the historical metadata payload."""

    model_name: str | None
    state: Any
    metadata: Mapping[str, Any]


def decode_checkpoint(checkpoint: Any, default_model_name: str | None = None) -> CheckpointParts:
    """Split raw state dictionaries and wrapped training checkpoints.

    Historically a checkpoint was either directly a state dictionary or a
    mapping containing ``model_state`` and optional metadata.  Do not validate
    or coerce the state object here: ``load_state_dict`` remains responsible
    for reporting its original compatibility errors.
    """

    if isinstance(checkpoint, Mapping) and "model_state" in checkpoint:
        model_name = checkpoint.get("model_key") or default_model_name
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state"}
        return CheckpointParts(str(model_name) if model_name is not None else None,
                               checkpoint["model_state"], metadata)
    return CheckpointParts(default_model_name, checkpoint, {})


# Keep this order in sync with the historical implementations.  The first four
# paths are identity/horizontal/vertical/both flips, followed by 90/180/270°.
TTA_PATHS = ("identity", "flip_h", "flip_v", "flip_hv", "rot90", "rot180", "rot270")


@dataclass(frozen=True)
class TTAPath:
    """Geometry contract for one seven-path TTA branch.

    Shapes use ``(height, width)`` order.  ``native_shape`` is the shape after
    the spatial transform and before any adapter/model constraint.  The
    ``inverse_steps`` explicitly describe how model output returns to source
    coordinates; adapters must not reconstruct this policy themselves.
    """

    name: str
    source_shape: tuple[int, int]
    native_shape: tuple[int, int]
    model_shape: tuple[int, int]
    inverse_steps: tuple[str, ...]

    @property
    def model_input_shape(self) -> tuple[int, int]:
        return self.model_shape

    @property
    def model_output_shape(self) -> tuple[int, int]:
        # The heatmap model is fully convolutional and historically preserves
        # its input spatial shape.  Adapters still pass actual output arrays to
        # ``inverse_tta`` so unusual model strides remain safely handled.
        return self.model_shape

    @property
    def inverse_geometry(self) -> tuple[str, ...]:
        return self.inverse_steps


def tta_plan(height: int, width: int, model_shape: tuple[int, int] | None = None) -> tuple[TTAPath, ...]:
    """Build the seven deterministic TTA paths and their shape geometry.

    ``model_shape`` is an optional fixed adapter constraint.  With no
    constraint, rotations retain their native ``(width, height)`` shape,
    avoiding the historical non-square coordinate drift.
    """

    source = (int(height), int(width))
    names = TTA_PATHS
    paths: list[TTAPath] = []
    for name in names:
        native = (width, height) if name in {"rot90", "rot270"} else source
        inverse = {
            "identity": (),
            "flip_h": ("flip_h",),
            "flip_v": ("flip_v",),
            "flip_hv": ("flip_hv",),
            "rot90": ("rot270",),
            "rot180": ("rot180",),
            "rot270": ("rot90",),
        }[name]
        paths.append(TTAPath(name, source, native, tuple(model_shape or native),
                             (("resize_to_native",) if model_shape and model_shape != native else ()) + inverse))
    return tuple(paths)


def _path_name(path: TTAPath | str) -> str:
    return path.name if isinstance(path, TTAPath) else path


def apply_tta(image: np.ndarray, path: TTAPath | str) -> np.ndarray:
    """Apply one TTA spatial transform, preserving channel and dtype."""

    name = _path_name(path)
    if name == "identity":
        return image
    if name == "flip_h":
        return image[:, ::-1, ...]
    if name == "flip_v":
        return image[::-1, :, ...]
    if name == "flip_hv":
        return image[::-1, ::-1, ...]
    if name == "rot90":
        return np.rot90(image, 1)
    if name == "rot180":
        return np.rot90(image, 2)
    if name == "rot270":
        return np.rot90(image, 3)
    raise ValueError(f"unknown TTA path: {name}")


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Small dependency-free fallback for non-square inverse TTA shapes."""

    target_h, target_w = shape
    if array.shape[:2] == (target_h, target_w):
        return array
    ys = np.minimum((np.arange(target_h) * array.shape[0] / target_h).astype(int), array.shape[0] - 1)
    xs = np.minimum((np.arange(target_w) * array.shape[1] / target_w).astype(int), array.shape[1] - 1)
    return array[np.ix_(ys, xs)]


def inverse_tta(
    heatmap: np.ndarray,
    path: TTAPath | str,
    target_shape: tuple[int, int] | None = None,
    resize_fn: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """Map a model heatmap from one TTA path back to image coordinates."""

    hm = np.asarray(heatmap).squeeze()
    name = _path_name(path)
    native_shape = path.native_shape if isinstance(path, TTAPath) else None
    if native_shape is not None and hm.shape[:2] != native_shape:
        if resize_fn is None:
            hm = _resize_nearest(hm, native_shape)
        else:
            hm = resize_fn(hm, (native_shape[1], native_shape[0]))
    if name == "identity":
        mapped = hm
    elif name == "flip_h":
        mapped = hm[:, ::-1]
    elif name == "flip_v":
        mapped = hm[::-1, :]
    elif name == "flip_hv":
        mapped = hm[::-1, ::-1]
    elif name == "rot90":
        mapped = np.rot90(hm, 3)
    elif name == "rot180":
        mapped = np.rot90(hm, 2)
    elif name == "rot270":
        mapped = np.rot90(hm, 1)
    else:
        raise ValueError(f"unknown TTA path: {path}")
    if target_shape is not None:
        mapped = _resize_nearest(mapped, (int(target_shape[0]), int(target_shape[1])))
    return mapped


def combine_tta_heatmaps(
    heatmaps: Sequence[np.ndarray],
    target_shape: tuple[int, int] | None = None,
    plan: Sequence[TTAPath] | None = None,
    resize_fn: Callable[..., np.ndarray] | None = None,
) -> np.ndarray:
    """Inverse-map and average seven heatmaps, returning ``(1, H, W)``."""

    if len(heatmaps) != len(TTA_PATHS):
        raise ValueError(f"expected {len(TTA_PATHS)} TTA heatmaps, got {len(heatmaps)}")
    if plan is None:
        if target_shape is None:
            target_shape = tuple(np.asarray(heatmaps[0]).squeeze().shape[:2])  # type: ignore[assignment]
        plan = tta_plan(*target_shape)
    if len(plan) != len(TTA_PATHS):
        raise ValueError(f"expected {len(TTA_PATHS)} TTA paths, got {len(plan)}")
    if target_shape is None:
        target_shape = plan[0].source_shape
    mapped = [inverse_tta(hm, path, target_shape, resize_fn=resize_fn)
              for hm, path in zip(heatmaps, plan)]
    return np.mean(mapped, axis=0)[np.newaxis, ...]


# Descriptive aliases make the policy easy to discover for adapters and tests.
build_tta_plan = tta_plan
inverse_tta_heatmap = inverse_tta
combine_tta = combine_tta_heatmaps


def detect_peaks(heatmap: np.ndarray, threshold: float = 0.5, cluster_dist: float = 3) -> list[tuple[int, int, float]]:
    """Find local maxima, cluster nearby peaks, and keep the best per cluster."""

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.ndimage import maximum_filter

    hm = np.asarray(heatmap).squeeze()
    local_max = maximum_filter(hm, size=5)
    peaks = (hm == local_max) & (hm > threshold)
    ys, xs = np.where(peaks)
    scores = hm[ys, xs]
    if len(xs) == 0:
        return []
    if len(xs) == 1:
        return [(int(xs[0]), int(ys[0]), float(scores[0]))]

    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    labels = fcluster(linkage(coords, method="complete", metric="euclidean"),
                      t=cluster_dist, criterion="distance")
    kept: list[tuple[int, int, float]] = []
    for cid in np.unique(labels):
        mask = labels == cid
        best = np.where(mask)[0][int(np.argmax(scores[mask]))]
        kept.append((int(xs[best]), int(ys[best]), float(scores[best])))
    return kept


__all__ = [
    "CheckpointParts", "decode_checkpoint", "TTA_PATHS", "TTAPath", "tta_plan", "build_tta_plan",
    "apply_tta", "inverse_tta", "inverse_tta_heatmap", "combine_tta_heatmaps", "combine_tta",
    "detect_peaks",
]
