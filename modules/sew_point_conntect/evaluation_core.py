"""Pure Sew Point Connect evaluation rules.

This module intentionally has no Torch, OpenCV, filesystem, or CLI imports.  It
operates only on explicit point/edge facts supplied by the imperative shells.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Iterable, Mapping, Sequence

Edge = tuple[int, int]
PointFact = Mapping[str, object]
PredictedPoint = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PointMatchResult:
    """One-to-one point matches aligned to the input prediction order."""

    matched_gt_ids: tuple[int | None, ...]
    radius: float

    @property
    def matched_count(self) -> int:
        return sum(item is not None for item in self.matched_gt_ids)


@dataclass(frozen=True, slots=True)
class EdgeComparison:
    """Canonical GT/prediction sets and their TP/FP/FN partition."""

    ground_truth: frozenset[Edge]
    predicted: frozenset[Edge]
    true_positive: frozenset[Edge]
    false_positive: frozenset[Edge]
    false_negative: frozenset[Edge]

    def metrics(self) -> dict[str, float | int]:
        counts = summarize_counts(
            len(self.true_positive),
            len(self.false_positive),
            len(self.false_negative),
        )
        return {
            "gt_edges": len(self.ground_truth),
            "pred_edges": len(self.predicted),
            **counts,
        }


def normalize_edges(edges: Iterable[Mapping[str, object]] | None) -> frozenset[Edge]:
    """Normalize valid edges to an undirected, de-duplicated immutable set."""

    normalized: set[Edge] = set()
    for edge in edges or ():
        try:
            src = int(edge["src"])
            dst = int(edge["dst"])
        except (KeyError, TypeError, ValueError):
            continue
        if src == dst:
            continue
        normalized.add((src, dst) if src < dst else (dst, src))
    return frozenset(normalized)


def normalize_edge_set(edges: Iterable[Mapping[str, object]] | None) -> frozenset[Edge]:
    """Named compatibility alias for callers that use the historical helper."""

    return normalize_edges(edges)


def compare_edges(
    gt_edges: Iterable[Mapping[str, object]] | None,
    predicted_edges: Iterable[Mapping[str, object]] | None,
) -> EdgeComparison:
    """Partition normalized edges once for metrics and visualization shells."""

    ground_truth = normalize_edges(gt_edges)
    predicted = normalize_edges(predicted_edges)
    return EdgeComparison(
        ground_truth=ground_truth,
        predicted=predicted,
        true_positive=predicted & ground_truth,
        false_positive=predicted - ground_truth,
        false_negative=ground_truth - predicted,
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def summarize_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """Return the canonical zero-safe precision/recall/F1 presentation."""

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def edge_metrics(
    gt_edges: Iterable[Mapping[str, object]] | None,
    predicted_edges: Iterable[Mapping[str, object]] | None,
) -> dict[str, float | int]:
    """Compute undirected edge confusion metrics with the historical schema."""

    return compare_edges(gt_edges, predicted_edges).metrics()


def estimate_point_spacing(
    gt_points: Sequence[PointFact],
    default_spacing: float = 28.0,
) -> float:
    """Estimate robust nearest-neighbour spacing from explicit point facts."""

    if len(gt_points) < 2:
        return float(default_spacing)
    nearest: list[float] = []
    for index, point in enumerate(gt_points):
        try:
            x = float(point["x"])
            y = float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        distances: list[float] = []
        for other_index, other in enumerate(gt_points):
            if index == other_index:
                continue
            try:
                distance = hypot(x - float(other["x"]), y - float(other["y"]))
            except (KeyError, TypeError, ValueError):
                continue
            if isfinite(distance):
                distances.append(distance)
        if distances:
            nearest.append(min(distances))
    if not nearest:
        return float(default_spacing)
    ordered = sorted(nearest)
    # Match numpy.percentile's linear interpolation for the quartiles used by
    # datasets.estimate_spacing, without importing numpy into the core.
    def percentile(values: Sequence[float], fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    q1 = percentile(ordered, 0.25)
    q3 = percentile(ordered, 0.75)
    iqr = max(q3 - q1, 1e-6)
    keep = [value for value in ordered if q1 - 1.5 * iqr <= value <= q3 + 1.5 * iqr]
    values = sorted(keep or ordered)
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2)


def match_points(
    gt_points: Sequence[PointFact],
    predicted_points: Sequence[PredictedPoint],
    *,
    match_radius: float | None = None,
    default_spacing: float = 28.0,
) -> PointMatchResult:
    """Perform score-prioritized, one-to-one geometric matching.

    The returned IDs are aligned with ``predicted_points`` (unlike the old
    helper's sorted intermediate list), making graph lifting deterministic.
    A point exactly on the radius is considered a match.
    """

    normalized_gt: list[tuple[int, float, float]] = []
    for index, point in enumerate(gt_points):
        try:
            gt_id = int(point.get("id", index))
            x = float(point["x"])
            y = float(point["y"])
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if isfinite(x) and isfinite(y):
            normalized_gt.append((gt_id, x, y))
    normalized_gt.sort(key=lambda item: item[0])
    spacing = estimate_point_spacing(gt_points, default_spacing)
    radius = max(4.0, spacing * 0.45) if match_radius is None else float(match_radius)
    if not isfinite(radius) or radius < 0:
        raise ValueError("match_radius must be a finite non-negative value")

    matched_gt: set[int] = set()
    result: list[int | None] = [None] * len(predicted_points)

    def prediction_score(index: int) -> float:
        try:
            score = float(predicted_points[index][2])
        except (IndexError, TypeError, ValueError):
            return float("-inf")
        return score if isfinite(score) else float("-inf")

    ordered_indices = sorted(
        range(len(predicted_points)),
        key=prediction_score,
        reverse=True,
    )
    for pred_index in ordered_indices:
        try:
            x, y, _score = predicted_points[pred_index]
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            continue
        if not isfinite(x) or not isfinite(y):
            continue
        best_id: int | None = None
        best_distance: float | None = None
        for gt_id, gt_x, gt_y in normalized_gt:
            if gt_id in matched_gt:
                continue
            distance = hypot(x - gt_x, y - gt_y)
            if distance > radius:
                continue
            if best_distance is None or distance < best_distance:
                best_id, best_distance = gt_id, distance
        if best_id is not None:
            matched_gt.add(best_id)
            result[pred_index] = best_id
    return PointMatchResult(tuple(result), radius)


def point_metrics(
    gt_points: Sequence[PointFact],
    predicted_points: Sequence[PredictedPoint],
    *,
    match_radius: float | None = None,
    default_spacing: float = 28.0,
) -> dict[str, float | int]:
    matches = match_points(gt_points, predicted_points, match_radius=match_radius, default_spacing=default_spacing)
    counts = summarize_counts(matches.matched_count, len(predicted_points) - matches.matched_count, len(gt_points) - matches.matched_count)
    return {"gt_points": len(gt_points), "pred_points": len(predicted_points), **counts}


def lift_predicted_edges(
    gt_points: Sequence[PointFact],
    predicted_points: Sequence[PointFact],
    predicted_edges: Iterable[Mapping[str, object]],
    *,
    match_radius: float | None = None,
    default_spacing: float = 28.0,
) -> tuple[dict[str, int | float], ...]:
    """Map predicted point IDs to GT IDs and return valid unique edges."""

    facts: list[PredictedPoint] = []
    for point in predicted_points:
        try:
            facts.append((float(point["x"]), float(point["y"]), float(point.get("score", 1.0))))
        except (AttributeError, KeyError, TypeError, ValueError):
            facts.append((float("nan"), float("nan"), float("-inf")))
    matches = match_points(gt_points, facts, match_radius=match_radius, default_spacing=default_spacing)
    point_id_to_gt: dict[int, int | None] = {}
    for index, point in enumerate(predicted_points):
        try:
            point_id_to_gt[int(point.get("id", index))] = matches.matched_gt_ids[index]
        except (AttributeError, TypeError, ValueError):
            continue

    lifted: dict[Edge, float] = {}
    for edge in predicted_edges or ():
        try:
            src_gt = point_id_to_gt.get(int(edge["src"]))
            dst_gt = point_id_to_gt.get(int(edge["dst"]))
            if src_gt is None or dst_gt is None or src_gt == dst_gt:
                continue
            normalized = (src_gt, dst_gt) if src_gt < dst_gt else (dst_gt, src_gt)
            score = float(edge.get("score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if not isfinite(score):
            score = 0.0
        lifted[normalized] = max(score, lifted.get(normalized, float("-inf")))
    return tuple({"src": src, "dst": dst, "score": score} for (src, dst), score in sorted(lifted.items()))


def graph_metrics_from_matches(
    gt_points: Sequence[PointFact],
    gt_edges: Iterable[Mapping[str, object]],
    predicted_points: Sequence[PointFact],
    predicted_edges: Iterable[Mapping[str, object]],
    *,
    match_radius: float | None = None,
    default_spacing: float = 28.0,
) -> dict[str, float | int]:
    lifted = lift_predicted_edges(
        gt_points,
        predicted_points,
        predicted_edges,
        match_radius=match_radius,
        default_spacing=default_spacing,
    )
    return edge_metrics(gt_edges, lifted)
