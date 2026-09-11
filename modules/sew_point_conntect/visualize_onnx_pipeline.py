"""Visualize the Sew Point -> Sew Point Connect ONNX pipeline.

The ``small`` mode sends the complete image through the point model once.  The
``large`` mode uses the generic Sew Point overlapping-tile detector.  This is a
pure point-then-edge pipeline and intentionally has no density-checker logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法编码可视化图片: {path}")
    encoded.tofile(str(path))


def _point_map(annotation: dict) -> dict[int, tuple[int, int]]:
    return {
        int(point.get("id", index)): (int(round(float(point["x"]))), int(round(float(point["y"]))))
        for index, point in enumerate(annotation.get("points", []))
    }


def _build_annotation(
    points: list[tuple[float, float, float]],
    image_path: Path,
    image: np.ndarray,
    point_model: Path,
    connector_model: Path,
    point_threshold: float,
) -> dict:
    height, width = image.shape[:2]
    return {
        "schema_version": "1.1",
        "sample_id": image_path.stem,
        "image_path": str(image_path),
        "image_size": {"width": int(width), "height": int(height)},
        "points": [
            {
                "id": index,
                "x": float(x),
                "y": float(y),
                "score": float(score),
                "source": "sew_point_detector",
            }
            for index, (x, y, score) in enumerate(points)
        ],
        "edges": [],
        "segments": [],
        "metadata": {
            "source": "cosmos_toolbox_onnx_pipeline_visualization",
            "point_model_path": str(point_model),
            "connect_model_path": str(connector_model),
            "point_threshold": float(point_threshold),
            "input_channel_order": "BGR",
        },
    }


def _drawing_sizes(image_shape: tuple[int, ...]) -> tuple[int, int, float]:
    """Return bounded edge width, point radius and text scale."""
    image_scale = max(image_shape[:2]) / 5000.0
    edge_width = max(1, min(3, int(round(image_scale))))
    point_radius = max(2, min(6, int(round(image_scale))))
    text_scale = max(0.4, min(1.2, max(image_shape[:2]) / 12000.0))
    return edge_width, point_radius, text_scale


def draw_pipeline_visualization(
    image: np.ndarray,
    annotation: dict,
    predicted_edges: list[dict],
) -> np.ndarray:
    """Draw predicted graph edges and points without requiring ground truth."""
    visual = image.copy()
    points = _point_map(annotation)
    edge_width, point_radius, text_scale = _drawing_sizes(image.shape)

    for edge in predicted_edges:
        src = int(edge["src"])
        dst = int(edge["dst"])
        if src in points and dst in points:
            cv2.line(visual, points[src], points[dst], (0, 220, 255), edge_width, cv2.LINE_AA)

    draw_ids = len(points) <= 300
    for point_id, position in points.items():
        cv2.circle(visual, position, point_radius, (80, 255, 80), -1, cv2.LINE_AA)
        cv2.circle(visual, position, point_radius + 1, (20, 20, 20), 1, cv2.LINE_AA)
        if draw_ids:
            cv2.putText(
                visual,
                str(point_id),
                (position[0] + point_radius + 2, position[1] - point_radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    labels = (f"Points: {len(points)}", f"Edges: {len(predicted_edges)}")
    y = max(22, int(round(24 * text_scale)))
    for label in labels:
        cv2.putText(
            visual,
            label,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            max(1, min(2, int(round(text_scale)))),
            cv2.LINE_AA,
        )
        y += max(24, int(round(28 * text_scale)))
    return visual


def run_pipeline(
    *,
    image_path: str,
    point_model: str,
    connector_model: str,
    patch_model: str,
    output_image: str,
    output_json: str,
    mode: str = "small",
    point_threshold: float = 0.5,
    connector_threshold: float = 0.5,
    cluster_dist: int = 3,
    tile_size: int = 256,
    stride: int = 192,
    batch_size: int = 16,
    patch_batch_size: int = 128,
) -> dict:
    from algo.models.ort_providers import get_ort_device
    from cosmos_toolbox.training.cab_f_project import project_entry
    from sew_point.inference_onnx import KeypointDetectorONNX
    from sew_point.tools.predict_large_image import detect_large_image

    SewPointConnector = project_entry().SewPointConnector

    source = Path(image_path).expanduser().resolve()
    image = _read_image(source)
    point_model_path = Path(point_model).expanduser().resolve()
    connector_model_path = Path(connector_model).expanduser().resolve()
    patch_model_path = Path(patch_model).expanduser().resolve()

    for label, path in (
        ("Sew Point ONNX", point_model_path),
        ("Connector ONNX", connector_model_path),
        ("Connector Patch ONNX", patch_model_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} 不存在: {path}")

    detector = KeypointDetectorONNX(
        model_path=str(point_model_path),
        device=get_ort_device(),
        threshold=float(point_threshold),
        cluster_dist=float(cluster_dist),
    )
    if mode == "large":
        points = detect_large_image(
            detector,
            image,
            tile_size=int(tile_size),
            stride=int(stride),
            cluster_dist=float(cluster_dist),
            use_tta=False,
            batch_size=int(batch_size),
        )
    elif mode == "small":
        points = detector.detect_numpy(np.ascontiguousarray(image), use_tta=False)
    else:
        raise ValueError(f"未知推理模式: {mode}")

    connector = SewPointConnector(
        {
            "path": connector_model_path,
            "patch_model_path": patch_model_path,
            "threshold": float(connector_threshold),
            "patch_batch_size": int(patch_batch_size),
        }
    )
    annotation = _build_annotation(
        points=points,
        image_path=source,
        image=image,
        point_model=point_model_path,
        connector_model=connector_model_path,
        point_threshold=float(point_threshold),
    )
    predicted_edges = connector.predict(annotation=annotation, image_bgr=image)
    annotation["edges"] = predicted_edges
    annotation["predicted_edges"] = predicted_edges
    annotation["metadata"].update(
        {
            "source": "cosmos_toolbox_onnx_pipeline_visualization",
            "inference_mode": mode,
            "patch_model_path": str(patch_model_path),
            "connector_threshold": float(connector_threshold),
            "point_model_channel_order": "BGR",
            "tile_size": int(tile_size) if mode == "large" else None,
            "stride": int(stride) if mode == "large" else None,
        }
    )

    output_image_path = Path(output_image).expanduser().resolve()
    output_json_path = Path(output_json).expanduser().resolve()
    visual = draw_pipeline_visualization(image, annotation, predicted_edges)
    _write_image(output_image_path, visual)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "mode": mode,
        "points": len(points),
        "edges": len(predicted_edges),
        "output_image": str(output_image_path),
        "output_json": str(output_json_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sew Point 与 Sew Point Connect ONNX 全流程可视化")
    parser.add_argument("--image-path", required=True, help="待推理图片")
    parser.add_argument("--point-model", required=True, help="Sew Point ONNX 模型")
    parser.add_argument("--connector-model", required=True, help="Connector 图网络 ONNX 模型")
    parser.add_argument("--patch-model", required=True, help="Connector Patch Encoder ONNX 模型")
    parser.add_argument("--output-image", required=True, help="输出可视化图片")
    parser.add_argument("--output-json", required=True, help="输出预测 JSON")
    parser.add_argument("--mode", choices=("small", "large"), default="small", help="small=训练尺寸图，large=普通重叠滑窗")
    parser.add_argument("--point-threshold", type=float, default=0.5, help="关键点阈值")
    parser.add_argument("--connector-threshold", type=float, default=0.5, help="连边阈值")
    parser.add_argument("--cluster-dist", type=int, default=3, help="关键点合并距离")
    parser.add_argument("--tile-size", type=int, default=256, help="大图滑窗尺寸")
    parser.add_argument("--stride", type=int, default=192, help="大图滑窗步长")
    parser.add_argument("--batch-size", type=int, default=16, help="关键点滑窗批大小")
    parser.add_argument("--patch-batch-size", type=int, default=128, help="边 Patch 编码批大小")
    return parser


def main() -> None:
    run_pipeline(**vars(build_parser().parse_args()))


if __name__ == "__main__":
    main()
