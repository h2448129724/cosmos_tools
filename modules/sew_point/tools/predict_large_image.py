r"""
大图滑窗关键点推理

用法：
    python -m sew_point.tools.predict_large_image --image <path> --model <path>
"""

import argparse
import os
import time

import cv2

# ruff: noqa: E402  -- direct-script bootstrap must run before project imports

# Bootstrap: allow direct execution of ``python sew_point/tools/predict_large_image.py``.
# No-op when launched via ``python -m sew_point.tools.predict_large_image``
# (which expects PYTHONPATH=modules;shared/cabf_common).
try:
    from ._bootstrap import ensure_module_paths
except ImportError:
    from _bootstrap import ensure_module_paths

ensure_module_paths(__file__)

from sew_point.inference_onnx import KeypointDetectorONNX
from sew_point.utils import (_imread, _imwrite, iter_tiles, merge_keypoints,
                             to_bgr, export_labelme_json)


def detect_large_image(detector, image, tile_size=256, stride=192,
                       cluster_dist=3, use_tta=True, batch_size=32):
    """对大图滑窗推理，返回 [(x, y, score), ...]。"""
    image_bgr = to_bgr(image)
    h, w = image_bgr.shape[:2]
    coords = iter_tiles(h, w, tile_size, stride)

    tiles = []
    for x0, y0 in coords:
        tile = image_bgr[y0:y0 + tile_size, x0:x0 + tile_size]
        th, tw = tile.shape[:2]
        if th != tile_size or tw != tile_size:
            tile = cv2.copyMakeBorder(tile, 0, tile_size - th, 0, tile_size - tw,
                                      cv2.BORDER_REFLECT_101)
        tiles.append(tile)

    all_points = []
    if use_tta:
        total_tiles = len(tiles)
        for tile_index, ((x0, y0), tile) in enumerate(zip(coords, tiles), start=1):
            print(f"滑窗推理 [TTA] step {tile_index:03d}/{total_tiles:03d} x={x0} y={y0}")
            for px, py, s in detector.detect_numpy(tile, use_tta=True):
                ax, ay = px + x0, py + y0
                if ax < w and ay < h:
                    all_points.append((float(ax), float(ay), float(s)))
    else:
        bs = batch_size
        batch_starts = list(range(0, len(tiles), bs))
        total_batches = len(batch_starts)
        for batch_index, i in enumerate(batch_starts, start=1):
            print(f"滑窗推理 [Batch] step {batch_index:03d}/{total_batches:03d} start={i} batch_size={len(tiles[i:i + bs])}")
            batch = tiles[i:i + bs]
            batch_xy = coords[i:i + bs]
            for (x0, y0), pts in zip(batch_xy, detector.detect_batch_numpy(batch)):
                for px, py, s in pts:
                    ax, ay = px + x0, py + y0
                    if ax < w and ay < h:
                        all_points.append((float(ax), float(ay), float(s)))

    return merge_keypoints(all_points, cluster_dist)


def main():
    parser = argparse.ArgumentParser(description="大图滑窗关键点检测")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cluster-dist", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--point-radius", type=int, default=2,
                        help="可视化点半径，默认 2 像素")
    parser.add_argument("--visualization-format", choices=["jpg", "png"], default="jpg",
                        help="可视化图片格式，默认使用压缩率更高的 jpg")
    parser.add_argument("--jpeg-quality", type=int, default=85,
                        help="JPEG 可视化质量，范围 0-100，默认 85")
    parser.add_argument("--png-compression", type=int, default=9,
                        help="PNG 压缩等级，范围 0-9，默认 9")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints", "best.onnx")

    detector = KeypointDetectorONNX(model_path=model_path, device=args.device,
                                    threshold=args.threshold, cluster_dist=args.cluster_dist)

    image = _imread(args.image)
    if image is None:
        print(f"[ERROR] 无法读取: {args.image}")
        return
    h, w = image.shape[:2]
    print(f"[INFO] {w}x{h}  tiles={len(iter_tiles(h, w, args.tile_size, args.stride))}")

    t0 = time.time()
    points = detect_large_image(detector, image, args.tile_size, args.stride,
                                args.cluster_dist, not args.no_tta, args.batch_size)
    print(f"[DETECT] {len(points)} 个关键点  ({time.time()-t0:.1f}s)")

    output_dir = args.output or os.path.dirname(args.image)
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.image))[0]

    vis = image.copy()
    for x, y, s in points:
        cv2.circle(vis, (int(x), int(y)), args.point_radius, (0, 0, 255), -1)
    vis_ext = args.visualization_format
    if vis_ext == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
    else:
        encode_params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
    _imwrite(os.path.join(output_dir, f"{name}_pred.{vis_ext}"), vis, encode_params)
    export_labelme_json(args.image, h, w, points, os.path.join(output_dir, f"{name}_pred.json"))


if __name__ == "__main__":
    main()
