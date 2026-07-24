"""sew_point 共享工具函数

文件读写、热力图峰值检测、滑窗坐标生成、关键点合并、Labelme JSON 导出。
"""

import json
import os

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_cabf_api():
    """Load the shared CAB-F API from the monorepo shared package."""
    from cabf import MASTER_SCHEMA_VERSION, make_empty_master_annotation, write_json
    return MASTER_SCHEMA_VERSION, make_empty_master_annotation, write_json


def _imread(path):
    """cv2.imread 的 Unicode 路径兼容版本。"""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _imwrite(path, img, params=None):
    """cv2.imwrite 的 Unicode 路径兼容版本。"""
    ext = os.path.splitext(path)[1]
    encode_params = [] if params is None else params
    success, encoded = cv2.imencode(ext, img, encode_params)
    if not success:
        raise OSError(f"图像编码失败: {path}")
    encoded.tofile(path)


def to_bgr(image):
    """将任意格式图像转为 BGR。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def detect_peaks(heatmap, threshold=0.5, cluster_dist=3):
    """在热力图中找局部最大值，层级聚类去重。返回 [(x, y, score), ...]。"""
    from scipy.ndimage import maximum_filter

    hm = heatmap.squeeze()
    local_max = maximum_filter(hm, size=5)
    peaks = (hm == local_max) & (hm > threshold)
    ys, xs = np.where(peaks)
    scores = hm[ys, xs]

    if len(xs) == 0:
        return []
    if len(xs) == 1:
        return [(int(xs[0]), int(ys[0]), float(scores[0]))]

    from scipy.cluster.hierarchy import fcluster, linkage

    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    Z = linkage(coords, method="complete", metric="euclidean")
    labels = fcluster(Z, t=cluster_dist, criterion="distance")

    kept = []
    for cid in np.unique(labels):
        mask = labels == cid
        idx = np.argmax(scores[mask])
        best = np.where(mask)[0][idx]
        kept.append((int(xs[best]), int(ys[best]), float(scores[best])))
    return kept


def iter_tiles(h, w, tile_size, stride):
    """生成 (x0, y0) 滑窗坐标列表，保证覆盖到图像边缘。"""
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]
    if xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))
    if ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    return [(x, y) for y in ys for x in xs]


def merge_keypoints(all_points, cluster_dist=3):
    """将重叠区域检测到的重复点合并，保留 score 最高的。"""
    if len(all_points) <= 1:
        return list(all_points)

    from scipy.cluster.hierarchy import fcluster, linkage

    coords = np.array([[p[0], p[1]] for p in all_points], dtype=np.float64)
    scores = np.array([p[2] for p in all_points])
    Z = linkage(coords, method="complete", metric="euclidean")
    labels = fcluster(Z, t=cluster_dist, criterion="distance")

    merged = []
    for cid in np.unique(labels):
        mask = labels == cid
        idx = np.argmax(scores[mask])
        best = np.where(mask)[0][idx]
        merged.append(all_points[best])
    return merged


def export_labelme_json(image_path, h, w, points, output_path, label_name="sew"):
    """导出 Labelme JSON。"""
    shapes = []
    for x, y, s in points:
        shapes.append({
            "label": label_name,
            "score": float(s),
            "points": [[float(x), float(y)]],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "point",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        })
    data = {
        "version": "3.3.9",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_master_json(image_path, h, w, points, output_path, sample_id=None, metadata=None):
    """导出 CAB-F 母格式 JSON。"""
    MASTER_SCHEMA_VERSION, make_empty_master_annotation, write_json = _load_cabf_api()
    if sample_id is None:
        sample_id = os.path.splitext(os.path.basename(image_path))[0]

    data = make_empty_master_annotation(
        image_path=os.path.basename(image_path),
        width=int(w),
        height=int(h),
        sample_id=str(sample_id),
    )

    point_items = []
    for idx, (x, y, s) in enumerate(points):
        point_items.append(
            {
                "id": idx,
                "x": float(x),
                "y": float(y),
                "score": float(s),
                "source": "model",
            }
        )

    merged_metadata = {
        "source": "sew_point_batch_infer",
        "point_count": len(point_items),
    }
    if metadata:
        merged_metadata.update(metadata)

    data["schema_version"] = MASTER_SCHEMA_VERSION
    data["points"] = point_items
    data["metadata"] = merged_metadata
    write_json(output_path, data)
