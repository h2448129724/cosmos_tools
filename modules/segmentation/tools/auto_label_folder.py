import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

# ruff: noqa: E402  -- direct-script bootstrap must run before project imports

# Bootstrap: allow direct execution of ``python segmentation/tools/auto_label_folder.py``.
# No-op when launched via ``python -m segmentation.tools.auto_label_folder``.
try:
    from ._bootstrap import ensure_module_paths
except ImportError:
    from _bootstrap import ensure_module_paths

ensure_module_paths(__file__)

import cv2
import numpy as np
from PIL import Image

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from cabf import IMAGE_SUFFIXES
from segmentation.model_registry import get_model


def _is_image_file(name: str) -> bool:
    return os.path.splitext(name.lower())[1] in IMAGE_SUFFIXES


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _iter_image_paths(img_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(img_dir):
        for fn in files:
            if _is_image_file(fn):
                yield os.path.join(root, fn)


def _configure_pillow_max_pixels(allow_large_images: bool, max_image_pixels: Optional[int]) -> None:
    if allow_large_images:
        Image.MAX_IMAGE_PIXELS = None
        return
    if max_image_pixels is not None:
        Image.MAX_IMAGE_PIXELS = int(max_image_pixels)


def _transform() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


@dataclass(frozen=True)
class PolyConfig:
    prob_thresh: float
    min_area: int
    approx_epsilon: float
    close_kernel: int
    open_kernel: int


def _mask_to_polygons(mask01: np.ndarray, cfg: PolyConfig) -> List[List[List[float]]]:
    """
    输入：mask01 (H,W) 0/1
    输出：polygons，每个 polygon 是 [[x,y], ...] float
    """
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)
    bin255 = (mask01 * 255).astype(np.uint8)

    if cfg.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        bin255 = cv2.morphologyEx(bin255, cv2.MORPH_CLOSE, k)
    if cfg.open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_kernel, cfg.open_kernel))
        bin255 = cv2.morphologyEx(bin255, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(bin255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[List[float]]] = []
    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < int(cfg.min_area):
            continue
        peri = float(cv2.arcLength(cnt, True))
        eps = float(cfg.approx_epsilon) * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        poly = [[float(x), float(y)] for x, y in pts]
        polys.append(poly)
    return polys


def _make_xanylabeling_json(image_basename: str, w: int, h: int, polygons: List[List[List[float]]]) -> Dict:
    shapes = []
    for poly in polygons:
        shapes.append(
            {
                "label": "glue",
                "score": None,
                "points": poly,
                "group_id": None,
                "description": "",
                "difficult": False,
                "shape_type": "polygon",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        )

    return {
        "version": "4.0.0-beta.5",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_basename,
        "imageData": None,
        "imageHeight": int(h),
        "imageWidth": int(w),
    }


def _annotation_output_path(image_path: str, image_root: str, output_root: str) -> str:
    """Return a collision-free annotation path while retaining source layout.

    ``auto_label_folder`` accepts recursive input trees.  Flattening every
    image to ``output_root/<stem>.json`` loses samples such as
    ``left/a.png`` and ``right/a.png``.  Mirroring the relative source path
    keeps both the stable stem and its source identity for downstream tools.
    """
    source = os.path.abspath(image_path)
    root = os.path.abspath(image_root)
    try:
        relative = os.path.relpath(source, root)
    except ValueError as exc:  # pragma: no cover - only possible across drives
        raise ValueError(f"图片目录与图片不在同一文件系统：{image_path}") from exc
    relative_path = os.path.normpath(relative)
    if relative_path == os.pardir or relative_path.startswith(os.pardir + os.sep):
        raise ValueError(f"图片不在输入目录内：{image_path}")
    return os.path.join(output_root, os.path.splitext(relative_path)[0] + ".json")


def _plan_annotation_outputs(image_paths: Iterable[str], image_root: str, output_root: str) -> Dict[str, str]:
    """Plan all output names and reject any remaining path collision.

    Mirroring directories handles the usual recursive basename collision.  A
    single directory can still contain ``sample.png`` and ``sample.jpg``;
    both naturally map to ``sample.json``.  Failing before model loading or
    writing prevents a silent overwrite and makes the source identity issue
    actionable for the caller.
    """
    planned: Dict[str, str] = {}
    collisions: Dict[str, List[str]] = {}
    for image_path in image_paths:
        output_path = _annotation_output_path(image_path, image_root, output_root)
        key = os.path.normcase(os.path.abspath(output_path))
        if key in planned:
            collisions.setdefault(key, [planned[key]]).append(image_path)
        else:
            planned[key] = image_path
    if collisions:
        details = "; ".join(
            f"{output}: {', '.join(sources)}" for output, sources in collisions.items()
        )
        raise ValueError(f"多个源图片映射到同一标注文件，拒绝覆盖：{details}")
    return {source: _annotation_output_path(source, image_root, output_root) for source in planned.values()}


def _image_path_for_annotation(image_path: str, annotation_path: str) -> str:
    """Make ``imagePath`` resolve from the emitted JSON to its source image.

    X-AnyLabeling resolves ``imagePath`` relative to the annotation file.  A
    relative path therefore preserves the exact source even when two nested
    directories contain the same basename, and remains portable when the
    input/output trees are moved together.
    """
    try:
        relative = os.path.relpath(
            os.path.abspath(image_path), os.path.dirname(os.path.abspath(annotation_path))
        )
    except ValueError:
        # Windows cannot express a relative path across drive letters.  An
        # absolute path still preserves the exact source for X-AnyLabeling.
        relative = os.path.abspath(image_path)
    return relative.replace(os.sep, "/")


def _load_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"无法读取图片: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def _predict_prob_glue(model: torch.nn.Module, img_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    t = _transform()
    x = t(image=img_rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1]
    return probs[0].detach().float().cpu().numpy()


def main():
    # python auto_label_folder.py ^
#   --img_dir "D:\project\changrui\CAB-F\260312\data\images" ^
#   --out_ann_dir "D:\project\changrui\CAB-F\260312\data\annotations_pred" ^
#   --ckpt "D:\project\tianwei\glue_extract\checkpoints\xxx_best_model.pth" ^
#   --model microunet ^
#   --device cuda ^
#   --prob_thresh 0.5 ^
#   --min_area 200
    p = argparse.ArgumentParser(description="用训练好的 pth 批量自动标注文件夹内 1024x1024 图片，输出 X-AnyLabeling/Labelme 风格 glue polygon JSON。")
    p.add_argument("--img_dir", type=str, required=True, help="输入图片文件夹（递归遍历）")
    p.add_argument("--out_ann_dir", type=str, required=True, help="输出 JSON 文件夹")
    p.add_argument("--ckpt", type=str, required=True, help="模型权重 .pth 路径")
    p.add_argument("--model", type=str, default="microunet", help="模型名（与 model_registry 一致）")
    p.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    p.add_argument("--prob_thresh", type=float, default=0.5, help="像素阈值，>该值认为是 glue")
    p.add_argument("--min_area", type=int, default=200, help="最小轮廓面积（像素）")
    p.add_argument("--approx_epsilon", type=float, default=0.003, help="轮廓简化系数（越小点越多）")
    p.add_argument("--close_kernel", type=int, default=5, help="闭运算核尺寸（<=1 关闭）")
    p.add_argument("--open_kernel", type=int, default=0, help="开运算核尺寸（<=1 关闭）")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的 json")
    p.add_argument("--allow_large_images", action="store_true", help="允许超大图（关闭 Pillow 像素上限检查）")
    p.add_argument("--max_image_pixels", type=int, default=None, help="自定义 Pillow 像素上限")
    args = p.parse_args()

    _configure_pillow_max_pixels(bool(args.allow_large_images), args.max_image_pixels)

    img_paths = sorted(list(_iter_image_paths(args.img_dir)))
    if not img_paths:
        raise FileNotFoundError(f"没有找到图片: {args.img_dir}")
    output_paths = _plan_annotation_outputs(img_paths, args.img_dir, args.out_ann_dir)

    device = torch.device("cpu" if str(args.device).lower() == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = get_model(str(args.model), in_channels=3, n_classes=2).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    _safe_makedirs(args.out_ann_dir)

    poly_cfg = PolyConfig(
        prob_thresh=float(args.prob_thresh),
        min_area=int(args.min_area),
        approx_epsilon=float(args.approx_epsilon),
        close_kernel=int(args.close_kernel),
        open_kernel=int(args.open_kernel),
    )

    n_done = 0
    total_images = len(img_paths)
    for image_index, ip in enumerate(img_paths, start=1):
        base = os.path.basename(ip)
        out_json = output_paths[ip]
        if (not args.overwrite) and os.path.exists(out_json):
            print(f"自动标注 step {image_index:03d}/{total_images:03d} skip existing {base}")
            continue

        img_rgb = _load_image_rgb(ip)
        h, w = img_rgb.shape[:2]
        if (h, w) != (1024, 1024):
            raise ValueError(f"图片不是 1024x1024：{ip} size={(w, h)}")

        prob = _predict_prob_glue(model, img_rgb, device=device)
        mask01 = (prob > poly_cfg.prob_thresh).astype(np.uint8)
        polygons = _mask_to_polygons(mask01, poly_cfg)

        payload = _make_xanylabeling_json(_image_path_for_annotation(ip, out_json), w, h, polygons)
        # Keep the original source identity available to consumers that do not
        # resolve LabelMe's imagePath field relative to the JSON file.
        payload["sourceImagePath"] = os.path.abspath(ip)
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        n_done += 1
        print(f"自动标注 step {image_index:03d}/{total_images:03d} done={n_done} file={base}")

    print(f"完成：输出json={n_done}，目录={args.out_ann_dir}")


if __name__ == "__main__":
    main()
