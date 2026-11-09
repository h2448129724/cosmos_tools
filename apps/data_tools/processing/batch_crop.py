"""批量裁剪 — 使用相同 ROI 矩形批量裁剪文件夹中的所有图片。"""
from __future__ import annotations

import os

from img_tools.core.crop_plan import CoordinateMode, CropRegion, build_image_crop_plan

from .image_io import read_image, write_image
from ..common.helpers import ensure_dir, get_image_files


def crop_single_image(
    image_path: str,
    rects: list[tuple[int, int, int, int]],
    ref_width: int,
    ref_height: int,
    output_dir: str,
    *,
    coordinate_mode: CoordinateMode = "scaled",
    output_names: list[str] | None = None,
) -> int:
    """Crop one image using ROI rectangles and flattened naming.

    ``scaled`` maps the rectangles from the reference size onto this image.
    ``absolute`` slices the given pixels and ignores the reference size.
    """
    if not rects:
        return 0
    if coordinate_mode not in {"absolute", "scaled"}:
        raise ValueError(f"未知的坐标模式：{coordinate_mode}")
    if coordinate_mode == "scaled" and (ref_width <= 0 or ref_height <= 0):
        return 0

    img = read_image(image_path)
    if img is None:
        return 0

    ensure_dir(output_dir)

    h, w = img.shape[:2]
    reference_size = (ref_width, ref_height)
    if coordinate_mode == "absolute" and (ref_width <= 0 or ref_height <= 0):
        reference_size = (w, h)
    plan = build_image_crop_plan(
        source_name=image_path,
        image_size=(w, h),
        regions=(
            CropRegion(x1, y1, x2, y2, f"roi_{index}")
            for index, (x1, y1, x2, y2) in enumerate(rects, start=1)
        ),
        reference_size=reference_size,
        coordinate_mode=coordinate_mode,
        scale_semantics="edges",
        suffix_mode="preserve",
    )
    if output_names is None:
        chosen_names = [operation.output_name for operation in plan.operations]
    else:
        if len(output_names) != len(plan.operations):
            raise ValueError("输出文件名数量与有效 ROI 不一致")
        chosen_names = [os.path.basename(name) for name in output_names]
    total = 0

    for operation, output_name in zip(plan.operations, chosen_names, strict=True):
        x1, y1, x2, y2 = operation.box
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(output_dir, output_name)
        write_image(out_path, crop)
        total += 1

    return total


def batch_crop(
    input_dir: str,
    rects: list[tuple[int, int, int, int]],
    ref_width: int,
    ref_height: int,
    output_dir: str,
    *,
    progress_callback=None,
) -> int:
    """Crop all images using the same ROI rects, scaling for different sizes.

    All crops are written into output_dir. Each file is suffixed with roi_N
    to avoid collisions when multiple ROI regions are exported.
    """
    files = get_image_files(input_dir)
    if not files or not rects:
        return 0

    ensure_dir(output_dir)

    total = 0
    for fi, fpath in enumerate(files):
        total += crop_single_image(fpath, rects, ref_width, ref_height, output_dir)

        if progress_callback:
            progress_callback(fi + 1, len(files))

    return total
