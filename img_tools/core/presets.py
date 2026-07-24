"""Portable JSON presets for reusable ROI collections."""
from __future__ import annotations

import json
from pathlib import Path

from .models import Roi


def save_roi_preset(path: str | Path, rois: list[Roi], *, reference_size: tuple[int, int] | None) -> None:
    """Save ROIs and their optional reference image size as UTF-8 JSON."""
    if not rois:
        raise ValueError("没有可保存的 ROI")
    output = Path(path)
    payload = {
        "version": 1,
        "reference_size": list(reference_size) if reference_size else None,
        "rois": [{"name": roi.name, "x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height} for roi in rois],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_roi_preset(path: str | Path) -> tuple[list[Roi], tuple[int, int] | None]:
    """Read a ROI preset and validate its minimum schema."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("rois"), list):
        raise ValueError("不是受支持的 ROI 预设文件")
    rois = [Roi(int(item["x"]), int(item["y"]), int(item["width"]), int(item["height"]), str(item.get("name") or "ROI")) for item in data["rois"]]
    reference = data.get("reference_size")
    if reference is not None and (not isinstance(reference, list) or len(reference) != 2):
        raise ValueError("ROI 预设中的参考尺寸无效")
    return rois, tuple(map(int, reference)) if reference else None
