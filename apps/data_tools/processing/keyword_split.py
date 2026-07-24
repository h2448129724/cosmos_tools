"""关键字划分 — 根据文件名中的关键字将图片分类到子文件夹。"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from ..common.helpers import ensure_dir, get_image_files

logger = logging.getLogger(__name__)


def classify_by_keywords(
    input_dir: str,
    keywords: list[str],
    output_dir: str,
    mode: str = "copy",
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Scan input_dir for images, match filename keywords, copy/move to keyword subfolders.

    Returns dict mapping keyword -> file count. Unmatched files go to '_unsorted'.
    """
    files = get_image_files(input_dir)
    keywords_lower = [kw.lower() for kw in keywords]
    counts: dict[str, int] = {kw: 0 for kw in keywords}
    counts["_unsorted"] = 0

    for fpath in files:
        stem = Path(fpath).stem.lower()
        matched = False
        for kw, kw_lower in zip(keywords, keywords_lower):
            if kw_lower in stem:
                counts[kw] += 1
                matched = True
                if not dry_run:
                    dest_dir = os.path.join(output_dir, kw)
                    ensure_dir(dest_dir)
                    _transfer(fpath, os.path.join(dest_dir, os.path.basename(fpath)), mode)
                break
        if not matched:
            counts["_unsorted"] += 1
            if not dry_run:
                dest_dir = os.path.join(output_dir, "_unsorted")
                ensure_dir(dest_dir)
                _transfer(fpath, os.path.join(dest_dir, os.path.basename(fpath)), mode)

    return counts


def _transfer(src: str, dst: str, mode: str):
    if os.path.exists(dst):
        logger.warning("目标文件已存在，跳过: %s", dst)
        return
    try:
        if mode == "move":
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)
    except OSError as exc:
        logger.error("文件传输失败 %s -> %s: %s", src, dst, exc)
