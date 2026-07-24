"""Consistent output collision and directory-layout policies."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

ConflictPolicy = Literal["rename", "skip", "overwrite"]


def resolve_output_path(path: Path, policy: ConflictPolicy) -> Path | None:
    """Return an output path, or ``None`` when an existing file should be skipped."""
    if not path.exists() or policy == "overwrite":
        return path
    if policy == "skip":
        return None
    if policy != "rename":
        raise ValueError(f"未知的重名策略：{policy}")
    for number in range(1, 100_000):
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"无法生成输出文件名：{path}")
