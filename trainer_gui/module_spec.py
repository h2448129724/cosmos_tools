"""module.json 校验工具。

为 GUI 自动发现模块提供结构护栏：检测 feature_name、actions、entry、output
等字段缺失或类型错误，并在扫描时报告可诊断错误而非静默跳过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    """单条校验错误。"""
    path: str          # module.json 的路径（用于定位）
    field: str         # 错误字段（点分路径，如 "actions[0].entry.type"）
    message: str       # 人类可读的错误描述

    def __str__(self) -> str:
        return f"{self.path}: {self.field} — {self.message}"


@dataclass
class ModuleSpec:
    """校验后的 module.json 结果。"""
    path: Path
    feature_name: str
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# 校验常量
# ---------------------------------------------------------------------------

VALID_ENTRY_TYPES = {"script", "module"}


# ---------------------------------------------------------------------------
# 校验函数
# ---------------------------------------------------------------------------

def validate_module_json(path: Path) -> ModuleSpec:
    """校验单个 module.json 文件。

    返回 ``ModuleSpec``，其中 ``errors`` 列表记录所有发现的问题。
    如果 JSON 解析失败则产生一条顶层错误。
    """
    spec = ModuleSpec(path=path, feature_name="")

    if not path.exists():
        spec.errors.append(ValidationError(str(path), "", "文件不存在"))
        return spec

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        spec.errors.append(ValidationError(str(path), "", f"JSON 解析失败: {exc}"))
        return spec

    if not isinstance(data, dict):
        spec.errors.append(ValidationError(str(path), "", "顶层结构必须是 JSON 对象"))
        return spec

    # --- feature_name ---
    feature_name = data.get("feature_name")
    if not feature_name or not isinstance(feature_name, str):
        spec.errors.append(ValidationError(str(path), "feature_name", "缺失或非字符串"))
    else:
        spec.feature_name = feature_name

    # --- actions ---
    actions = data.get("actions")
    if actions is None:
        # 允许无 actions（使用默认训练入口）
        pass
    elif not isinstance(actions, list):
        spec.errors.append(ValidationError(str(path), "actions", "必须是数组"))
    else:
        for i, raw in enumerate(actions):
            if not isinstance(raw, dict):
                spec.errors.append(
                    ValidationError(str(path), f"actions[{i}]", "每个 action 必须是对象")
                )
                continue
            _validate_action(spec, data, raw, i)

    return spec


def _validate_action(spec: ModuleSpec, top: dict, raw: dict, index: int) -> None:
    """校验单个 action 项。"""
    prefix = f"actions[{index}]"
    pstr = str(spec.path)

    # action_name
    action_name = raw.get("action_name") or raw.get("name")
    if not action_name or not isinstance(action_name, str):
        spec.errors.append(ValidationError(pstr, f"{prefix}.action_name", "缺失或非字符串"))

    # entry
    entry_data = raw.get("entry") or top.get("entry")
    if entry_data is None:
        # 允许缺失——FeatureScanner 会给出默认值 "script" / "train.py"
        pass
    elif not isinstance(entry_data, dict):
        spec.errors.append(ValidationError(pstr, f"{prefix}.entry", "必须是对象"))
    else:
        entry_type = entry_data.get("type")
        if entry_type not in VALID_ENTRY_TYPES:
            spec.errors.append(
                ValidationError(pstr, f"{prefix}.entry.type", f"必须是 {VALID_ENTRY_TYPES} 之一，实际为 {entry_type!r}")
            )
        entry_value = entry_data.get("value")
        if not entry_value or not isinstance(entry_value, str):
            spec.errors.append(ValidationError(pstr, f"{prefix}.entry.value", "缺失或非字符串"))

    # output
    output_data = raw.get("output") or top.get("output")
    if output_data is not None and not isinstance(output_data, dict):
        spec.errors.append(ValidationError(pstr, f"{prefix}.output", "必须是对象"))
    elif isinstance(output_data, dict):
        arg_name = output_data.get("arg_name")
        if arg_name is not None and not isinstance(arg_name, str):
            spec.errors.append(ValidationError(pstr, f"{prefix}.output.arg_name", "非字符串"))
        # extra_outputs
        extra = output_data.get("extra_outputs")
        if extra is not None:
            if not isinstance(extra, list):
                spec.errors.append(ValidationError(pstr, f"{prefix}.output.extra_outputs", "必须是数组"))
            else:
                for j, item in enumerate(extra):
                    if not isinstance(item, dict):
                        spec.errors.append(
                            ValidationError(pstr, f"{prefix}.output.extra_outputs[{j}]", "必须是对象")
                        )
                    elif not item.get("arg_name"):
                        spec.errors.append(
                            ValidationError(pstr, f"{prefix}.output.extra_outputs[{j}].arg_name", "缺失")
                        )

    # field_overrides
    fo = raw.get("field_overrides")
    if fo is not None and not isinstance(fo, dict):
        spec.errors.append(ValidationError(pstr, f"{prefix}.field_overrides", "必须是对象"))
    elif isinstance(fo, dict):
        for key, val in fo.items():
            if not isinstance(val, dict):
                spec.errors.append(ValidationError(pstr, f"{prefix}.field_overrides.{key}", "必须是对象"))

    # ui
    ui = raw.get("ui")
    if ui is not None and not isinstance(ui, dict):
        spec.errors.append(ValidationError(pstr, f"{prefix}.ui", "必须是对象"))


def validate_all_modules(modules_root: Path) -> list[ModuleSpec]:
    """扫描 modules_root 下所有 module.json 并返回校验结果列表。"""
    results: list[ModuleSpec] = []
    if not modules_root.is_dir():
        return results
    for child in sorted(modules_root.iterdir()):
        if not child.is_dir():
            continue
        json_path = child / "module.json"
        if json_path.exists():
            results.append(validate_module_json(json_path))
    return results
