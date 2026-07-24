from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from .paths import COSMOS_ROOT


class CabfConfigError(ValueError):
    """Raised when a CAB-F configuration cannot support the requested action."""


def _atomic_write_with_backup(path: Path, text: str) -> Path | None:
    """Atomically replace a production config and keep its previous bytes."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_suffix(path.suffix + ".bak") if path.exists() else None
    backup_tmp: Path | None = None
    output_tmp: Path | None = None
    try:
        if backup is not None:
            handle, name = tempfile.mkstemp(prefix=f".{backup.name}.", suffix=".tmp", dir=path.parent)
            os.close(handle)
            backup_tmp = Path(name)
            shutil.copy2(path, backup_tmp)
            os.replace(backup_tmp, backup)
            backup_tmp = None

        handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        output_tmp = Path(name)
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(output_tmp, path)
        output_tmp = None
        return backup
    finally:
        for temporary in (backup_tmp, output_tmp):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class RoiField:
    path_parts: tuple[Any, ...]
    display_name: str
    side: str
    is_multi: bool

    @property
    def path_key(self) -> str:
        return ".".join(str(part) for part in self.path_parts)


@dataclass(frozen=True, slots=True)
class RoiIssue:
    field: str
    roi_index: int
    message: str


@dataclass(slots=True)
class TemplateGenerationResult:
    image: np.ndarray
    model_path: str
    source_size: tuple[int, int]
    template_size: tuple[int, int]
    calibrated_image: np.ndarray | None = None
    calibration_offset: tuple[int, int] = (0, 0)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_rect_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(_is_number(item) for item in value)


def _is_multi_rect_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_is_rect_list(item) for item in value)


def normalize_rects(value: Any) -> list[tuple[int, int, int, int]]:
    if _is_rect_list(value):
        return [tuple(int(item) for item in value)]
    if _is_multi_rect_list(value):
        return [tuple(int(item) for item in rect) for rect in value]
    return []


def _detect_side(path_parts: tuple[Any, ...]) -> str:
    lowered = {str(part).lower() for part in path_parts}
    if "top" in lowered:
        return "top"
    if "bottom" in lowered:
        return "bottom"
    return "unknown"


def collect_roi_fields(node: Any, path_parts: tuple[Any, ...] = ()) -> list[RoiField]:
    fields: list[RoiField] = []
    if isinstance(node, dict):
        for key, value in node.items():
            next_path = path_parts + (key,)
            if str(key).lower() in {"roi", "rois"} and (_is_rect_list(value) or _is_multi_rect_list(value)):
                fields.append(
                    RoiField(
                        path_parts=next_path,
                        display_name=".".join(str(part) for part in next_path),
                        side=_detect_side(next_path),
                        is_multi=_is_multi_rect_list(value),
                    )
                )
            fields.extend(collect_roi_fields(value, next_path))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            fields.extend(collect_roi_fields(value, path_parts + (index,)))
    return fields


def _get_value(data: Any, path_parts: tuple[Any, ...]) -> Any:
    current = data
    for part in path_parts:
        current = current[part]
    return current


def _set_rects(data: Any, field: RoiField, rects: list[tuple[int, int, int, int]]) -> None:
    current = data
    for part in field.path_parts[:-1]:
        current = current[part]
    if field.is_multi:
        current[field.path_parts[-1]] = [[int(value) for value in rect] for rect in rects]
    else:
        current[field.path_parts[-1]] = [int(value) for value in rects[0]] if rects else []


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _find_yaml_key_line(lines: list[str], path_parts: tuple[Any, ...]) -> int:
    if not path_parts or any(not isinstance(part, str) for part in path_parts):
        return -1
    start = 0
    end = len(lines)
    parent_indent = -1
    for depth, key in enumerate(path_parts):
        found_index = -1
        found_indent = -1
        expected_prefix = f"{key}:"
        for index in range(start, end):
            stripped = lines[index].strip()
            if not stripped or stripped.startswith("#") or not stripped.startswith(expected_prefix):
                continue
            indent = _line_indent(lines[index])
            if indent <= parent_indent:
                continue
            if found_index == -1 or indent < found_indent:
                found_index = index
                found_indent = indent
        if found_index < 0:
            return -1
        if depth == len(path_parts) - 1:
            return found_index
        start = found_index + 1
        end = len(lines)
        for index in range(start, len(lines)):
            stripped = lines[index].strip()
            if stripped and not stripped.startswith("#") and _line_indent(lines[index]) <= found_indent:
                end = index
                break
        parent_indent = found_indent
    return -1


def _format_roi_value(field: RoiField, rects: list[tuple[int, int, int, int]]) -> str:
    if field.is_multi:
        return "[" + ", ".join(f"[{x1}, {y1}, {x2}, {y2}]" for x1, y1, x2, y2 in rects) + "]"
    if not rects:
        return "[]"
    return "[" + ", ".join(str(value) for value in rects[0]) + "]"


def _replace_yaml_value_line(line: str, value: str) -> str:
    newline = "\n" if line.endswith("\n") else ""
    body = line[:-1] if newline else line
    comment = ""
    if "#" in body:
        body, comment = body.split("#", 1)
        comment = "#" + comment
    prefix, separator, _old_value = body.partition(":")
    if not separator:
        raise CabfConfigError(f"无法更新 YAML 行：{line.rstrip()}")
    replaced = f"{prefix}: {value}"
    if comment:
        replaced += f"  {comment}"
    return replaced + newline


def _patch_roi_text(text: str, data: dict[str, Any], fields: list[RoiField]) -> str:
    lines = text.splitlines(keepends=True)
    for field in fields:
        index = _find_yaml_key_line(lines, field.path_parts)
        if index >= 0:
            lines[index] = _replace_yaml_value_line(
                lines[index],
                _format_roi_value(field, normalize_rects(_get_value(data, field.path_parts))),
            )
    return "".join(lines)


_LIST_ITEM_RE = re.compile(r"^(?P<indent>\s*)-(?P<rest>.*)$")
_PATH_VALUE_RE = re.compile(r"^(?P<prefix>\s*(?:-\s*)?path\s*:\s*)(?P<value>.*?)(?P<comment>\s+#.*)?$")


def _quote_like(old_value: str, new_value: str) -> str:
    old_value = old_value.strip()
    if len(old_value) >= 2 and old_value[0] == old_value[-1] and old_value[0] in {"'", '"'}:
        quote = old_value[0]
        escaped = new_value.replace(quote, f"\\{quote}")
        return f"{quote}{escaped}{quote}"
    return f'"{new_value}"'


def _patch_match_template_path(text: str, side_index: int, new_value: str) -> str:
    lines = text.splitlines(keepends=True)
    header_index = _find_yaml_key_line(lines, ("inspection", "match_template"))
    if header_index < 0:
        raise CabfConfigError("配置中没有找到 inspection.match_template")
    base_indent = _line_indent(lines[header_index])
    current_item = -1
    for index in range(header_index + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped and not stripped.startswith("#") and _line_indent(lines[index]) <= base_indent:
            break
        item_match = _LIST_ITEM_RE.match(lines[index].rstrip("\n"))
        if item_match and len(item_match.group("indent")) > base_indent:
            current_item += 1
            rest = item_match.group("rest").strip()
            if current_item == side_index and rest and not rest.startswith("path:"):
                newline = "\n" if lines[index].endswith("\n") else ""
                comment = ""
                scalar = rest
                if " #" in scalar:
                    scalar, comment = scalar.split(" #", 1)
                    comment = "  #" + comment
                lines[index] = f"{item_match.group('indent')}- {_quote_like(scalar, new_value)}{comment}{newline}"
                return "".join(lines)
        if current_item == side_index:
            newline = "\n" if lines[index].endswith("\n") else ""
            body = lines[index][:-1] if newline else lines[index]
            match = _PATH_VALUE_RE.match(body)
            if match:
                quoted = _quote_like(match.group("value"), new_value)
                lines[index] = f"{match.group('prefix')}{quoted}{match.group('comment') or ''}{newline}"
                return "".join(lines)
    raise CabfConfigError(f"inspection.match_template 缺少索引 {side_index} 的模板路径")


def resolve_config_path(value: str, config_path: Path, cosmos_root: Path = COSMOS_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (cosmos_root / path, config_path.parent / path)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def path_for_config(path: Path, cosmos_root: Path = COSMOS_ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(cosmos_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


class CabfConfigDocument:
    SIDE_INDEX = {"top": 0, "bottom": 1}

    def __init__(self, path: Path, data: dict[str, Any], original_text: str) -> None:
        self.path = path.resolve()
        self.data = data
        self.original_text = original_text
        self._roi_fields = collect_roi_fields(data)

    @classmethod
    def load(cls, path: str | Path) -> "CabfConfigDocument":
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"配置文件不存在：{resolved}")
        text = resolved.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or not isinstance(data.get("inspection"), dict):
            raise CabfConfigError("不是有效的 CAB-F 配置：缺少 inspection")
        return cls(resolved, data, text)

    @property
    def project_name(self) -> str:
        inspection = self.data.get("inspection", {})
        project = str(inspection.get("project", "CAB-F"))
        product = str(inspection.get("product", ""))
        return " / ".join(item for item in (project, product) if item)

    def roi_fields(self, side: str) -> list[RoiField]:
        return [field for field in self._roi_fields if field.side == side]

    def rects(self, field: RoiField) -> list[tuple[int, int, int, int]]:
        return normalize_rects(_get_value(self.data, field.path_parts))

    def set_rects(self, field: RoiField, rects: list[tuple[int, int, int, int]]) -> None:
        if not field.is_multi and len(rects) > 1:
            raise CabfConfigError(f"{field.display_name} 是单 ROI 字段")
        _set_rects(self.data, field, rects)

    def template_value(self, side: str) -> str:
        index = self.SIDE_INDEX[side]
        templates = self.data.get("inspection", {}).get("match_template", [])
        if not isinstance(templates, list) or index >= len(templates):
            return ""
        value = templates[index]
        if isinstance(value, dict):
            return str(value.get("path", "") or "")
        return str(value or "")

    def template_path(self, side: str) -> Path | None:
        value = self.template_value(side)
        return resolve_config_path(value, self.path) if value else None

    def validate_rois(self, side: str, image_size: tuple[int, int]) -> list[RoiIssue]:
        width, height = image_size
        issues: list[RoiIssue] = []
        for field in self.roi_fields(side):
            for index, (x1, y1, x2, y2) in enumerate(self.rects(field)):
                if x2 <= x1 or y2 <= y1:
                    issues.append(RoiIssue(field.display_name, index, "坐标顺序无效"))
                elif x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    issues.append(RoiIssue(field.display_name, index, f"超出基准图 {width}×{height}"))
        return issues

    def save_rois(self) -> None:
        patched = _patch_roi_text(self.original_text, self.data, self._roi_fields)
        _atomic_write_with_backup(self.path, patched)
        self.original_text = patched

    def update_template_path(self, side: str, template_path: str | Path) -> str:
        index = self.SIDE_INDEX[side]
        config_value = path_for_config(Path(template_path))
        templates = self.data.get("inspection", {}).get("match_template", [])
        if not isinstance(templates, list) or index >= len(templates):
            raise CabfConfigError(f"inspection.match_template 缺少 {side.upper()} 配置")
        if isinstance(templates[index], dict):
            templates[index]["path"] = config_value
        else:
            templates[index] = config_value
        patched = _patch_match_template_path(self.original_text, index, config_value)
        _atomic_write_with_backup(self.path, patched)
        self.original_text = patched
        return config_value


def read_image(path: str | Path) -> np.ndarray:
    source = Path(path)
    encoded = np.fromfile(source, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise CabfConfigError(f"无法读取图片：{source}")
    return image


def write_template(path: str | Path, image: np.ndarray) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".png":
        target = target.with_suffix(".png")
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise CabfConfigError(f"模板编码失败：{target}")
    encoded.tofile(target)
    return target.resolve()


def write_reference(path: str | Path, image: np.ndarray) -> Path:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise CabfConfigError("校准基准图必须是三通道彩色图片")
    return write_template(path, image)


def resolve_backend_glue_model_path(backend_loader: Callable[[], dict[str, Any]] | None = None) -> str:
    if backend_loader is None:
        def backend_loader() -> dict[str, Any]:
            from biz.config_loader import backend_config

            return backend_config

    backend = backend_loader()
    try:
        value = str(backend["cab_f"]["glue_segment"]["path"])
    except (KeyError, TypeError) as exc:
        raise CabfConfigError("backend_config.yaml 缺少 cab_f.glue_segment.path") from exc
    if not value or value.startswith("@"):
        raise CabfConfigError(f"胶体分割模型尚未解析为本地路径：{value or '空'}")
    path = Path(value)
    if not path.is_absolute():
        path = COSMOS_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"胶体分割模型不存在：{path}")
    return str(path.resolve())


class CabfTemplateGenerator:
    """Lazy CAB-F template generator backed by the same config as production."""

    def __init__(
        self,
        model_path_loader: Callable[[], str] = resolve_backend_glue_model_path,
        extractor_factory: Callable[[str], Any] | None = None,
        matcher_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._model_path_loader = model_path_loader
        self._extractor_factory = extractor_factory
        self._matcher_factory = matcher_factory
        self._extractor = None
        self._matcher = None
        self._loaded_model_path = ""

    def _ensure_components(self, model_path: str) -> tuple[Any, Any]:
        if self._extractor_factory is None or self._matcher_factory is None:
            from algo.cab_f import CADMatcher, GlueExtractor

            extractor_factory = self._extractor_factory or GlueExtractor
            matcher_factory = self._matcher_factory or CADMatcher
        else:
            extractor_factory = self._extractor_factory
            matcher_factory = self._matcher_factory
        if self._extractor is None or self._loaded_model_path != model_path:
            self._extractor = extractor_factory(model_path)
            self._loaded_model_path = model_path
        if self._matcher is None:
            self._matcher = matcher_factory()
        return self._extractor, self._matcher

    def generate_from_path(self, image_path: str | Path) -> TemplateGenerationResult:
        image = read_image(image_path)
        return self.generate(image)

    def generate(self, image: np.ndarray) -> TemplateGenerationResult:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise CabfConfigError("初始原图必须是三通道彩色图片")
        channel_spread = np.ptp(image, axis=2)
        gray = image[:, :, 0]
        near_binary = (gray <= 10) | (gray >= 245)
        if int(channel_spread.max()) <= 1 and float(near_binary.mean()) >= 0.995:
            raise CabfConfigError(
                "当前图片是黑白二值匹配模板，不是现场原始基准图；"
                "请选择相机采集的原始彩色基准图重新生成模板"
            )
        model_path = self._model_path_loader()
        extractor, matcher = self._ensure_components(model_path)
        source_height, source_width = image.shape[:2]
        # Resize before the channel conversion so a full-resolution RGB copy
        # is not kept next to a very large CAB-F source image.
        resized_bgr = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        mask = extractor.glue_extract(resized_bgr)
        if not np.any(mask < 128):
            raise CabfConfigError(
                "胶体分割未检测到前景；请确认选择的是现场原始基准图，"
                "并检查 backend_config.yaml 中的 cab_f.glue_segment 模型"
            )
        template, half_offset = matcher.center_and_refine_template_with_offset(mask)
        template = np.ascontiguousarray(template)
        template_height, template_width = template.shape[:2]
        scale_y = source_height / max(template_height, 1)
        scale_x = source_width / max(template_width, 1)
        full_offset = (
            int(round(half_offset[0] * scale_y)),
            int(round(half_offset[1] * scale_x)),
        )
        from algo.common.cad_match_core import transform

        calibrated_image = np.ascontiguousarray(transform(image, full_offset, 0.0, border_value=0))
        return TemplateGenerationResult(
            image=template,
            model_path=model_path,
            source_size=(source_width, source_height),
            template_size=(template_width, template_height),
            calibrated_image=calibrated_image,
            calibration_offset=full_offset,
        )
