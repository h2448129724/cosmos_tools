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
from cabf.roi import (
    CabfRoiDocument,
    RoiCardinalityError,
    RoiField,
    RoiIssue,
    collect_roi_fields as collect_roi_fields,
    normalize_rects as normalize_rects,
    _find_yaml_key_line as _find_yaml_key_line,
    _line_indent as _line_indent,
)

from .cabf_calibration import (
    CalibrationTransformPlan,
    ForegroundFacts,
    SourceImageFacts,
    decide_foreground,
    decide_source_image,
    plan_calibration_transform,
    plan_half_scale,
)
from .paths import COSMOS_ROOT
from .training.cab_f_project import project_entry


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


@dataclass(slots=True)
class TemplateGenerationResult:
    image: np.ndarray
    model_path: str
    source_size: tuple[int, int]
    template_size: tuple[int, int]
    calibrated_image: np.ndarray | None = None
    calibration_offset: tuple[int, int] = (0, 0)
    calibration_plan: CalibrationTransformPlan | None = None


_LIST_ITEM_RE = re.compile(r"^(?P<indent>\s*)-(?P<rest>.*)$")
_PATH_VALUE_RE = re.compile(r"^(?P<prefix>\s*(?:-\s*)?path\s*:\s*)(?P<value>.*?)(?P<comment>\s+#.*)?$")


def _quote_like(old_value: str, new_value: str) -> str:
    old_value = old_value.strip()
    if len(old_value) >= 2 and old_value[0] == old_value[-1] and old_value[0] in {"'", '"'}:
        quote = old_value[0]
        if quote == "'":
            # YAML single-quoted scalars escape a quote by doubling it; a
            # backslash escape would silently produce invalid/changed paths.
            escaped = new_value.replace("'", "''")
        else:
            # Double-quoted YAML scalars treat backslashes as escapes.
            escaped = (
                new_value.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
        return f"{quote}{escaped}{quote}"
    escaped = (
        new_value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _split_yaml_inline_comment(value: str) -> tuple[str, str]:
    """Split an inline YAML value from its comment without parsing/reformatting."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip(), value[index:].rstrip()
    return value.rstrip(), ""


def _inline_sequence_item_spans(value: str) -> list[tuple[int, int]]:
    """Return item spans inside an inline YAML sequence.

    This deliberately only identifies top-level commas; quoted strings and
    nested flow values are kept intact so a path containing a comma remains a
    single item.
    """
    if not (value.startswith("[") and value.endswith("]")):
        return []
    inner_start, inner_end = 1, len(value) - 1
    spans: list[tuple[int, int]] = []
    start = inner_start
    quote: str | None = None
    escaped = False
    depth = 0
    for index in range(inner_start, inner_end):
        char = value[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "[{(":
            depth += 1
        elif char in "]})":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            spans.append((start, index))
            start = index + 1
    if value[inner_start:inner_end].strip() or spans:
        spans.append((start, inner_end))
    return spans


def _patch_inline_match_template_path(line: str, side_index: int, new_value: str) -> str | None:
    """Patch a flow-style ``match_template: [top, bottom]`` line in place."""
    key_match = re.search(r"\bmatch_template\s*:\s*", line)
    if key_match is None:
        return None
    prefix_end = key_match.end()
    value, comment = _split_yaml_inline_comment(line[prefix_end:])
    if not (value.lstrip().startswith("[") and value.rstrip().endswith("]")):
        return None
    leading = value[: len(value) - len(value.lstrip())]
    flow = value.lstrip()
    trailing = value[len(value.rstrip()) :]
    spans = _inline_sequence_item_spans(flow)
    if side_index < 0 or side_index >= len(spans):
        raise CabfConfigError(f"inspection.match_template 缺少索引 {side_index} 的模板路径")
    start, end = spans[side_index]
    item = flow[start:end]
    item_left = item[: len(item) - len(item.lstrip())]
    item_right = item[len(item.rstrip()) :]
    scalar = item.strip()
    if scalar.startswith("{"):
        path_match = re.search(r"\bpath\s*:\s*", scalar)
        if path_match is None:
            raise CabfConfigError(f"inspection.match_template 缺少索引 {side_index} 的模板路径")
        value_start = path_match.end()
        quote: str | None = None
        escaped = False
        value_end = len(scalar)
        for index in range(value_start, len(scalar)):
            char = scalar[index]
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
            elif char in {",", "}"}:
                value_end = index
                break
        old_scalar = scalar[value_start:value_end].strip()
        raw_value = scalar[value_start:value_end]
        value_left = raw_value[: len(raw_value) - len(raw_value.lstrip())]
        value_right = raw_value[len(raw_value.rstrip()) :]
        scalar = (
            scalar[:value_start]
            + value_left
            + _quote_like(old_scalar, new_value)
            + value_right
            + scalar[value_end:]
        )
        flow = flow[:start] + item_left + scalar + item_right + flow[end:]
        return line[:prefix_end] + leading + flow + trailing + (f" {comment}" if comment else "")
    flow = flow[:start] + item_left + _quote_like(scalar, new_value) + item_right + flow[end:]
    return line[:prefix_end] + leading + flow + trailing + (f" {comment}" if comment else "")


def _patch_match_template_path(text: str, side_index: int, new_value: str) -> str:
    lines = text.splitlines(keepends=True)
    header_index = _find_yaml_key_line(lines, ("inspection", "match_template"))
    if header_index < 0:
        raise CabfConfigError("配置中没有找到 inspection.match_template")
    header_line = lines[header_index]
    newline = "\n" if header_line.endswith("\n") else ""
    header_body = header_line[:-1] if newline else header_line
    inline = _patch_inline_match_template_path(header_body, side_index, new_value)
    if inline is not None:
        lines[header_index] = inline + newline
        return "".join(lines)
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
    """YAML persistence shell around the pure CAB-F ROI document."""

    SIDE_INDEX = {"top": 0, "bottom": 1}

    def __init__(self, path: Path, data: dict[str, Any], original_text: str) -> None:
        self.path = path.resolve()
        self.data = data
        self.original_text = original_text
        self.roi_document = CabfRoiDocument(data)

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
        return self.roi_document.fields(side)

    def rects(self, field: RoiField) -> list[tuple[int, int, int, int]]:
        return self.roi_document.rects(field)

    def set_rects(self, field: RoiField, rects: list[tuple[int, int, int, int]]) -> None:
        try:
            self.roi_document.set_rects(field, rects)
        except RoiCardinalityError as exc:
            raise CabfConfigError(f"{field.display_name} 是单 ROI 字段") from exc

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
        return self.roi_document.validate(
            side,
            calibrated_reference_size=image_size,
        )

    def save_rois(self) -> None:
        patched = self.roi_document.patch_text(self.original_text)
        _atomic_write_with_backup(self.path, patched)
        self.original_text = patched

    def update_template_path(self, side: str, template_path: str | Path) -> str:
        config_value = self.validate_template_path_update(side, template_path)
        index = self.SIDE_INDEX[side]
        templates = self.data.get("inspection", {}).get("match_template", [])
        patched = _patch_match_template_path(self.original_text, index, config_value)
        _atomic_write_with_backup(self.path, patched)
        # Mutate the decoded document only after the textual patch succeeded;
        # callers can safely retry after a malformed/unsupported YAML layout.
        if isinstance(templates[index], dict):
            templates[index]["path"] = config_value
        else:
            templates[index] = config_value
        self.original_text = patched
        return config_value

    def validate_template_path_update(self, side: str, template_path: str | Path) -> str:
        """Validate and render a template-path update without writing files."""
        if side not in self.SIDE_INDEX:
            raise CabfConfigError(f"不支持的侧别：{side}")
        index = self.SIDE_INDEX[side]
        config_value = path_for_config(Path(template_path))
        templates = self.data.get("inspection", {}).get("match_template", [])
        if not isinstance(templates, list) or index >= len(templates):
            raise CabfConfigError(f"inspection.match_template 缺少 {side.upper()} 配置")
        _patch_match_template_path(self.original_text, index, config_value)
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
            cab_f = project_entry()

            extractor_factory = self._extractor_factory or cab_f.GlueExtractor
            matcher_factory = self._matcher_factory or cab_f.CADMatcher
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
        shape = tuple(int(value) for value in getattr(image, "shape", ()))
        if len(shape) == 3 and shape[0] > 0 and shape[1] > 0 and shape[2] == 3:
            channel_spread = np.ptp(image, axis=2)
            gray = image[:, :, 0]
            near_binary = (gray <= 10) | (gray >= 245)
            source_decision = decide_source_image(
                SourceImageFacts(
                    shape=shape,
                    channel_spread_max=float(channel_spread.max()),
                    near_binary_ratio=float(near_binary.mean()),
                )
            )
        else:
            source_decision = decide_source_image(
                SourceImageFacts(shape=shape, channel_spread_max=0.0, near_binary_ratio=0.0)
            )
        if not source_decision.accepted and source_decision.rejection == "binary_template":
            raise CabfConfigError(
                "当前图片是黑白二值匹配模板，不是现场原始基准图；"
                "请选择相机采集的原始彩色基准图重新生成模板"
            )
        if not source_decision.accepted:
            raise CabfConfigError("初始原图必须是三通道彩色图片")
        model_path = self._model_path_loader()
        extractor, matcher = self._ensure_components(model_path)
        source_height, source_width = image.shape[:2]
        # Resize before the channel conversion so a full-resolution RGB copy
        # is not kept next to a very large CAB-F source image.
        half_scale = plan_half_scale((source_width, source_height))
        # Keep OpenCV's fx/fy path (including its behaviour for degenerate
        # one-pixel inputs); the pure plan exposes the dimensions it computes
        # for ordinary odd/even images without changing this shell effect.
        resized_bgr = cv2.resize(
            image,
            (0, 0),
            fx=half_scale.scale,
            fy=half_scale.scale,
            interpolation=cv2.INTER_AREA,
        )
        mask = extractor.glue_extract(resized_bgr)
        foreground_decision = decide_foreground(
            ForegroundFacts(
                foreground_pixels=int(np.count_nonzero(mask < 128)),
                total_pixels=int(mask.size),
            )
        )
        if not foreground_decision.accepted:
            raise CabfConfigError(
                "胶体分割未检测到前景；请确认选择的是现场原始基准图，"
                "并检查 backend_config.yaml 中的 cab_f.glue_segment 模型"
            )
        template, half_offset = matcher.center_and_refine_template_with_offset(mask)
        template = np.ascontiguousarray(template)
        template_height, template_width = template.shape[:2]
        calibration_plan = plan_calibration_transform(
            (source_width, source_height),
            (template_width, template_height),
            half_offset,
        )
        from algo.common.cad_match_core import transform

        calibrated_image = np.ascontiguousarray(
            transform(
                image,
                calibration_plan.full_offset,
                calibration_plan.angle_degrees,
                border_value=calibration_plan.border_value,
            )
        )
        return TemplateGenerationResult(
            image=template,
            model_path=model_path,
            source_size=(source_width, source_height),
            template_size=(template_width, template_height),
            calibrated_image=calibrated_image,
            calibration_offset=calibration_plan.full_offset,
            calibration_plan=calibration_plan,
        )
