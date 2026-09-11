"""Pure planning and validation for complete-inspection requests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class InputSlot:
    name: str
    min_files: int = 1
    max_files: int = 1


@dataclass(frozen=True, slots=True)
class PipelineDescriptor:
    project: str
    product: str
    backend_key: str
    engine_default: str
    slots: tuple[InputSlot, ...]


@dataclass(frozen=True, slots=True)
class InputCasePlan:
    case_id: str
    inputs: Mapping[str, tuple[str, ...]]
    color_space: str = "rgb"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inputs",
            {str(name): tuple(str(value) for value in values) for name, values in self.inputs.items()},
        )


class ProjectFamily(str, Enum):
    CAB = "cab"
    CAB_F = "cab_f"
    DAB = "dab"
    OS_DAB = "os_dab"


@dataclass(frozen=True, slots=True)
class ProjectExecutionRoute:
    family: ProjectFamily
    local_supported: bool
    service_mode: str | None


def route_project(project: str) -> ProjectExecutionRoute:
    """Classify one Cosmos project once for all execution adapters."""

    if project == "CAB":
        return ProjectExecutionRoute(ProjectFamily.CAB, True, "images")
    if project == "CAB-F":
        return ProjectExecutionRoute(ProjectFamily.CAB_F, True, None)
    if project.startswith("DAB"):
        return ProjectExecutionRoute(ProjectFamily.DAB, True, "image")
    if project == "OS-DAB":
        return ProjectExecutionRoute(ProjectFamily.OS_DAB, True, None)
    raise ValueError(f"不支持的 Cosmos 项目：{project}")


def backend_key_for_project(project: str) -> str:
    normalized = "_".join(str(project).lower().split("-"))
    return "dab" if normalized.startswith("dab") else normalized


def describe_product_config(data: Mapping[str, Any]) -> PipelineDescriptor:
    inspection = data.get("inspection")
    if not isinstance(inspection, Mapping):
        raise ValueError("产品配置缺少 inspection")
    project = str(inspection.get("project") or "").strip()
    product = str(inspection.get("product") or "").strip()
    conf = inspection.get("conf")
    if not project or not isinstance(conf, Mapping):
        raise ValueError("产品配置缺少 inspection.project 或 inspection.conf")

    if project == "CAB":
        loader = data.get("image_loader") or {}
        nested = loader.get("loaders") if isinstance(loader, Mapping) else None
        file_count = max(1, len(nested)) if isinstance(nested, list) else 1
        slots = (InputSlot("combined", file_count, file_count),)
        engine = "service"
    else:
        preferred = {"top", "bottom"} if project == "CAB-F" else {"front", "back"}
        face_names = tuple(
            str(name)
            for name, value in conf.items()
            if name in preferred and isinstance(value, Mapping)
        )
        if not face_names:
            face_names = tuple(str(name) for name, value in conf.items() if isinstance(value, Mapping))
        if not face_names:
            raise ValueError("inspection.conf 没有输入侧别")
        slots = tuple(InputSlot(name) for name in face_names)
        engine = "service" if project.startswith("DAB") else "local"
    return PipelineDescriptor(project, product, backend_key_for_project(project), engine, slots)


def plan_input_manifest(
    data: Mapping[str, Any],
    descriptor: PipelineDescriptor,
) -> tuple[InputCasePlan, ...]:
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("输入清单必须包含非空 cases 列表")
    slot_map = {slot.name: slot for slot in descriptor.slots}
    cases: list[InputCasePlan] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("inputs"), Mapping):
            raise ValueError(f"cases[{index - 1}] 缺少 inputs")
        case_id = str(raw.get("id") or f"case-{index:03d}").strip()
        if case_id in seen:
            raise ValueError(f"输入清单 case id 重复：{case_id}")
        seen.add(case_id)
        raw_inputs = raw["inputs"]
        unknown = set(raw_inputs) - set(slot_map)
        if unknown:
            raise ValueError(f"{case_id} 包含未知输入槽：{', '.join(sorted(unknown))}")
        planned: dict[str, tuple[str, ...]] = {}
        for name, slot in slot_map.items():
            values = raw_inputs.get(name)
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                raise ValueError(f"{case_id}.{name} 必须是文件列表")
            if not slot.min_files <= len(values) <= slot.max_files:
                raise ValueError(f"{case_id}.{name} 需要 {slot.min_files} 个文件，当前 {len(values)} 个")
            planned[name] = tuple(str(value) for value in values)
        color_space = str(raw.get("color_space") or data.get("color_space") or "rgb").lower()
        if color_space not in {"rgb", "bgr", "gray"}:
            raise ValueError(f"{case_id}.color_space 仅支持 rgb/bgr/gray")
        cases.append(InputCasePlan(case_id, planned, color_space))
    return tuple(cases)


def select_engine(requested: str, descriptor: PipelineDescriptor) -> str:
    if requested not in {"auto", "local", "service"}:
        raise ValueError(f"未知执行引擎：{requested}")
    return descriptor.engine_default if requested == "auto" else requested


__all__ = [
    "InputCasePlan",
    "InputSlot",
    "PipelineDescriptor",
    "ProjectExecutionRoute",
    "ProjectFamily",
    "backend_key_for_project",
    "describe_product_config",
    "plan_input_manifest",
    "route_project",
    "select_engine",
]
