"""Pure capability specifications and catalog projection rules.

The imperative shell binds the returned plans to Qt page factories and
workspace activators.  This module owns navigation ordering, grouping,
search, default visibility, and training-action selection without importing
or probing those adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Iterable


class ActivityStage(StrEnum):
    PROJECT = "project"
    CABF = "cabf"
    ANNOTATION = "annotation"
    FIELD = "field"
    DATA = "data"
    ARTIFACTS = "artifacts"
    CUSTOM = "custom"


STAGE_TITLES: dict[ActivityStage, str] = {
    ActivityStage.PROJECT: "项目",
    ActivityStage.CABF: "CAB-F",
    ActivityStage.ANNOTATION: "标注与复核",
    ActivityStage.FIELD: "现场数据",
    ActivityStage.DATA: "图片处理",
    ActivityStage.ARTIFACTS: "产物",
    ActivityStage.CUSTOM: "定制功能",
}
_STAGE_ORDER = {stage: index for index, stage in enumerate(ActivityStage)}


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """Inert facts used by navigation, search, and presentation."""

    key: str
    title: str
    description: str
    stage: ActivityStage
    order: int = 100
    keywords: tuple[str, ...] = ()
    visible: bool = True

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("Capability key cannot be empty")
        if not isinstance(self.stage, ActivityStage):
            raise TypeError("Capability stage must be an ActivityStage")


@dataclass(frozen=True, slots=True)
class CapabilityProjection:
    """Immutable, ordered source of truth for capability presentation."""

    items: tuple[CapabilitySpec, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for item in self.items:
            if item.key in seen:
                raise KeyError(f"Capability already registered: {item.key}")
            seen.add(item.key)

    def registered(self, item: CapabilitySpec) -> CapabilityProjection:
        if self.find(item.key) is not None:
            raise KeyError(f"Capability already registered: {item.key}")
        return CapabilityProjection((*self.items, item))

    def get(self, key: str) -> CapabilitySpec:
        item = self.find(key)
        if item is None:
            raise KeyError(f"Unknown capability: {key}")
        return item

    def find(self, key: str) -> CapabilitySpec | None:
        return next((item for item in self.items if item.key == key), None)

    def visible(self) -> tuple[CapabilitySpec, ...]:
        return tuple(sorted((item for item in self.items if item.visible), key=_sort_key))

    def grouped(self) -> tuple[tuple[ActivityStage, tuple[CapabilitySpec, ...]], ...]:
        visible = self.visible()
        return tuple(
            (stage, items)
            for stage in ActivityStage
            if (items := tuple(item for item in visible if item.stage == stage))
        )

    def search(self, query: str) -> tuple[CapabilitySpec, ...]:
        needle = query.strip().casefold()
        if not needle:
            return self.visible()
        return tuple(
            item
            for item in self.visible()
            if needle in " ".join((item.title, item.description, *item.keywords)).casefold()
        )


class ActivationKind(StrEnum):
    CALL_WORKSPACE_METHOD = "call_workspace_method"


@dataclass(frozen=True, slots=True)
class ActivationPlan:
    kind: ActivationKind
    target: str
    action: str = ""

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ValueError("Activation target cannot be empty")


@dataclass(frozen=True, slots=True)
class PlannedCapability:
    """A specification plus an inert key describing its shell adapter."""

    spec: CapabilitySpec
    page_factory_key: str | None = None
    workspace_key: str | None = None
    activation: ActivationPlan | None = None

    def __post_init__(self) -> None:
        if (self.page_factory_key is None) == (self.workspace_key is None):
            raise ValueError(
                "Planned capability must provide exactly one page factory key or workspace key"
            )
        if self.page_factory_key is not None and not self.page_factory_key.strip():
            raise ValueError("Page factory key cannot be empty")
        if self.workspace_key is not None and not self.workspace_key.strip():
            raise ValueError("Workspace key cannot be empty")
        if self.activation is not None and self.workspace_key is None:
            raise ValueError("Only workspace capabilities can declare activation")


@dataclass(frozen=True, slots=True)
class WorkspaceCapabilityFact:
    key: str
    title: str
    description: str


def plan_default_capabilities(
    workspaces: Iterable[WorkspaceCapabilityFact],
    *,
    include_project_capabilities: bool,
) -> tuple[PlannedCapability, ...]:
    """Project the default capability catalog from explicit workspace facts."""

    workspace_facts = tuple(workspaces)
    planned: list[PlannedCapability] = [
        _page(
            "overview",
            "项目概览",
            "查看项目就绪状态和常用工作路径。",
            ActivityStage.PROJECT,
            page_factory_key="overview",
            order=0,
        )
    ]

    stage_by_workspace = {
        "image": ActivityStage.DATA,
    }
    for workspace in workspace_facts:
        planned.append(
            _workspace(
                workspace.key,
                workspace.title,
                workspace.description,
                stage_by_workspace.get(workspace.key, ActivityStage.CUSTOM),
                workspace_key=workspace.key,
                order=900,
            )
        )

    if not include_project_capabilities:
        return _validated(planned)

    generic_labeling_tools = (
        (
            "data.keyword_split",
            "关键字划分",
            "根据文件名关键字将图片归类到子目录。",
            ActivityStage.DATA,
            "labeling_tool:keyword",
        ),
        (
            "data.batch_crop",
            "批量裁剪",
            "按统一区域批量裁剪图片。",
            ActivityStage.DATA,
            "labeling_tool:batch_crop",
        ),
        (
            "data.single_crop",
            "逐张裁剪",
            "按固定尺寸逐张调整 ROI 位置并裁剪当前图片。",
            ActivityStage.DATA,
            "labeling_tool:single_crop",
        ),
        (
            "data.tile_crop",
            "自动分块",
            "将大图自动切分为规则小图。",
            ActivityStage.DATA,
            "labeling_tool:tile_crop",
        ),
        (
            "data.roi_editor",
            "YAML ROI 编辑（通用）",
            "在样本图上编辑其他项目 YAML 的 ROI；CAB-F 配置会自动转到“CAB-F ROI 配置”。",
            ActivityStage.DATA,
            "labeling_tool:roi",
        ),
        (
            "data.inner_mask",
            "内侧 Mask",
            "生成和检查内侧区域 Mask。",
            ActivityStage.DATA,
            "labeling_tool:inner_mask",
        ),
        (
            "annotation.visualize",
            "标注可视化",
            "浏览图片与标注叠加结果。",
            ActivityStage.ANNOTATION,
            "labeling_tool:label_visualization",
        ),
        (
            "annotation.point_mark",
            "点位标注导出",
            "在图片上滚轮缩放并按顺序标点，导出二维坐标列表。",
            ActivityStage.ANNOTATION,
            "labeling_tool:point_mark",
        ),
    )
    for order, (key, title, description, stage, factory_key) in enumerate(
        generic_labeling_tools,
        start=10,
    ):
        planned.append(
            _page(
                key,
                title,
                description,
                stage,
                page_factory_key=factory_key,
                order=order,
            )
        )

    planned.extend(
        (
            _page(
                "images.png_compress",
                "图片压缩 · PNG / JPG",
                "PNG 无损原地压缩或转 JPG 有损压缩，校验后可选删除源 PNG。",
                ActivityStage.DATA,
                page_factory_key="png_compress",
                order=3,
                keywords=("PNG", "JPG", "JPEG", "压缩", "存储", "无损", "有损", "原地"),
            ),
            _page(
                "data.ng_extract",
                "NG 原图提取",
                "从数据库或导出的 Excel 筛选 NG 产品记录，检查后复制原图；也可把数据库导出为 Excel。",
                ActivityStage.DATA,
                page_factory_key="ng_extract",
                order=2,
                keywords=("数据库", "SQLite", "Excel", "xlsx", "NG", "原图", "复制", "导出", "日志"),
            ),
            _page(
                "cabf.config_studio",
                "CAB-F ROI 配置",
                "加载产品 YAML 与 TOP/BOTTOM 校准基准图，在校准坐标系中查看和编辑 ROI。",
                ActivityStage.DATA,
                page_factory_key="cabf_config",
                order=0,
                keywords=("CAB-F", "ROI", "YAML", "配置", "基准图"),
            ),
            _page(
                "data.database",
                "数据库浏览（只读）",
                "查看 SQLite 表数据与字段结构，支持筛选、排序和分页。",
                ActivityStage.DATA,
                page_factory_key="database",
                order=40,
                keywords=("数据库", "SQLite", "只读", "检测记录"),
            ),
            _page(
                "cabf.workflow",
                "CAB-F 数据与标注流程",
                "数据集准备、筛选、点边标注、校验与数据集导出。",
                ActivityStage.ANNOTATION,
                page_factory_key="cabf_workflow",
                order=10,
                keywords=("CAB-F", "流程", "标注", "校验", "导出"),
            ),
            _page(
                "cabf.graph_editor",
                "点边标注",
                "修正缝纫点和点之间的连边关系。",
                ActivityStage.ANNOTATION,
                page_factory_key="cabf_graph_editor",
                order=20,
            ),
            _page(
                "data.sample_review",
                "样本审阅",
                "浏览图片和可选标注，保留有效样本或移除异常样本。",
                ActivityStage.DATA,
                page_factory_key="dataset_review",
                order=30,
                keywords=("图片", "筛选", "复核", "标注", "数据清理"),
            ),
        )
    )

    planned.extend(
        (
            _page(
                "project.artifacts",
                "项目产物",
                "查看处理与导出任务产生的文件。",
                ActivityStage.ARTIFACTS,
                page_factory_key="artifacts",
                order=10,
            ),
            _page(
                "project.tasks",
                "任务与日志",
                "集中查看处理任务的状态、进度与日志。",
                ActivityStage.ARTIFACTS,
                page_factory_key="tasks",
                order=20,
            ),
        )
    )
    return _validated(_place(planned))


# One table decides which navigation group each capability lives in and its
# position there, so related entries stay together as the catalog grows.
_NAV_PLACEMENT: dict[str, tuple[ActivityStage, int]] = {
    "cabf.workflow": (ActivityStage.CABF, 1),
    "cabf.graph_editor": (ActivityStage.CABF, 2),
    "cabf.config_studio": (ActivityStage.CABF, 3),
    "data.sample_review": (ActivityStage.ANNOTATION, 1),
    "annotation.visualize": (ActivityStage.ANNOTATION, 2),
    "annotation.point_mark": (ActivityStage.ANNOTATION, 3),
    "data.ng_extract": (ActivityStage.FIELD, 1),
    "data.database": (ActivityStage.FIELD, 2),
    "image": (ActivityStage.DATA, 1),
    "images.png_compress": (ActivityStage.DATA, 2),
    "data.batch_crop": (ActivityStage.DATA, 3),
    "data.single_crop": (ActivityStage.DATA, 4),
    "data.tile_crop": (ActivityStage.DATA, 5),
    "data.keyword_split": (ActivityStage.DATA, 6),
    "data.inner_mask": (ActivityStage.DATA, 7),
    "data.roi_editor": (ActivityStage.DATA, 8),
}


def _place(planned: Iterable[PlannedCapability]) -> list[PlannedCapability]:
    placed = []
    for item in planned:
        placement = _NAV_PLACEMENT.get(item.spec.key)
        if placement is not None:
            stage, order = placement
            item = replace(item, spec=replace(item.spec, stage=stage, order=order))
        placed.append(item)
    return placed


def _page(
    key: str,
    title: str,
    description: str,
    stage: ActivityStage,
    *,
    page_factory_key: str,
    order: int = 100,
    keywords: tuple[str, ...] = (),
    visible: bool = True,
) -> PlannedCapability:
    return PlannedCapability(
        CapabilitySpec(key, title, description, stage, order, keywords, visible),
        page_factory_key=page_factory_key,
    )


def _workspace(
    key: str,
    title: str,
    description: str,
    stage: ActivityStage,
    *,
    workspace_key: str,
    order: int = 100,
    activation: ActivationPlan | None = None,
    keywords: tuple[str, ...] = (),
    visible: bool = True,
) -> PlannedCapability:
    return PlannedCapability(
        CapabilitySpec(key, title, description, stage, order, keywords, visible),
        workspace_key=workspace_key,
        activation=activation,
    )


def _validated(planned: Iterable[PlannedCapability]) -> tuple[PlannedCapability, ...]:
    result = tuple(planned)
    CapabilityProjection(tuple(item.spec for item in result))
    return result


def _sort_key(item: CapabilitySpec) -> tuple[int, int, str]:
    return (_STAGE_ORDER[item.stage], item.order, item.title.casefold())
