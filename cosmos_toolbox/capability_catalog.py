"""Pure capability specifications and catalog projection rules.

The imperative shell binds the returned plans to Qt page factories and
workspace activators.  This module owns navigation ordering, grouping,
search, default visibility, and training-action selection without importing
or probing those adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class ActivityStage(StrEnum):
    PROJECT = "project"
    DATA = "data"
    ANNOTATION = "annotation"
    TRAINING = "training"
    EVALUATION = "evaluation"
    ARTIFACTS = "artifacts"
    CUSTOM = "custom"


STAGE_TITLES: dict[ActivityStage, str] = {
    ActivityStage.PROJECT: "项目",
    ActivityStage.DATA: "数据准备",
    ActivityStage.ANNOTATION: "标注与复核",
    ActivityStage.TRAINING: "训练与推理",
    ActivityStage.EVALUATION: "评估",
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
    OPEN_TRAINING_FEATURE = "open_training_feature"


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


@dataclass(frozen=True, slots=True)
class TrainingFeatureFact:
    feature_name: str
    display_name: str
    description: str = ""
    enabled: bool = True
    action_names: tuple[str, ...] = ()


def select_default_training_action(action_names: Iterable[str]) -> str:
    """Preserve scanner order while providing a deterministic empty fallback."""

    return next((name for name in action_names if name.strip()), "train")


def plan_default_capabilities(
    workspaces: Iterable[WorkspaceCapabilityFact],
    *,
    include_project_capabilities: bool,
    training_features: Iterable[TrainingFeatureFact] = (),
) -> tuple[PlannedCapability, ...]:
    """Project the default capability catalog from explicit external facts.

    A discovered training feature replaces the broad training workspace in
    navigation.  If discovery fails or yields no enabled feature, the native
    training workspace remains visible as a safe fallback.
    """

    workspace_facts = tuple(workspaces)
    enabled_training = tuple(feature for feature in training_features if feature.enabled)
    planned: list[PlannedCapability] = [
        _page(
            "overview",
            "项目概览",
            "查看项目就绪状态和从数据到模型的工作路径。",
            ActivityStage.PROJECT,
            page_factory_key="overview",
            order=0,
        )
    ]

    stage_by_workspace = {
        "image": ActivityStage.DATA,
        "labeling": ActivityStage.ANNOTATION,
        "training": ActivityStage.TRAINING,
    }
    for workspace in workspace_facts:
        replaced = include_project_capabilities and (
            workspace.key == "labeling"
            or (workspace.key == "training" and bool(enabled_training))
        )
        planned.append(
            _workspace(
                workspace.key,
                workspace.title,
                workspace.description,
                stage_by_workspace.get(workspace.key, ActivityStage.CUSTOM),
                workspace_key=workspace.key,
                order=900,
                visible=not replaced,
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
            "data.tile_crop",
            "自动分块",
            "将大图自动切分为规则小图。",
            ActivityStage.DATA,
            "labeling_tool:tile_crop",
        ),
        (
            "data.roi_editor",
            "ROI 编辑",
            "在图片上配置和复用 ROI。",
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
            "data.image_filter",
            "图像筛选",
            "按图像条件批量筛选素材。",
            ActivityStage.DATA,
            "labeling_tool:image_filter",
        ),
        (
            "annotation.visualize",
            "标注可视化",
            "浏览图片与标注叠加结果。",
            ActivityStage.ANNOTATION,
            "labeling_tool:label_visualization",
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
                "cabf.field_dataset",
                "CAB-F 现场数据集生成",
                "独立勾选模型，从现场原图生成裁片、伪标签和复核数据，支持暂停续跑。",
                ActivityStage.DATA,
                page_factory_key="cabf_field_dataset",
                order=1,
                keywords=("CAB-F", "数据集", "现场", "批量", "ONNX", "裁片"),
            ),
            _page(
                "cabf.config_studio",
                "CAB-F 基准图与模板生成",
                "从 TOP/BOTTOM 初始原图生成全尺寸校准基准图与匹配模板，并在校准坐标系中编辑 ROI。",
                ActivityStage.DATA,
                page_factory_key="cabf_config",
                order=0,
                keywords=("CAB-F", "ROI", "模板", "配置", "基准图"),
            ),
            _page(
                "cosmos.pipeline_test",
                "完整流程测试",
                "跳过生产页面，按配置运行项目检测、规则判定、产品汇总和结果输出。",
                ActivityStage.EVALUATION,
                page_factory_key="cosmos_pipeline",
                order=10,
                keywords=("Cosmos", "CAB-F", "Pipeline", "完整流程", "测试", "配置"),
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
                "cabf.sew_point_connect",
                "Sew Point Connect",
                "连接数据准备、CAB-F 点边复核、模型训练和产物。",
                ActivityStage.PROJECT,
                page_factory_key="sew_point_connect",
                order=20,
                keywords=("CAB-F", "连边", "点边", "流程"),
            ),
            _workspace(
                "cabf.workflow",
                "CAB-F 数据与标注流程",
                "筛选、预测、点边修正、校验和训练数据导出。",
                ActivityStage.ANNOTATION,
                workspace_key="labeling",
                order=10,
                activation=ActivationPlan(
                    ActivationKind.CALL_WORKSPACE_METHOD,
                    "_show_cabf_workflow_dialog",
                ),
                keywords=("CAB-F", "9步流程", "标注"),
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
            _workspace(
                "cabf.point_filter",
                "数据筛选（兼容入口）",
                "旧 CAB-F 数据筛选入口。",
                ActivityStage.ANNOTATION,
                workspace_key="labeling",
                order=30,
                activation=ActivationPlan(
                    ActivationKind.CALL_WORKSPACE_METHOD,
                    "_show_point_filter_dialog",
                ),
                visible=False,
            ),
            _page(
                "cabf.dataset_export",
                "数据集校验与导出",
                "校验母数据并导出训练数据集。",
                ActivityStage.ANNOTATION,
                page_factory_key="cabf_dataset_export",
                order=40,
            ),
        )
    )

    for order, feature in enumerate(enabled_training, start=10):
        action = select_default_training_action(feature.action_names)
        planned.append(
            _workspace(
                f"training.{feature.feature_name}",
                feature.display_name,
                feature.description or f"配置并运行 {feature.display_name}。",
                ActivityStage.TRAINING,
                workspace_key="training",
                order=order,
                activation=ActivationPlan(
                    ActivationKind.OPEN_TRAINING_FEATURE,
                    feature.feature_name,
                    action,
                ),
                keywords=(feature.feature_name, "训练", "推理", "导出"),
            )
        )

    planned.extend(
        (
            _page(
                "project.artifacts",
                "项目产物",
                "查看训练、推理和导出产生的模型与文件。",
                ActivityStage.ARTIFACTS,
                page_factory_key="artifacts",
                order=10,
            ),
            _page(
                "project.tasks",
                "任务与日志",
                "集中查看训练和处理任务的状态、进度与日志。",
                ActivityStage.ARTIFACTS,
                page_factory_key="tasks",
                order=20,
            ),
        )
    )
    return _validated(planned)


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
