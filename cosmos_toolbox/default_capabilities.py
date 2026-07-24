from __future__ import annotations

from collections.abc import Sequence

from .activities import ArtifactsActivity, ProjectOverviewPage, SewPointConnectActivity, TaskCenterActivity
from .capabilities import ActivityStage, Capability, CapabilityCatalog
from .paths import TOOLBOX_ROOT
from .workspaces import WorkspaceAdapter


def _call(method_name: str):
    def activate(workspace, _runtime) -> None:
        method = getattr(workspace, method_name, None)
        if callable(method):
            method()

    return activate


def _open_training(feature_name: str, action_name: str = "train"):
    def activate(workspace, _runtime) -> None:
        opener = getattr(workspace, "open_feature", None)
        if callable(opener):
            opener(feature_name, action_name)
            return
        fallback = getattr(workspace, "_select_feature_action", None)
        if callable(fallback):
            fallback(feature_name, action_name)

    return activate


def build_capability_catalog(
    workspace_adapters: Sequence[WorkspaceAdapter],
    *,
    include_project_capabilities: bool,
) -> CapabilityCatalog:
    catalog = CapabilityCatalog(
        (
            Capability(
                key="overview",
                title="项目概览",
                description="查看项目就绪状态和从数据到模型的工作路径。",
                stage=ActivityStage.PROJECT,
                order=0,
                page_factory=lambda runtime, parent: ProjectOverviewPage(runtime, parent),
            ),
        )
    )

    stage_by_workspace = {
        "image": ActivityStage.DATA,
        "labeling": ActivityStage.ANNOTATION,
        "training": ActivityStage.TRAINING,
    }
    for adapter in workspace_adapters:
        catalog.register(
            Capability(
                key=adapter.key,
                title=adapter.title,
                description=adapter.description,
                stage=stage_by_workspace.get(adapter.key, ActivityStage.CUSTOM),
                order=900,
                workspace_key=adapter.key,
                visible=(adapter.key not in {"labeling", "training"}) or not include_project_capabilities,
            )
        )

    if not include_project_capabilities:
        return catalog

    generic_labeling_tools = (
        ("data.keyword_split", "关键字划分", "根据文件名关键字将图片归类到子目录。", "keyword"),
        ("data.batch_crop", "批量裁剪", "按统一区域批量裁剪图片。", "batch_crop"),
        ("data.tile_crop", "自动分块", "将大图自动切分为规则小图。", "tile_crop"),
        ("data.roi_editor", "ROI 编辑", "在图片上配置和复用 ROI。", "roi"),
        ("data.inner_mask", "内侧 Mask", "生成和检查内侧区域 Mask。", "inner_mask"),
        ("data.image_filter", "图像筛选", "按图像条件批量筛选素材。", "image_filter"),
        ("annotation.visualize", "标注可视化", "浏览图片与标注叠加结果。", "label_visualization"),
    )
    for order, (key, title, description, tool_name) in enumerate(generic_labeling_tools, start=10):
        catalog.register(
            Capability(
                key=key,
                title=title,
                description=description,
                stage=ActivityStage.ANNOTATION if key.startswith("annotation.") else ActivityStage.DATA,
                order=order,
                page_factory=lambda runtime, parent, name=tool_name: _create_labeling_tool_activity(
                    runtime, parent, name
                ),
            )
        )

    catalog.register(
        Capability(
            key="cabf.config_studio",
            title="CAB-F 基准图与模板生成",
            description="从 TOP/BOTTOM 初始原图生成全尺寸校准基准图与匹配模板，并在校准坐标系中编辑 ROI。",
            stage=ActivityStage.DATA,
            order=0,
            page_factory=lambda runtime, parent: _create_cabf_config_activity(runtime, parent),
            keywords=("CAB-F", "ROI", "模板", "配置", "基准图"),
        )
    )
    catalog.register(
        Capability(
            key="cosmos.pipeline_test",
            title="完整流程测试",
            description="跳过生产页面，按配置运行项目检测、规则判定、产品汇总和结果输出。",
            stage=ActivityStage.EVALUATION,
            order=10,
            page_factory=lambda runtime, parent: _create_cosmos_pipeline_activity(runtime, parent),
            keywords=("Cosmos", "CAB-F", "Pipeline", "完整流程", "测试", "配置"),
        )
    )
    catalog.register(
        Capability(
            key="cabf.sew_point_connect",
            title="Sew Point Connect",
            description="连接数据准备、CAB-F 点边复核、模型训练和产物。",
            stage=ActivityStage.PROJECT,
            order=20,
            page_factory=lambda runtime, parent: SewPointConnectActivity(runtime, parent),
            keywords=("CAB-F", "连边", "点边", "流程"),
        )
    )
    catalog.register(
        Capability(
            key="cabf.workflow",
            title="CAB-F 数据与标注流程",
            description="筛选、预测、点边修正、校验和训练数据导出。",
            stage=ActivityStage.ANNOTATION,
            order=10,
            workspace_key="labeling",
            activate=_call("_show_cabf_workflow_dialog"),
            keywords=("CAB-F", "9步流程", "标注"),
        )
    )
    catalog.register(
        Capability(
            key="cabf.graph_editor",
            title="点边标注",
            description="修正缝纫点和点之间的连边关系。",
            stage=ActivityStage.ANNOTATION,
            order=20,
            page_factory=lambda runtime, parent: _create_cabf_graph_editor_activity(runtime, parent),
        )
    )
    catalog.register(
        Capability(
            key="data.sample_review",
            title="样本审阅",
            description="浏览图片和可选标注，保留有效样本或移除异常样本。",
            stage=ActivityStage.DATA,
            order=30,
            page_factory=lambda runtime, parent: _create_dataset_review_activity(runtime, parent),
            keywords=("图片", "筛选", "复核", "标注", "数据清理"),
        )
    )
    # Compatibility route for saved navigation state and older launchers.  It
    # is intentionally hidden; CAB-F composes the generic review capability.
    catalog.register(
        Capability(
            key="cabf.point_filter",
            title="数据筛选（兼容入口）",
            description="旧 CAB-F 数据筛选入口。",
            stage=ActivityStage.ANNOTATION,
            order=30,
            workspace_key="labeling",
            activate=_call("_show_point_filter_dialog"),
            visible=False,
        )
    )
    catalog.register(
        Capability(
            key="cabf.dataset_export",
            title="数据集校验与导出",
            description="校验母数据并导出训练数据集。",
            stage=ActivityStage.ANNOTATION,
            order=40,
            page_factory=lambda runtime, parent: _create_cabf_dataset_export_activity(runtime, parent),
        )
    )

    try:
        from trainer_gui.feature_scanner import FeatureScanner

        features = FeatureScanner(TOOLBOX_ROOT).scan()
    except Exception:
        features = []
    for order, feature in enumerate((item for item in features if item.enabled), start=10):
        default_action = feature.actions[0].action_name if feature.actions else "train"
        catalog.register(
            Capability(
                key=f"training.{feature.feature_name}",
                title=feature.display_name,
                description=feature.description or f"配置并运行 {feature.display_name}。",
                stage=ActivityStage.TRAINING,
                order=order,
                workspace_key="training",
                activate=_open_training(feature.feature_name, default_action),
                keywords=(feature.feature_name, "训练", "推理", "导出"),
            )
        )

    catalog.register(
        Capability(
            key="project.artifacts",
            title="项目产物",
            description="查看训练、推理和导出产生的模型与文件。",
            stage=ActivityStage.ARTIFACTS,
            order=10,
            page_factory=lambda runtime, parent: ArtifactsActivity(runtime, parent),
        )
    )
    catalog.register(
        Capability(
            key="project.tasks",
            title="任务与日志",
            description="集中查看训练和处理任务的状态、进度与日志。",
            stage=ActivityStage.ARTIFACTS,
            order=20,
            page_factory=lambda runtime, parent: TaskCenterActivity(runtime, parent),
        )
    )
    return catalog


def _create_dataset_review_activity(runtime, parent):
    from apps.labeling_ui.app.data_review import DatasetReviewPage

    page = DatasetReviewPage(parent)
    state = runtime.project_context.state
    page.configure_paths(
        image_dir=state.image_dir or state.dataset_root,
        label_dir=state.annotation_dir,
        auto_load=False,
    )
    page.reviewCompleted.connect(
        lambda result: runtime.project_context.update(image_dir=str(result.active_image_dir))
    )
    return page


def _create_labeling_tool_activity(runtime, parent, tool_name: str):
    from apps.labeling_ui.app.tools.batch_crop_page import BatchCropPage
    from apps.labeling_ui.app.tools.image_filter_page import ImageFilterPage
    from apps.labeling_ui.app.tools.inner_mask_page import InnerMaskPage
    from apps.labeling_ui.app.tools.keyword_split_page import KeywordSplitPage
    from apps.labeling_ui.app.tools.label_visualization_page import LabelVisualizationPage
    from apps.labeling_ui.app.tools.roi_editor_page import RoiConfigEditorPage
    from apps.labeling_ui.app.tools.tile_crop_page import AutoTileCropPage

    page_types = {
        "keyword": KeywordSplitPage,
        "batch_crop": BatchCropPage,
        "tile_crop": AutoTileCropPage,
        "roi": RoiConfigEditorPage,
        "inner_mask": InnerMaskPage,
        "image_filter": ImageFilterPage,
        "label_visualization": LabelVisualizationPage,
    }
    page = page_types[tool_name](runtime.window, parent)
    page.setProperty("ownsPageHeader", True)
    return page


def _as_native_page(page):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialog

    if isinstance(page, QDialog):
        page.setModal(False)
        page.setWindowFlag(Qt.WindowType.Dialog, False)
        page.setWindowFlag(Qt.WindowType.Window, False)
        page.setWindowFlag(Qt.WindowType.Widget, True)
    page.setProperty("ownsPageHeader", True)
    return page


def _create_cabf_graph_editor_activity(runtime, parent):
    from apps.labeling_ui.app.graph_annotation_dialog import StitchGraphEditorDialog

    page = StitchGraphEditorDialog(parent)
    state = runtime.project_context.state
    image_dir = state.image_dir or state.dataset_root
    page.configure_paths(
        image_dir=image_dir,
        label_dir=state.annotation_dir,
        output_dir=state.annotation_dir,
        auto_open=bool(image_dir and state.annotation_dir),
    )
    return _as_native_page(page)


def _create_cabf_dataset_export_activity(runtime, parent):
    from apps.labeling_ui.app.dataset_export_dialog import CabfDatasetToolDialog

    page = CabfDatasetToolDialog(parent)
    state = runtime.project_context.state
    image_dir = state.image_dir or state.dataset_root
    if image_dir:
        page.edit_image_dir.setText(image_dir)
    if state.annotation_dir:
        page.edit_annotation_dir.setText(state.annotation_dir)
    if state.output_root:
        from pathlib import Path

        root = Path(state.output_root)
        page.edit_model_a_output_dir.setText(str(root / "model_a"))
        page.edit_model_b_output_dir.setText(str(root / "model_b"))
    return _as_native_page(page)


def _create_cabf_config_activity(runtime, parent):
    from .cabf_activity import CabfConfigActivity

    return CabfConfigActivity(runtime, parent)


def _create_cosmos_pipeline_activity(runtime, parent):
    from .cosmos_pipeline_activity import CosmosPipelineActivity

    return CosmosPipelineActivity(runtime, parent)
