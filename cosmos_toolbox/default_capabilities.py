from __future__ import annotations

import logging
from collections.abc import Sequence

from .activities import ArtifactsActivity, ProjectOverviewPage, SewPointConnectActivity, TaskCenterActivity
from .capabilities import Capability, CapabilityCatalog
from .capability_catalog import (
    ActivationKind,
    ActivationPlan,
    PlannedCapability,
    TrainingFeatureFact,
    WorkspaceCapabilityFact,
    plan_default_capabilities,
)
from .paths import TOOLBOX_ROOT
from .workspaces import WorkspaceAdapter


logger = logging.getLogger(__name__)


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
    training_features: Sequence[TrainingFeatureFact] | None = None,
) -> CapabilityCatalog:
    """Bind a pure capability plan to concrete Qt/workspace adapters."""

    feature_facts = (
        tuple(training_features)
        if training_features is not None
        else _scan_training_feature_facts()
    )
    workspace_facts = tuple(
        WorkspaceCapabilityFact(adapter.key, adapter.title, adapter.description)
        for adapter in workspace_adapters
    )
    planned = plan_default_capabilities(
        workspace_facts,
        include_project_capabilities=include_project_capabilities,
        training_features=feature_facts,
    )
    return CapabilityCatalog(_bind_planned_capability(item) for item in planned)


def _scan_training_feature_facts() -> tuple[TrainingFeatureFact, ...]:
    """Imperative adapter for module discovery; failures keep workspace fallback."""

    try:
        from trainer_gui.feature_scanner import FeatureScanner

        features = FeatureScanner(TOOLBOX_ROOT).scan()
    except Exception:
        logger.warning(
            "Training feature discovery failed; keeping the native training workspace visible.",
            exc_info=True,
        )
        return ()
    return tuple(
        TrainingFeatureFact(
            feature_name=feature.feature_name,
            display_name=feature.display_name,
            description=feature.description,
            enabled=feature.enabled,
            action_names=tuple(action.action_name for action in feature.actions),
        )
        for feature in features
    )


def _bind_planned_capability(planned: PlannedCapability) -> Capability:
    spec = planned.spec
    return Capability(
        key=spec.key,
        title=spec.title,
        description=spec.description,
        stage=spec.stage,
        order=spec.order,
        page_factory=(
            _page_factory_for(planned.page_factory_key)
            if planned.page_factory_key is not None
            else None
        ),
        workspace_key=planned.workspace_key,
        activate=_activation_for(planned.activation),
        keywords=spec.keywords,
        visible=spec.visible,
    )


def _page_factory_for(factory_key: str):
    if factory_key.startswith("labeling_tool:"):
        tool_name = factory_key.partition(":")[2]
        return lambda runtime, parent: _create_labeling_tool_activity(
            runtime,
            parent,
            tool_name,
        )
    factories = {
        "overview": lambda runtime, parent: ProjectOverviewPage(runtime, parent),
        "cabf_config": _create_cabf_config_activity,
        "cabf_field_dataset": _create_field_dataset_activity,
        "database": _create_database_activity,
        "cosmos_pipeline": _create_cosmos_pipeline_activity,
        "sew_point_connect": lambda runtime, parent: SewPointConnectActivity(
            runtime,
            parent,
        ),
        "cabf_graph_editor": _create_cabf_graph_editor_activity,
        "dataset_review": _create_dataset_review_activity,
        "cabf_dataset_export": _create_cabf_dataset_export_activity,
        "artifacts": lambda runtime, parent: ArtifactsActivity(runtime, parent),
        "tasks": lambda runtime, parent: TaskCenterActivity(runtime, parent),
    }
    try:
        return factories[factory_key]
    except KeyError as exc:
        raise KeyError(f"Unknown page factory key: {factory_key}") from exc


def _activation_for(plan: ActivationPlan | None):
    if plan is None:
        return None
    if plan.kind is ActivationKind.CALL_WORKSPACE_METHOD:
        return _call(plan.target)
    if plan.kind is ActivationKind.OPEN_TRAINING_FEATURE:
        return _open_training(plan.target, plan.action)
    raise ValueError(f"Unsupported capability activation: {plan.kind}")


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


def _create_database_activity(runtime, parent):
    from .database_ui import DatabasePage

    page = DatabasePage(parent)
    page.protect_window_close(runtime.window)
    return page


def _create_field_dataset_activity(runtime, parent):
    from .field_dataset_ui import FieldDatasetPage

    page = FieldDatasetPage(parent)
    state = runtime.project_context.state
    page.source.setText(state.image_dir or state.dataset_root or "")
    if state.output_root:
        page.output.setText(state.output_root)
    page.protect_window_close(runtime.window)
    return page
