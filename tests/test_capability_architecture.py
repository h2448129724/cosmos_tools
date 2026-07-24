from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication, QWidget

from cosmos_toolbox.capabilities import ActivityStage, Capability, CapabilityCatalog
from cosmos_toolbox.project_context import ProjectContext
from cosmos_toolbox.project_session import ArtifactKind, ProjectSession
from cosmos_toolbox.task_center import TaskCenter, TaskStatus


def get_qapp():
    return QApplication.instance() or QApplication([])


def _page(_runtime, parent):
    return QWidget(parent)


def test_capability_catalog_is_the_navigation_and_search_source():
    catalog = CapabilityCatalog(
        (
            Capability("project", "项目概览", "项目状态", ActivityStage.PROJECT, page_factory=_page),
            Capability("train", "连边训练", "Sew Point Connect", ActivityStage.TRAINING, page_factory=_page),
        )
    )

    assert [item.key for item in catalog.visible()] == ["project", "train"]
    assert catalog.grouped()[1][0] == ActivityStage.TRAINING
    assert catalog.search("connect")[0].key == "train"


def test_project_session_persists_artifact_lineage(tmp_path):
    path = tmp_path / "session.json"
    session = ProjectSession(path)
    artifact = session.register_artifact(
        kind=ArtifactKind.MODEL,
        name="connector.onnx",
        path=tmp_path / "connector.onnx",
        source_capability="training.sew_point_conntect",
        source_task_id="run-1",
        inputs=("dataset-v2",),
    )

    restored = ProjectSession(path)
    assert restored.artifacts == (artifact,)
    assert restored.recent()[0].inputs == ("dataset-v2",)


class _RunManager(QObject):
    run_started = Signal(object)
    log_received = Signal(str, str, str)
    run_finished = Signal(object)


class _Record:
    run_id = "run-1"
    action_display_name = "训练"
    feature_name = "sew_point_conntect"
    artifacts_dir = "D:/runs/run-1/artifacts"
    output_dir = "D:/runs/run-1"
    status = "success"


def test_task_center_adapts_training_runs_without_exposing_run_manager_details():
    get_qapp()
    manager = _RunManager()
    center = TaskCenter()
    center.bind_training_manager(manager)

    manager.run_started.emit(_Record())
    manager.log_received.emit("run-1", "stdout", "epoch 1")
    manager.run_finished.emit(_Record())

    task = center.get("run-1")
    assert task is not None
    assert task.status == TaskStatus.SUCCESS
    assert task.logs == ["epoch 1"]
    assert task.capability_key == "training.sew_point_conntect"


def test_project_context_and_session_remain_separate_persistence_concerns(tmp_path):
    context = ProjectContext(tmp_path / "workspace.json")
    session = ProjectSession(tmp_path / "session.json")
    context.update(project_name="cabf")
    session.register_artifact(kind=ArtifactKind.DATASET, name="dataset", path=tmp_path)

    assert ProjectContext(tmp_path / "workspace.json").state.project_name == "cabf"
    assert ProjectSession(tmp_path / "session.json").artifacts[0].kind == ArtifactKind.DATASET
