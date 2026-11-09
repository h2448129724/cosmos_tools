from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

from trainer_gui import run_manager as run_manager_module
from trainer_gui.history_manager import HistoryManager
from trainer_gui.models import CondaEnvInfo, EntryConfig, FeatureAction, FeatureModule
from trainer_gui.run_manager import RunManager


def _feature_and_action(tmp_path: Path) -> tuple[FeatureModule, FeatureAction]:
    feature = FeatureModule(
        feature_name="feature",
        display_name="Feature",
        module_dir=tmp_path,
    )
    action = FeatureAction(
        action_name="train",
        display_name="Train",
        script_path=tmp_path / "train.py",
        entry=EntryConfig(type="script", value="train.py"),
    )
    return feature, action


def _command_manager(tmp_path: Path, history: HistoryManager, code: str) -> RunManager:
    manager = RunManager(tmp_path, history)
    manager._build_command = lambda feature, action, params, conda_env: (
        [sys.executable, "-c", code],
        tmp_path,
        dict(os.environ),
        sys.executable,
    )
    return manager


def _wait_until_idle(manager: RunManager, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while manager.active_runs and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not manager.active_runs


def test_run_paths_are_contained_and_unique(tmp_path: Path) -> None:
    history = HistoryManager(tmp_path)
    feature, action = _feature_and_action(tmp_path)
    feature.feature_name = "../feature"
    action.action_name = r"..\action"

    first = history.create_run_paths(feature, action, r"../project\escape")
    second = history.create_run_paths(feature, action, r"../project\escape")

    runs_root = history.runs_root.resolve()
    assert first["run_dir"].resolve().is_relative_to(runs_root)
    assert second["run_dir"].resolve().is_relative_to(runs_root)
    assert first["run_dir"] != second["run_dir"]
    assert first["project_name"].name == "_project_escape"
    assert first["run_dir"].parent == history.runs_root / "_project_escape" / "_feature" / "_action"


def test_startup_failure_closes_logs_and_finalizes_failed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    history = HistoryManager(tmp_path)
    feature, action = _feature_and_action(tmp_path)
    manager = RunManager(tmp_path, history)

    def fail_to_start(*args, **kwargs):
        raise OSError("executable unavailable")

    monkeypatch.setattr(run_manager_module.subprocess, "Popen", fail_to_start)
    with pytest.raises(OSError, match="executable unavailable"):
        manager.start_run(feature, action, "project", {})

    entries = history.load_history()
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    meta = history.load_run_meta(entries[0]["output_dir"])
    assert meta["status"] == "failed"
    assert meta["end_time"]
    assert Path(meta["stderr_log"]).read_text(encoding="utf-8").startswith("Failed to start process:")
    assert not manager.active_runs


def test_completion_waits_for_both_readers_before_closing_logs(tmp_path: Path) -> None:
    history = HistoryManager(tmp_path)
    feature, action = _feature_and_action(tmp_path)
    manager = _command_manager(
        tmp_path,
        history,
        "import sys; print('stdout complete'); print('stderr complete', file=sys.stderr)",
    )

    record = manager.start_run(feature, action, "project", {})
    _wait_until_idle(manager)

    assert Path(record.stdout_log).read_text(encoding="utf-8") == "stdout complete\n"
    assert Path(record.stderr_log).read_text(encoding="utf-8") == "stderr complete\n"
    assert history.load_run_meta(record.output_dir)["status"] == "success"


def test_start_signal_precedes_finish_for_fast_child(tmp_path: Path) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    history = HistoryManager(tmp_path)
    feature, action = _feature_and_action(tmp_path)
    manager = _command_manager(tmp_path, history, "print('done')")
    events: list[str] = []
    manager.run_started.connect(lambda record: events.append("started"))
    manager.run_finished.connect(lambda record: events.append("finished"))

    manager.start_run(feature, action, "project", {})
    deadline = time.monotonic() + 5.0
    while len(events) < 2 and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    assert events == ["started", "finished"]


def test_synchronous_start_callback_can_shutdown(tmp_path: Path) -> None:
    history = HistoryManager(tmp_path)
    feature, action = _feature_and_action(tmp_path)
    manager = _command_manager(tmp_path, history, "import time; time.sleep(30)")

    manager.run_started.connect(lambda record: manager.shutdown())
    record = manager.start_run(feature, action, "project", {})

    assert not manager.active_runs
    assert history.load_run_meta(record.output_dir)["status"] == "stopped"


def test_shutdown_cancels_process_and_uses_platform_path_separator(tmp_path: Path) -> None:
    history = HistoryManager(tmp_path)
    feature, action = _feature_and_action(tmp_path)
    manager = _command_manager(tmp_path, history, "import time; time.sleep(30)")
    record = manager.start_run(feature, action, "project", {})
    manager.shutdown()

    assert not manager.active_runs
    assert history.load_run_meta(record.output_dir)["status"] == "stopped"

    conda = CondaEnvInfo("test-env", str(tmp_path / "env"), sys.executable)
    env = RunManager._build_process_env(conda, tmp_path)
    expected = [str(tmp_path / "env")]
    if os.name == "nt":
        expected.extend((str(tmp_path / "env" / "Scripts"), str(tmp_path / "env" / "Library" / "bin")))
    else:
        expected.append(str(tmp_path / "env" / "bin"))
    assert env["PATH"].split(os.pathsep)[: len(expected)] == expected
