from __future__ import annotations

from trainer_gui import run_manager as run_manager_module
from trainer_gui.history_manager import HistoryManager
from trainer_gui.run_manager import RunManager


class FakeProcess:
    def __init__(self, pid: int):
        self.pid = pid
        self.terminate_calls = 0

    def poll(self):
        return None

    def terminate(self):
        self.terminate_calls += 1


def test_run_manager_shutdown_stops_every_active_run(tmp_path, monkeypatch):
    manager = RunManager(tmp_path, HistoryManager(tmp_path))
    first = FakeProcess(101)
    second = FakeProcess(202)
    manager.active_runs.update({"first": first, "second": second})
    monkeypatch.setattr(manager, "_terminate_process_tree", lambda process: process.terminate())

    manager.shutdown()

    assert manager.stopped_runs == {"first", "second"}
    assert first.terminate_calls == 1
    assert second.terminate_calls == 1


def test_windows_stop_run_terminates_the_process_tree(tmp_path, monkeypatch):
    manager = RunManager(tmp_path, HistoryManager(tmp_path))
    process = FakeProcess(303)
    manager.active_runs["run"] = process
    commands = []

    class Result:
        returncode = 0

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return Result()

    monkeypatch.setattr(run_manager_module.os, "name", "nt")
    monkeypatch.setattr(run_manager_module.subprocess, "run", fake_run)

    manager.stop_run("run")

    assert commands[0][0] == ["taskkill", "/PID", "303", "/T", "/F"]
    assert process.terminate_calls == 0


def test_windows_stop_run_falls_back_when_taskkill_fails(tmp_path, monkeypatch):
    manager = RunManager(tmp_path, HistoryManager(tmp_path))
    process = FakeProcess(404)
    manager.active_runs["run"] = process

    class Result:
        returncode = 1

    monkeypatch.setattr(run_manager_module.os, "name", "nt")
    monkeypatch.setattr(run_manager_module.subprocess, "run", lambda *_args, **_kwargs: Result())

    manager.stop_run("run")

    assert process.terminate_calls == 1
