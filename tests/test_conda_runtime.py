from __future__ import annotations

from shared.conda_runtime import CondaEnvManager


def test_conda_payload_maps_names_and_active_environment(monkeypatch):
    monkeypatch.setattr("shared.conda_runtime.os.name", "nt")
    environments = CondaEnvManager._parse_envs(
        {
            "root_prefix": r"C:\Miniconda3",
            "active_prefix": r"C:\Miniconda3\envs\onnx-gpu",
            "envs": [r"C:\Miniconda3", r"C:\Miniconda3\envs\onnx-gpu"],
        }
    )

    assert {item.name for item in environments} == {"base", "onnx-gpu"}
    selected = next(item for item in environments if item.name == "onnx-gpu")
    assert selected.is_active
    assert selected.python_executable.endswith("python.exe")


def test_selected_environment_updates_child_process_environment(monkeypatch):
    manager = CondaEnvManager()
    monkeypatch.setattr(
        manager,
        "list_envs",
        lambda: CondaEnvManager._parse_envs(
            {
                "root_prefix": r"C:\Miniconda3",
                "active_prefix": "",
                "envs": [r"C:\Miniconda3\envs\train"],
            }
        ),
    )

    environment = manager.process_environment("train", {"PATH": "system-path"})

    assert environment["CONDA_DEFAULT_ENV"] == "train"
    assert environment["CONDA_PREFIX"] == r"C:\Miniconda3\envs\train"
    assert "system-path" in environment["PATH"]
