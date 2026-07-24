from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from apps.cosmos_pipeline.runner import (
    PipelineRequest,
    backend_key_for_project,
    describe_config,
    load_input_manifest,
    validate_request,
)
from apps.cosmos_pipeline import runner
from cosmos_toolbox.cosmos_pipeline_activity import CosmosPipelineActivity, build_pipeline_command
from cosmos_toolbox.default_capabilities import build_capability_catalog
from cosmos_toolbox.project_context import ProjectContext
from cosmos_toolbox.project_session import ProjectSession
from cosmos_toolbox.task_center import TaskCenter


TOOLBOX_ROOT = Path(__file__).resolve().parents[1]
COSMOS_ROOT = TOOLBOX_ROOT.parents[1]


@pytest.mark.parametrize(
    ("relative", "project", "slots", "engine"),
    [
        ("conf/CAB/CAB_D01.yaml", "CAB", ("combined",), "service"),
        ("conf/CAB-F/D01-L.local.yaml", "CAB-F", ("top", "bottom"), "local"),
        ("conf/DAB-SF/800B.yaml", "DAB-SF", ("back", "front"), "service"),
        ("conf/OS-DAB/800B-NB.local.yaml", "OS-DAB", ("front", "back"), "local"),
    ],
)
def test_descriptor_recognizes_cosmos_projects(relative: str, project: str, slots: tuple[str, ...], engine: str):
    descriptor = describe_config(COSMOS_ROOT / relative)

    assert descriptor.project == project
    assert tuple(slot.name for slot in descriptor.slots) == slots
    assert descriptor.engine_default == engine


def test_backend_key_normalizes_all_dab_variants():
    assert backend_key_for_project("CAB-F") == "cab_f"
    assert backend_key_for_project("DAB-GF-D7") == "dab"
    assert backend_key_for_project("OS-DAB") == "os_dab"


def test_rgb_input_adapter_allows_cosmos_line_scan_sized_images(tmp_path: Path):
    path = tmp_path / "large-for-test.png"
    Image.new("RGB", (20, 20), (1, 2, 3)).save(path)
    original_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 10
    try:
        loaded = runner._load_image(path, "rgb")
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit

    assert loaded.shape == (20, 20, 3)


def test_pipeline_executes_from_cosmos_root_and_restores_caller_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    seen: list[Path] = []
    monkeypatch.setattr(runner, "_execute", lambda _request: seen.append(Path.cwd()) or 0)

    assert runner.execute(object()) == 0
    assert seen == [COSMOS_ROOT]
    assert Path.cwd() == caller


def _pipeline_files(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    product = tmp_path / "product.yaml"
    product.write_text(
        "inspection:\n  project: CAB-F\n  product: demo\n  conf:\n    top: {}\n    bottom: {}\n",
        encoding="utf-8",
    )
    backend = tmp_path / "backend.yaml"
    backend.write_text("cab_f: {}\n", encoding="utf-8")
    top = tmp_path / "top.png"
    bottom = tmp_path / "bottom.png"
    top.write_bytes(b"image")
    bottom.write_bytes(b"image")
    manifest = tmp_path / "inputs.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {"cases": [{"id": "one", "inputs": {"top": [str(top)], "bottom": [str(bottom)]}}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return product, backend, manifest, top, bottom


def test_manifest_validates_slot_names_and_file_counts(tmp_path: Path):
    product, _backend, manifest, _top, _bottom = _pipeline_files(tmp_path)
    descriptor = describe_config(product)

    cases = load_input_manifest(manifest, descriptor)

    assert cases[0].case_id == "one"
    assert set(cases[0].inputs) == {"top", "bottom"}


def test_headless_cli_dry_run_does_not_create_output(tmp_path: Path):
    product, backend, manifest, _top, _bottom = _pipeline_files(tmp_path)
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLBOX_ROOT / "scripts" / "cosmos_pipeline_test.py"),
            "--config",
            str(product),
            "--backend-config",
            str(backend),
            "--input-manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--dry-run",
        ],
        cwd=TOOLBOX_ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"type": "describe"' in result.stdout
    assert "未加载模型" in result.stdout
    assert not output.exists()


def test_pipeline_command_is_shell_free_and_complete(tmp_path: Path):
    product, backend, manifest, _top, _bottom = _pipeline_files(tmp_path)
    request = PipelineRequest(product, backend, manifest, tmp_path / "out", "local", True, False, False)

    command = build_pipeline_command(request)

    assert command[:2] == [sys.executable, "-u"]
    assert command[command.index("--engine") + 1] == "local"
    assert command[command.index("--input-manifest") + 1] == str(manifest)
    assert "--dry-run" in command


def test_pipeline_command_accepts_selected_python(tmp_path: Path):
    product, backend, manifest, _top, _bottom = _pipeline_files(tmp_path)
    request = PipelineRequest(product, backend, manifest, tmp_path / "out")

    command = build_pipeline_command(request, r"C:\conda\envs\cabf\python.exe")

    assert command[0] == r"C:\conda\envs\cabf\python.exe"


class _Runtime:
    def __init__(self, tmp_path: Path):
        self.task_center = TaskCenter()
        self.project_context = ProjectContext(tmp_path / "workspace.json")
        self.project_session = ProjectSession(tmp_path / "session.json")

    def open_capability(self, _key: str) -> None:
        pass


def test_pipeline_page_uses_separate_persisted_context(tmp_path: Path):
    product, backend, manifest, _top, _bottom = _pipeline_files(tmp_path)
    runtime = _Runtime(tmp_path)
    runtime.project_context.update(
        pipeline_config_path=product,
        pipeline_backend_config_path=backend,
        pipeline_input_manifest_path=manifest,
    )

    page = CosmosPipelineActivity(runtime)

    assert page.config_edit.text() == str(product.resolve())
    assert page.backend_edit.text() == str(backend.resolve())
    assert page.manifest_edit.text() == str(manifest.resolve())
    assert ProjectContext(tmp_path / "workspace.json").state.pipeline_config_path == str(product.resolve())


def test_catalog_exposes_generic_cosmos_pipeline():
    capability = build_capability_catalog((), include_project_capabilities=True).get("cosmos.pipeline_test")

    assert capability.title == "完整流程测试"
    assert capability.page_factory is not None


def test_validate_request_checks_backend_node(tmp_path: Path):
    product, backend, manifest, _top, _bottom = _pipeline_files(tmp_path)
    backend.write_text("cab: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cab_f"):
        validate_request(PipelineRequest(product, backend, manifest, tmp_path / "out"))
