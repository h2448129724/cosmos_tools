from __future__ import annotations

import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from apps.cosmos_pipeline.runner import (
    InputCase,
    InputSlot,
    PipelineDescriptor,
    PipelineRequest,
    backend_key_for_project,
    describe_config,
    load_input_manifest,
    validate_request,
)
from apps.cosmos_pipeline import runner
from cosmos_toolbox.cosmos_pipeline_activity import CosmosPipelineActivity, build_pipeline_command
from cosmos_toolbox.app import ToolboxWindow
from cosmos_toolbox.default_capabilities import build_capability_catalog
from cosmos_toolbox.project_context import ProjectContext
from cosmos_toolbox.project_session import ProjectSession
from cosmos_toolbox.task_center import TaskCenter, TaskStatus


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
    assert "--allow-ng" in command


def test_pipeline_command_accepts_selected_python(tmp_path: Path):
    product, backend, manifest, _top, _bottom = _pipeline_files(tmp_path)
    request = PipelineRequest(product, backend, manifest, tmp_path / "out")

    command = build_pipeline_command(request, r"C:\conda\envs\cabf\python.exe")

    assert command[0] == r"C:\conda\envs\cabf\python.exe"
    assert "--fail-on-ng" in command


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
    assert page.fail_on_ng_check.isChecked()


def test_catalog_exposes_generic_cosmos_pipeline():
    capability = build_capability_catalog((), include_project_capabilities=True).get("cosmos.pipeline_test")

    assert capability.title == "完整流程测试"
    assert capability.page_factory is not None


def test_validate_request_checks_backend_node(tmp_path: Path):
    product, backend, manifest, _top, _bottom = _pipeline_files(tmp_path)
    backend.write_text("cab: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cab_f"):
        validate_request(PipelineRequest(product, backend, manifest, tmp_path / "out"))


def test_configure_cosmos_uses_custom_backend_on_first_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    product, backend, _manifest, _top, _bottom = _pipeline_files(tmp_path)
    backend.write_text("cab_f:\n  marker: custom-backend\n", encoding="utf-8")
    descriptor = describe_config(product)
    request = PipelineRequest(product, backend, tmp_path / "inputs.yaml", tmp_path / "out")
    from biz import config_loader

    monkeypatch.setattr(config_loader, "_backend_config", None)
    runner._configure_cosmos(request, descriptor)

    assert config_loader._backend_config == {"cab_f": {"marker": "custom-backend"}}


@pytest.mark.parametrize('path_key', ['path', 'point_model_path', 'connect_model_path', '14_14_path'])
def test_configure_cosmos_rejects_unresolved_model_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_key: str
):
    product, backend, _manifest, _top, _bottom = _pipeline_files(tmp_path)
    backend.write_text(
        f"cab_f:\n  detector:\n    {path_key}: '@cab_f/missing.onnx#stable'\n",
        encoding="utf-8",
    )
    descriptor = describe_config(product)
    request = PipelineRequest(product, backend, tmp_path / "inputs.yaml", tmp_path / "out")
    from biz import config_loader

    monkeypatch.setattr(config_loader, "_backend_config", None)
    monkeypatch.setattr(config_loader, "_resolve_registry_paths", lambda data: data)

    with pytest.raises(ValueError, match="未解析"):
        runner._configure_cosmos(request, descriptor)


def test_pipeline_default_returns_nonzero_for_business_ng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    descriptor = PipelineDescriptor("CAB-F", "demo", "cab_f", "local", (InputSlot("top"),))
    case = InputCase("one", {"top": (tmp_path / "top.png",)})
    monkeypatch.setattr(runner, "validate_request", lambda _request: (descriptor, [case]))
    monkeypatch.setattr(runner, "_configure_cosmos", lambda _request, _descriptor: None)

    def fake_run_case(*_args, **_kwargs):
        return {"passed": False}

    monkeypatch.setattr(runner, "_run_case", fake_run_case)
    request = PipelineRequest(Path("config.yaml"), Path("backend.yaml"), Path("manifest.yaml"), tmp_path / "out")

    assert runner._execute(request) == 2
    allow_request = PipelineRequest(
        Path("config.yaml"), Path("backend.yaml"), Path("manifest.yaml"), tmp_path / "out-allow", fail_on_ng=False
    )
    assert runner._execute(allow_request) == 0


def test_run_case_releases_previous_slot_images_before_loading_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class ImageToken:
        pass

    descriptor = PipelineDescriptor(
        "CAB-F", "demo", "cab_f", "local", (InputSlot("top"), InputSlot("bottom"))
    )
    case = InputCase(
        "one",
        {
            "top": (tmp_path / "top.png",),
            "bottom": (tmp_path / "bottom.png",),
        },
    )
    request = PipelineRequest(
        Path("config.yaml"),
        Path("backend.yaml"),
        Path("manifest.yaml"),
        tmp_path / "out",
        save_annotated_images=False,
    )
    first_image_ref = None
    load_count = 0

    def fake_load_image(_path, _color_space):
        nonlocal first_image_ref, load_count
        load_count += 1
        if load_count == 2:
            assert first_image_ref is not None
            assert first_image_ref() is None
        image = ImageToken()
        if load_count == 1:
            first_image_ref = weakref.ref(image)
        return image

    monkeypatch.setattr(runner, "_load_image", fake_load_image)
    monkeypatch.setattr(runner, "_detect_local", lambda _descriptor, _slot, images, _index: (images[0], {}))
    monkeypatch.setattr(
        runner,
        "_evaluate_and_draw",
        lambda _descriptor, _result, image, _index, **_kwargs: (True, {}, [], image),
    )
    monkeypatch.setattr(runner, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_event", lambda *_args, **_kwargs: None)

    report = runner._run_case(case, descriptor, request, tmp_path / "out")

    assert load_count == 2
    assert report["passed"] is True


def test_skip_annotated_images_does_not_call_drawing(monkeypatch: pytest.MonkeyPatch):
    from algo import cab_f_drawing
    from biz.config_loader import config

    descriptor = PipelineDescriptor("CAB-F", "demo", "cab_f", "local", (InputSlot("bottom"),))
    monkeypatch.setattr(config, "inspection", {"ok_checkers": [[]], "conf": {}})
    monkeypatch.setattr(
        cab_f_drawing,
        "draw",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("drawing must be skipped")),
    )

    passed, checks, messages, annotated = runner._evaluate_and_draw(
        descriptor,
        {"face": "bottom"},
        object(),
        0,
        draw_output=False,
    )

    assert passed is True
    assert checks == {}
    assert messages == []
    assert annotated is None


def test_cli_defaults_to_failing_on_business_ng_and_allows_compatibility_flag():
    parser = runner.build_parser()
    base = ["--config", "config.yaml", "--input-manifest", "manifest.yaml"]

    assert runner.request_from_args(parser.parse_args(base)).fail_on_ng is True
    assert runner.request_from_args(parser.parse_args([*base, "--allow-ng"])).fail_on_ng is False


def test_pipeline_ui_records_business_ng_as_distinct_status(tmp_path: Path):
    runtime = _Runtime(tmp_path)
    page = CosmosPipelineActivity(runtime)
    page._task_id = 'pipeline-ng'
    page._task_finished = False
    runtime.task_center.start(
        page._task_id,
        'pipeline',
        page.capability_key,
        str(tmp_path / 'output'),
    )

    page._on_finished(2, None)

    assert runtime.task_center.get('pipeline-ng').status == TaskStatus.BUSINESS_NG
    assert page.status_label.text() == '业务 NG'


def test_business_ng_output_is_registered_as_artifact(tmp_path: Path):
    output = tmp_path / 'output'
    output.mkdir()
    session = ProjectSession(tmp_path / 'session.json')
    shell = SimpleNamespace(project_session=session)
    task = SimpleNamespace(
        status=TaskStatus.BUSINESS_NG,
        output_path=str(output),
        task_id='pipeline-ng',
        title='pipeline',
        capability_key='cosmos.pipeline_test',
    )

    ToolboxWindow._on_task_finished(shell, task)

    assert len(session.artifacts) == 1
    assert session.artifacts[0].source_task_id == 'pipeline-ng'
