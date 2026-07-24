from __future__ import annotations

from cosmos_toolbox.project_context import ProjectContext, ProjectState


def test_project_context_persists_and_emits_changes(tmp_path):
    state_path = tmp_path / "workspace.json"
    context = ProjectContext(state_path)
    changes = []
    context.changed.connect(changes.append)

    context.update(
        project_name="cabf-demo",
        model_path=tmp_path / "model.onnx",
        cabf_config_path=tmp_path / "product.yaml",
        cabf_source_top=tmp_path / "top-source.png",
        cabf_source_bottom=tmp_path / "bottom-source.png",
        cabf_reference_top=tmp_path / "top-reference.png",
        cabf_reference_bottom=tmp_path / "bottom-reference.png",
        cabf_template_top=tmp_path / "top-template.png",
        cabf_template_bottom=tmp_path / "bottom-template.png",
    )

    assert changes[-1].project_name == "cabf-demo"
    restored = ProjectContext(state_path)
    assert restored.state.project_name == "cabf-demo"
    assert restored.state.model_path == str((tmp_path / "model.onnx").resolve())
    assert restored.state.cabf_config_path == str((tmp_path / "product.yaml").resolve())
    assert restored.state.cabf_source_top == str((tmp_path / "top-source.png").resolve())
    assert restored.state.cabf_source_bottom == str((tmp_path / "bottom-source.png").resolve())
    assert restored.state.cabf_reference_top == str((tmp_path / "top-reference.png").resolve())
    assert restored.state.cabf_reference_bottom == str((tmp_path / "bottom-reference.png").resolve())
    assert restored.state.cabf_template_top == str((tmp_path / "top-template.png").resolve())
    assert restored.state.cabf_template_bottom == str((tmp_path / "bottom-template.png").resolve())
    assert restored.state.runtime_profile == "onnx-gpu"


def test_project_context_migrates_v1_reference_paths_to_source_paths(tmp_path):
    state_path = tmp_path / "workspace.json"
    top = tmp_path / "top.png"
    bottom = tmp_path / "bottom.png"
    state_path.write_text(
        (
            '{"schema_version": 1, '
            f'"cabf_reference_top": "{top.as_posix()}", '
            f'"cabf_reference_bottom": "{bottom.as_posix()}"'
            "}"
        ),
        encoding="utf-8",
    )

    state = ProjectContext(state_path).state

    assert state.cabf_source_top == str((tmp_path / "top.png").resolve())
    assert state.cabf_source_bottom == str((tmp_path / "bottom.png").resolve())
    assert state.cabf_reference_top == ""
    assert state.cabf_reference_bottom == ""


def test_dataset_root_derives_common_project_directories(tmp_path):
    dataset = tmp_path / "sample-project"
    images = dataset / "images"
    annotations = dataset / "annotations"
    outputs = dataset / "outputs"
    for directory in (images, annotations, outputs):
        directory.mkdir(parents=True, exist_ok=True)

    context = ProjectContext(tmp_path / "workspace.json")
    state = context.set_dataset_root(dataset)

    assert state.project_name == "sample-project"
    assert state.dataset_root == str(dataset.resolve())
    assert state.image_dir == str(images.resolve())
    assert state.annotation_dir == str(annotations.resolve())
    assert state.output_root == str(outputs.resolve())


def test_invalid_or_future_state_falls_back_to_empty_context(tmp_path):
    state_path = tmp_path / "workspace.json"
    state_path.write_text('{"schema_version": 99, "project_name": "future"}', encoding="utf-8")

    assert ProjectContext(state_path).state == ProjectState()
