from __future__ import annotations

import json
from pathlib import Path

from apps.cabf_flow import config_model, flow

REPO_ROOT = Path(__file__).resolve().parents[1]
CABF_FLOW_SCRIPT = REPO_ROOT / "scripts" / "cabf_flow.py"


def test_cabf_flow_script_uses_shared_bootstrap():
    source = CABF_FLOW_SCRIPT.read_text(encoding="utf-8")

    assert "ensure_import_paths" in source
    assert "sys.path.insert" not in source


def test_apply_defaults_fills_missing_keys():
    partial = {"dataset_root": "D:/data/cabf"}
    result = config_model.apply_defaults(partial)
    assert result["dataset_root"] == "D:/data/cabf"
    assert result["train_model_modules_root"].endswith("cosmos_toolbox\\modules")
    assert result["weights"]["sew_point_onnx"] == ""
    assert result["predict"]["edge_postprocess_preset"] == "balanced"


def test_apply_defaults_ignores_empty_string_overrides():
    result = config_model.apply_defaults({"repo_root": "", "train_model_modules_root": ""})
    assert result["repo_root"].endswith("cosmos_toolbox")
    assert result["train_model_modules_root"].endswith("cosmos_toolbox\\modules")


def test_load_and_nested_config_helpers(tmp_path: Path):
    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps({"weights": {"sew_point_onnx": "model.onnx"}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cfg = config_model.load_config(config_path)
    assert config_model.get_nested(cfg, "weights.sew_point_onnx") == "model.onnx"
    config_model.set_nested(cfg, "outputs.sew_point_train_out", "runs/sew_point")
    assert cfg["outputs"]["sew_point_train_out"] == "runs/sew_point"


def test_init_config_writes_template(tmp_path: Path):
    output = tmp_path / "default_paths.json"
    code = flow.cmd_init_config({}, type("Args", (), {"output": str(output), "force": False})())
    assert code == 0
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["predict"]["point_threshold"] == "0.30"


def test_doctor_and_show_config_output(capsys):
    cfg = config_model.apply_defaults({})
    flow.cmd_show_config(cfg, object())
    printed = capsys.readouterr().out
    assert '"train_model_modules_root"' in printed

    flow.cmd_doctor(cfg, object())
    printed = capsys.readouterr().out
    assert "train_model_modules_root" in printed


def test_pipeline_dry_run(tmp_path: Path, capsys):
    flow.LOG_DIR = tmp_path / "logs"
    cfg = config_model.apply_defaults(
        {
            "master_images_dir": "D:/data/master/images",
            "master_annotations_dir": "D:/data/master/annotations",
            "point_predictions_dir": "D:/data/predictions/points",
            "edge_predictions_dir": "D:/data/predictions/edges",
            "model_a_export_root": "D:/data/model_a_export",
            "model_b_export_root": "D:/data/model_b_export",
            "outputs": {
                "sew_point_train_out": "D:/data/runs/sew_point",
                "sew_point_conntect_train_out": "D:/data/runs/sew_point_conntect",
            },
        }
    )
    args = type(
        "Args",
        (),
        {
            "image_dir": "",
            "point_output_dir": "",
            "point_model": "",
            "point_threshold": 0.3,
            "point_distance_threshold": 0.0,
            "edge_annotation_dir": "",
            "edge_output_dir": "",
            "edge_model": "",
            "postprocess_preset": "balanced",
            "no_compare_gt": False,
            "validate_annotation_dir": "",
            "export_annotation_dir": "",
            "report_path": "",
            "show_samples": False,
            "model_a_output": "",
            "model_b_output": "",
            "include_train": True,
            "model_a_images": "",
            "model_a_annotations": "",
            "model_a_out": "",
            "model_b_images": "",
            "model_b_annotations": "",
            "model_b_out": "",
            "dry_run": True,
        },
    )()
    code = flow.cmd_pipeline(cfg, args)
    assert code == 0
    printed = capsys.readouterr().out
    assert "predict-points" in printed
    assert "predict-edges" in printed
    assert "validate" in printed
    assert "export" in printed
    assert "train" in printed


def test_pipeline_hands_predicted_annotations_to_later_steps(monkeypatch, tmp_path: Path):
    flow.LOG_DIR = tmp_path / "logs"
    cfg = config_model.apply_defaults(
        {
            "master_images_dir": "D:/data/master/images",
            "master_annotations_dir": "D:/data/master/annotations",
            "point_predictions_dir": "D:/data/predictions/points",
            "edge_predictions_dir": "D:/data/predictions/edges",
            "model_a_export_root": "D:/data/model_a_export",
            "model_b_export_root": "D:/data/model_b_export",
            "outputs": {
                "sew_point_train_out": "D:/data/runs/sew_point",
                "sew_point_conntect_train_out": "D:/data/runs/sew_point_conntect",
            },
        }
    )
    recorded: list[list[str]] = []

    def fake_run_command(args: list[str], cwd: str, dry_run: bool) -> int:
        recorded.append(args)
        return 0

    monkeypatch.setattr(flow, "run_command", fake_run_command)
    args = type(
        "Args",
        (),
        {
            "image_dir": "",
            "point_output_dir": "",
            "point_model": "",
            "point_threshold": 0.3,
            "point_distance_threshold": 0.0,
            "edge_annotation_dir": "",
            "edge_output_dir": "",
            "edge_model": "",
            "postprocess_preset": "balanced",
            "no_compare_gt": False,
            "validate_annotation_dir": "",
            "export_annotation_dir": "",
            "report_path": "",
            "show_samples": False,
            "model_a_output": "",
            "model_b_output": "",
            "include_train": False,
            "model_a_images": "",
            "model_a_annotations": "",
            "model_a_out": "",
            "model_b_images": "",
            "model_b_annotations": "",
            "model_b_out": "",
            "dry_run": True,
        },
    )()

    assert flow.cmd_pipeline(cfg, args) == 0

    edge_cmd = next(command for command in recorded if "sew_point_conntect.batch_predict" in command)
    validate_cmd = next(command for command in recorded if str(flow.SCRIPTS_DIR / "cabf_validate.py") in command)
    export_cmds = [
        command
        for command in recorded
        if str(flow.SCRIPTS_DIR / "cabf_export_model_a.py") in command
        or str(flow.SCRIPTS_DIR / "cabf_export_model_b.py") in command
    ]

    assert edge_cmd[edge_cmd.index("--annotation_dir") + 1] == "D:/data/predictions/points"
    assert validate_cmd[validate_cmd.index("--annotation_dir") + 1] == "D:/data/predictions/edges"
    assert {command[command.index("--annotation_dir") + 1] for command in export_cmds} == {"D:/data/predictions/edges"}
