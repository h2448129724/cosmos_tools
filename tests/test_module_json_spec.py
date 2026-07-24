"""测试 module.json 校验、FeatureScanner 加载和 RunManager 导出命令。"""

from __future__ import annotations

# ruff: noqa: E402  -- repository constants intentionally precede project imports

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from trainer_gui.module_spec import validate_all_modules, validate_module_json
from trainer_gui.feature_scanner import FeatureScanner
from trainer_gui.models import CondaEnvInfo

MODULES_ROOT = REPO_ROOT / "modules"
SHARED_CABF_ROOT = REPO_ROOT / "shared" / "cabf_common"
DIRECT_TOOL_SCRIPTS = [
    MODULES_ROOT / "segmentation" / "tools" / "auto_label_folder.py",
    MODULES_ROOT / "segmentation" / "tools" / "predict_single_image.py",
    MODULES_ROOT / "sew_point" / "tools" / "batch_infer.py",
    MODULES_ROOT / "sew_point" / "tools" / "predict_large_image.py",
]


# ---------------------------------------------------------------------------
# 1. module.json 结构校验
# ---------------------------------------------------------------------------


class TestModuleJsonStructure:
    """遍历仓库中所有 module.json，确保格式合规。"""

    @pytest.fixture(autouse=True)
    def _scan(self):
        self.specs = validate_all_modules(MODULES_ROOT)

    def test_all_module_jsons_are_valid(self):
        errors: list[str] = []
        for spec in self.specs:
            if not spec.is_valid:
                for err in spec.errors:
                    errors.append(str(err))
        assert not errors, "module.json 校验失败:\n" + "\n".join(errors)

    def test_feature_name_matches_directory(self):
        for spec in self.specs:
            dir_name = spec.path.parent.name
            assert spec.feature_name == dir_name, (
                f"{spec.path}: feature_name={spec.feature_name!r} != 目录名 {dir_name!r}"
            )


class TestModuleJsonValidationDetails:
    """对校验器本身做单元测试，确保边界条件覆盖。"""

    def test_missing_file(self, tmp_path: Path):
        spec = validate_module_json(tmp_path / "nope.json")
        assert not spec.is_valid
        assert any("文件不存在" in e.message for e in spec.errors)

    def test_invalid_json(self, tmp_path: Path):
        p = tmp_path / "module.json"
        p.write_text("{bad json", encoding="utf-8")
        spec = validate_module_json(p)
        assert not spec.is_valid
        assert any("JSON 解析失败" in e.message for e in spec.errors)

    def test_top_level_not_object(self, tmp_path: Path):
        p = tmp_path / "module.json"
        p.write_text("[]", encoding="utf-8")
        spec = validate_module_json(p)
        assert not spec.is_valid
        assert any("顶层结构" in e.message for e in spec.errors)

    def test_missing_feature_name(self, tmp_path: Path):
        p = tmp_path / "module.json"
        p.write_text(json.dumps({"actions": []}), encoding="utf-8")
        spec = validate_module_json(p)
        assert not spec.is_valid
        assert any("feature_name" in e.field for e in spec.errors)

    def test_bad_entry_type(self, tmp_path: Path):
        p = tmp_path / "module.json"
        p.write_text(json.dumps({
            "feature_name": "test_mod",
            "actions": [{"action_name": "a", "entry": {"type": "bad", "value": "x.py"}}],
        }), encoding="utf-8")
        spec = validate_module_json(p)
        assert not spec.is_valid
        assert any("entry.type" in e.field for e in spec.errors)

    def test_valid_minimal(self, tmp_path: Path):
        p = tmp_path / "module.json"
        p.write_text(json.dumps({
            "feature_name": "my_mod",
            "actions": [{"action_name": "train", "entry": {"type": "script", "value": "train.py"}}],
        }), encoding="utf-8")
        spec = validate_module_json(p)
        assert spec.is_valid
        assert spec.feature_name == "my_mod"

    def test_field_overrides_not_dict(self, tmp_path: Path):
        p = tmp_path / "module.json"
        p.write_text(json.dumps({
            "feature_name": "mod",
            "actions": [{"action_name": "a", "field_overrides": "oops"}],
        }), encoding="utf-8")
        spec = validate_module_json(p)
        assert not spec.is_valid
        assert any("field_overrides" in e.field for e in spec.errors)


# ---------------------------------------------------------------------------
# 2. FeatureScanner 加载测试
# ---------------------------------------------------------------------------


class TestFeatureScannerLoad:
    """确保每个 module.json 可被 FeatureScanner 正确加载。"""

    @pytest.fixture(autouse=True)
    def _scan(self):
        self.scanner = FeatureScanner(REPO_ROOT)
        self.modules = self.scanner.scan()

    def test_all_module_dirs_discovered(self):
        """每个有 module.json 的模块目录都应该被扫描到。"""
        json_dirs = {
            d.name for d in MODULES_ROOT.iterdir()
            if d.is_dir() and (d / "module.json").exists()
        }
        # 有 module.json 的也可能同时有 train.py，所以两者都检查
        found = {m.feature_name for m in self.modules}
        missing = json_dirs - found
        assert not missing, f"FeatureScanner 未发现以下模块: {missing}"

    def test_every_action_has_script_path(self):
        """每个 action 的 script_path 必须能被解析（不一定要求存在）。"""
        for mod in self.modules:
            for act in mod.actions:
                assert act.script_path is not None, (
                    f"{mod.feature_name}/{act.action_name}: script_path 为 None"
                )

    def test_entry_value_resolves_to_existing_file_for_script_type(self):
        """entry.type=script 的 action，对应脚本文件应存在。"""
        for mod in self.modules:
            for act in mod.actions:
                if act.entry.type == "script":
                    assert act.script_path.exists(), (
                        f"{mod.feature_name}/{act.action_name}: "
                        f"脚本 {act.script_path} 不存在"
                    )


# ---------------------------------------------------------------------------
# Sew Point 模型结构默认值
# ---------------------------------------------------------------------------


def test_sew_point_model_structure_defaults_to_registered_unet():
    """导入常量无法被静态解析时，module.json 仍应提供可执行的模型默认值。"""
    from trainer_gui.argparse_parser import ArgparseSchemaParser

    sew_point = next(
        module
        for module in FeatureScanner(REPO_ROOT).scan()
        if module.feature_name == "sew_point"
    )
    parser = ArgparseSchemaParser()
    for action_name in ("train", "single_inference", "export_onnx"):
        action = next(item for item in sew_point.actions if item.action_name == action_name)
        fields = parser.parse_action(action)
        model_name = next(field for field in fields if field.name == "model_name")
        assert model_name.default == "sew_point_unet"


# ---------------------------------------------------------------------------
# 3. RunManager.build_export_command_text 测试
# ---------------------------------------------------------------------------


class TestBuildExportCommandText:
    """对主要 action 调用 build_export_command_text 不应抛出异常。"""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from trainer_gui.history_manager import HistoryManager
        from trainer_gui.run_manager import RunManager

        self.scanner = FeatureScanner(REPO_ROOT)
        self.modules = self.scanner.scan()
        self.hm = HistoryManager(REPO_ROOT)
        self.rm = RunManager(REPO_ROOT, self.hm)

    def test_all_actions_export_without_error(self):
        """对所有模块的所有 action，生成导出命令不应抛出异常。"""
        conda = CondaEnvInfo(name="test_env", prefix="/tmp/test_env", python_executable="python")
        errors = []
        for mod in self.modules:
            for act in mod.actions:
                # 构造最小参数集
                params = {}
                for f in act.schema:
                    if f.name not in params:
                        params[f.name] = ""
                # YOLO predict 需要 model 和 source 占位值
                if mod.feature_name == "yolo" and act.action_name == "predict":
                    params.setdefault("model", "best.pt")
                    params.setdefault("source", "image.jpg")
                try:
                    self.rm.build_export_command_text(
                        feature=mod,
                        action=act,
                        params=params,
                        conda_env=conda,
                        target_platform="linux",
                    )
                except Exception as exc:
                    errors.append(f"{mod.feature_name}/{act.action_name}: {exc}")
        assert not errors, "build_export_command_text 失败:\n" + "\n".join(errors)

    def test_export_command_contains_entry_value(self):
        """导出命令应包含 entry.value（模块名或脚本名），YOLO 除外。"""
        conda = None
        for mod in self.modules:
            for act in mod.actions:
                # YOLO 使用 `yolo` CLI 而非 python -m，跳过 entry.value 检查
                if mod.feature_name == "yolo":
                    continue
                params = {f.name: "" for f in act.schema}
                text = self.rm.build_export_command_text(
                    feature=mod, action=act, params=params,
                    conda_env=conda, target_platform="linux",
                )
                assert act.entry.value in text, (
                    f"{mod.feature_name}/{act.action_name}: "
                    f"导出命令中未包含 entry.value={act.entry.value!r}"
                )

    def test_module_export_command_includes_shared_cabf_pythonpath(self):
        """模块入口导出命令应显式包含 modules 与 shared/cabf_common。"""
        conda = None
        for mod in self.modules:
            if mod.feature_name == "yolo":
                continue
            for act in mod.actions:
                if act.entry.type != "module":
                    continue
                params = {f.name: "" for f in act.schema}
                linux_text = self.rm.build_export_command_text(
                    feature=mod,
                    action=act,
                    params=params,
                    conda_env=conda,
                    target_platform="linux",
                )
                windows_text = self.rm.build_export_command_text(
                    feature=mod,
                    action=act,
                    params=params,
                    conda_env=conda,
                    target_platform="windows",
                )

                assert "modules" in linux_text
                assert "shared/cabf_common" in linux_text
                assert "modules" in windows_text
                assert "shared\\cabf_common" in windows_text

    def test_runtime_env_includes_shared_cabf_pythonpath(self):
        """GUI 运行环境应让模块入口直接 import cabf。"""
        from trainer_gui.run_manager import RunManager

        env = RunManager._build_process_env(None, REPO_ROOT)
        pythonpath = env["PYTHONPATH"].split(os.pathsep)

        assert str(MODULES_ROOT.resolve()) in pythonpath
        assert str(SHARED_CABF_ROOT.resolve()) in pythonpath

    def test_resolve_yolo_model_name_pose(self):
        """pose 任务应映射到 *-pose.pt 模型文件名。"""
        from trainer_gui.run_manager import RunManager

        params = {"task": "pose", "model_series": "yolo11", "model_size": "n", "custom_model": ""}
        assert RunManager._resolve_yolo_model_name(params) == "yolo11n-pose.pt"

        params["model_series"] = "yolo26"
        params["model_size"] = "s"
        assert RunManager._resolve_yolo_model_name(params) == "yolo26s-pose.pt"

    def test_yolo_predict_export_command(self):
        """YOLO predict action 应生成包含 yolo ... predict 的命令。"""
        conda = None
        for mod in self.modules:
            if mod.feature_name != "yolo":
                continue
            for act in mod.actions:
                if act.action_name != "predict":
                    continue
                params = {f.name: "" for f in act.schema}
                params["model"] = "best.pt"
                params["source"] = "test.jpg"
                params["task"] = "detect"
                text = self.rm.build_export_command_text(
                    feature=mod, action=act, params=params,
                    conda_env=conda, target_platform="linux",
                )
                assert "yolo" in text, "predict 命令应包含 yolo"
                assert "predict" in text, "predict 命令应包含 predict"
                assert "model=best.pt" in text, "predict 命令应包含 model=best.pt"
                assert "source=test.jpg" in text, "predict 命令应包含 source=test.jpg"

    def test_yolo_predict_export_command_with_flags(self):
        """YOLO predict 导出命令应正确传递布尔标志。"""
        conda = None
        for mod in self.modules:
            if mod.feature_name != "yolo":
                continue
            for act in mod.actions:
                if act.action_name != "predict":
                    continue
                params = {f.name: "" for f in act.schema}
                params["model"] = "best.pt"
                params["source"] = "test.jpg"
                params["task"] = "detect"
                params["save"] = True
                params["save_txt"] = True
                params["conf"] = 0.5
                text = self.rm.build_export_command_text(
                    feature=mod, action=act, params=params,
                    conda_env=conda, target_platform="linux",
                )
                assert "save=True" in text
                assert "save_txt=True" in text
                assert "conf=0.5" in text

    def test_yolo_predict_export_command_disables_save_explicitly(self):
        """YOLO CLI 默认保存预测图，GUI 取消保存时必须显式传 save=False。"""
        conda = None
        for mod in self.modules:
            if mod.feature_name != "yolo":
                continue
            for act in mod.actions:
                if act.action_name != "predict":
                    continue
                params = {f.name: "" for f in act.schema}
                params["model"] = "best.pt"
                params["source"] = "test.jpg"
                params["task"] = "detect"
                params["save"] = False
                text = self.rm.build_export_command_text(
                    feature=mod, action=act, params=params,
                    conda_env=conda, target_platform="linux",
                )
                assert "save=False" in text
                runtime_command = self.rm._build_yolo_runtime_command(act, params)
                assert "save=False" in runtime_command

    def test_yolo_export_onnx_uses_ultralytics_cli(self):
        """YOLO ONNX 导出应走 yolo export CLI，而不是普通 python -m 模块入口。"""
        conda = None
        yolo_mod = next(mod for mod in self.modules if mod.feature_name == "yolo")
        export_action = next(act for act in yolo_mod.actions if act.action_name == "export_onnx")
        params = {f.name: "" for f in export_action.schema}
        params["model"] = "best.pt"
        params["imgsz"] = 1024
        params["opset"] = 11
        params["output"] = "runs/out/model.onnx"

        text = self.rm.build_export_command_text(
            feature=yolo_mod,
            action=export_action,
            params=params,
            conda_env=conda,
            target_platform="linux",
        )

        assert "yolo export" in text
        assert "python -m yolo.scripts.export_onnx" not in text
        assert "format=onnx" in text


def test_requirements_include_yolo_export_onnx_dependency():
    """Ultralytics 导出 ONNX 时需要 onnx 包，不能只安装 onnxruntime。"""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    package_names = {
        line.strip().split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].lower()
        for line in requirements
        if line.strip() and not line.strip().startswith("#")
    }

    assert "onnx" in package_names


# ---------------------------------------------------------------------------
# 4. 运行记录 meta.json / config.json 完整性测试
# ---------------------------------------------------------------------------


class TestRunRecordMeta:
    """验证 HistoryManager 生成的 meta.json 包含所有必要字段。"""

    def test_init_run_record_contains_env_fields(self, tmp_path: Path):
        from trainer_gui.history_manager import HistoryManager
        from trainer_gui.models import FeatureAction, FeatureModule, EntryConfig, OutputConfig

        hm = HistoryManager(tmp_path)
        mod = FeatureModule(
            feature_name="test_mod",
            display_name="Test",
            module_dir=tmp_path / "modules" / "test_mod",
        )
        act = FeatureAction(
            action_name="train",
            display_name="训练",
            script_path=tmp_path / "train.py",
            entry=EntryConfig(),
            output=OutputConfig(),
        )
        paths = hm.create_run_paths(mod, act, "proj")
        record = hm.init_run_record(
            feature=mod,
            action=act,
            project_name="proj",
            command=["python", "train.py"],
            cwd=tmp_path,
            run_dir=paths["run_dir"],
            artifacts_dir=paths["artifacts_dir"],
            conda_env_name="pytorch",
            conda_prefix="/opt/conda/envs/pytorch",
            python_executable="python",
        )

        # 验证新字段
        assert record.repo_root == str(tmp_path.resolve())
        assert record.python_version  # sys.version 非空即可
        assert record.platform

        # 写入 meta.json 并验证内容
        hm.write_meta(record)
        meta = json.loads(Path(record.meta_path).read_text(encoding="utf-8"))
        assert meta["repo_root"] == str(tmp_path.resolve())
        assert meta["python_version"]
        assert meta["platform"]
        assert meta["run_id"]
        assert meta["start_time"]
        assert meta["cwd"]
        assert meta["conda_env_name"] == "pytorch"

    def test_config_json_written_correctly(self, tmp_path: Path):
        from trainer_gui.history_manager import HistoryManager
        from trainer_gui.models import FeatureAction, FeatureModule, EntryConfig, OutputConfig

        hm = HistoryManager(tmp_path)
        mod = FeatureModule(
            feature_name="fm",
            display_name="FM",
            module_dir=tmp_path / "modules" / "fm",
        )
        act = FeatureAction(
            action_name="train",
            display_name="Train",
            script_path=tmp_path / "train.py",
            entry=EntryConfig(),
            output=OutputConfig(),
        )
        paths = hm.create_run_paths(mod, act, "proj")
        record = hm.init_run_record(
            feature=mod, action=act, project_name="proj",
            command=["python", "train.py"], cwd=tmp_path,
            run_dir=paths["run_dir"], artifacts_dir=paths["artifacts_dir"],
            conda_env_name=None, conda_prefix=None, python_executable="python",
        )
        config_payload = {
            "project_name": "proj",
            "feature_name": "fm",
            "params": {"lr": "0.001"},
        }
        hm.write_config(record.config_path, config_payload)
        config = json.loads(Path(record.config_path).read_text(encoding="utf-8"))
        assert config["params"]["lr"] == "0.001"

    def test_finalize_updates_status_and_timing(self, tmp_path: Path):
        from trainer_gui.history_manager import HistoryManager
        from trainer_gui.models import FeatureAction, FeatureModule, EntryConfig, OutputConfig

        hm = HistoryManager(tmp_path)
        mod = FeatureModule(
            feature_name="fm2",
            display_name="FM2",
            module_dir=tmp_path / "modules" / "fm2",
        )
        act = FeatureAction(
            action_name="train",
            display_name="Train",
            script_path=tmp_path / "train.py",
            entry=EntryConfig(),
            output=OutputConfig(),
        )
        paths = hm.create_run_paths(mod, act, "proj")
        record = hm.init_run_record(
            feature=mod, action=act, project_name="proj",
            command=["python", "train.py"], cwd=tmp_path,
            run_dir=paths["run_dir"], artifacts_dir=paths["artifacts_dir"],
            conda_env_name=None, conda_prefix=None, python_executable="python",
        )
        finalized = hm.finalize_run(record, status="success", exit_code=0)
        assert finalized.status == "success"
        assert finalized.exit_code == 0
        assert finalized.end_time is not None
        assert finalized.duration_seconds is not None
        assert finalized.duration_seconds >= 0

        # meta.json 应被更新
        meta = json.loads(Path(finalized.meta_path).read_text(encoding="utf-8"))
        assert meta["status"] == "success"
        assert meta["exit_code"] == 0
        assert meta["end_time"]


def test_training_modules_import_cabf_directly():
    """训练模块不应再通过 modules/cabf_shared.py 间接依赖共享层。"""
    checked = [
        MODULES_ROOT / "sew_point" / "utils.py",
        MODULES_ROOT / "sew_point_conntect" / "batch_predict.py",
    ]
    offenders = []
    for path in checked:
        source = path.read_text(encoding="utf-8")
        if "cabf_shared" in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "仍依赖 cabf_shared bridge: " + ", ".join(offenders)


def test_sew_point_direct_script_bootstrap_includes_shared_cabf_root():
    """允许直接运行的 sew_point 工具脚本也必须能找到 cabf 共享层。"""
    checked = [
        MODULES_ROOT / "sew_point" / "tools" / "batch_infer.py",
        MODULES_ROOT / "sew_point" / "tools" / "predict_large_image.py",
    ]
    offenders = []
    for path in checked:
        source = path.read_text(encoding="utf-8")
        if "shared" not in source or "cabf_common" not in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "直接脚本 bootstrap 缺少 shared/cabf_common: " + ", ".join(offenders)


def test_direct_tool_scripts_use_local_bootstrap_helpers():
    """可直接运行的工具脚本不应各自内联 sys.path 修改。"""
    offenders = []
    for path in DIRECT_TOOL_SCRIPTS:
        source = path.read_text(encoding="utf-8")
        if "sys.path.insert" in source or "_bootstrap" not in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, "工具脚本应复用本地 bootstrap helper: " + ", ".join(offenders)
