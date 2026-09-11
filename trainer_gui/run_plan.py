from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping

from modules.yolo.action_plan import YoloActionPlan

if TYPE_CHECKING:
    from .models import FeatureAction, FeatureModule


PathStyle = Literal["native", "posix"]


@dataclass(frozen=True, slots=True)
class CommandPlan:
    """A deterministic description of one training command.

    The plan contains facts only.  It does not inspect the process
    environment, create output directories, write history, or launch a
    process; those effects belong to the RunManager shell.
    """

    argv: tuple[str, ...]
    cwd: Path
    python_executable: str


@dataclass(frozen=True, slots=True)
class OutputTargetPlan:
    """Describe how the shell should materialize one declared output."""

    arg_name: str
    target: Path | None
    is_directory: bool
    normalize_user_path: bool = False
    clear_existing: bool = False


def module_pythonpath_entries(project_root: Path) -> tuple[Path, ...]:
    return (
        project_root / "modules",
        project_root / "shared" / "cabf_common",
    )


def resolve_yolo_model_name(params: Mapping[str, object]) -> str:
    return YoloActionPlan.from_params("train", params).model_name


def build_yolo_argv(
    action: FeatureAction,
    params: Mapping[str, object],
    *,
    path_style: PathStyle,
) -> tuple[str, ...]:
    """Build one YOLO command for both runtime and command export callers."""

    if action.action_name in {"train", "predict", "export_onnx"}:
        return YoloActionPlan.from_params(action.action_name, params).cli_argv(path_style=path_style)
    return ("yolo",)


def build_runtime_plan(
    feature: FeatureModule,
    action: FeatureAction,
    params: Mapping[str, object],
    *,
    python_executable: str,
    project_root: Path,
) -> CommandPlan:
    if feature.feature_name == "yolo" and action.action_name in {"train", "predict"}:
        return CommandPlan(
            build_yolo_argv(action, params, path_style="native"),
            project_root,
            "yolo",
        )
    if feature.feature_name == "yolo" and action.action_name == "export_onnx":
        return CommandPlan(
            YoloActionPlan.from_params(action.action_name, params).module_argv(
                python_executable=python_executable
            ),
            project_root,
            python_executable,
        )

    argv = build_schema_argv(action, params, python_executable=python_executable)
    cwd = project_root if action.entry.type == "module" else feature.module_dir
    return CommandPlan(argv, cwd, python_executable)


def build_export_argv(
    feature: FeatureModule,
    action: FeatureAction,
    params: Mapping[str, object],
) -> tuple[str, ...]:
    if feature.feature_name == "yolo" and action.action_name == "export_onnx":
        return YoloActionPlan.from_params(action.action_name, params).module_argv()
    if feature.feature_name == "yolo" and action.action_name in {"train", "predict"}:
        return build_yolo_argv(action, params, path_style="posix")
    return build_schema_argv(action, params, python_executable="python")


def build_schema_argv(
    action: FeatureAction,
    params: Mapping[str, object],
    *,
    python_executable: str,
) -> tuple[str, ...]:
    command = (
        [python_executable, "-m", action.entry.value]
        if action.entry.type == "module"
        else [python_executable, action.entry.value]
    )
    for field in action.schema:
        if field.name not in params:
            continue
        value = params[field.name]
        if field.action == "store_true":
            if value:
                command.append(field.cli_flag)
            continue
        if field.action == "store_false":
            if not value:
                command.append(field.cli_flag)
            continue
        if value in (None, ""):
            continue
        command.extend([field.cli_flag, str(value)])
    return tuple(command)


def plan_output_target(
    action: FeatureAction,
    arg_name: str,
    params: Mapping[str, object],
    artifacts_dir: Path,
    *,
    default_file_name: str | None,
    use_artifacts_subdir: bool,
) -> OutputTargetPlan:
    field = next((item for item in action.schema if item.name == arg_name), None)
    is_directory = _is_directory_output_target(arg_name, field.path_mode if field else None)
    force_managed_output = bool(field and field.hidden)
    user_value = "" if force_managed_output else str(params.get(arg_name, "") or "").strip()

    if user_value:
        return OutputTargetPlan(
            arg_name,
            Path(user_value),
            is_directory,
            normalize_user_path=True,
            clear_existing=force_managed_output,
        )
    if not use_artifacts_subdir:
        return OutputTargetPlan(arg_name, None, is_directory, clear_existing=force_managed_output)
    if is_directory:
        return OutputTargetPlan(arg_name, artifacts_dir, True, clear_existing=force_managed_output)
    if default_file_name:
        return OutputTargetPlan(
            arg_name,
            artifacts_dir / default_file_name,
            False,
            clear_existing=force_managed_output,
        )
    return OutputTargetPlan(arg_name, None, False, clear_existing=force_managed_output)


def _append_values(command: list[str], params: Mapping[str, object], names: tuple[str, ...]) -> None:
    for name in names:
        value = params.get(name)
        if value not in (None, ""):
            command.append(f"{name}={value}")


def _append_optional_value(command: list[str], params: Mapping[str, object], name: str) -> None:
    value = params.get(name)
    if value not in (None, ""):
        command.append(f"{name}={value}")


def _path_text(path: Path, style: PathStyle) -> str:
    return path.as_posix() if style == "posix" else str(path)


def _is_directory_output_target(arg_name: str, path_mode: str | None) -> bool:
    if path_mode == "dir":
        return True
    if path_mode in {"save_file", "open_file"}:
        return False
    return arg_name.endswith("_dir") or arg_name in {"save_dir", "output_dir", "out_dir"}
