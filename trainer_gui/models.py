from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.conda_runtime import CondaEnvInfo as CondaEnvInfo


@dataclass
class EntryConfig:
    type: str = "script"
    value: str = "train.py"


@dataclass
class OutputTargetConfig:
    arg_name: str
    default_file_name: str | None = None


@dataclass
class OutputConfig:
    arg_name: str | None = "save_dir"
    use_artifacts_subdir: bool = True
    default_file_name: str | None = None
    extra_outputs: list[OutputTargetConfig] = field(default_factory=list)


@dataclass
class FieldOverride:
    label: str | None = None
    help: str | None = None
    widget: str | None = None
    path_mode: str | None = None
    group: str | None = None
    placeholder: str | None = None
    required: bool | None = None
    default: Any = None
    choices: list[Any] | None = None
    hidden: bool | None = None
    read_only: bool | None = None


@dataclass
class FieldSchema:
    name: str
    cli_flag: str
    field_type: str = "string"
    required: bool = False
    default: Any = None
    help: str = ""
    choices: list[Any] | None = None
    action: str | None = None
    widget: str | None = None
    path_mode: str | None = None
    group: str | None = None
    label: str | None = None
    placeholder: str | None = None
    hidden: bool = False
    read_only: bool = False


@dataclass
class ModuleUiConfig:
    field_order: list[str] = field(default_factory=list)
    hidden_fields: list[str] = field(default_factory=list)
    read_only_fields: list[str] = field(default_factory=list)


@dataclass
class FeatureAction:
    action_name: str
    display_name: str
    script_path: Path
    entry: EntryConfig = field(default_factory=EntryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    description: str = ""
    preferred_conda_env: str | None = None
    ui: ModuleUiConfig = field(default_factory=ModuleUiConfig)
    field_overrides: dict[str, FieldOverride] = field(default_factory=dict)
    schema: list[FieldSchema] = field(default_factory=list)


@dataclass
class FeatureModule:
    feature_name: str
    display_name: str
    module_dir: Path
    description: str = ""
    enabled: bool = True
    readme_path: Path | None = None
    module_json_path: Path | None = None
    default_project_name: str = "default"
    preferred_conda_env: str | None = None
    actions: list[FeatureAction] = field(default_factory=list)


@dataclass
class RunRecord:
    run_id: str
    project_name: str
    feature_name: str
    action_name: str
    action_display_name: str
    status: str
    cwd: str
    command: list[str]
    start_time: str
    end_time: str | None
    duration_seconds: float | None
    output_dir: str
    artifacts_dir: str
    conda_env_name: str | None
    conda_prefix: str | None
    python_executable: str
    exit_code: int | None
    stdout_log: str
    stderr_log: str
    config_path: str
    meta_path: str
    repo_root: str = ""
    python_version: str = ""
    platform: str = ""
