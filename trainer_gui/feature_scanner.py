from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import EntryConfig, FeatureAction, FeatureModule, FieldOverride, ModuleUiConfig, OutputConfig, OutputTargetConfig
from .module_spec import validate_module_json

logger = logging.getLogger(__name__)


class FeatureScanner:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.modules_root = self.project_root / "modules"

    def scan(self) -> list[FeatureModule]:
        features: list[FeatureModule] = []
        if not self.modules_root.exists():
            return features

        for child in sorted(self.modules_root.iterdir()):
            if not child.is_dir():
                continue

            module_json_path = child / "module.json"
            metadata = self._load_module_json(module_json_path)
            train_script = child / "train.py"
            has_actions = bool(metadata.get("actions"))
            if not train_script.exists() and not has_actions:
                continue

            # 校验 module.json 结构，记录问题但不停机
            if module_json_path.exists():
                spec = validate_module_json(module_json_path)
                if not spec.is_valid:
                    for err in spec.errors:
                        logger.warning("module.json 校验问题: %s", err)

            feature_name = str(metadata.get("feature_name") or child.name)
            display_name = str(metadata.get("display_name") or feature_name)
            readme_path = child / str(metadata.get("readme_path") or "README.md")
            if not readme_path.exists():
                readme_path = None

            features.append(
                FeatureModule(
                    feature_name=feature_name,
                    display_name=display_name,
                    module_dir=child,
                    description=str(metadata.get("description") or ""),
                    enabled=bool(metadata.get("enabled", True)),
                    readme_path=readme_path,
                    module_json_path=module_json_path if module_json_path.exists() else None,
                    default_project_name=str(metadata.get("default_project_name") or "default"),
                    preferred_conda_env=metadata.get("preferred_conda_env"),
                    actions=self._build_actions(child, metadata),
                )
            )
        return features

    def _build_actions(self, module_dir: Path, metadata: dict) -> list[FeatureAction]:
        action_items = metadata.get("actions") or []
        if not action_items:
            return [self._default_action(module_dir, metadata)]

        actions: list[FeatureAction] = []
        for raw in action_items:
            if not isinstance(raw, dict):
                continue
            actions.append(self._build_action(module_dir, metadata, raw))
        return actions

    def _default_action(self, module_dir: Path, metadata: dict) -> FeatureAction:
        return self._build_action(
            module_dir,
            metadata,
            {
                "action_name": "train",
                "display_name": "训练",
                "entry": metadata.get("entry") or {"type": "script", "value": "train.py"},
                "output": metadata.get("output") or {},
                "description": metadata.get("description") or "",
                "preferred_conda_env": metadata.get("preferred_conda_env"),
                "ui": metadata.get("ui") or {},
                "field_overrides": metadata.get("field_overrides") or {},
                "script_path": metadata.get("script_path"),
            },
        )

    def _build_action(self, module_dir: Path, metadata: dict, raw: dict) -> FeatureAction:
        entry_data = raw.get("entry") or metadata.get("entry") or {"type": "script", "value": "train.py"}
        output_data = raw.get("output") or metadata.get("output") or {}
        ui_data = raw.get("ui") or {}
        entry = EntryConfig(
            type=str(entry_data.get("type") or "script"),
            value=str(entry_data.get("value") or "train.py"),
        )
        return FeatureAction(
            action_name=str(raw.get("action_name") or raw.get("name") or entry.value),
            display_name=str(raw.get("display_name") or raw.get("action_name") or "动作"),
            script_path=self._resolve_script_path(module_dir, entry, raw.get("script_path")),
            entry=entry,
            output=OutputConfig(
                arg_name=output_data.get("arg_name", "save_dir"),
                use_artifacts_subdir=bool(output_data.get("use_artifacts_subdir", True)),
                default_file_name=output_data.get("default_file_name"),
                extra_outputs=[
                    OutputTargetConfig(
                        arg_name=str(item.get("arg_name") or ""),
                        default_file_name=item.get("default_file_name"),
                    )
                    for item in list(output_data.get("extra_outputs") or [])
                    if isinstance(item, dict) and item.get("arg_name")
                ],
            ),
            description=str(raw.get("description") or metadata.get("description") or ""),
            preferred_conda_env=raw.get("preferred_conda_env") or metadata.get("preferred_conda_env"),
            ui=ModuleUiConfig(
                field_order=list(ui_data.get("field_order") or []),
                hidden_fields=list(ui_data.get("hidden_fields") or []),
                read_only_fields=list(ui_data.get("read_only_fields") or []),
            ),
            field_overrides=self._parse_field_overrides(raw.get("field_overrides") or {}),
        )

    def _resolve_script_path(self, module_dir: Path, entry: EntryConfig, explicit_script_path: str | None) -> Path:
        if explicit_script_path:
            return (module_dir / explicit_script_path).resolve()
        if entry.type == "script":
            return (module_dir / entry.value).resolve()
        return (self.modules_root / Path(*entry.value.split("."))).with_suffix(".py")

    @staticmethod
    def _load_module_json(path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _parse_field_overrides(data: dict) -> dict[str, FieldOverride]:
        overrides: dict[str, FieldOverride] = {}
        for name, raw in data.items():
            if not isinstance(raw, dict):
                continue
            overrides[name] = FieldOverride(
                label=raw.get("label"),
                help=raw.get("help"),
                widget=raw.get("widget"),
                path_mode=raw.get("path_mode"),
                group=raw.get("group"),
                placeholder=raw.get("placeholder"),
                required=raw.get("required"),
                default=raw.get("default"),
                choices=raw.get("choices"),
                hidden=raw.get("hidden"),
                read_only=raw.get("read_only"),
            )
        return overrides
