from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AppSettings:
    default_project_name: str = ""
    default_command_platform: str = "linux"
    remember_last_selection: bool = True
    last_feature_name: str = ""
    last_action_name: str = ""
    last_conda_env_name: str = ""


class SettingsManager:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.settings_path = self.project_root / "artifacts" / "gui_settings.json"

    def load(self) -> AppSettings:
        if not self.settings_path.exists():
            return AppSettings()
        try:
            with self.settings_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            logger.warning("无法加载 GUI 设置文件: %s", self.settings_path, exc_info=True)
            return AppSettings()

        if not isinstance(payload, dict):
            return AppSettings()

        return AppSettings(
            default_project_name=str(payload.get("default_project_name", "") or ""),
            default_command_platform=str(payload.get("default_command_platform", "linux") or "linux"),
            remember_last_selection=bool(payload.get("remember_last_selection", True)),
            last_feature_name=str(payload.get("last_feature_name", "") or ""),
            last_action_name=str(payload.get("last_action_name", "") or ""),
            last_conda_env_name=str(payload.get("last_conda_env_name", "") or ""),
        )

    def save(self, settings: AppSettings) -> None:
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        with self.settings_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(settings), f, ensure_ascii=False, indent=2)
