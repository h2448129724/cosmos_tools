from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CondaEnvInfo:
    name: str
    prefix: str
    python_executable: str
    is_active: bool = False


class CondaEnvManager:
    """Discover Conda environments and prepare isolated child processes."""

    _shared_envs: tuple[CondaEnvInfo, ...] | None = None

    def __init__(self) -> None:
        self._envs: tuple[CondaEnvInfo, ...] | None = None

    def list_envs(self, *, refresh: bool = False) -> list[CondaEnvInfo]:
        if self._envs is not None and not refresh:
            return list(self._envs)
        if self._shared_envs is not None and not refresh:
            self._envs = self._shared_envs
            return list(self._envs)
        payload = self._load_environment_payload()
        self._envs = tuple(self._parse_envs(payload)) if payload is not None else ()
        type(self)._shared_envs = self._envs
        return list(self._envs)

    def find(self, name: str | None) -> CondaEnvInfo | None:
        target = str(name or "").strip().casefold()
        if not target:
            return None
        return next((item for item in self.list_envs() if item.name.casefold() == target), None)

    def python_executable(self, name: str | None) -> str:
        environment = self.find(name)
        if environment is not None and Path(environment.python_executable).is_file():
            return environment.python_executable
        return sys.executable

    def process_environment(
        self,
        name: str | None,
        base: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        environment = dict(base or os.environ)
        selected = self.find(name)
        if selected is None:
            return environment

        prefix = Path(selected.prefix)
        existing_path = environment.get("PATH", "")
        path_parts = (
            str(prefix),
            str(prefix / "Scripts"),
            str(prefix / "Library" / "bin"),
            existing_path,
        )
        environment["PATH"] = os.pathsep.join(part for part in path_parts if part)
        environment["CONDA_DEFAULT_ENV"] = selected.name
        environment["CONDA_PREFIX"] = selected.prefix
        return environment

    def _load_environment_payload(self) -> dict | None:
        command = self._conda_command()
        if command is None:
            logger.warning("无法定位 conda 命令")
            return None
        try:
            result = subprocess.run(
                [command, "env", "list", "--json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            payload = json.loads(result.stdout)
            return payload if isinstance(payload, dict) else None
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            logger.warning("无法获取 conda 环境列表", exc_info=True)
            return None

    @staticmethod
    def _parse_envs(payload: dict) -> list[CondaEnvInfo]:
        root_prefix = str(payload.get("root_prefix", ""))
        active_prefix = str(payload.get("active_prefix", ""))
        envs: list[CondaEnvInfo] = []
        for raw_prefix in payload.get("envs", []):
            prefix = Path(str(raw_prefix))
            python = prefix / ("python.exe" if os.name == "nt" else "bin/python")
            name = "base" if _same_path(prefix, Path(root_prefix)) else prefix.name
            envs.append(
                CondaEnvInfo(
                    name=name,
                    prefix=str(prefix),
                    python_executable=str(python),
                    is_active=_same_path(prefix, Path(active_prefix)),
                )
            )
        envs.sort(key=lambda item: (not item.is_active, item.name != "base", item.name.casefold()))
        return envs

    @staticmethod
    def _conda_command() -> str | None:
        candidates: list[Path] = []
        configured = os.environ.get("CONDA_EXE")
        if configured:
            candidates.append(Path(configured))
        discovered = shutil.which("conda")
        if discovered:
            candidates.append(Path(discovered))

        executable = Path(sys.executable).resolve()
        roots = [executable.parent, executable.parent.parent]
        if executable.parent.parent.name.casefold() == "envs":
            roots.append(executable.parent.parent.parent)
        roots.extend((Path.home() / "miniconda3", Path.home() / "anaconda3"))
        for root in roots:
            candidates.extend((root / "Scripts" / "conda.exe", root / "condabin" / "conda.bat", root / "bin" / "conda"))

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return None


def _same_path(left: Path, right: Path) -> bool:
    if not str(left) or not str(right):
        return False
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return os.path.normcase(str(left)) == os.path.normcase(str(right))
