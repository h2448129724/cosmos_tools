from __future__ import annotations

import json
import platform
import re
import shutil
import sys
import unicodedata
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import FeatureAction, FeatureModule, RunRecord


class HistoryManager:
    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.runs_root = self.project_root / "runs"
        self.legacy_runs_root = self.project_root / "artifacts" / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.history_path = self.runs_root / "history.jsonl"
        if not self.history_path.exists():
            self.history_path.write_text("", encoding="utf-8")

    def create_run_paths(self, feature: FeatureModule, action: FeatureAction, project_name: str) -> dict[str, Path]:
        # These values are displayed in the GUI and come from module metadata or
        # user input.  Treat each one as a single path component: a name such as
        # ``../outside`` must never be able to redirect a run out of ``runs``.
        safe_project = self._safe_component(
            project_name or feature.default_project_name or "default", "default"
        )
        safe_feature = self._safe_component(feature.feature_name, "feature")
        safe_action = self._safe_component(action.action_name, "action")
        parent_dir = self.runs_root / safe_project / safe_feature / safe_action
        self._assert_contained(parent_dir, self.runs_root)

        # ``exist_ok=True`` with a second-resolution timestamp caused concurrent
        # launches to share one output directory and overwrite each other's logs.
        # Keep the familiar timestamp as the visible prefix, adding microseconds
        # (and a deterministic collision suffix for mocked clocks) to make each
        # directory allocation atomic and unique.
        timestamp_base = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        for collision in range(1000):
            timestamp = timestamp_base if collision == 0 else f"{timestamp_base}-{collision}"
            run_dir = parent_dir / timestamp
            self._assert_contained(run_dir, self.runs_root)
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            break
        else:  # pragma: no cover - requires a pathological pre-existing directory set
            raise FileExistsError(f"无法为训练运行分配唯一目录: {parent_dir}")

        return {
            "project_name": Path(safe_project),
            "run_dir": run_dir,
            "artifacts_dir": run_dir,
            "timestamp": Path(timestamp),
        }

    @staticmethod
    def _safe_component(value: object, fallback: str) -> str:
        """Return a safe, readable single path component.

        Both slash variants are replaced explicitly because metadata can be
        authored on another operating system.  Windows-invalid characters,
        control characters, trailing dots/spaces, and reserved device names are
        handled as well; this keeps the same metadata usable on either platform.
        """

        text = unicodedata.normalize("NFC", str(value or "")).strip()
        text = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", text)
        text = text.strip(" .")
        if not text or text in {".", ".."}:
            return fallback

        stem = text.split(".", 1)[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"COM[1-9]", stem) or re.fullmatch(
            r"LPT[1-9]", stem
        ):
            return f"_{text}"
        return text

    @staticmethod
    def _assert_contained(path: Path, root: Path) -> None:
        """Reject paths escaping ``root``, including through symlinked parents."""

        resolved_root = root.resolve()
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"运行路径必须位于 runs 目录内: {path}") from exc

    def init_run_record(
        self,
        feature: FeatureModule,
        action: FeatureAction,
        project_name: str,
        command: list[str],
        cwd: Path,
        run_dir: Path,
        artifacts_dir: Path,
        conda_env_name: str | None,
        conda_prefix: str | None,
        python_executable: str,
    ) -> RunRecord:
        timestamp = run_dir.name
        started_at = datetime.now().astimezone().isoformat()
        run_id = f"{project_name}__{feature.feature_name}__{action.action_name}__{timestamp}"
        return RunRecord(
            run_id=run_id,
            project_name=project_name,
            feature_name=feature.feature_name,
            action_name=action.action_name,
            action_display_name=action.display_name,
            status="running",
            cwd=str(cwd),
            command=command,
            start_time=started_at,
            end_time=None,
            duration_seconds=None,
            output_dir=str(run_dir),
            artifacts_dir=str(artifacts_dir),
            conda_env_name=conda_env_name,
            conda_prefix=conda_prefix,
            python_executable=python_executable,
            exit_code=None,
            stdout_log=str(run_dir / "stdout.log"),
            stderr_log=str(run_dir / "stderr.log"),
            config_path=str(run_dir / "config.json"),
            meta_path=str(run_dir / "meta.json"),
            repo_root=str(self.project_root),
            python_version=sys.version,
            platform=platform.platform(),
        )

    def write_config(self, config_path: str | Path, payload: dict[str, Any]) -> None:
        self._write_json(Path(config_path), payload)

    def write_meta(self, record: RunRecord) -> None:
        self._write_json(Path(record.meta_path), asdict(record))

    def finalize_run(self, record: RunRecord, status: str, exit_code: int | None) -> RunRecord:
        ended = datetime.now().astimezone()
        started = datetime.fromisoformat(record.start_time)
        record.end_time = ended.isoformat()
        record.duration_seconds = round((ended - started).total_seconds(), 3)
        record.status = status
        record.exit_code = exit_code
        self.write_meta(record)
        self.append_history(record)
        return record

    def append_history(self, record: RunRecord) -> None:
        line = json.dumps(
            {
                "run_id": record.run_id,
                "project_name": record.project_name,
                "feature_name": record.feature_name,
                "action_name": record.action_name,
                "action_display_name": record.action_display_name,
                "status": record.status,
                "start_time": record.start_time,
                "end_time": record.end_time,
                "duration_seconds": record.duration_seconds,
                "output_dir": record.output_dir,
                "artifacts_dir": record.artifacts_dir,
                "exit_code": record.exit_code,
            },
            ensure_ascii=False,
        )
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def load_history(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in self._history_paths():
            items.extend(self._read_history_file(path))

        deduped: dict[str, dict[str, Any]] = {}
        for item in items:
            key = str(item.get("run_id") or item.get("output_dir") or "")
            if key:
                deduped[key] = item

        return sorted(
            deduped.values(),
            key=lambda item: str(item.get("start_time", "")),
            reverse=True,
        )

    def load_run_meta(self, output_dir: str | Path) -> dict[str, Any]:
        path = Path(output_dir) / "meta.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_run_config(self, output_dir: str | Path) -> dict[str, Any]:
        path = Path(output_dir) / "config.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _history_paths(self) -> list[Path]:
        paths = [self.history_path]
        legacy_path = self.legacy_runs_root / "history.jsonl"
        if legacy_path != self.history_path and legacy_path.exists():
            paths.append(legacy_path)
        return paths

    @staticmethod
    def _read_history_file(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        items: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items

    def move_run_to_trash(self, output_dir: str | Path) -> Path:
        source = Path(output_dir).resolve()
        if not source.exists():
            raise FileNotFoundError(f"训练记录目录不存在: {source}")

        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        trash_root = self.project_root / ".trash" / timestamp
        trash_root.mkdir(parents=True, exist_ok=True)
        destination = trash_root / source.name
        if destination.exists():
            destination = trash_root / f"{source.name}_{datetime.now().astimezone().strftime('%H%M%S')}"

        shutil.move(str(source), str(destination))
        self._rewrite_history_without_output_dir(str(source))
        return destination

    def _rewrite_history_without_output_dir(self, output_dir: str) -> None:
        records = []
        for item in self.load_history():
            if str(item.get("output_dir", "")) == output_dir:
                continue
            records.append(item)

        records.reverse()
        with self.history_path.open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
