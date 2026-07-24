"""Shared cancellation and machine-readable reports for folder tasks."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskSummary:
    task_type: str
    processed_files: int
    written_files: int
    errors: tuple[str, ...]
    cancelled: bool = False


def write_task_report(output_dir: str | Path, summary: TaskSummary, *, parameters: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and CSV task reports next to generated results."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    payload = {"created_at": now, "parameters": parameters, **asdict(summary)}
    json_path, csv_path = root / "report.json", root / "report.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task_type", "processed_files", "written_files", "cancelled", "error"])
        if summary.errors:
            for error in summary.errors:
                writer.writerow([summary.task_type, summary.processed_files, summary.written_files, summary.cancelled, error])
        else:
            writer.writerow([summary.task_type, summary.processed_files, summary.written_files, summary.cancelled, ""])
    return json_path, csv_path
