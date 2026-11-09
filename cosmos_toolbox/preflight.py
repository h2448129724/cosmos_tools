from __future__ import annotations

import importlib
import json
import sys
from dataclasses import asdict, dataclass

from .paths import COSMOS_ROOT, TOOLBOX_ROOT


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    available: bool
    version: str = ""
    error: str = ""


REQUIRED_MODULES = (
    "numpy",
    "cv2",
    "PIL",
    "PySide6",
    "yaml",
    "openpyxl",
)


def dependency_status() -> list[DependencyStatus]:
    result: list[DependencyStatus] = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            result.append(DependencyStatus(name, True, str(getattr(module, "__version__", ""))))
        except Exception as exc:  # environment boundary
            result.append(DependencyStatus(name, False, error=f"{type(exc).__name__}: {exc}"))
    return result


def build_report() -> dict:
    report = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "toolbox_root": str(TOOLBOX_ROOT),
        "cosmos_root": str(COSMOS_ROOT),
        "dependencies": [asdict(item) for item in dependency_status()],
    }
    return report


def main() -> int:
    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["available"] for item in report["dependencies"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
