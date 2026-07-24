"""Enforce the monorepo's architectural invariants.

Two concerns are guarded:

1. **Layering direction** — each layer may depend only on layers at or below
   it, plus stdlib and third-party packages::

       trainer_gui ─┐
       apps        ─┼──► shared/cabf_common ──► (stdlib + 3rd-party)
       modules     ─┘

   Concrete bans (verified clean as of this commit):

     - shared      : never imports apps / modules / trainer_gui
     - modules     : never imports apps / trainer_gui   (may import ``cabf``)
     - apps        : never imports modules / trainer_gui (may import ``cabf``)
     - trainer_gui : never imports apps                 (may import ``cabf``;
                     launches training modules as subprocesses, not imports)

2. **Dependency weight** — keep each layer's third-party footprint appropriate
   to its role:

     - modules    : no GUI frameworks (training must stay headless-capable so
                    it runs on servers without a display/PySide6 installed)
     - shared     : no heavy ML runtimes (torch/onnx/etc.) so the master-
                    annotation protocol stays lightweight for every consumer

Imports are caught statically via AST, so a dynamic ``importlib`` load with a
string argument would evade this — keep such loads within the same layer.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# layer directory -> top-level module names it must not import
BANNED_IMPORTS: dict[str, set[str]] = {
    "shared": {"apps", "modules", "trainer_gui"},
    "modules": {"apps", "trainer_gui"},
    "apps": {"modules", "trainer_gui"},
    "trainer_gui": {"apps"},
}

# GUI frameworks — training modules must not pull these in (server-side runs).
GUI_FRAMEWORKS = {"PySide2", "PySide6", "PyQt5", "PyQt6", "PyQt7", "qtpy"}
# Heavy ML runtimes — the shared protocol layer must stay lightweight.
HEAVY_ML_RUNTIMES = {"torch", "torchvision", "tensorflow", "keras", "onnx", "onnxruntime", "ultralytics"}


def _absolute_import_top_levels(source: str):
    """Yield the top-level module name for every absolute (non-relative) import."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            # node.level == 0 means absolute; relative imports stay intra-layer.
            if node.level == 0 and node.module:
                yield node.module.split(".")[0]


def _iter_source_files(layer: str):
    for path in (REPO_ROOT / layer).rglob("*.py"):
        if "__pycache__" in path.parts or "egg-info" in path.parts:
            continue
        yield path


def _import_violations(layer: str, forbidden: set[str]) -> list[str]:
    violations = []
    for path in _iter_source_files(layer):
        source = path.read_text(encoding="utf-8")
        for top_level in _absolute_import_top_levels(source):
            if top_level in forbidden:
                violations.append(f"{path.relative_to(REPO_ROOT).as_posix()}: imports '{top_level}'")
    return violations


def test_layers_do_not_import_upward():
    violations = []
    for layer, banned in BANNED_IMPORTS.items():
        violations.extend(_import_violations(layer, banned))

    assert not violations, "layering violations found:\n  " + "\n  ".join(sorted(set(violations)))


def test_training_modules_do_not_import_gui_frameworks():
    """Training code must stay headless-capable (no PySide6/PyQt dependency)."""
    violations = _import_violations("modules", GUI_FRAMEWORKS)
    assert not violations, "modules pull in GUI frameworks:\n  " + "\n  ".join(sorted(set(violations)))


def test_shared_layer_has_no_heavy_ml_runtime():
    """shared/cabf_common must stay lightweight so every consumer can import it cheaply."""
    violations = _import_violations("shared", HEAVY_ML_RUNTIMES)
    assert not violations, "shared layer pulls in heavy ML runtimes:\n  " + "\n  ".join(sorted(set(violations)))
