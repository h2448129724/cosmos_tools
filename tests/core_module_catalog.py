"""Shared inventory and static checks for functional-core architecture gates.

The catalog is intentionally explicit.  A new functional core is not covered by
accident: it must be added here once for the source-effect gate and once for the
inert-import gate.  The entries point at source files rather than importing the
modules, so the static gate can run even when an optional runtime is absent.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class CoreModule:
    """One registered functional-core module and its source location."""

    import_name: str
    source: str

    @property
    def source_path(self) -> Path:
        return REPOSITORY_ROOT / self.source


# Keep this list scoped to modules whose public job is to calculate domain
# facts, state transitions, projections, comparisons, or execution plans.
# Adapters and UI modules belong outside this inventory even when they contain
# helper functions.
CORE_MODULES: tuple[CoreModule, ...] = (
    CoreModule("cosmos_toolbox.capability_catalog", "cosmos_toolbox/capability_catalog.py"),
    CoreModule("cosmos_toolbox.route_state", "cosmos_toolbox/route_state.py"),
    CoreModule("cosmos_toolbox.task_ledger", "cosmos_toolbox/task_ledger.py"),
    CoreModule("cosmos_toolbox.task_presentation", "cosmos_toolbox/task_presentation.py"),
    CoreModule("cosmos_toolbox.layout_presentation", "cosmos_toolbox/layout_presentation.py"),
    CoreModule("cosmos_toolbox.workbench_state", "cosmos_toolbox/workbench_state.py"),
    CoreModule("cosmos_toolbox.cabf_calibration", "cosmos_toolbox/cabf_calibration.py"),
    CoreModule("cosmos_toolbox.training.model_resolution", "cosmos_toolbox/training/model_resolution.py"),
    CoreModule("img_tools.core.crop_plan", "img_tools/core/crop_plan.py"),
    CoreModule("img_tools.core.fixed_roi", "img_tools/core/fixed_roi.py"),
    CoreModule("trainer_gui.run_plan", "trainer_gui/run_plan.py"),
    CoreModule("apps.cabf_flow.annotation_source_core", "apps/cabf_flow/annotation_source_core.py"),
    CoreModule("apps.cabf_flow.plan", "apps/cabf_flow/plan.py"),
    CoreModule("apps.cosmos_pipeline.outcome", "apps/cosmos_pipeline/outcome.py"),
    CoreModule("apps.cosmos_pipeline.request_plan", "apps/cosmos_pipeline/request_plan.py"),
    CoreModule("apps.point_marking.point_core", "apps/point_marking/point_core.py"),
    CoreModule("modules.sew_point.inference_core", "modules/sew_point/inference_core.py"),
    CoreModule("modules.sew_point_conntect.evaluation_core", "modules/sew_point_conntect/evaluation_core.py"),
    CoreModule("modules.sew_point_conntect.inference_core", "modules/sew_point_conntect/inference_core.py"),
    CoreModule("modules.yolo.action_plan", "modules/yolo/action_plan.py"),
)


# These are runtime boundaries, not ordinary dependency-light Python helpers.
# The prefix comparison also catches submodules such as ``torch.nn`` and
# ``PySide6.QtWidgets``.
FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "cv2",
        "onnx",
        "onnxruntime",
        "torch",
        "torchaudio",
        "torchvision",
        "ultralytics",
        "tensorflow",
        "keras",
    }
)

# Imports which provide direct imperative boundaries.  ``pathlib`` remains
# allowed because pure plans may carry Path values without reading or writing.
FORBIDDEN_EFFECT_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "builtins.open",
        "io",
        "os",
        "sqlite3",
        "socket",
        "subprocess",
        "threading",
        "multiprocessing",
        "concurrent.futures",
        "shutil",
        "tempfile",
        "requests",
        "httpx",
        "urllib.request",
        "yaml",
    }
)

# A small whitelist for module-level constant construction.  Calls inside
# function/class bodies are deliberately ignored: they execute only when the
# core API is invoked and are checked by the core's behavioural tests.
PURE_TOP_LEVEL_CALLS: frozenset[str] = frozenset(
    {
        "bool",
        "dict",
        "enumerate",
        "float",
        "frozenset",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
)


@dataclass(frozen=True, slots=True)
class SourceEffect:
    """A source-level violation with enough location for a useful failure."""

    module: str
    line: int
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.module}:{self.line}: {self.kind}: {self.detail}"


def _root(name: str) -> str:
    return name.split(".", 1)[0]


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _is_forbidden_import(name: str) -> tuple[str, str] | None:
    root = _root(name)
    if root in FORBIDDEN_IMPORT_ROOTS:
        return "runtime-import", name
    if name in FORBIDDEN_EFFECT_IMPORT_ROOTS or root in FORBIDDEN_EFFECT_IMPORT_ROOTS:
        return "effect-import", name
    # ``from urllib import request`` is equivalent to importing
    # ``urllib.request`` for this boundary check.
    if name == "urllib":
        return "effect-import", name
    return None


def _top_level_calls(tree: ast.Module) -> Iterable[ast.Call]:
    """Yield calls in module-level expressions, excluding definitions."""

    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Decorators run at import time, but the only permitted decorator in
            # registered cores is dataclass/frozen/slots.  Check those below.
            for decorator in statement.decorator_list:
                for node in ast.walk(decorator):
                    if isinstance(node, ast.Call):
                        yield node
            continue
        yield from (node for node in ast.walk(statement) if isinstance(node, ast.Call))


def find_source_effects(core: CoreModule, source: str | None = None) -> tuple[SourceEffect, ...]:
    """Return forbidden imports and import-time effects in one core source."""

    text = core.source_path.read_text(encoding="utf-8") if source is None else source
    tree = ast.parse(text, filename=str(core.source_path))
    effects: list[SourceEffect] = []

    # Imports are forbidden even when hidden behind a function.  The source
    # gate protects the core's dependency boundary; the inert-import probe
    # separately checks which operations actually execute during import.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                forbidden = _is_forbidden_import(alias.name)
                if forbidden:
                    effects.append(SourceEffect(core.import_name, node.lineno, *forbidden))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "__future__":
                continue
            forbidden = _is_forbidden_import(module)
            if forbidden:
                effects.append(SourceEffect(core.import_name, node.lineno, *forbidden))
            elif module == "urllib" and any(alias.name == "request" for alias in node.names):
                effects.append(SourceEffect(core.import_name, node.lineno, "effect-import", "urllib.request"))

    # Function bodies are the normal execution surface of a functional core.
    # Path value objects are allowed, but their I/O methods are not.
    io_methods = {'open', 'read_text', 'read_bytes', 'write_text', 'write_bytes',
                  'touch', 'mkdir', 'rmdir', 'unlink', 'rename', 'replace',
                  'iterdir', 'glob', 'rglob', 'exists', 'is_file', 'is_dir', 'stat'}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == 'open':
            effects.append(SourceEffect(core.import_name, node.lineno, 'function-effect', 'open'))
        elif isinstance(node.func, ast.Attribute) and node.func.attr in io_methods - {'replace'}:
            effects.append(SourceEffect(core.import_name, node.lineno, 'function-effect', node.func.attr))

    for call in _top_level_calls(tree):
        function_name = _dotted_name(call.func) or "<dynamic>"
        short_name = function_name.rsplit(".", 1)[-1]
        if short_name in {"dataclass", "dataclass_transform"}:
            continue
        # Any explicit call through a known effect namespace is forbidden even
        # if the namespace was imported under an alias.
        if function_name == "importlib.import_module":
            literal = call.args[0] if call.args and isinstance(call.args[0], ast.Constant) else None
            imported = literal.value if literal and isinstance(literal.value, str) else "<dynamic>"
            effects.append(SourceEffect(core.import_name, call.lineno, "runtime-import", imported))
            continue
        if short_name in {"open", "socket", "create_connection", "Popen", "run", "call", "check_call", "check_output"}:
            effects.append(SourceEffect(core.import_name, call.lineno, "import-time-effect", function_name))
            continue
        if function_name.startswith(("os.", "shutil.", "sqlite3.", "subprocess.", "socket.", "threading.")):
            effects.append(SourceEffect(core.import_name, call.lineno, "import-time-effect", function_name))
            continue
        if short_name not in PURE_TOP_LEVEL_CALLS:
            # Do not permit an arbitrary factory at import time.  An explicit
            # registration constant can use one of the small pure builtins
            # above; all domain construction belongs behind an API function.
            effects.append(SourceEffect(core.import_name, call.lineno, "import-time-call", function_name))

    return tuple(effects)
