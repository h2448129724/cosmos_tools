"""Contract tests for the per-module tool bootstrap shims.

Each ``modules/<name>/tools/`` directory carries a byte-identical ``_bootstrap.py``
so direct script execution works before ``sys.path`` is configured. These tests
lock in the unified contract and guard against the drift that once left the
sew_point direct-execution path broken (caller referenced a function name that
did not match the definition, so only ``python -m`` worked).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES_ROOT = REPO_ROOT / "modules"
SHARED_CABF_ROOT = REPO_ROOT / "shared" / "cabf_common"
BOOTSTRAPS = [
    MODULES_ROOT / "segmentation" / "tools" / "_bootstrap.py",
    MODULES_ROOT / "sew_point" / "tools" / "_bootstrap.py",
]


def _load_bootstrap(path: Path):
    spec = importlib.util.spec_from_file_location(f"_test_bootstrap_{path.parents[1].name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bootstrap_shims_are_identical():
    contents = [path.read_text(encoding="utf-8") for path in BOOTSTRAPS]
    assert len(set(contents)) == 1, "module tool _bootstrap.py shims must stay identical"


def test_bootstrap_exposes_unified_name_only():
    for path in BOOTSTRAPS:
        module = _load_bootstrap(path)
        assert hasattr(module, "ensure_module_paths"), f"{path} must expose ensure_module_paths"
        for stale in ("ensure_module_import_path", "ensure_direct_scriptpaths", "ensure_direct_script_paths"):
            assert not hasattr(module, stale), f"{path} must not expose stale name {stale!r}"


def test_ensure_module_paths_puts_modules_and_cabf_on_syspath():
    snapshot = list(sys.path)
    try:
        for path in BOOTSTRAPS:
            module = _load_bootstrap(path)
            # the shim only inspects parents[2]/[3], so passing itself is representative
            module.ensure_module_paths(str(path))
        assert str(MODULES_ROOT) in sys.path
        assert str(SHARED_CABF_ROOT) in sys.path
    finally:
        sys.path[:] = snapshot
