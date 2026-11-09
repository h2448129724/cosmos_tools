"""Inert-import gates for registered functional-core modules.

Each module is imported in a fresh interpreter with runtime imports blocked and
an audit hook that rejects filesystem, network, process, or thread operations
originating from the module being imported.  Running in a child process keeps
the test independent of pytest, Qt fixtures, and modules imported by siblings.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from core_module_catalog import CORE_MODULES


_IMPORT_PROBE = r'''
import builtins
import importlib
import json
import os
from pathlib import Path
import sys
import threading

module_name = __MODULE_NAME__
target_path = str(Path(__TARGET_PATH__).resolve()).lower()
forbidden_roots = set(__FORBIDDEN_ROOTS__) | set(__FORBIDDEN_EFFECT_ROOTS__)

def _reject_path_effect(*args, **kwargs):
    raise RuntimeError('import-time filesystem effect through pathlib')

for _path_method in ('open', 'read_text', 'read_bytes', 'write_text', 'write_bytes',
                     'touch', 'mkdir', 'rmdir', 'unlink', 'rename', 'replace'):
    setattr(Path, _path_method, _reject_path_effect)

class BlockForbiddenImports:
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.', 1)[0]
        if root in forbidden_roots:
            raise ImportError(f'functional core imported forbidden runtime: {fullname}')
        return None

sys.meta_path.insert(0, BlockForbiddenImports())
sys.dont_write_bytecode = True

active_filename = None
def trace(frame, event, arg):
    # ``sys._getframe`` itself emits an audit event, so an audit hook cannot
    # inspect its own stack without recursively re-entering.  A trace callback
    # records the currently executing source frame instead.  Importlib's file
    # reads update this to an importlib frame, while a direct target call keeps
    # it set to the target source.
    global active_filename
    active_filename = frame.f_code.co_filename.lower()
    return trace

sys.settrace(trace)

def audit(event, args):
    if active_filename != target_path:
        return
    blocked = {
        'open', 'os.open', 'os.mkdir', 'os.makedirs', 'os.remove', 'os.unlink',
        'os.rename', 'os.replace', 'os.rmdir', 'os.scandir', 'os.listdir',
        'socket.__new__', 'socket.connect', 'socket.getaddrinfo',
        'subprocess.Popen', 'threading.Thread.start',
    }
    if event in blocked:
        raise RuntimeError(f'import-time side effect {event}')

sys.addaudithook(audit)
before_env = dict(os.environ)
before_cwd = os.getcwd()
before_threads = {thread.ident for thread in threading.enumerate()}
before_modules = set(sys.modules)
try:
    importlib.import_module(module_name)
except BaseException as exc:
    print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}))
    raise
after_threads = {thread.ident for thread in threading.enumerate()}
forbidden_loaded = sorted(
    name for name in sys.modules
    if name not in before_modules and name.split('.', 1)[0] in forbidden_roots
)
problems = []
if before_env != dict(os.environ):
    problems.append('environment changed during import')
if before_cwd != os.getcwd():
    problems.append('working directory changed during import')
if before_threads != after_threads:
    problems.append('thread set changed during import')
if forbidden_loaded:
    problems.append('forbidden modules loaded: ' + ', '.join(forbidden_loaded))
if problems:
    raise RuntimeError('; '.join(problems))
print(json.dumps({'ok': True}))
'''


def _probe_import(entry, *, extra_pythonpath: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    repository_root = Path(__file__).resolve().parents[1]
    path_entries = [str(repository_root), str(repository_root / "modules"), str(repository_root / "shared" / "cabf_common")]
    if extra_pythonpath is not None:
        path_entries.insert(0, str(extra_pythonpath))
    existing = env.get("PYTHONPATH")
    if existing:
        path_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(path_entries)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = _IMPORT_PROBE.replace("__MODULE_NAME__", repr(entry.import_name))
    code = code.replace("__TARGET_PATH__", repr(str(entry.source_path)))
    from core_module_catalog import FORBIDDEN_EFFECT_IMPORT_ROOTS, FORBIDDEN_IMPORT_ROOTS

    code = code.replace("__FORBIDDEN_ROOTS__", repr(sorted(FORBIDDEN_IMPORT_ROOTS)))
    effect_roots = sorted({item.split(".", 1)[0] for item in FORBIDDEN_EFFECT_IMPORT_ROOTS})
    code = code.replace("__FORBIDDEN_EFFECT_ROOTS__", repr(effect_roots))
    return subprocess.run(
        [sys.executable, "-c", code],
        # A fixture runs in its own temporary directory so a failing probe
        # cannot leave artifacts in the worktree.  Registered cores use the
        # repository root as their neutral working directory.
        cwd=str(extra_pythonpath or repository_root),
        env=env,
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )


@pytest.mark.parametrize("entry", CORE_MODULES, ids=lambda item: item.import_name)
def test_registered_core_imports_are_inert(entry) -> None:
    result = _probe_import(entry)
    assert result.returncode == 0, (
        f"{entry.import_name} was not inert on import\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert '"ok": true' in result.stdout.lower(), result.stdout


@pytest.mark.parametrize(
    "source",
    [
        "from PySide6.QtWidgets import QWidget\n",
        "import cv2\n",
        "from pathlib import Path\nPath('import-side-effect.txt').write_text('bad')\n",
        "import socket\nsocket.socket()\n",
    ],
)
def test_import_probe_fails_for_deliberately_invalid_fixtures(tmp_path: Path, source: str) -> None:
    fixture = tmp_path / "invalid_core_fixture.py"
    fixture.write_text(textwrap.dedent(source), encoding="utf-8")
    entry = type("Fixture", (), {"import_name": "invalid_core_fixture", "source_path": fixture})
    result = _probe_import(entry, extra_pythonpath=tmp_path)
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert combined, result
    assert not (tmp_path / "import-side-effect.txt").exists()
    assert not (tmp_path / "import-side-effect-dir").exists()
