from __future__ import annotations

import os
import sys
from pathlib import Path


TOOLBOX_ROOT = Path(__file__).resolve().parents[1]
COSMOS_ROOT = Path(os.environ.get("COSMOS_ROOT", TOOLBOX_ROOT.parents[1])).resolve()
MODULES_ROOT = TOOLBOX_ROOT / "modules"
SHARED_CABF_ROOT = TOOLBOX_ROOT / "shared" / "cabf_common"


def ensure_import_paths() -> None:
    """Expose the migrated packages and the containing Cosmos checkout."""
    for path in reversed((TOOLBOX_ROOT, MODULES_ROOT, SHARED_CABF_ROOT, COSMOS_ROOT)):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    entries = [str(TOOLBOX_ROOT), str(MODULES_ROOT), str(SHARED_CABF_ROOT), str(COSMOS_ROOT)]
    current = env.get("PYTHONPATH")
    if current:
        entries.append(current)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    env["COSMOS_ROOT"] = str(COSMOS_ROOT)
    env["COSMOS_TOOLBOX_ROOT"] = str(TOOLBOX_ROOT)
    return env
