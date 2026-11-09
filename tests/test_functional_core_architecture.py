"""Static architecture gates for dependency-light functional cores."""

from __future__ import annotations

import textwrap

import pytest

from core_module_catalog import (
    CORE_MODULES,
    CoreModule,
    find_source_effects,
)


EXPECTED_CORE_NAMES = {
    "cosmos_toolbox.capability_catalog",
    "cosmos_toolbox.route_state",
    "cosmos_toolbox.task_ledger",
    "cosmos_toolbox.task_presentation",
    "cosmos_toolbox.layout_presentation",
    "cosmos_toolbox.workbench_state",
    "img_tools.core.crop_plan",
    "img_tools.core.fixed_roi",
    "apps.cabf_flow.plan",
    "apps.point_marking.point_core",
}


def test_catalog_is_explicit_and_points_at_existing_sources() -> None:
    names = {entry.import_name for entry in CORE_MODULES}
    assert names == EXPECTED_CORE_NAMES
    assert len(names) == len(CORE_MODULES), "a core must be registered exactly once"
    for entry in CORE_MODULES:
        assert entry.source_path.is_file(), entry


def test_registered_core_sources_have_no_effectful_imports_or_import_time_calls() -> None:
    effects = [effect for entry in CORE_MODULES for effect in find_source_effects(entry)]
    assert not effects, "functional-core source-effect gate failed:\n" + "\n".join(map(str, effects))


@pytest.mark.parametrize(
    ("source", "expected_kind"),
    [
        ("from PySide6.QtWidgets import QWidget\n", "runtime-import"),
        ("import torch\n", "runtime-import"),
        ("from cv2 import imread\n", "runtime-import"),
        ("import os\n", "effect-import"),
        ("import socket\n", "effect-import"),
        ("from pathlib import Path\nPath('state.json').read_text()\n", "import-time-call"),
        ("open('state.json', 'w').write('bad')\n", "import-time-effect"),
        ("import socket\nsocket.socket()\n", "import-time-effect"),
    ],
)
def test_source_gate_rejects_deliberately_invalid_fixtures(source: str, expected_kind: str) -> None:
    fixture = CoreModule("tests.invalid_fixture", "<fixture>")
    effects = find_source_effects(fixture, textwrap.dedent(source))
    assert any(effect.kind == expected_kind for effect in effects), effects


def test_source_gate_allows_dependency_light_value_definitions() -> None:
    fixture = CoreModule("tests.valid_fixture", "<fixture>")
    effects = find_source_effects(
        fixture,
        textwrap.dedent(
            """
            from dataclasses import dataclass
            from pathlib import PurePath
            from typing import Mapping
            VALUES = tuple(enumerate(("a", "b")))
            @dataclass(frozen=True, slots=True)
            class Plan:
                output: PurePath
            """,
        ),
    )
    assert effects == ()


def test_source_gate_checks_deferred_imports() -> None:
    fixture = CoreModule("tests.valid_runtime_fixture", "<fixture>")
    effects = find_source_effects(
        fixture,
        textwrap.dedent(
            """
            def execute_shell():
                import subprocess
                return subprocess.run(['echo', 'deferred'])
            """,
        ),
    )
    assert any(effect.kind == "effect-import" for effect in effects), effects


@pytest.mark.parametrize('call', ["open('state.json')", "Path('state.json').read_text()"])
def test_source_gate_checks_deferred_file_effects(call):
    fixture = CoreModule('tests.bad_function', '<fixture>')
    effects = find_source_effects(fixture, f'from pathlib import Path\ndef calculate():\n    return {call}\n')
    assert any(effect.kind == 'function-effect' for effect in effects)
