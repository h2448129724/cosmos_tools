"""Text-preserving CAB-F match-template config tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from types import SimpleNamespace
import numpy as np


TOOLBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLBOX))
sys.path.insert(0, str(TOOLBOX / "shared" / "cabf_common"))

from cosmos_toolbox.cabf_config import (  # noqa: E402
    CabfConfigDocument,
    CabfConfigError,
    _patch_match_template_path,
)


def test_patch_inline_match_template_preserves_yaml_and_comments() -> None:
    text = 'inspection:\n  match_template: ["top.png", \'bottom.png\'] # retain this\n'
    patched = _patch_match_template_path(text, 1, r"D:\templates\bottom#1's.png")
    parsed = yaml.safe_load(patched)
    assert parsed["inspection"]["match_template"][1] == r"D:\templates\bottom#1's.png"
    assert "# retain this" in patched


def test_update_template_path_does_not_mutate_document_on_patch_failure(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("inspection:\n  match_template: [top.png]\n", encoding="utf-8")
    document = CabfConfigDocument.load(config_path)
    before = document.original_text
    with pytest.raises(CabfConfigError):
        document.update_template_path("bottom", tmp_path / "bottom.png")
    assert document.original_text == before
    assert document.template_value("top") == "top.png"


@pytest.mark.parametrize('existing', [False, True])
def test_template_save_rolls_back_if_config_commit_fails(tmp_path, monkeypatch, existing):
    from cosmos_toolbox import cabf_activity
    target = tmp_path / 'template.png'
    if existing:
        target.write_bytes(b'previous-template')
    def fail(*args):
        raise OSError('config write denied')
    document = SimpleNamespace(template_path=lambda side: target,
                               validate_template_path_update=lambda *args: None,
                               update_template_path=fail)
    errors = []
    page = SimpleNamespace(side='top', document=document,
                           _generated_templates={'top': SimpleNamespace(image=np.zeros((4, 4), np.uint8))},
                           update_config_check=SimpleNamespace(isChecked=lambda: True),
                           _show_error=lambda title, exc: errors.append(str(exc)))
    monkeypatch.setattr(cabf_activity.QFileDialog, 'getSaveFileName', lambda *args: (str(target), 'PNG'))
    cabf_activity.CabfConfigActivity._save_template(page)
    assert errors == ['config write denied']
    if existing:
        assert target.read_bytes() == b'previous-template'
    else:
        assert not target.exists()


def test_flow_mapping_preserves_other_fields():
    text = 'inspection:\n  match_template: [{path: top.png, score: 0.5}, {path: bottom.png}] # keep\n'
    patched = _patch_match_template_path(text, 0, 'new.png')
    assert yaml.safe_load(patched)['inspection']['match_template'][0] == {'path': 'new.png', 'score': .5}
    assert '# keep' in patched
