from types import SimpleNamespace
import pytest
from cosmos_toolbox.training import cab_f_project


@pytest.fixture(autouse=True)
def clear_adapter_cache():
    cab_f_project.project_entry.cache_clear()
    yield
    cab_f_project.project_entry.cache_clear()


def test_modern_entry_is_preferred(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(cab_f_project, 'ensure_import_paths', lambda: None)
    monkeypatch.setattr(cab_f_project.importlib, 'import_module',
                        lambda name: SimpleNamespace(load_project_package=lambda project: sentinel))
    assert cab_f_project.project_entry() is sentinel


def test_legacy_entry_loads_only_requested_symbol(monkeypatch):
    calls = []
    sentinel = object()
    def load(name):
        calls.append(name)
        return SimpleNamespace() if name == 'projects' else SimpleNamespace(EdgeGraphCore=sentinel)
    monkeypatch.setattr(cab_f_project, 'ensure_import_paths', lambda: None)
    monkeypatch.setattr(cab_f_project.importlib, 'import_module', load)
    entry = cab_f_project.project_entry()
    assert calls == ['projects']
    assert entry.EdgeGraphCore is sentinel
    assert calls[-1] == 'train.models.sew_point_connector'


def test_modern_errors_are_not_silently_fallback(monkeypatch):
    def fail(project):
        raise RuntimeError('invalid project config')
    monkeypatch.setattr(cab_f_project, 'ensure_import_paths', lambda: None)
    monkeypatch.setattr(cab_f_project.importlib, 'import_module',
                        lambda name: SimpleNamespace(load_project_package=fail))
    with pytest.raises(RuntimeError, match='invalid project'):
        cab_f_project.project_entry()
