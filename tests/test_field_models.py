"""Adapter contract tests with no model files, sessions, or registry access."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from cosmos_toolbox.field_models import FieldModels


@pytest.mark.parametrize('modern,residual', [(True, False), (False, False), (False, True)])
def test_config_layouts(tmp_path, monkeypatch, modern, residual):
    import cosmos_toolbox.field_models as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setenv('WEIGHTREG_CACHE', str(tmp_path / 'cache'))
    backend = tmp_path / ('projects/CAB-F/config/backend.yaml' if modern else 'assets/config/backend_config.yaml')
    product = tmp_path / ('projects/CAB-F/config/products/D01-R.yaml' if modern else 'conf/cabf/D01-R.local.yaml')
    backend.parent.mkdir(parents=True)
    product.parent.mkdir(parents=True)
    backend.write_text('cab_f:\n  knife_checker: {}\n  sew_point_density:\n    connect_model_path: missing.onnx\n    patch_model_path: missing_patch.onnx\n', encoding='utf-8')
    product.write_text('inspection:\n  match_template: []\n', encoding='utf-8')
    if modern:
        runtime = tmp_path / 'projects/CAB-F/algorithms/yolo_adapter.py'
        runtime.parent.mkdir(parents=True)
        runtime.touch()
    if residual:
        (tmp_path / 'projects/CAB-F/config').mkdir(parents=True)
    model = FieldModels('D01-R')
    assert model.project_layout is modern
    assert 'knife_segment' in model.config
    assert model.template_root == (product.parent if modern else tmp_path)


def test_missing_config_group_has_actionable_error(tmp_path, monkeypatch):
    import cosmos_toolbox.field_models as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    backend = tmp_path / 'assets/config/backend_config.yaml'
    backend.parent.mkdir(parents=True)
    backend.write_text('cab: {}', encoding='utf-8')
    with pytest.raises(ValueError, match='cab_f.*backend_config'):
        FieldModels('D01-R')


@pytest.mark.parametrize('modern', [True, False])
def test_runtime_uses_matching_branch_namespace(monkeypatch, modern):
    import sys
    import cosmos_toolbox.field_models as module
    model = object.__new__(FieldModels)
    model.project_layout = modern
    monkeypatch.setitem(sys.modules, 'projects', SimpleNamespace(
        load_project_package=lambda name: SimpleNamespace(__name__='projects._loaded_cab_f')))
    monkeypatch.setattr(module.importlib, 'import_module', lambda name: name)
    assert model._runtime('algo.cab_f.knife_checker', '.algorithms.knife') == (
        'projects._loaded_cab_f.algorithms.knife' if modern else 'algo.cab_f.knife_checker')


@pytest.fixture
def adapter():
    model = object.__new__(FieldModels)
    model.product = 'D01-R'
    model.config = {}
    model.inspection = {}
    model.sessions = {}
    model.cache = {}
    model.errors = {}
    calls = Counter()

    def predictor(name, result):
        def predict(image):
            calls[name] += 1
            return result
        model.sessions[name] = SimpleNamespace(predict=predict)

    predictor('roi_detector', [{'box': [50, 50, 50, 50], 'label': 'tail', 'score': .9}])
    predictor('tail_roi_detector', [{'box': [200, 200, 100, 100], 'label': 'cloth', 'score': .9}])
    predictor('tail_cloth_roi_detector', [{'box': [40, 40, 10, 10], 'label': 'point', 'score': .8}])
    return model, calls


def test_multiselect_reuses_predictions_and_preserves_local_coordinates(adapter):
    model, calls = adapter
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    result = model.generate(image, ['tail_roi_detector', 'tail_cloth_roi_detector'])
    assert not model.errors
    assert calls == {'roi_detector': 2, 'tail_roi_detector': 2, 'tail_cloth_roi_detector': 2}
    first, second = result['tail_cloth_roi_detector']
    assert first['image'].shape == (100, 100, 3)
    assert first['shapes'][0]['points'] == [[35., 35.], [45., 45.]]
    assert first['metadata']['transform_chain'] == [
        {'parent': 'source', 'crop_xyxy': [250, 250, 750, 750]},
        {'parent': 'tail_0', 'crop_xyxy': [150, 150, 250, 250]},
    ]
    assert second['metadata']['transform_chain'][0]['crop_xyxy'] == [1250, 250, 1750, 750]
    # Target labels cannot pollute the tail input reused by the next task.
    assert len(first['shapes']) == 1
    single = model.generate(image, ['tail_cloth_roi_detector'])
    assert single['tail_cloth_roi_detector'][0]['shapes'] == first['shapes']


def test_images_mode_runs_only_localization_and_resets_between_images(adapter):
    model, calls = adapter
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)
    result = model.generate(image, ['tail_cloth_roi_detector'], mode='images', face='bottom')
    assert calls == {'roi_detector': 2, 'tail_roi_detector': 2}
    sample = result['tail_cloth_roi_detector'][0]
    assert sample['metadata']['face'] == 'bottom'
    assert sample['metadata']['annotation_status'] == 'unlabeled'
    assert sample['shapes'] == []
    assert 'masks' not in sample
    model.generate(image, ['tail_cloth_roi_detector'], mode='images')
    assert calls['roi_detector'] == 4


def test_partial_model_failure_is_not_committed_and_other_branch_continues(adapter):
    model, calls = adapter
    count = 0

    def fail_second(image):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('second crop failed')
        return []

    model.sessions['tail_cloth_roi_detector'] = SimpleNamespace(predict=fail_second)
    result = model.generate(np.zeros((1000, 2000, 3), dtype=np.uint8),
                            ['tail_cloth_roi_detector', 'roi_detector'])
    assert 'tail_cloth_roi_detector' not in result
    assert len(result['roi_detector']) == 2
    assert 'second crop failed' in model.errors['tail_cloth_roi_detector']


def test_failed_dependency_is_attempted_once_and_independent_export_survives(adapter):
    model, calls = adapter

    def fail(image):
        calls['failed_tail'] += 1
        raise RuntimeError('tail unavailable')

    model.sessions['tail_roi_detector'] = SimpleNamespace(predict=fail)
    result = model.generate(np.zeros((1000, 2000, 3), dtype=np.uint8),
                            ['tail_cloth_roi_detector', 'hook_detector', 'roi_detector'])
    assert calls['failed_tail'] == 1
    assert set(model.errors) == {'tail_cloth_roi_detector', 'hook_detector'}
    assert set(result) == {'roi_detector'}


def test_failed_calibration_cached_for_fixed_roi_branches(adapter, monkeypatch):
    model, calls = adapter

    def fail():
        calls['calibration'] += 1
        raise ValueError('glue Dice below threshold')

    monkeypatch.setattr(model, '_calibrate_once', fail)
    for _ in range(2):
        with pytest.raises(ValueError, match='glue Dice'):
            model._calibrated()
    assert calls['calibration'] == 1


def test_glue_pseudo_mask_uses_rgb_without_changing_source(adapter):
    model, _ = adapter
    model.face = 'top'
    model.config['glue_segment'] = {'conf': .87}
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[..., 0] = 123
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:8, 3:9] = 255

    def predict_mask(actual_image, threshold):
        np.testing.assert_array_equal(actual_image, image[..., ::-1])
        assert actual_image is not image
        assert threshold == .87
        return mask

    model.sessions['glue_segment'] = SimpleNamespace(predict_mask=predict_mask)
    sample = model._annotate('glue_segment', model._sample(image, 'glue_fixture'))
    # Training mask is foreground-positive; only CAD calibration inverts it.
    np.testing.assert_array_equal(sample['masks']['glue'], mask)
    assert sample['shapes'][0]['label'] == 'glue'
    assert image[0, 0].tolist() == [123, 0, 0]


def test_tool_calibration_rgb_adapter_preserves_session_and_bgr_input():
    from cosmos_toolbox.field_models import _ToolGlueRGBAdapter
    source = np.array([[[10, 20, 30]]], dtype=np.uint8)
    session = object()
    calls = []
    segmenter = SimpleNamespace(session=session, predict_proba_batch=lambda images: calls.extend(images))
    wrapper = _ToolGlueRGBAdapter(segmenter)
    wrapper.predict_proba_batch([source])
    assert wrapper.session is session
    assert calls[0][0, 0].tolist() == [30, 20, 10]
    assert source[0, 0].tolist() == [10, 20, 30]
