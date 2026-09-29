from types import SimpleNamespace as NS

import pytest

from cosmos_toolbox.field_dataset_ng import decide, MODEL_CHECKS
from cosmos_toolbox.field_dataset import MODEL_IDS


def test_ng_runtime_swaps_only_glue_input_once_and_preserves_source():
    import numpy as np
    from cosmos_toolbox.field_dataset_ng import NGGate
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[:] = [17, 83, 201]  # BGR; asymmetric channels detect a missed/double swap.
    original = image.copy()
    received = []
    segmenter = NS(session=object(), predict_proba_batch=lambda batch: received.append(batch[0].copy()))
    extractor = NS(segmenter=segmenter)
    def evaluate(actual, face):
        assert actual is image  # Other inspection models still receive the BGR original.
        extractor.segmenter.predict_proba_batch([actual])
        return NS(results={}, outcomes={})
    gate = object.__new__(NGGate)
    gate.runtime = NS(resolve_component=lambda name: extractor if name == 'glue_extractor' else None,
                      evaluate_face_report=evaluate,
                      evaluation_provenance=NS(plan=NS(for_face=lambda face: NS(item_sequence=NS(planned=[])))))
    gate.evaluate(image, 'bottom', ['hook_detector'])
    gate.evaluate(image, 'bottom', ['hook_detector'])
    for converted in received:
        np.testing.assert_array_equal(converted, original[..., ::-1])
    np.testing.assert_array_equal(image, original)
    assert extractor.segmenter.segmenter is segmenter


def entry(item, instance=None, requires=(), provides=()):
    return NS(item_id=item, instance=instance or item, requires=requires, provides=provides, checker_binding=True)


def outcome(passed, cause=None):
    return NS(passed=passed, message='reason', annotations={'cause': cause} if cause else {})


def test_all_models_have_explicit_check_mapping():
    assert set(MODEL_CHECKS) == set(MODEL_IDS)


def test_independent_and_shared_ng_rules():
    plan = [entry('hook'), entry('tail_placement')]
    report = NS(results={}, outcomes={'hook': outcome(False), 'tail_placement': outcome(True)})
    result = decide(report, plan, ['hook_detector', 'tail_roi_detector', 'tail_placement_classifier', 'glue_segment'])
    assert result['hook_detector']['save']
    assert result['tail_roi_detector']['save']
    assert result['glue_segment']['save']
    assert result['tail_placement_classifier']['reason'] == 'OK'


@pytest.mark.parametrize('failed,missing,cause', [(True, False, None), (False, True, None), (False, False, 'algorithm')])
def test_failure_and_missing_are_not_business_ng(failed, missing, cause):
    report = NS(results={'tail': {'hook': {'failed': failed}}},
                outcomes={} if missing else {'custom_hook': outcome(False, cause)})
    result = decide(report, [entry('hook', 'custom_hook', requires=('tail',))], ['hook_detector'])
    assert not result['hook_detector']['save']
    assert result['hook_detector']['reason'] in {'execution_error', 'not_executed'}


def test_multiple_instances_any_real_ng_wins_and_disabled_does_not_save():
    report = NS(results={}, outcomes={'one': outcome(True), 'two': outcome(False)})
    result = decide(report, [entry('hook', 'one'), entry('hook', 'two')], ['hook_detector', 'qr_yolo_cut'])
    assert result['hook_detector']['save']
    assert result['qr_yolo_cut']['reason'] == 'not_executed'


def legacy_inspection():
    return {'conf': {'bottom': {'hook': {'expected_label': 'up'}, 'tail': {'placement': {'number': 1}}}},
            'ok_checkers': [[], [{'check_method': 'check_hook'}, {'check_method': 'check_tail_placement'}]]}


def test_legacy_uses_formal_checker_and_face_specific_enablement():
    from cosmos_toolbox.paths import ensure_import_paths
    ensure_import_paths()
    from biz.checker import CABFChecker
    from cosmos_toolbox.field_dataset_ng import decide_legacy
    results = {'face': 'bottom', 'hook': {'count': 1, 'detected_label': 'down', 'boxes': []},
               'tail': {'placement': [{'label': '1'}]}}
    inspection = legacy_inspection()
    selected = ['hook_detector', 'tail_roi_detector', 'tail_placement_classifier', 'qr_yolo_cut']
    decisions = decide_legacy(results, inspection, 'bottom', selected, CABFChecker)
    assert decisions['hook_detector']['save']
    assert decisions['tail_roi_detector']['save']
    assert decisions['tail_placement_classifier']['reason'] == 'OK'
    assert decisions['qr_yolo_cut']['reason'] == 'not_executed'
    assert not decide_legacy(results, inspection, 'top', selected, CABFChecker)['hook_detector']['save']


@pytest.mark.parametrize('value,reason', [
    ({'failed': True, 'status': False}, 'execution_error'),
    ({'boxes': [{'sew_point_failed': True}], 'status': False}, 'execution_error'),
    (None, 'not_executed'),
])
def test_legacy_failed_or_missing_not_saved(value, reason):
    from cosmos_toolbox.field_dataset_ng import decide_legacy
    results = {} if value is None else {'hook': value}
    decisions = decide_legacy(results, legacy_inspection(), 'bottom', ['hook_detector'],
                              NS(evaluate=lambda *args: outcome(False)))
    assert decisions['hook_detector']['reason'] == reason
    assert not decisions['hook_detector']['save']


def test_legacy_checker_exception_not_business_ng():
    from cosmos_toolbox.field_dataset_ng import decide_legacy
    def broken(*args):
        raise ValueError('bad configuration')
    result = decide_legacy({'hook': {}}, legacy_inspection(), 'bottom', ['hook_detector'], NS(evaluate=broken))
    assert result['hook_detector']['reason'] == 'execution_error'


def test_legacy_additional_checks_keep_real_names_and_shared_dependencies():
    from cosmos_toolbox.field_dataset_ng import decide_legacy
    inspection = {'conf': {'bottom': {'tail_sew_points': {}}},
                  'ok_checkers': [[], [{'check_method': 'check_tail_sew_points'}]]}
    result = decide_legacy({'tail_sew_points': {'count': 0}}, inspection, 'bottom',
                           ['sew_point_detector', 'tail_roi_detector', 'hook_detector'],
                           NS(evaluate=lambda *args: outcome(False)))
    assert result['sew_point_detector']['save']
    assert result['tail_roi_detector']['save']
    assert result['tail_roi_detector']['checks'][0]['item'] == 'tail_sew_points'
    assert not result['hook_detector']['save']


def test_legacy_unknown_checker_fails_explicitly():
    from cosmos_toolbox.field_dataset_ng import decide_legacy
    inspection = {'ok_checkers': [[{'check_method': 'check_future_item'}]]}
    with pytest.raises(ValueError, match='check_future_item'):
        decide_legacy({}, inspection, 'top', ['glue_segment'], NS())


def test_legacy_gate_routes_without_projects_and_reuses_image(monkeypatch):
    import numpy as np
    import cosmos_toolbox.field_dataset_ng as module
    source = np.full((2, 2, 3), [17, 83, 201], dtype=np.uint8)
    received = []
    extractor = NS(segmenter=NS(session=object(), predict_proba_batch=lambda batch: received.append(batch[0])))
    def evaluate(image, params):
        assert image is source
        assert params['face'] == 1
        assert params['bottom']['hook']['expected_label'] == 'up'
        extractor.segmenter.predict_proba_batch([image])
        return image, {'face': 'bottom', 'hook': {'count': 1, 'detected_label': 'down'}}
    runtime = NS(_get_component=lambda name: extractor, cab_f_evaluation=evaluate)
    monkeypatch.setattr(module, '_load_legacy_runtime', lambda models: runtime)
    models = NS(project_layout=False, inspection=legacy_inspection())
    gate = module.NGGate(models)
    for _ in range(2):
        assert gate.evaluate(source, 'bottom', ['hook_detector'])['hook_detector']['save']
    for rgb in received:
        np.testing.assert_array_equal(rgb, source[..., ::-1])
    assert 'face' not in models.inspection['conf']
    gate.close()
    assert gate.legacy.runtime is None


@pytest.mark.parametrize('fail', [False, True])
def test_private_legacy_import_restores_config_even_on_failure(monkeypatch, fail):
    from cosmos_toolbox.paths import ensure_import_paths
    ensure_import_paths()
    import biz.config_loader as loader
    import cosmos_toolbox.field_dataset_ng as module
    original_config, original_backend = loader.config, loader._backend_config
    models = NS(inspection={'product': 'D01-L'}, config={'hook_detector': {'path': 'local.onnx'}})
    def execute(runtime):
        assert loader.config.inspection == models.inspection
        assert loader._backend_config == {'cab_f': models.config}
        runtime.config = loader.config
        if fail:
            raise ImportError('fixture import failure')
    spec = NS(loader=NS(exec_module=execute))
    monkeypatch.setattr(module.importlib.util, 'spec_from_file_location', lambda *args: spec)
    monkeypatch.setattr(module.importlib.util, 'module_from_spec', lambda *args: NS())
    if fail:
        with pytest.raises(ImportError, match='fixture'):
            module._load_legacy_runtime(models)
    else:
        runtime = module._load_legacy_runtime(models)
        assert runtime.config is not original_config
    assert loader.config is original_config
    assert loader._backend_config is original_backend


@pytest.mark.parametrize('legacy', [False, True])
def test_ng_knife_rgb_is_lazy_instance_local_and_converted_once(legacy):
    import numpy as np
    from cosmos_toolbox.field_dataset_ng import NGGate, LegacyNGGate

    source = np.full((4, 5, 3), [17, 83, 201], dtype=np.uint8)
    original = source.copy()
    predictions, resolutions = [], []

    class Checker:
        def predict_mask(self, image_bgr, threshold=.5):
            predictions.append(image_bgr.copy())
            assert threshold == .73
            return {'ear': np.ones(source.shape[:2], dtype=np.uint8)}

        def evaluate(self, ear_image, params, px_per_mm):
            assert px_per_mm == 12
            return self.predict_mask(ear_image, params['threshold'])

    checker = Checker()
    extractor = NS(segmenter=NS(session=object(), predict_proba_batch=lambda batch: None))
    other = object()
    components = {'knife_checker': checker, 'glue_extractor': extractor, 'other': other}
    def resolve(name):
        resolutions.append(name)
        return components[name]

    resolver_name = '_get_component' if legacy else 'resolve_component'
    runtime = NS(**{resolver_name: resolve})
    def evaluate(image, face):
        assert image is source
        # No eager knife model initialization for unrelated/disabled checks.
        assert 'knife_checker' not in resolutions
        resolver = getattr(runtime, resolver_name)
        assert resolver('other') is other
        resolver('knife_checker').evaluate(image, {'threshold': .73}, 12)
        return (image, {}) if legacy else NS(results={}, outcomes={})

    if legacy:
        gate = object.__new__(LegacyNGGate)
        gate.inspection = {'conf': {}}
        gate.checker = NS()
        runtime.cab_f_evaluation = evaluate
    else:
        gate = object.__new__(NGGate)
        runtime.evaluate_face_report = evaluate
        runtime.evaluation_provenance = NS(plan=NS(
            for_face=lambda face: NS(item_sequence=NS(planned=[]))))
    gate.runtime = runtime
    for _ in range(2):
        resolutions.clear()
        gate.evaluate(source, 'top', ['knife_segment'])
    assert len(predictions) == 2
    for actual in predictions:
        np.testing.assert_array_equal(actual, original[..., ::-1])
    np.testing.assert_array_equal(source, original)
    # The underlying production checker has not been patched.
    checker.evaluate(source, {'threshold': .73}, 12)
    np.testing.assert_array_equal(predictions[-1], original)


def test_legacy_yolo_adapter_converts_only_predict_once():
    import numpy as np
    from cosmos_toolbox.field_dataset_ng import _adapt_knife_resolver
    image = np.full((3, 4, 3), [17, 83, 201], dtype=np.uint8)
    seen = []
    detector = NS(predict=lambda pixels: seen.append(pixels.copy()))
    runtime = NS(_get_component=lambda name: detector)
    for _ in range(2):
        _adapt_knife_resolver(runtime, '_get_component', adapt_yolo=True)
        runtime._get_component('roi_detector').predict(image)
    assert len(seen) == 2
    for actual in seen:
        np.testing.assert_array_equal(actual, image[..., ::-1])
