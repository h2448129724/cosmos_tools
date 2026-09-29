import numpy as np

from cosmos_toolbox.field_prediction_cache import FramePredictionCache


def test_reuse_requires_identical_pixels_parameters_and_kind():
    cache = FramePredictionCache()
    image = np.full((3, 4, 3), [17, 83, 201], dtype=np.uint8)
    calls = []
    def compute():
        calls.append(1)
        return {'mask': np.ones((3, 4), dtype=np.uint8), 'box': [1, 2]}
    first = cache.call('knife', {'threshold': .5}, image, compute)
    first['mask'][:] = 0
    first['box'][0] = 99
    reused = cache.call('knife', {'threshold': .5}, image.copy(), compute)
    assert reused['mask'].all() and reused['box'] == [1, 2]
    cache.call('knife', {'threshold': .6}, image, compute)
    cache.call('knife', {'threshold': .5}, image[..., ::-1], compute)
    cache.call('other', {'threshold': .5}, image, compute)
    assert len(calls) == 4 and cache.hits == 1
    cache.reset()
    assert not cache.entries and cache.hits == 0


def test_cache_memory_is_bounded():
    cache = FramePredictionCache(max_bytes=128)
    for value in range(10):
        image = np.full((2, 2, 3), value, dtype=np.uint8)
        cache.call('mask', {}, image, lambda: np.zeros(100, dtype=np.uint8))
        assert cache.bytes <= 128
    assert len(cache.entries) == 1


def test_calibration_reuse_requires_matching_pixels_and_params():
    import hashlib
    from pathlib import Path
    from cosmos_toolbox.field_models import FieldModels
    model = object.__new__(FieldModels)
    model.face = 'top'
    model.template_root = Path.cwd()
    model.inspection = {'match_template': [{'path': 'template.png'}]}
    model.image = np.full((4, 5, 3), 17, dtype=np.uint8)
    model.cache = {}
    model.calibration_reused = 0
    expected = model.image.copy()
    model.shared_calibration = {
        'params': {'path': str((Path.cwd() / 'template.png').resolve())},
        'pixels': hashlib.sha256(memoryview(model.image).cast('B')).hexdigest(),
        'metadata': {'glue_dice': .9, 'offset': (0, 0), 'angle': 0, 'scale': 1},
        'transform': lambda *args, **kwargs: expected,
    }
    assert model._calibrate_once() is expected
    assert model.calibration_reused == 1
    model.cache = {}
    model.image[0, 0] = 23
    def fallback(*args):
        raise RuntimeError('different pixels must recompute')
    model._runtime = fallback
    import pytest
    with pytest.raises(RuntimeError, match='must recompute'):
        model._calibrate_once()
