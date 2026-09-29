"""Optional business-NG gate using the project's formal inspection outcomes."""
from collections.abc import Mapping
import importlib
import importlib.util
import copy
import threading
from types import SimpleNamespace


MODEL_CHECKS = {
    'glue_segment': None,  # Calibration is shared by all enabled checks.
    'roi_detector': {'ear_placement', 'ear_distance', 'ear_sewed', 'knife', 'tail_placement', 'tail_cloth', 'hook'},
    'ear_placement_classifier': {'ear_placement'},
    'knife_segment': {'knife', 'ear_distance'},
    'tail_roi_detector': {'tail_placement', 'tail_cloth', 'hook'},
    'tail_placement_classifier': {'tail_placement'},
    'tail_cloth_roi_detector': {'tail_cloth'},
    'tail_cloth_seam_classifier': {'tail_cloth'},
    'hook_detector': {'hook'},
    'sew_point_detector': {'density', 'tail_cloth', 'ear_sewed'},
    'sew_point_connector': {'density'},
    'reinforcement_placement_detector': {'reinforcement'},
    'error_detector': {'error_detector'},
    'qr_yolo_cut': {'qr'},
}


def has_failure(value):
    if isinstance(value, Mapping):
        return any((key == 'failed' or str(key).endswith('_failed')) and item is True
                   for key, item in value.items()) or any(has_failure(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_failure(v) for v in value)
    return False


def decide(report, planned, selected, model_checks=None):
    """Keep only explicit business NG; missing/failed execution is never NG."""
    facts = []
    for entry in planned:
        if entry.checker_binding is None:
            continue
        outcome = report.outcomes.get(entry.instance)
        results = [report.results.get(port) for port in (*entry.requires, *entry.provides)]
        annotations = getattr(outcome, 'annotations', {}) or {}
        if outcome is None:
            state = 'not_executed'
        elif has_failure(results) or annotations.get('cause') == 'algorithm':
            state = 'error'
        elif outcome.passed is False:
            state = 'NG'
        elif outcome.passed is True:
            state = 'OK'
        else:
            state = 'not_executed'
        facts.append({'item': entry.item_id, 'instance': entry.instance, 'state': state,
                      'message': getattr(outcome, 'message', '')})
    decisions = {}
    for model in selected:
        checks = (MODEL_CHECKS if model_checks is None else model_checks)[model]
        related = [f for f in facts if checks is None or f['item'] in checks]
        ng = [f for f in related if f['state'] == 'NG']
        reason = ('NG' if ng else 'execution_error' if any(f['state'] == 'error' for f in related)
                  else 'not_executed' if not related or any(f['state'] == 'not_executed' for f in related)
                  else 'OK')
        decisions[model] = {'save': bool(ng), 'reason': reason, 'checks': related}
    return decisions


class NGGate:
    def __init__(self, models):
        from .paths import ensure_import_paths
        ensure_import_paths()
        self.legacy = None
        if not models.project_layout:
            self.legacy = LegacyNGGate(models)
            return
        from projects import load_project_package
        package = load_project_package('CAB-F')
        runtime_type = importlib.import_module(package.__name__ + '.algorithms.evaluation.runtime').CABFEvaluationRuntime
        inspection = copy.deepcopy(models.inspection)
        inspection.pop('model_params', None)  # Already merged and locally resolved by FieldModels.
        self.runtime = runtime_type(inspection_config=inspection, backend_models=models.config,
                                    inspection_configuration_path=models.product_config_path,
                                    expected_product=models.product)

    def evaluate(self, image, face, selected):
        if getattr(self, 'legacy', None) is not None:
            return self.legacy.evaluate(image, face, selected)
        from .field_models import _ToolGlueRGBAdapter
        # Match the tool's crop-calibration path. Only this runtime instance's
        # glue and knife networks see RGB; source pixels remain BGR.
        _adapt_knife_resolver(self.runtime, 'resolve_component')
        extractor = self.runtime.resolve_component('glue_extractor')
        if not isinstance(extractor.segmenter, _ToolGlueRGBAdapter):
            extractor.segmenter = _ToolGlueRGBAdapter(extractor.segmenter)
        report = self.runtime.evaluate_face_report(image, 0 if face == 'top' else 1)
        planned = self.runtime.evaluation_provenance.plan.for_face(face).item_sequence.planned
        return decide(report, planned, selected)

    def close(self):
        if getattr(self, 'legacy', None) is not None:
            self.legacy.close()
        else:
            self.runtime.close()


# Result keys consumed by the current dev branch's formal CABFChecker.
LEGACY_CHECK_RESULTS = {
    'ear_placement': ('ear',), 'ear_sewed': ('ear_sewed',),
    'knife': ('knife',), 'ear_distance': ('ear_distance',),
    'tail_placement': ('tail',), 'tail_cloth': ('tail',),
    'tail_sew_points': ('tail_sew_points',), 'hook': ('hook',),
    'density': ('density',), 'reinforcement': ('reinforcement',),
    'error_detector': ('error_detector',), 'qr': ('decode_image',),
    'locating_hole': ('locating_hole',),
    'glue_side_distance': ('glue_side_distance',),
    'glue_consistency': ('glue_consistency',),
}
_legacy_load_lock = threading.RLock()


def _adapt_knife_resolver(runtime, name, *, adapt_yolo=False, models=None):
    """Adapt only this tool-owned runtime; keep component loading lazy."""
    from .field_models import _ToolKnifeRGBAdapter

    resolve = getattr(runtime, name)
    if getattr(resolve, '_tool_knife_rgb', False):
        return

    def resolve_tool_component(component_name, *args, **kwargs):
        if models is not None and component_name == 'knife_checker':
            component = models._model('knife_segment')
        elif models is not None and component_name == 'glue_extractor':
            if runtime.glue_extractor is None:
                from .field_models import _ToolGlueRGBAdapter
                extractor = object.__new__(runtime.GlueExtractor)
                extractor.model_path = models.config['glue_segment']['path']
                extractor.segmenter = _ToolGlueRGBAdapter(models._model('glue_segment'))
                extractor.thresh = models.config['glue_segment'].get('conf', .9)
                runtime.glue_extractor = extractor
            component = runtime.glue_extractor
        else:
            component = resolve(component_name, *args, **kwargs)
        if adapt_yolo:
            from .field_models import CLASSES
            if component_name in CLASSES and not getattr(component, '_tool_yolo_rgb', False):
                if models is not None:
                    models.shared_yolo[component_name] = component
                import cv2
                predict = component.predict
                component._tool_raw_predict = predict
                def predict_rgb(image, *args, **kwargs):
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if models is not None and not args and not kwargs:
                        from .field_prediction_cache import yolo_predict
                        return yolo_predict(models.inference_cache, component, rgb)
                    return predict(rgb, *args, **kwargs)
                component.predict = predict_rgb
                component._tool_yolo_rgb = True
        if component_name == 'knife_checker' and not isinstance(component, _ToolKnifeRGBAdapter):
            return _ToolKnifeRGBAdapter(component)
        return component

    resolve_tool_component._tool_knife_rgb = True
    setattr(runtime, name, resolve_tool_component)


def _load_legacy_runtime(models):
    """Load dev's evaluator into a private namespace, with local-only weights.

    The evaluator imports config objects at module load. Temporarily bind these
    imports without notifying observers, and restore them even if loading fails.
    Its component globals then belong solely to this dataset worker's instance.
    No registry resolution or main-application configuration update is invoked.
    """
    from .paths import COSMOS_ROOT
    import biz.config_loader as loader

    spec = importlib.util.spec_from_file_location(
        '_field_dataset_cabf_eval', COSMOS_ROOT / 'algo/cab_f_eval.py')
    runtime = importlib.util.module_from_spec(spec)
    with _legacy_load_lock:
        original_config, original_backend = loader.config, loader._backend_config
        try:
            inspection = copy.deepcopy(models.inspection)
            inspection.pop('model_params', None)
            loader.config = SimpleNamespace(inspection=inspection)
            loader._backend_config = {'cab_f': copy.deepcopy(models.config)}
            spec.loader.exec_module(runtime)
        finally:
            loader.config, loader._backend_config = original_config, original_backend
    # QR uses a class-owned detector, outside the component resolver. Subclass
    # locally so configuration and cached models never touch the production class.
    if hasattr(runtime, 'QRDetectDecodeProcessor'):
        original_qr = runtime.QRDetectDecodeProcessor
        class ToolQRProcessor(original_qr):
            _detector = None
            _decoder = None
            _default_config = {}

            @classmethod
            def detect_decode_image(cls, image, *args, **kwargs):
                import cv2
                return super().detect_decode_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), *args, **kwargs)
        runtime.QRDetectDecodeProcessor = ToolQRProcessor
    return runtime


def decide_legacy(results, inspection, face, selected, checker):
    """Use exactly the enabled production checkers, never infer NG from status alone."""
    if face not in ('top', 'bottom'):
        raise ValueError(f'Unsupported face: {face}')
    index = 0 if face == 'top' else 1
    configured = inspection.get('ok_checkers') or []
    enabled = (configured[index] or []) if index < len(configured) else []
    conf = inspection.get('conf') or {}
    face_conf = conf.get(face) or {}
    planned, outcomes = [], {}
    for number, entry in enumerate(enabled):
        method = entry['check_method']
        item = method.removeprefix('check_')
        keys = LEGACY_CHECK_RESULTS.get(item)
        instance = f'{method}:{number}'
        planned.append(SimpleNamespace(item_id=item, instance=instance, requires=keys or (),
                                       provides=(), checker_binding=True))
        if keys is None:
            raise ValueError(f'NG 筛选尚未适配当前检查项：{method}')
        if any(key not in face_conf or key not in results for key in keys):
            continue
        try:
            outcomes[instance] = checker.evaluate(method, results, conf)
        except Exception as exc:
            outcomes[instance] = SimpleNamespace(passed=False, message=str(exc), annotations={'cause': 'algorithm'})
    model_checks = {key: None if value is None else set(value) for key, value in MODEL_CHECKS.items()}
    for model in ('roi_detector', 'tail_roi_detector', 'tail_cloth_roi_detector', 'sew_point_detector'):
        model_checks[model].add('tail_sew_points')
    for model in ('sew_point_detector', 'sew_point_connector'):
        model_checks[model].add('glue_side_distance')
    return decide(SimpleNamespace(results=results, outcomes=outcomes), planned, selected, model_checks)


class LegacyNGGate:
    def __init__(self, models):
        self.inspection = copy.deepcopy(models.inspection)
        self.runtime = _load_legacy_runtime(models)
        self.models = models if hasattr(models, 'inference_cache') else None
        if self.models is not None:
            import hashlib
            import numpy as np
            from pathlib import Path
            calibrate = self.runtime.calibrated_pic
            def capture_calibration(image, params):
                result = calibrate(image, params)
                meta = result[2]
                if result[0] is not None and meta:
                    normalized = dict(params)
                    normalized['path'] = str((models.template_root / params['path']).resolve())
                    models.shared_calibration = {
                        'params': normalized,
                        'pixels': hashlib.sha256(memoryview(np.ascontiguousarray(image)).cast('B')).hexdigest(),
                        'metadata': {key: value for key, value in meta.items() if not isinstance(value, np.ndarray)},
                        'transform': self.runtime.transform}
                return result
            self.runtime.calibrated_pic = capture_calibration
        from biz.checker import CABFChecker
        self.checker = CABFChecker

    def evaluate(self, image, face, selected):
        from .field_models import _ToolGlueRGBAdapter
        if face not in ('top', 'bottom'):
            raise ValueError(f'Unsupported face: {face}')
        _adapt_knife_resolver(self.runtime, '_get_component', adapt_yolo=True, models=getattr(self, 'models', None))
        extractor = self.runtime._get_component('glue_extractor')
        if not isinstance(extractor.segmenter, _ToolGlueRGBAdapter):
            extractor.segmenter = _ToolGlueRGBAdapter(extractor.segmenter)
        params = copy.deepcopy(self.inspection.get('conf') or {})
        params['face'] = 0 if face == 'top' else 1
        _, results = self.runtime.cab_f_evaluation(image, params)
        return decide_legacy(results, self.inspection, face, selected, self.checker)

    def close(self):
        # Private module and sessions are not registered in sys.modules.
        self.runtime = None
        self.models = None
