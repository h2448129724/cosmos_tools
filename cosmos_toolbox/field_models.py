"""Local CAB-F model adapters; no registry downloads or source-image writes."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
CLASSES = {
    'roi_detector': ['ear', 'tail', 'dm'],
    'tail_roi_detector': ['cloth', 'hook', 'placement'],
    'tail_cloth_roi_detector': ['point', 'seam', 'line'],
    'ear_placement_classifier': ['0', '1', '2', '3'],
    'tail_placement_classifier': ['0', '1', '2'],
    'tail_cloth_seam_classifier': ['0', '1'],
    'hook_detector': ['down', 'up'],
    'qr_yolo_cut': ['qr'],
    'error_detector': ['defective_fabric', 'defect_mark_line', 'stain', 'pulled_thread',
                       'loose_thread', 'broken_thread', 'skipped_stitch', 'foreign_object'],
}
CLASSIFIERS = {'ear_placement_classifier', 'tail_placement_classifier',
               'tail_cloth_seam_classifier', 'hook_detector'}
# The tool stores BGR pixels. These are the current legacy model call contracts,
# not an instruction to swap an entire source image or change production code.
INPUT_CONTRACTS = {
    **{name: {'input': 'RGB', 'normalization': '0..1',
              'resize': 'center_crop' if name in CLASSIFIERS else 'letterbox'} for name in CLASSES},
    'glue_segment': {'input': 'RGB', 'normalization': 'ImageNet', 'resize': 'UNet'},
    'knife_segment': {'input': 'RGB', 'normalization': 'ImageNet', 'resize': 'UNet'},
    'reinforcement_placement_detector': {'input': 'BGR', 'normalization': 'internal RGB/ImageNet', 'resize': 'configured input_size'},
    'sew_point_detector': {'input': 'BGR', 'normalization': 'internal RGB/0..1', 'resize': 'tiled'},
    'sew_point_connector': {'input': 'BGR', 'normalization': 'patch BGR/0..1', 'resize': 'affine patches'},
}


def shape(label, points, kind='rectangle', score=None, group_id=None):
    return {'label': str(label), 'points': [[float(x), float(y)] for x, y in points],
            'shape_type': kind, 'flags': {}, 'group_id': group_id,
            'description': 'model pseudo-label', 'score': score}


class _ToolGlueRGBAdapter:
    """Tool-local BGR boundary; never modify the shared Cosmos segmenter."""

    def __init__(self, segmenter):
        self.segmenter = segmenter
        self.session = segmenter.session

    def predict_proba_batch(self, images):
        return self.segmenter.predict_proba_batch(
            [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])


class _ToolKnifeRGBAdapter:
    """Convert tool BGR crops at the boundary, not inside production code."""

    def __init__(self, checker):
        self.checker = checker

    def __getattr__(self, name):
        return getattr(self.checker, name)

    def predict_mask(self, image_bgr, threshold=.5):
        return self.checker.predict_mask(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), threshold)

    def evaluate(self, image_bgr, *args, **kwargs):
        # evaluate calls the original checker's predict_mask internally. Convert
        # once here; wrapping that method too would swap the channels twice.
        return self.checker.evaluate(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), *args, **kwargs)


class FieldModels:
    """Lazy sessions shared across images, intermediate crops shared within an image."""

    def __init__(self, product: str, product_config=None):
        if product not in {'D01-L', 'D01-R'}:
            raise ValueError(f'Unsupported CAB-F product: {product}')
        self.product = product
        project_root = ROOT / 'projects/CAB-F'
        backend_path = project_root / 'config/backend.yaml'
        products_root = project_root / 'config/products'
        # Branch switches can leave ignored files/directories behind. A directory
        # alone is not evidence of an available project configuration/runtime.
        manifest_path = project_root / 'bundle_manifest.json'
        if manifest_path.is_file():
            paths = json.loads(manifest_path.read_text(encoding='utf-8'))['configuration']
            backend_path = project_root / paths['backend']
            products_root = project_root / paths['products']
        self.project_layout = (backend_path.is_file() and
                               (project_root / 'algorithms/yolo_adapter.py').is_file())
        if self.project_layout:
            product_path = products_root / f'{product}.yaml'
            self.template_root = products_root
        else:
            backend_path = ROOT / 'assets/config/backend_config.yaml'
            product_path = ROOT / f'conf/cabf/{product}.yaml'
            self.template_root = ROOT
        if product_config:
            product_path = Path(product_config).expanduser().resolve()
        self.product_config_path = product_path.resolve()
        self.config = self._read_group(backend_path, 'cab_f')
        self.inspection = self._read_group(product_path, 'inspection')
        for name, overrides in (self.inspection.get('model_params') or {}).items():
            if not isinstance(overrides, dict):
                raise ValueError(f'model_params.{name} must be a mapping')
            self.config[name] = {**self.config.get(name, {}), **overrides}
        self.config['knife_segment'] = dict(self.config['knife_checker'])
        density = self.config['sew_point_density']
        self.config['sew_point_connector'] = {
            **density, 'path': density['connect_model_path'], 'patch_model_path': density['patch_model_path']}
        cache_dir = Path(os.environ.get('WEIGHTREG_CACHE', Path.home() / '.cache/weightreg'))
        manifest_path = cache_dir / 'resolved.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        self.snapshot = {'product': product, 'product_config_path': str(self.product_config_path),
                         'product_sha256': self._hash(product_path),
                         'backend_sha256': self._hash(backend_path), 'models': {}}
        self.snapshot['input_contracts'] = INPUT_CONTRACTS if not self.project_layout else {'runtime': 'project_adapter'}
        self.snapshot['templates'] = {}
        for params in self.inspection.get('match_template', []):
            if params and params.get('path'):
                path = (self.template_root / params['path']).resolve()
                self.snapshot['templates'][str(path)] = self._hash(path) if path.is_file() else 'missing'
        # Hash actual source contents, including uncommitted changes. A Git HEAD
        # alone cannot identify the algorithms running in a dirty working tree.
        roots = [ROOT / 'algo', ROOT / 'biz', ROOT / 'utils']
        if self.project_layout:
            roots.extend([project_root / 'algorithms', ROOT / 'projects/_shared'])
        self.snapshot['runtime_sources'] = {
            str(path.relative_to(ROOT)): self._hash(path)
            for root in roots for path in sorted(root.rglob('*.py')) if path.is_file()}
        for name, config in self.config.items():
            if not isinstance(config, dict):
                continue
            model_paths = {}
            for key, ref in list(config.items()):
                if not key.endswith('path') or not isinstance(ref, str):
                    continue
                if ref.startswith('@'):
                    registry, _, tag = ref[1:].partition('#')
                    digest = manifest.get(f'{registry.removesuffix(".onnx")}@{tag or "stable"}')
                    path = cache_dir / digest if digest else None
                else:
                    path = Path(ref)
                    if not path.is_absolute():
                        path = ROOT / path
                if path is not None and path.is_file():
                    config[key] = str(path)
                    model_paths[key] = {'reference': ref, 'path': str(path), 'sha256': self._hash(path)}
                else:
                    model_paths[key] = {'reference': ref, 'missing': True}
            self.snapshot['models'][name] = {'paths': model_paths, 'config': config.copy()}
        self.sessions = {}
        self.shared_yolo = {}
        self.errors = {}
        self.cache = {}
        from .field_prediction_cache import FramePredictionCache
        self.inference_cache = FramePredictionCache()
        self._frame_image = None

    def begin_image(self, image):
        self.inference_cache.reset()
        self._frame_image = image
        self.shared_calibration = None
        self.calibration_reused = 0

    def close(self):
        # Remove instance-local callable closures before dropping the sessions;
        # otherwise closures retaining bound methods can keep ORT alive until GC.
        held = {id(model): model for model in [*self.sessions.values(), *self.shared_yolo.values()]}
        for model in held.values():
            if getattr(model, '_tool_cached_mask', False):
                del model.predict_mask
                del model._tool_cached_mask
            if getattr(model, '_tool_yolo_rgb', False):
                del model.predict
                del model._tool_raw_predict
                del model._tool_yolo_rgb
        self.sessions.clear()
        self.shared_yolo.clear()
        self.cache.clear()
        self.inference_cache.reset()
        self._frame_image = self.image = None
        self.shared_calibration = None

    @staticmethod
    def _read_group(path, group):
        document = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not isinstance(document, dict) or not isinstance(document.get(group), dict):
            raise ValueError(f'Missing or invalid {group} configuration in {path}')
        return document[group]

    def _runtime(self, legacy, modern):
        from .paths import ensure_import_paths
        ensure_import_paths()
        if not self.project_layout:
            return importlib.import_module(legacy)
        if modern.startswith('.'):
            from projects import load_project_package
            package = load_project_package('CAB-F')
            modern = package.__name__ + modern
        return importlib.import_module(modern)

    @staticmethod
    def _hash(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _model(self, name):
        if name in self.sessions:
            return self.sessions[name]
        cfg = self.config[name]
        if not Path(cfg.get('path', '')).is_file():
            raise FileNotFoundError(f'{name}: local model unavailable: {cfg.get("path")}')
        if name in CLASSES:
            task = 'cls' if name in CLASSIFIERS else ('obb' if name == 'qr_yolo_cut' else 'det')
            shared = getattr(self, 'shared_yolo', {}).get(name)
            if (shared is not None and shared.model_path == cfg['path'] and shared.task == task
                    and shared.classes == cfg.get('classes', CLASSES[name])
                    and shared.conf_thres == cfg.get('conf', .5)
                    and shared.iou_thres == cfg.get('iou', .45)
                    and shared.agnostic == cfg.get('agnostic', False)
                    and not shared.filter_classes and not cfg.get('filter_classes')
                    and shared.epsilon_factor == cfg.get('epsilon_factor', .005)):
                self.sessions[name] = shared
                return shared
            runtime = self._runtime('algo.models.yolo', '.algorithms.yolo_adapter')
            YOLO = getattr(runtime, 'CABFOnnxYolo' if self.project_layout else 'YOLO')
            model = YOLO({**cfg, 'model_path': cfg['path'], 'classes': cfg.get('classes', CLASSES[name]), 'task': task,
                          'conf_threshold': cfg.get('conf', .5), 'iou_threshold': cfg.get('iou', .45)})
        else:
            module, cls = {
                'knife_segment': ('knife_checker', 'KnifeChecker'),
                'reinforcement_placement_detector': ('reinforcement_placement_detector', 'ReinforcementPlacementDetector'),
                'sew_point_detector': ('sew_point_detector', 'SewPointDetector'),
                'sew_point_connector': ('sew_point_connector', 'SewPointConnector'),
            }.get(name, (None, None))
            if name == 'glue_segment':
                OnnxUNetSegmenter = self._runtime('algo.models.unet_segmenter',
                    'projects._shared.vision.inference').OnnxUNetSegmenter
                model = OnnxUNetSegmenter(cfg['path'])
            else:
                modern_module = 'knife' if module == 'knife_checker' else module
                model = getattr(self._runtime(f'algo.cab_f.{module}', f'.algorithms.{modern_module}'), cls)(cfg)
                if name == 'knife_segment' and hasattr(self, 'inference_cache'):
                    predict_mask = model.predict_mask
                    def cached_mask(image_bgr, threshold=.5):
                        return self.inference_cache.call('knife', {**cfg, 'threshold': threshold}, image_bgr,
                                                         lambda: predict_mask(image_bgr, threshold))
                    model.predict_mask = cached_mask
                    model._tool_cached_mask = True
        self.sessions[name] = model
        return model

    def _predict(self, name, sample):
        key = ('prediction', name, sample['metadata']['crop_key'])
        failure_key = ('prediction_error', name, sample['metadata']['crop_key'])
        if failure_key in self.cache:
            raise self.cache[failure_key]
        if key not in self.cache:
            try:
                image = sample['image']
                if not getattr(self, 'project_layout', False):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                model = self._model(name)
                if hasattr(self, 'inference_cache') and not self.project_layout:
                    from .field_prediction_cache import yolo_predict
                    self.cache[key] = yolo_predict(self.inference_cache, model, image)
                else:
                    self.cache[key] = model.predict(image)
            except Exception as exc:
                self.cache[failure_key] = exc
                raise
        return self.cache[key]

    def _sample(self, image, key, **metadata):
        return {'image': image, 'shapes': [], 'metadata': {
            'crop_key': key, 'product': self.product, 'face': self.face,
            'coordinate_space': 'source', 'annotation_status': 'unlabeled', **metadata}}

    def _crop(self, sample, box, key, fixed=None):
        cx, cy, width, height = box[:4]
        if fixed:
            width, height = fixed
        h, w = sample['image'].shape[:2]
        x1, y1 = max(0, int(cx-width/2)), max(0, int(cy-height/2))
        x2, y2 = min(w, int(cx+width/2)), min(h, int(cy+height/2))
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Empty crop')
        ancestry = list(sample['metadata'].get('transform_chain', []))
        ancestry.append({'parent': sample['metadata']['crop_key'], 'crop_xyxy': [x1, y1, x2, y2]})
        return self._sample(sample['image'][y1:y2, x1:x2], key,
                            parent=sample['metadata']['crop_key'], crop_xyxy=[x1, y1, x2, y2],
                            transform_chain=ancestry,
                            calibration=sample['metadata'].get('calibration'),
                            coordinate_space=sample['metadata'].get('coordinate_space', 'source'))

    def _roi_inputs(self):
        if 'roi_inputs' not in self.cache:
            h, w = self.image.shape[:2]
            small = cv2.resize(self.image, (max(2, int(w*.1)), max(1, int(h*.1))), interpolation=cv2.INTER_AREA)
            half = small.shape[1]//2
            self.cache['roi_inputs'] = [self._sample(small[:, x1:x2], f'roi_half_{i}',
                source_scale=[small.shape[1]/w, small.shape[0]/h], scaled_x_offset=x1)
                for i, (x1, x2) in enumerate([(0, half), (half, small.shape[1])])]
        return self.cache['roi_inputs']

    def _regions(self, label):
        key = ('regions', label)
        if key not in self.cache:
            results = []
            source = self._sample(self.image, 'source')
            for sample in self._roi_inputs():
                sx, sy = sample['metadata']['source_scale']
                for det in self._predict('roi_detector', sample):
                    if str(det['label']) != label:
                        continue
                    cx, cy, bw, bh = det['box']
                    box = [(cx+sample['metadata']['scaled_x_offset'])/sx, cy/sy, bw/sx, bh/sy]
                    params = self.inspection.get('conf', {}).get(self.face, {})
                    fixed = params.get('ear', {}).get('placement', {}).get('wh', [700, 700]) if label == 'ear' else None
                    results.append(self._crop(source, box, f'{label}_{len(results)}', fixed))
            self.cache[key] = results
        return self.cache[key]

    def _children(self, parent_label, model, label, fixed=None):
        key = ('children', parent_label, model, label)
        if key not in self.cache:
            parents = self._regions(parent_label) if parent_label == 'tail' else self._cloth()
            crops = []
            for parent in parents:
                for index, det in enumerate(self._predict(model, parent)):
                    if str(det['label']) == label:
                        crops.append(self._crop(parent, det['box'], f'{parent["metadata"]["crop_key"]}_{label}_{index}', fixed))
            self.cache[key] = crops
        return self.cache[key]

    def _cloth(self):
        return self._children('tail', 'tail_roi_detector', 'cloth')

    def _tiles(self, image, size, prefix, stride=None):
        iter_tiles = self._runtime('algo.utils', 'projects._shared.vision.ops.imaging').iter_tiles
        h, w = image.shape[:2]
        for x, y in iter_tiles(h, w, size, stride or size):
            tile = image[y:y+size, x:x+size]
            th, tw = tile.shape[:2]
            if th != size or tw != size:
                tile = cv2.copyMakeBorder(tile, 0, size-th, 0, size-tw, cv2.BORDER_CONSTANT)
            yield self._sample(tile, f'{prefix}_{x}_{y}', crop_xyxy=[x, y, x+tw, y+th], valid_size=[tw, th])

    def _calibrated(self):
        if 'calibration_error' in self.cache:
            raise self.cache['calibration_error']
        try:
            return self._calibrate_once()
        except Exception as exc:
            self.cache['calibration_error'] = exc
            raise

    def _calibrate_once(self):
        if 'calibrated' not in self.cache:
            shared = getattr(self, 'shared_calibration', None)
            if shared is not None:
                params = dict(self.inspection['match_template'][0 if self.face == 'top' else 1])
                params['path'] = str((self.template_root / params['path']).resolve())
                digest = hashlib.sha256(memoryview(np.ascontiguousarray(self.image)).cast('B')).hexdigest()
                if params == shared['params'] and digest == shared['pixels']:
                    meta = shared['metadata']
                    if meta.get('glue_dice', 0) > .5:
                        self.cache['calibrated'] = shared['transform'](
                            self.image, meta['offset'], meta['angle'], scale=meta.get('scale', 1.0), border_value=0)
                        self.cache['calibration_metadata'] = dict(meta)
                        self.calibration_reused += 1
                        return self.cache['calibrated']
            GlueExtractor = self._runtime('algo.cab_f.glue_extraction', '.algorithms.glue_extraction').GlueExtractor
            CADMatcher = self._runtime('algo.cab_f.match_glue', '.algorithms.match_glue').CADMatcher
            calibrated_pic = self._runtime('algo.cab_f.run_calibration', '.algorithms.calibration').calibrated_pic
            model = self._model('glue_segment')
            # Reuse the existing segmentation session rather than loading twice.
            extractor = object.__new__(GlueExtractor)
            extractor.model_path = self.config['glue_segment']['path']
            extractor.segmenter = _ToolGlueRGBAdapter(model)
            extractor.thresh = self.config['glue_segment'].get('conf', .9)
            params = dict(self.inspection['match_template'][0 if self.face == 'top' else 1])
            params['path'] = str((self.template_root / params['path']).resolve())
            calibrated, _, meta = calibrated_pic(self.image, params, extractor, CADMatcher())
            if calibrated is None or not meta or meta['glue_dice'] < .5:
                raise ValueError('Calibration failed or glue Dice below 0.5; fixed ROI not exported')
            self.cache['calibrated'] = calibrated
            self.cache['calibration_metadata'] = {k: v for k, v in meta.items() if not isinstance(v, np.ndarray)}
        return self.cache['calibrated']

    def _inputs(self, name):
        if name == 'roi_detector':
            return self._roi_inputs()
        if name in {'ear_placement_classifier', 'knife_segment'}:
            return self._regions('ear')
        if name == 'tail_roi_detector':
            return self._regions('tail')
        if name == 'tail_cloth_roi_detector':
            return self._cloth()
        if name == 'tail_placement_classifier':
            params = self.inspection.get('conf', {}).get(self.face, {})
            fixed = params.get('tail', {}).get('placement', {}).get('wh', [350, 350])
            return self._children('tail', 'tail_roi_detector', 'placement', fixed)
        if name == 'hook_detector':
            return self._children('tail', 'tail_roi_detector', 'hook')
        if name == 'tail_cloth_seam_classifier':
            return self._children('cloth', 'tail_cloth_roi_detector', 'seam')
        if name == 'qr_yolo_cut':
            params = self.inspection['conf'].get(self.face, {}).get('decode_image', {})
            roi = params.get('rois')
            if not roi or len(roi) != 4 or not all(isinstance(v, (int, float)) for v in roi):
                raise ValueError(f'{self.product}/{self.face}: QR ROI not configured')
            image = self._calibrated()
            x1, y1, x2, y2 = roi
            source = self._sample(image, 'calibrated', coordinate_space='calibrated',
                                  calibration=self.cache['calibration_metadata'])
            return [self._crop(source, [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], 'qr')]
        if name == 'glue_segment':
            half = cv2.resize(self.image, (0, 0), fx=.5, fy=.5)
            return [dict(sample, metadata={**sample['metadata'], 'source_scale': [.5, .5]})
                    for sample in self._tiles(half, 256, 'glue')]
        if name == 'error_detector':
            config = self.config[name]
            return list(self._tiles(self.image, int(config.get('tile_size', 1024)), 'error',
                                    int(config.get('tile_stride', 896))))
        if name in {'sew_point_detector', 'sew_point_connector'}:
            # Explicit tail-cloth scope, preserved in each exported sample.
            return [self._sample(tile['image'], f'{cloth["metadata"]["crop_key"]}_{tile["metadata"]["crop_key"]}',
                parent=cloth['metadata']['crop_key'], crop_xyxy=tile['metadata']['crop_xyxy'], roi_scope='tail_cloth',
                valid_size=tile['metadata']['valid_size'],
                transform_chain=[*cloth['metadata'].get('transform_chain', []),
                    {'parent': cloth['metadata']['crop_key'], 'crop_xyxy': tile['metadata']['crop_xyxy']}])
                for cloth in self._cloth() for tile in self._tiles(cloth['image'], 256, 'stitch')]
        if name == 'reinforcement_placement_detector':
            params = self.inspection['conf'].get(self.face, {}).get('reinforcement')
            if not params or 'roi' not in params:
                raise ValueError(f'{self.product}/{self.face}: reinforcement ROI not configured')
            image = self._calibrated()
            x1, y1, x2, y2 = params['roi']
            source = self._sample(image, 'calibrated', coordinate_space='calibrated')
            sample = self._crop(source, [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], 'reinforcement')
            sample['metadata']['calibration'] = self.cache['calibration_metadata']
            return [sample]
        raise ValueError(f'Unknown dataset model: {name}')

    @staticmethod
    def _simplify_contour(contour, epsilon_px=1.0, min_iou=.98):
        """Reduce vertices without removing objects; compare to the old polygon.

        Raster IoU protects thin/small features from a perimeter-only tolerance.
        Coordinates are crop pixels, independent of preview scaling. Masks stay
        lossless; this only changes the editable polygon representation.
        """
        if epsilon_px <= 0 or not 0 < min_iou <= 1:
            raise ValueError('Invalid polygon simplification tolerance')
        if len(contour) <= 4 or cv2.contourArea(contour) <= 0:
            return contour
        x, y, width, height = cv2.boundingRect(contour)
        offset = np.array([[[x, y]]], dtype=contour.dtype)
        reference = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(reference, [contour - offset], 1)
        candidate_mask = np.zeros_like(reference)
        for tolerance in (epsilon_px, epsilon_px / 2, epsilon_px / 4, epsilon_px / 8):
            candidate = cv2.approxPolyDP(contour, tolerance, True)
            if len(candidate) < 3 or len(candidate) >= len(contour) or cv2.contourArea(candidate) <= 0:
                continue
            candidate_mask.fill(0)
            cv2.fillPoly(candidate_mask, [candidate - offset], 1)
            union = np.count_nonzero(reference | candidate_mask)
            overlap = np.count_nonzero(reference & candidate_mask)
            if union and overlap / union >= min_iou:
                return candidate
        return contour

    @staticmethod
    def _mask_shapes(masks):
        shapes = []
        for label, mask in masks.items():
            contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = FieldModels._simplify_contour(contour).reshape(-1, 2)
                if len(points) >= 3:
                    shapes.append(shape(label, points, 'polygon'))
        return shapes

    def _annotate(self, name, sample):
        image = sample['image']
        if name in CLASSES:
            result = self._predict(name, sample)
            if name in CLASSIFIERS:
                sample['classification'] = str(result['label'])
                sample['metadata']['score'] = float(result['score'])
            else:
                h, w = image.shape[:2]
                for det in result:
                    cx, cy, bw, bh = det['box'][:4]
                    if name == 'qr_yolo_cut':
                        pts = cv2.boxPoints(((cx, cy), (bw, bh), np.degrees(det['box'][4])))
                        pts[:, 0] = np.clip(pts[:, 0], 0, w)
                        pts[:, 1] = np.clip(pts[:, 1], 0, h)
                        sample['shapes'].append(shape(det['label'], pts, 'polygon', float(det['score'])))
                        sample['metadata']['yolo_task'] = 'obb'
                    else:
                        pts = [[max(0, cx-bw/2), max(0, cy-bh/2)], [min(w, cx+bw/2), min(h, cy+bh/2)]]
                        if pts[1][0] > pts[0][0] and pts[1][1] > pts[0][1]:
                            sample['shapes'].append(shape(det['label'], pts, score=float(det['score'])))
        elif name in {'glue_segment', 'knife_segment', 'reinforcement_placement_detector'}:
            model = self._model(name)
            if name == 'knife_segment':
                params = self.inspection.get('conf', {}).get(self.face, {}).get('knife', {})
                masks = _ToolKnifeRGBAdapter(model).predict_mask(
                    image, params.get('threshold', self.config[name].get('threshold', .5)))
            elif name == 'glue_segment':
                # Only this tool converts its OpenCV crops to the model's RGB input.
                masks = {'glue': model.predict_mask(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                                    self.config[name].get('conf', .9))}
            else:
                masks = {'reinforcement': model.detect(image)}
            masks = {label: cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                     for label, mask in masks.items()}
            sample['masks'] = masks
            sample['shapes'] = self._mask_shapes(masks)
            sample['metadata']['polygon_simplification'] = {
                'method': 'approxPolyDP', 'max_epsilon_px': 1.0,
                'min_raster_iou': .98, 'reference': 'original_external_contour',
                'mask_modified': False}
        else:
            point_key = ('points', sample['metadata']['crop_key'])
            if point_key not in self.cache:
                self.cache[point_key] = self._model('sew_point_detector').detect(image)
            points = self.cache[point_key]
            annotation = {'points': [{'id': i, 'x': float(p[0]), 'y': float(p[1]), 'score': float(p[2])}
                                     for i, p in enumerate(points)], 'edges': []}
            sample['shapes'] = [shape('point', [[p['x'], p['y']]], 'point', p['score'], p['id']) for p in annotation['points']]
            if name == 'sew_point_connector':
                annotation['edges'] = self._model(name).predict(annotation, image_bgr=image)
                for edge in annotation['edges']:
                    p, q = annotation['points'][edge['src']], annotation['points'][edge['dst']]
                    sample['shapes'].append(shape('connection', [[p['x'], p['y']], [q['x'], q['y']]], 'line', edge['score']))
            sample['metadata']['graph'] = annotation
        sample['metadata']['annotation_status'] = 'review'
        sample['metadata']['label_source'] = 'model'
        return sample

    def generate(self, image: np.ndarray, selected: list[str], mode: str = 'auto', face: str = 'top') -> dict:
        if mode not in {'auto', 'images'}:
            raise ValueError(f'Unsupported generation mode: {mode}')
        if face not in {'top', 'bottom'}:
            raise ValueError(f'Unsupported face: {face}')
        if hasattr(self, 'inference_cache') and self._frame_image is not image:
            self.begin_image(image)
        self.face, self.image, self.cache, self.errors = face, image, {}, {}
        self.review_reasons = {}
        output = {}
        for name in selected:
            try:
                inputs = self._inputs(name)
                if not inputs:
                    self.review_reasons[name] = '未检测到所需 ROI，无法生成裁片；请复核上游定位，不自动作为负样本。'
                    continue
                output[name] = []
                for item in inputs:
                    # Intermediate cache entries must never receive target labels.
                    sample = {**item, 'shapes': [], 'metadata': dict(item['metadata'])}
                    output[name].append(self._annotate(name, sample) if mode == 'auto' else sample)
            except Exception as exc:
                self.errors[name] = f'{type(exc).__name__}: {exc}'
                output.pop(name, None)
        self.image, self.cache = None, {}
        self._frame_image = None
        return output
