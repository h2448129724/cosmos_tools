"""Bounded per-image inference reuse; equality includes pixels and model settings."""
import copy
import hashlib
import json
from collections import OrderedDict

import numpy as np


class FramePredictionCache:
    def __init__(self, max_bytes=64 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.reset()

    def reset(self):
        self.entries = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    @staticmethod
    def size(value):
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(FramePredictionCache.size(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(FramePredictionCache.size(v) for v in value)
        return 64

    def call(self, kind, settings, image, compute):
        pixels = np.ascontiguousarray(image)
        digest = hashlib.sha256(memoryview(pixels).cast('B')).hexdigest()
        key = (kind, json.dumps(settings, sort_keys=True, default=str), pixels.shape, pixels.dtype.str, digest)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return copy.deepcopy(self.entries[key][0])
        self.misses += 1
        result = compute()
        size = self.size(result)
        if size <= self.max_bytes:
            while self.entries and self.bytes + size > self.max_bytes:
                _, (_, removed) = self.entries.popitem(last=False)
                self.bytes -= removed
            self.entries[key] = (copy.deepcopy(result), size)
            self.bytes += size
        return result


def yolo_predict(cache, model, rgb):
    # Subclasses with different preprocessing deliberately do not share entries.
    preprocess = type(model).preprocess
    settings = {'path': model.model_path, 'classes': model.classes, 'task': model.task,
                'conf': model.conf_thres, 'iou': model.iou_thres,
                'agnostic': model.agnostic, 'filter': model.filter_classes,
                'input_shape': list(model.input_shape), 'epsilon': model.epsilon_factor,
                'preprocess': f'{preprocess.__module__}.{preprocess.__qualname__}'}
    predict = getattr(model, '_tool_raw_predict', model.predict)
    return cache.call('yolo', settings, rgb, lambda: predict(rgb))
