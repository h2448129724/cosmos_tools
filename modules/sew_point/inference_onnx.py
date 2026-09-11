"""
ONNX 推理（推荐用于部署）

用法：
    python sew_point/inference_onnx.py --image test.png --model best.onnx
"""

import argparse
import os

import cv2
import numpy as np


import onnxruntime as ort

try:
    from .utils import _imread, _imwrite
    from .inference_core import apply_tta, combine_tta_heatmaps, detect_peaks, tta_plan
except ImportError:
    from utils import _imread, _imwrite
    from inference_core import apply_tta, combine_tta_heatmaps, detect_peaks, tta_plan


class KeypointDetectorONNX:
    """ONNX 关键点检测器。"""

    def __init__(self, model_path=None, device="cpu", threshold=0.5, cluster_dist=3):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")

        self.threshold = threshold
        self.cluster_dist = cluster_dist

        providers = ["CPUExecutionProvider"]
        if device == "cuda" or (device is None and "CUDAExecutionProvider" in ort.get_available_providers()):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) >= 4 and all(isinstance(value, int) for value in input_shape[2:4]):
            self._model_shape = (int(input_shape[2]), int(input_shape[3]))
        else:
            self._model_shape = None

        provider_name = self.session.get_providers()[0]
        print(f"[INFO] ONNX provider: {provider_name}")

    # ── 预处理 ──

    def _preprocess(self, img_bgr):
        """单图预处理 → (1, C, H, W)。"""
        img = img_bgr.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))[np.newaxis]

    def _preprocess_batch(self, images_bgr):
        """批量预处理 → (B, C, H, W)。"""
        batch = [np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
                 for img in images_bgr]
        return np.stack(batch, axis=0)

    # ── 推理 ──

    def _inference(self, img_np):
        inp = self._preprocess(img_np)
        return self.session.run([self.output_name], {self.input_name: inp})[0]

    def _inference_batch(self, images_bgr):
        inp = self._preprocess_batch(images_bgr)
        return self.session.run([self.output_name], {self.input_name: inp})[0]

    def _inference_tta(self, img_np):
        """TTA: original + 4 翻转 + 3 旋转，共 7 路。"""
        h, w = img_np.shape[:2]
        heatmaps = []
        plan = tta_plan(h, w, model_shape=self._model_shape)
        for path in plan:
            transformed = apply_tta(img_np, path).copy()
            if path.model_shape != path.native_shape:
                transformed = cv2.resize(transformed, (path.model_shape[1], path.model_shape[0]))
            hm = self._inference(transformed).squeeze()
            heatmaps.append(hm)
        return combine_tta_heatmaps(heatmaps, (h, w), plan=plan, resize_fn=cv2.resize)

    # ── 检测接口 ──

    def detect_numpy(self, img_bgr, use_tta=True):
        """从 BGR 数组检测。返回 [(x, y, score), ...]。"""
        hm = self._inference_tta(img_bgr) if use_tta else self._inference(img_bgr)
        return detect_peaks(hm, self.threshold, self.cluster_dist)

    def detect_batch_numpy(self, images_bgr):
        """批量检测（无 TTA）。返回 list of [(x, y, score), ...]。"""
        if not images_bgr:
            return []
        heatmaps = self._inference_batch(images_bgr)
        return [detect_peaks(heatmaps[i], self.threshold, self.cluster_dist)
                for i in range(len(images_bgr))]

    def detect(self, image_path, use_tta=True):
        """从文件路径检测。"""
        img = _imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect_numpy(img, use_tta)

    def detect_and_visualize(self, image_path, output_path=None, use_tta=True):
        """检测并保存可视化。"""
        img = _imread(image_path)
        hm = self._inference_tta(img) if use_tta else self._inference(img)
        peaks = detect_peaks(hm, self.threshold, self.cluster_dist)

        vis = img.copy()
        for x, y, s in peaks:
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

        hm_color = cv2.applyColorMap(
            (hm.squeeze() * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        for x, y, s in peaks:
            cv2.circle(hm_color, (x, y), 3, (255, 255, 255), -1)

        if output_path is None:
            output_path = os.path.splitext(image_path)[0] + "_pred_onnx.png"
        _imwrite(output_path, np.hstack([vis, hm_color]))
        print(f"[SAVE] {output_path}")
        return peaks


def main():
    parser = argparse.ArgumentParser(description="ONNX Keypoint Detection")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cluster_dist", type=float, default=3)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.onnx")

    detector = KeypointDetectorONNX(model_path=model_path, device=args.device,
                                    threshold=args.threshold, cluster_dist=args.cluster_dist)
    peaks = detector.detect_and_visualize(args.image, args.output, use_tta=not args.no_tta)
    print(f"[DETECT] {len(peaks)} keypoints:")
    for x, y, s in peaks:
        print(f"  ({x}, {y}) score={s:.3f}")


if __name__ == "__main__":
    main()
