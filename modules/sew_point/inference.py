"""
PyTorch 推理（调试用，部署推荐用 inference_onnx.py）

用法：
    python sew_point/inference.py --image test.png --model best.pth
"""

import argparse
import os

import cv2
import numpy as np
import torch

try:
    from .model_registry import DEFAULT_MODEL, get_model, model_choices
    from .utils import _imread, _imwrite
    from .inference_core import apply_tta, combine_tta_heatmaps, decode_checkpoint, detect_peaks, tta_plan
except ImportError:
    from model_registry import DEFAULT_MODEL, get_model, model_choices
    from utils import _imread, _imwrite
    from inference_core import apply_tta, combine_tta_heatmaps, decode_checkpoint, detect_peaks, tta_plan


class KeypointDetector:
    """PyTorch 关键点检测器。"""

    def __init__(self, model_path=None, device=None, threshold=0.5, cluster_dist=3, model_name=DEFAULT_MODEL):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pth")
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.threshold = threshold
        self.cluster_dist = cluster_dist
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        decoded = decode_checkpoint(checkpoint, model_name)
        model_name = decoded.model_name or model_name
        state = decoded.state
        self.model = get_model(model_name).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def _predict_heatmap(self, img_bgr, use_tta=True):
        h, w = img_bgr.shape[:2]

        def to_tensor(img):
            return torch.from_numpy(
                np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
            ).unsqueeze(0).to(self.device)

        def infer(t):
            with torch.no_grad():
                return self.model(t).cpu().numpy()[0, 0]

        if not use_tta:
            return infer(to_tensor(img_bgr))[np.newaxis]

        heatmaps = []
        plan = tta_plan(h, w)
        for path in plan:
            transformed = apply_tta(img_bgr, path).copy()
            if path.model_shape != path.native_shape:
                transformed = cv2.resize(transformed, (path.model_shape[1], path.model_shape[0]))
            hm = infer(to_tensor(transformed))
            heatmaps.append(hm)
        return combine_tta_heatmaps(heatmaps, (h, w), plan=plan, resize_fn=cv2.resize)

    def detect(self, image_path, use_tta=True):
        img_bgr = _imread(image_path)
        hm = self._predict_heatmap(img_bgr, use_tta)
        return detect_peaks(hm, self.threshold, self.cluster_dist)

    def detect_numpy(self, img_bgr, use_tta=True):
        hm = self._predict_heatmap(img_bgr, use_tta)
        return detect_peaks(hm, self.threshold, self.cluster_dist)

    def detect_and_visualize(self, image_path, output_path=None, use_tta=True):
        img = _imread(image_path)
        hm = self._predict_heatmap(img, use_tta)
        peaks = detect_peaks(hm, self.threshold, self.cluster_dist)

        vis = img.copy()
        for x, y, s in peaks:
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)

        hm_color = cv2.applyColorMap(
            (hm.squeeze() * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        for x, y, s in peaks:
            cv2.circle(hm_color, (x, y), 3, (255, 255, 255), -1)

        if output_path is None:
            output_path = os.path.splitext(image_path)[0] + "_pred.png"
        _imwrite(output_path, np.hstack([vis, hm_color]))
        return peaks


def main():
    parser = argparse.ArgumentParser(description="PyTorch Keypoint Detection")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image_dir", type=str, help="Batch: directory of images")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL, choices=model_choices())
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cluster_dist", type=float, default=3)
    args = parser.parse_args()

    det = KeypointDetector(model_path=args.model, threshold=args.threshold,
                           cluster_dist=args.cluster_dist, model_name=args.model_name)

    if args.image:
        pts = det.detect_and_visualize(args.image)
        print(f"Detected {len(pts)} keypoints:")
        for x, y, s in pts:
            print(f"  ({x}, {y}) score={s:.3f}")
    elif args.image_dir:
        import glob
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        imgs = sorted(glob.glob(os.path.join(args.image_dir, "*.bmp")) +
                      glob.glob(os.path.join(args.image_dir, "*.png")) +
                      glob.glob(os.path.join(args.image_dir, "*.jpg")))
        for p in imgs:
            name = os.path.splitext(os.path.basename(p))[0]
            out = os.path.join(args.output_dir or args.image_dir, name + "_pred.png")
            pts = det.detect_and_visualize(p, out)
            print(f"  {name}: {len(pts)} pts")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
