import numpy as np

from modules.segmentation.tools import predict_single_image as prediction


class _TensorMeta:
    def __init__(self, name: str):
        self.name = name


class _FakeOnnxSession:
    def __init__(self):
        self.last_input = None

    def get_inputs(self):
        return [_TensorMeta("input")]

    def get_outputs(self):
        return [_TensorMeta("logits")]

    def run(self, output_names, inputs):
        assert output_names == ["logits"]
        self.last_input = inputs["input"]
        batch, _, height, width = self.last_input.shape
        logits = np.zeros((batch, 2, height, width), dtype=np.float32)
        logits[:, 1] = 2.0
        return [logits]


def test_model_format_dispatches_onnx_and_pytorch():
    assert prediction._model_format("model.onnx") == "onnx"
    assert prediction._model_format("model.pth") == "pytorch"


def test_tiled_inference_accepts_onnx_backend(monkeypatch):
    session = _FakeOnnxSession()
    monkeypatch.setattr(
        prediction,
        "_load_inference_backend",
        lambda model_path, model_name: ("onnx", session, None),
    )

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = prediction.run_inference_tiled(
        image_bgr=image,
        model_path="model.onnx",
        tile_size=8,
        center_size=8,
        batch_size=1,
        thresh=0.8,
    )

    assert session.last_input.shape == (1, 3, 8, 8)
    assert session.last_input.dtype == np.float32
    assert np.all(mask == 255)
