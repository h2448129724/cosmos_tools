from pathlib import Path

from modules.yolo.scripts import export_onnx


def test_move_export_result_uses_shutil_move(monkeypatch, tmp_path: Path):
    calls = []

    def fake_move(src: str, dst: str):
        calls.append((src, dst))

    result_path = tmp_path / "best.onnx"
    out_path = tmp_path / "runs" / "model.onnx"

    monkeypatch.setattr(export_onnx.shutil, "move", fake_move)

    export_onnx.move_export_result(result_path, out_path)

    assert calls == [(str(result_path), str(out_path))]
