from img_tools.core.models import Roi
from img_tools.core.presets import load_roi_preset, save_roi_preset


def test_roi_preset_round_trip(tmp_path):
    path = tmp_path / "常用ROI.json"
    rois = [Roi(1, 2, 30, 40, "左侧"), Roi(5, 6, 7, 8, "右侧")]

    save_roi_preset(path, rois, reference_size=(100, 200))
    loaded, reference = load_roi_preset(path)

    assert loaded == rois
    assert reference == (100, 200)
