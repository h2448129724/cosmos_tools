from types import SimpleNamespace

from apps.labeling_ui.app.tools.roi_editor_page import RoiConfigEditorPage, _collect_roi_fields


def test_collects_empty_cabf_roi_as_multi_field():
    data = {
        "inspection": {
            "project": "CAB-F",
            "conf": {"bottom": {"density": {"dense_stitch": {"roi": []}}}},
        }
    }

    fields = _collect_roi_fields(data)

    assert [field.path_key for field in fields] == ["inspection.conf.bottom.density.dense_stitch.roi"]
    assert fields[0].is_multi is True


def test_does_not_change_empty_roi_discovery_for_other_projects():
    data = {
        "inspection": {
            "project": "OS-DAB",
            "conf": {"bottom": {"check": {"roi": []}}},
        }
    }

    assert _collect_roi_fields(data) == []


def test_cabf_does_not_collect_legacy_rois_field():
    data = {
        "inspection": {
            "project": "CAB-F",
            "conf": {"top": {"decode_image": {"rois": [10, 20, 110, 120]}}},
        }
    }

    assert _collect_roi_fields(data) == []


def test_side_change_keeps_badge_and_canvas_in_sync():
    page = RoiConfigEditorPage(SimpleNamespace())
    page._config_data = {
        "inspection": {
            "conf": {
                "top": {"decode": {"roi": [1, 2, 11, 12]}},
                "bottom": {"decode": {"roi": [101, 102, 111, 112]}},
            }
        }
    }
    page._roi_fields = _collect_roi_fields(page._config_data)
    page._refresh_field_list()

    page._side_combo.setCurrentIndex(1)

    assert page._side_combo.currentData() == "bottom"
    assert page._current_image_side == "bottom"
    assert page._side_badge.text() == "BOTTOM"
    assert page._preview.get_roi_rects() == [(101, 102, 111, 112)]
