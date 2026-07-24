from __future__ import annotations

import inspect

from PySide6.QtWidgets import QApplication, QDialog, QFormLayout, QLineEdit, QWidget

from trainer_gui.main_window import MainWindow
from trainer_gui.models import FieldSchema
from trainer_gui.theme import build_app_stylesheet
from trainer_gui.widgets.form_builder import FormBuilder


class _RouterSpy:
    def __init__(self):
        self.dialogs = []

    def open_dialog(self, dialog, **kwargs):
        self.dialogs.append((dialog, kwargs))
        return dialog


def get_qapp():
    return QApplication.instance() or QApplication([])


def test_trainer_theme_does_not_paint_every_child_widget():
    assert "QMainWindow, QWidget {" not in build_app_stylesheet()


def test_text_details_are_embedded_when_router_is_bound(tmp_path):
    get_qapp()
    window = MainWindow(tmp_path)
    router = _RouterSpy()
    window.bind_workspace_router(router)

    window._show_text_dialog("完整日志", "hello")

    dialog, metadata = router.dialogs[0]
    assert isinstance(dialog, QDialog)
    assert metadata["source_key"] == "training"
    assert metadata["title"] == "完整日志"
    window.close()


def test_settings_apply_only_after_embedded_accept(tmp_path):
    get_qapp()
    window = MainWindow(tmp_path)
    router = _RouterSpy()
    window.bind_workspace_router(router)

    window._open_settings()
    dialog, metadata = router.dialogs[0]
    dialog.default_project_name_edit.setText("embedded")
    metadata["on_finished"](int(QDialog.DialogCode.Accepted))

    assert window.settings.default_project_name == "embedded"
    window.close()


def test_command_generation_has_no_secondary_dialog():
    source = inspect.getsource(MainWindow._generate_command)

    assert "CommandDialog" not in source
    assert "right_tabs.setCurrentIndex" in source


def test_required_fields_are_neutral_until_validation():
    get_qapp()
    host = QWidget()
    layout = QFormLayout(host)
    builder = FormBuilder(host)
    field = FieldSchema(name="image_dir", cli_flag="--image_dir", label="图片目录", required=True, widget="path")

    builder.build(layout, [field])
    target = builder._widgets["image_dir"]._value_widget
    assert "#ef4444" not in target.styleSheet()

    assert builder.validate_visible() == ["图片目录: 必填字段为空"]
    assert "#ef4444" in target.styleSheet()


def test_required_field_validates_after_user_finishes_editing():
    get_qapp()
    host = QWidget()
    layout = QFormLayout(host)
    builder = FormBuilder(host)
    field = FieldSchema(name="project", cli_flag="--project", label="项目", required=True, widget="text")

    builder.build(layout, [field])
    target = builder._widgets["project"]
    assert isinstance(target, QLineEdit)
    assert "#ef4444" not in target.styleSheet()

    target.editingFinished.emit()
    assert "#ef4444" in target.styleSheet()


def test_embedded_status_card_does_not_expand_to_fill_right_column(tmp_path):
    get_qapp()
    window = MainWindow(tmp_path)
    window.bind_workspace_router(_RouterSpy())

    assert window.run_status_panel.sizePolicy().verticalPolicy().name == "Maximum"
    assert window.run_status_panel.maximumHeight() < 16777215
    assert window.minimumWidth() == 0
    window.close()
