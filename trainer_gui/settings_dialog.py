from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QVBoxLayout,
)

from .settings_manager import AppSettings


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.resize(420, 220)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.default_project_name_edit = QLineEdit()
        self.default_project_name_edit.setText(settings.default_project_name)
        self.default_project_name_edit.setPlaceholderText("为空时继续使用各模块自己的默认项目名")

        self.command_platform_combo = QComboBox()
        self.command_platform_combo.addItem("Linux", "linux")
        self.command_platform_combo.addItem("Windows", "windows")
        selected = self.command_platform_combo.findData(settings.default_command_platform or "linux")
        self.command_platform_combo.setCurrentIndex(selected if selected >= 0 else 0)

        self.remember_last_selection_check = QCheckBox("记住上次选择的功能 / 子功能 / Conda 环境")
        self.remember_last_selection_check.setChecked(settings.remember_last_selection)

        form.addRow("默认项目名", self.default_project_name_edit)
        form.addRow("默认命令格式", self.command_platform_combo)
        form.addRow("", self.remember_last_selection_check)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> AppSettings:
        return AppSettings(
            default_project_name=self.default_project_name_edit.text().strip(),
            default_command_platform=str(self.command_platform_combo.currentData() or "linux"),
            remember_last_selection=self.remember_last_selection_check.isChecked(),
        )
