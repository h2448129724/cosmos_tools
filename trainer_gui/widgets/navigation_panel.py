"""导航面板：模块列表与项目选择。"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import FONT_SIZE_SECTION


class NavigationPanel(QFrame):
    """左侧模块导航面板。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("navPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 12)

        nav_title = QLabel("训练功能")
        nav_title.setObjectName("titleLabel")
        nav_title.setStyleSheet(f"font-size: {FONT_SIZE_SECTION}px;")

        nav_hint = QLabel("选择模块后配置并运行。")
        nav_hint.setObjectName("mutedLabel")
        nav_hint.setWordWrap(True)

        self.feature_list = QListWidget()

        layout.addWidget(nav_title)
        layout.addWidget(nav_hint)
        layout.addWidget(self.feature_list)


class BasicConfigGroup(QGroupBox):
    """基础配置区：子功能、项目名、Conda 环境、命令格式。"""

    def __init__(self, default_platform: str = "linux", parent: QWidget | None = None) -> None:
        super().__init__("基础配置", parent)
        form = QFormLayout(self)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.action_combo = QComboBox()
        self.project_name_edit = QLineEdit()
        self.project_name_edit.setPlaceholderText("项目名，不填则使用 default")
        self.conda_env_combo = QComboBox()
        self.command_platform_combo = QComboBox()
        self.settings_button = QPushButton("设置")
        self.settings_button.setProperty("buttonRole", "secondary")

        for widget in [self.project_name_edit, self.action_combo, self.conda_env_combo, self.command_platform_combo]:
            widget.setMinimumHeight(38)

        self.command_platform_combo.addItem("Linux", "linux")
        self.command_platform_combo.addItem("Windows", "windows")
        default_index = self.command_platform_combo.findData(default_platform or "linux")
        self.command_platform_combo.setCurrentIndex(default_index if default_index >= 0 else 0)

        form.addRow("子功能", self.action_combo)
        form.addRow("项目名", self.project_name_edit)
        form.addRow("Conda 环境", self.conda_env_combo)
        form.addRow("命令格式", self.command_platform_combo)
        form.addRow("", self.settings_button)
