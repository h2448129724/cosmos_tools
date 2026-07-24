"""运行状态面板：当前任务状态卡，主次信息分层展示。"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import FONT_SIZE_SECTION, TEXT_PRIMARY, status_badge_stylesheet


class RunStatusPanel(QFrame):
    """当前任务状态卡。

    主要信息直接展示：状态、Run ID、动作、开始时间、耗时、退出码、输出目录。
    次要信息放入可折叠的详情区：cwd、python、conda env、repo root、command、artifacts。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # --- 标题行 ---
        header = QHBoxLayout()
        title = QLabel("当前任务状态")
        title.setStyleSheet(f"font-size: {FONT_SIZE_SECTION}px; font-weight: 700; color: {TEXT_PRIMARY};")
        self.status_badge = QLabel("未开始")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setStyleSheet(status_badge_stylesheet("idle"))
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_badge)

        # --- 主要信息 ---
        primary_form = QFormLayout()
        primary_form.setHorizontalSpacing(18)
        primary_form.setVerticalSpacing(6)
        primary_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.status_run_id = QLabel("-")
        self.status_action = QLabel("-")
        self.status_start = QLabel("-")
        self.status_duration = QLabel("-")
        self.status_exit = QLabel("-")
        self.status_output = QLabel("-")
        self.status_output.setWordWrap(True)

        primary_form.addRow("Run ID", self.status_run_id)
        primary_form.addRow("子功能", self.status_action)
        primary_form.addRow("开始时间", self.status_start)
        primary_form.addRow("耗时", self.status_duration)
        primary_form.addRow("退出码", self.status_exit)
        primary_form.addRow("输出目录", self.status_output)

        # --- 详情折叠区（次要信息）---
        self._detail_toggle = QToolButton()
        self._detail_toggle.setCheckable(True)
        self._detail_toggle.setChecked(False)
        self._detail_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._detail_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._detail_toggle.setText("详情")
        self._detail_toggle.setStyleSheet("font-size: 12px; color: #64748b;")
        self._detail_toggle.clicked.connect(self._toggle_detail)

        self._detail_widget = QWidget()
        detail_form = QFormLayout(self._detail_widget)
        detail_form.setContentsMargins(12, 4, 0, 4)
        detail_form.setHorizontalSpacing(18)
        detail_form.setVerticalSpacing(4)
        detail_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.status_project = QLabel("-")
        self.status_feature = QLabel("-")
        self.status_end = QLabel("-")
        self.status_artifacts = QLabel("-")
        self.status_artifacts.setWordWrap(True)
        self.status_cwd = QLabel("-")
        self.status_cwd.setWordWrap(True)
        self.status_python = QLabel("-")
        self.status_conda = QLabel("-")
        self.status_conda.setWordWrap(True)
        self.status_repo_root = QLabel("-")
        self.status_repo_root.setWordWrap(True)
        self.status_command = QLabel("-")
        self.status_command.setWordWrap(True)
        self.status_command.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        detail_form.addRow("项目名", self.status_project)
        detail_form.addRow("功能", self.status_feature)
        detail_form.addRow("结束时间", self.status_end)
        detail_form.addRow("结果目录", self.status_artifacts)
        detail_form.addRow("工作目录", self.status_cwd)
        detail_form.addRow("Python", self.status_python)
        detail_form.addRow("Conda", self.status_conda)
        detail_form.addRow("仓库根", self.status_repo_root)
        detail_form.addRow("完整命令", self.status_command)

        self._detail_widget.setVisible(False)
        self._detail_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        # --- 操作按钮 ---
        buttons = QGridLayout()
        buttons.setHorizontalSpacing(6)
        buttons.setVerticalSpacing(6)
        self.open_output_button = QPushButton("打开输出目录")
        self.open_output_button.setProperty("buttonRole", "ghost")
        self.copy_output_button = QPushButton("复制输出路径")
        self.copy_output_button.setProperty("buttonRole", "ghost")
        self.copy_command_button = QPushButton("复制命令")
        self.copy_command_button.setProperty("buttonRole", "ghost")
        self.view_full_log_button = QPushButton("查看完整日志")
        self.view_full_log_button.setProperty("buttonRole", "ghost")
        buttons.addWidget(self.open_output_button, 0, 0)
        buttons.addWidget(self.copy_output_button, 0, 1)
        buttons.addWidget(self.copy_command_button, 1, 0)
        buttons.addWidget(self.view_full_log_button, 1, 1)

        layout.addLayout(header)
        layout.addLayout(primary_form)
        layout.addWidget(self._detail_toggle)
        layout.addWidget(self._detail_widget)
        layout.addLayout(buttons)

    def set_compact_mode(self, compact: bool) -> None:
        """Keep the card at content height when the surrounding shell hides the log."""
        if compact:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
            self.setMaximumHeight(760 if self._detail_toggle.isChecked() else 360)
        else:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self.setMaximumHeight(16777215)
        self.updateGeometry()

    def _toggle_detail(self, checked: bool) -> None:
        self._detail_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self._detail_widget.setVisible(checked)
        self._detail_widget.setMaximumHeight(16777215 if checked else 0)
        self._detail_widget.setMinimumHeight(0)
        if self.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Maximum:
            self.setMaximumHeight(760 if checked else 360)
        self.updateGeometry()
