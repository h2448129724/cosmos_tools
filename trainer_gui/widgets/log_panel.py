"""日志面板：实时日志（颜色分级 + 行数上限）、输出信息 tabs。"""

from __future__ import annotations

import html
import re

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..theme import log_stylesheet

# 日志行数上限，防止长训练时内存/渲染膨胀
MAX_LOG_LINES = 1000

# 日志颜色分级规则
_LOG_COLORS = [
    (re.compile(r"\[(?:ERROR|CRITICAL|FATAL)\b", re.IGNORECASE), "#ef4444"),  # 红色
    (re.compile(r"\[(?:WARN|WARNING)\b", re.IGNORECASE), "#f59e0b"),            # 橙色
    (re.compile(r"\[(?:DONE|SUCCESS|COMPLETE)\b", re.IGNORECASE), "#22c55e"),    # 绿色
    (re.compile(r"\[(?:INFO)\b", re.IGNORECASE), "#94a3b8"),                     # 蓝灰
]

# stderr 行一律浅红
_STDERR_COLOR = "#f87171"
_STDOUT_COLOR = "#e2e8f0"


class LogPanel(QTabWidget):
    """日志/输出 tab 面板，支持颜色分级和行数上限。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._line_count = 0

        # --- 实时日志 tab ---
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(0, 0, 0, 0)

        log_toolbar = QHBoxLayout()
        self.auto_scroll_checkbox = QCheckBox("自动滚动")
        self.auto_scroll_checkbox.setChecked(True)
        self.line_count_label = QLabel("0 行")
        self.line_count_label.setStyleSheet("color: #64748b; font-size: 11px;")
        self.copy_log_button = QPushButton("复制日志")
        self.copy_log_button.setProperty("buttonRole", "ghost")
        self.clear_log_button = QPushButton("清空日志")
        self.clear_log_button.setProperty("buttonRole", "ghost")
        self.export_log_button = QPushButton("导出日志")
        self.export_log_button.setProperty("buttonRole", "ghost")
        log_toolbar.addWidget(self.auto_scroll_checkbox)
        log_toolbar.addWidget(self.line_count_label)
        log_toolbar.addStretch(1)
        log_toolbar.addWidget(self.copy_log_button)
        log_toolbar.addWidget(self.clear_log_button)
        log_toolbar.addWidget(self.export_log_button)

        self.log_output = QTextEdit()
        self.log_output.setObjectName("logOutput")
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.log_output.setPlaceholderText("暂无日志，点击开始执行后将在这里显示实时输出。")
        self.log_output.setStyleSheet(log_stylesheet())

        log_layout.addLayout(log_toolbar)
        log_layout.addWidget(self.log_output)

        # --- 输出信息 tab ---
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.history_detail = QPlainTextEdit()
        self.history_detail.setReadOnly(True)
        self.history_detail.setPlaceholderText("这里会显示当前任务输出位置，或所选历史记录的详细信息。")
        output_layout.addWidget(self.history_detail)

        self.addTab(log_tab, "实时日志")
        self.addTab(output_tab, "输出信息")

    def append_log_line(self, text: str, stream: str = "stdout") -> None:
        """追加一行带颜色的日志。自动管理行数上限。"""
        color = _STDERR_COLOR if stream == "stderr" else self._line_color(text)
        escaped = html.escape(text)
        self.log_output.append(f'<span style="color:{color}">{escaped}</span>')
        self._line_count += 1
        self._trim_excess_lines()
        self.line_count_label.setText(f"{self._line_count} 行")

    def clear_log(self) -> None:
        """清空日志。"""
        self.log_output.clear()
        self._line_count = 0
        self.line_count_label.setText("0 行")

    def raw_log_text(self) -> str:
        """返回纯文本日志（无 HTML 标记）。"""
        return self.log_output.toPlainText()

    def error_summary(self, tail_lines: int = 50) -> str:
        """返回 stderr 最后 N 行摘要。"""
        text = self.raw_log_text()
        lines = text.splitlines()
        return "\n".join(lines[-tail_lines:]) if len(lines) > tail_lines else text

    @staticmethod
    def _line_color(text: str) -> str:
        """根据日志内容匹配颜色。"""
        for pattern, color in _LOG_COLORS:
            if pattern.search(text):
                return color
        return _STDOUT_COLOR

    def _trim_excess_lines(self) -> None:
        """超过上限时裁剪最早的行。"""
        if self._line_count <= MAX_LOG_LINES:
            return
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        # 删除前面的多余行
        excess = self._line_count - MAX_LOG_LINES
        for _ in range(excess):
            cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor)
        cursor.movePosition(cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        self._line_count = MAX_LOG_LINES
