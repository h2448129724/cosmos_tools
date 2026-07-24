from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
)


class CommandDialog(QDialog):
    def __init__(self, command_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("服务器可执行命令")
        self.resize(900, 520)

        layout = QVBoxLayout(self)

        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setPlainText(command_text)
        layout.addWidget(self.command_preview)

        button_row = QHBoxLayout()
        self.copy_button = QPushButton("复制命令")
        self.copy_button.clicked.connect(self._copy_command)
        button_row.addWidget(self.copy_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)
        layout.addWidget(buttons)

    def _copy_command(self) -> None:
        command_text = self.command_preview.toPlainText().strip()
        if not command_text:
            QMessageBox.information(self, "暂无命令", "当前没有可复制的命令。")
            return
        QApplication.clipboard().setText(command_text)
        QMessageBox.information(self, "已复制", "命令已复制到剪贴板。")
