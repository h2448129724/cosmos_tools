from __future__ import annotations

import subprocess
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

try:
    from scripts._bootstrap import REPO_ROOT, python_env
except ImportError:  # pragma: no cover - direct script execution from scripts/
    from _bootstrap import REPO_ROOT, python_env


def _launch(args: list[str]) -> None:
    subprocess.Popen(
        args,
        cwd=str(REPO_ROOT),
        env=python_env(),
    )


class LauncherWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CABF 启动器")
        self.resize(420, 220)
        self._launching = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("选择要打开的界面")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        hint = QLabel("标注、筛选、导出用 CABF 标注工具；训练、运行命令管理用训练 UI。")
        hint.setWordWrap(True)
        hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(hint)

        self.label_button = QPushButton("打开 CABF 标注工具")
        self.label_button.setMinimumHeight(44)
        self.label_button.clicked.connect(self.open_labeling_ui)
        layout.addWidget(self.label_button)

        self.trainer_button = QPushButton("打开训练 UI")
        self.trainer_button.setMinimumHeight(44)
        self.trainer_button.clicked.connect(self.open_trainer_ui)
        layout.addWidget(self.trainer_button)

        layout.addStretch(1)

    def open_labeling_ui(self) -> None:
        if self._launching:
            return
        try:
            self._set_launching()
            _launch([sys.executable, str(REPO_ROOT / "scripts" / "labeling_ui.py")])
            self.close()
        except Exception as exc:
            self._clear_launching()
            QMessageBox.critical(self, "启动失败", f"无法启动 CABF 标注工具：\n{exc}")

    def open_trainer_ui(self) -> None:
        if self._launching:
            return
        try:
            self._set_launching()
            _launch([sys.executable, "-m", "trainer_gui.app"])
            self.close()
        except Exception as exc:
            self._clear_launching()
            QMessageBox.critical(self, "启动失败", f"无法启动训练 UI：\n{exc}")

    def _set_launching(self) -> None:
        self._launching = True
        self.label_button.setEnabled(False)
        self.trainer_button.setEnabled(False)

    def _clear_launching(self) -> None:
        self._launching = False
        self.label_button.setEnabled(True)
        self.trainer_button.setEnabled(True)


def main() -> int:
    app = QApplication(sys.argv)
    font = QFont(app.font())
    if font.pointSize() <= 0:
        font.setPointSize(10)
    app.setFont(font)
    window = LauncherWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
