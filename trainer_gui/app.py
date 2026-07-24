from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from shared.project_paths import repo_root

from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    font = QFont(app.font())
    if font.pointSize() <= 0:
        font.setPointSize(10)
    app.setFont(font)
    window = MainWindow(repo_root())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
