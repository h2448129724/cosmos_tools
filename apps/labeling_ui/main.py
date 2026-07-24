"""CAB-F工具 — 入口。"""
import os
import sys
import logging
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase
from .app.main_window import MainWindow, STYLESHEET


def main() -> int:
    log_level = os.environ.get("CABF_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.WARNING),
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    families = set(QFontDatabase().families())
    font_family = ""
    for candidate in ("Microsoft YaHei UI", "Segoe UI", "Microsoft YaHei"):
        if candidate in families:
            font_family = candidate
            break
    app_font = QFont(font_family, 10) if font_family else QFont()
    app_font.setPointSize(10)
    app.setFont(app_font)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
