"""统一视觉系统：颜色、间距、按钮角色、状态色、全局 stylesheet。

使用方式::

    from .theme import build_app_stylesheet, STATUS_COLORS

    self.setStyleSheet(build_app_stylesheet())
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 颜色常量
# ---------------------------------------------------------------------------

# 背景
BG_WINDOW = "#ffffff"
BG_CARD = "#ffffff"
BG_HOVER = "#e7e9eb"
BG_SELECTED = "#d8e3ee"
BG_INPUT = "#ffffff"
BG_LOG = "#0f172a"
BG_HEADER = "#eeeeef"
BG_TAB = "#e5e7e9"

# 边框
BORDER_CARD = "#cfd3d7"
BORDER_INPUT = "#bfc5ca"
GRID_LINE = "#dfe2e4"

# 文字
TEXT_PRIMARY = "#202428"
TEXT_SECONDARY = "#3f464c"
TEXT_MUTED = "#697077"
TEXT_ON_PRIMARY = "#ffffff"
TEXT_ON_DARK = "#e2e8f0"

# 主色（蓝）
PRIMARY = "#356a9a"
PRIMARY_HOVER = "#2c5d89"

# 按钮
BTN_SECONDARY_BG = "#eeeeef"
BTN_SECONDARY_FG = "#30363b"
BTN_SECONDARY_HOVER = "#dfe2e4"
BTN_DANGER = "#a83b32"
BTN_DANGER_HOVER = "#8e332c"
BTN_GHOST_BG = "#ffffff"
BTN_GHOST_FG = "#30363b"
BTN_GHOST_BORDER = "#bfc5ca"
BTN_GHOST_HOVER = "#e7e9eb"
BTN_DISABLED_BG = "#d7dadd"
BTN_DISABLED_FG = "#92989d"

# 状态色
STATUS_COLORS = {
    "running": ("#1e40af", "#dbeafe"),       # 深蓝 / 浅蓝
    "success": ("#15803d", "#dcfce7"),        # 深绿 / 浅绿
    "failed": ("#dc2626", "#fee2e2"),          # 红 / 浅红
    "stopped": ("#d97706", "#fef3c7"),         # 橙 / 浅橙
    "pending": ("#64748b", "#f1f5f9"),         # 灰 / 浅灰
}

# 圆角
RADIUS_CARD = 3
RADIUS_INPUT = 3
RADIUS_BUTTON = 3
RADIUS_BADGE = 2
RADIUS_COMPACT = 2

# 间距
PADDING_BUTTON = "6px 12px"
PADDING_COMPACT = "4px 7px"
PADDING_INPUT = "5px"
PADDING_LIST_ITEM = "8px 9px"
PADDING_TAB = "6px 10px"
PADDING_HEADER = "6px 5px"
PADDING_LOG = "10px"

# 字号
FONT_SIZE_TITLE = 20
FONT_SIZE_SECTION = 16
FONT_SIZE_LOG = 12
FONT_SIZE_COMPACT = 11


# ---------------------------------------------------------------------------
# 状态 badge 样式辅助
# ---------------------------------------------------------------------------

def status_badge_stylesheet(status: str) -> str:
    """根据运行状态返回 badge stylesheet。"""
    fg, bg = STATUS_COLORS.get(status, STATUS_COLORS["pending"])
    return (
        f"color: {fg}; background: {bg}; "
        f"padding: 4px 10px; border-radius: {RADIUS_BADGE}px; font-weight: 700;"
    )


# ---------------------------------------------------------------------------
# 全局 stylesheet 构建
# ---------------------------------------------------------------------------

_APP_STYLESHEET = f"""\
QMainWindow {{
    background: {BG_WINDOW};
    color: #1f2937;
}}

QWidget {{
    background: transparent;
    color: #1f2937;
}}

QGroupBox {{
    background: {BG_CARD};
    border: 1px solid {BORDER_CARD};
    border-radius: {RADIUS_CARD}px;
    margin-top: 14px;
    font-weight: 600;
    padding-top: 12px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_SECONDARY};
}}

QFrame#statusCard, QFrame#navPanel {{
    background: {BG_CARD};
    border: 1px solid {BORDER_CARD};
    border-radius: {RADIUS_CARD}px;
}}

QListWidget {{
    background: transparent;
    border: none;
    outline: none;
    padding: 8px 0;
}}

QListWidget::item {{
    padding: {PADDING_LIST_ITEM};
    margin: 2px 8px;
    border-radius: {RADIUS_INPUT}px;
    color: {TEXT_SECONDARY};
    border-left: 4px solid transparent;
}}

QListWidget::item:selected {{
    background: {BG_SELECTED};
    color: {TEXT_PRIMARY};
    border-left: 4px solid {PRIMARY};
}}

QListWidget::item:hover {{
    background: {BG_HOVER};
}}

QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QTableWidget, QTabWidget::pane {{
    background: {BG_INPUT};
    border: 1px solid {BORDER_INPUT};
    border-radius: {RADIUS_INPUT}px;
    padding: {PADDING_INPUT};
}}

QComboBox {{
    padding-right: 24px;
}}

QCheckBox {{
    spacing: 7px;
    color: {TEXT_PRIMARY};
    background: transparent;
}}

QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 3px;
}}

QCheckBox::indicator:unchecked {{
    background: #ffffff;
    border: 1px solid #747d85;
}}

QCheckBox::indicator:unchecked:hover {{
    border: 2px solid {PRIMARY};
}}

QCheckBox::indicator:checked {{
    background: {PRIMARY};
    border: 1px solid {PRIMARY_HOVER};
}}

QCheckBox::indicator:disabled {{
    background: {BTN_DISABLED_BG};
    border: 1px solid {BTN_DISABLED_FG};
}}

QCheckBox:disabled {{
    color: {BTN_DISABLED_FG};
}}

QPushButton {{
    background: {BG_CARD};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER_INPUT};
    border-radius: {RADIUS_BUTTON}px;
    padding: {PADDING_BUTTON};
    font-weight: 600;
}}

QPushButton:hover {{
    background: {BG_HOVER};
}}

QPushButton:disabled {{
    background: {BTN_DISABLED_BG};
    color: {BTN_DISABLED_FG};
}}

QPushButton[buttonRole="primary"] {{
    background: {PRIMARY};
    color: {TEXT_ON_PRIMARY};
    border-color: {PRIMARY};
}}

QPushButton[buttonRole="primary"]:hover {{
    background: {PRIMARY_HOVER};
}}

QPushButton[buttonRole="secondary"] {{
    background: {BTN_SECONDARY_BG};
    color: {BTN_SECONDARY_FG};
}}

QPushButton[buttonRole="secondary"]:hover {{
    background: {BTN_SECONDARY_HOVER};
}}

QPushButton[buttonRole="danger"] {{
    background: {BTN_DANGER};
}}

QPushButton[buttonRole="danger"]:hover {{
    background: {BTN_DANGER_HOVER};
}}

QPushButton[buttonRole="ghost"] {{
    background: {BTN_GHOST_BG};
    color: {BTN_GHOST_FG};
    border: 1px solid {BTN_GHOST_BORDER};
    padding: 7px 12px;
}}

QPushButton[buttonRole="ghost"]:hover {{
    background: {BTN_GHOST_HOVER};
}}

QPushButton[compact="true"] {{
    padding: {PADDING_COMPACT};
    font-size: {FONT_SIZE_COMPACT}px;
    border-radius: {RADIUS_COMPACT}px;
}}

QToolButton {{
    background: transparent;
    border: none;
    font-weight: 600;
    color: {TEXT_PRIMARY};
    padding: 6px 2px;
}}

QLabel#titleLabel {{
    font-size: {FONT_SIZE_TITLE}px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}

QLabel#mutedLabel {{
    color: {TEXT_MUTED};
}}

QLabel#statusBadge {{
    font-weight: 700;
    padding: 4px 10px;
    border-radius: {RADIUS_BADGE}px;
    border: 1px solid {BORDER_CARD};
}}

QHeaderView::section {{
    background: {BG_HEADER};
    border: none;
    border-bottom: 1px solid {BORDER_CARD};
    padding: {PADDING_HEADER};
    color: {TEXT_MUTED};
    font-weight: 600;
}}

QTableWidget {{
    gridline-color: {GRID_LINE};
    alternate-background-color: #fafcff;
}}

QTableWidget::item:selected {{
    background: #eaf3ff;
    color: {TEXT_PRIMARY};
}}

QTabBar::tab {{
    background: {BG_TAB};
    border: 1px solid {BORDER_CARD};
    border-bottom: none;
    padding: {PADDING_TAB};
    margin-right: 4px;
    border-top-left-radius: {RADIUS_INPUT}px;
    border-top-right-radius: {RADIUS_INPUT}px;
}}

QTabBar::tab:selected {{
    background: {BG_CARD};
    color: {TEXT_PRIMARY};
}}

QTextEdit#logOutput {{
    padding: {PADDING_LOG};
}}
"""

_LOG_STYLESHEET = (
    f"font-family: Consolas, 'Courier New', monospace; "
    f"font-size: {FONT_SIZE_LOG}px; "
    f"background: {BG_LOG}; color: {TEXT_ON_DARK};"
)


def build_app_stylesheet() -> str:
    """返回全局应用 stylesheet。"""
    return _APP_STYLESHEET


def log_stylesheet() -> str:
    """返回日志控件的专用 stylesheet。"""
    return _LOG_STYLESHEET
