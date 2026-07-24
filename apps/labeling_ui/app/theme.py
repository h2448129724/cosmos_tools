"""Shared theme tokens and QSS for the CAB-F desktop UI."""
from __future__ import annotations


TOKENS = {
    "bg_page": "#FFFFFF",
    "bg_card": "#FFFFFF",
    "border_main": "#CFD3D7",
    "border_weak": "#DFE2E4",
    "primary": "#356A9A",
    "text_main": "#202428",
    "text_secondary": "#697077",
    "text_weak": "#92989D",
    "success": "#356848",
    "warning": "#8A611F",
    "danger": "#A83B32",
    "success_bg": "#E3EEE7",
    "warning_bg": "#F6EAD8",
    "danger_bg": "#F4E2E0",
    "neutral_bg": "#E9EBED",
    "neutral_text": "#555D64",
    "sidebar_selected": "#D8E3EE",
    "sidebar_hover": "#E7E9EB",
    "overview_bg": "#F4F5F5",
}


APP_STYLESHEET = f"""
QMainWindow {{
    background: {TOKENS["bg_page"]};
}}

QWidget {{
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {TOKENS["text_main"]};
}}

QMenuBar {{
    background: {TOKENS["bg_card"]};
    border-bottom: 1px solid {TOKENS["border_weak"]};
    padding: 3px 8px;
}}
QMenuBar::item {{
    padding: 5px 9px;
    border-radius: 2px;
}}
QMenuBar::item:selected {{
    background: {TOKENS["sidebar_hover"]};
}}
QMenu {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 2px;
    padding: 3px;
}}
QMenu::item {{
    padding: 6px 12px;
    border-radius: 2px;
}}
QMenu::item:selected {{
    background: {TOKENS["sidebar_selected"]};
    color: {TOKENS["primary"]};
}}

#sidebar {{
    background: {TOKENS["bg_card"]};
    border-right: 1px solid {TOKENS["border_main"]};
}}
#sidebarHeader,
#sidebar QLabel {{
    color: {TOKENS["text_secondary"]};
}}
#sidebarFootnote {{
    color: {TOKENS["text_secondary"]};
    font-size: 13px;
    line-height: 1.45;
}}

#toolList {{
    border: none;
    background: transparent;
    outline: none;
    padding: 2px;
}}
#toolList::item {{
    background: transparent;
    color: {TOKENS["text_main"]};
    padding: 6px 9px;
    margin: 0;
    border-radius: 0;
}}
#toolList::item:selected {{
    background: {TOKENS["sidebar_selected"]};
    color: {TOKENS["primary"]};
    border-left: 3px solid {TOKENS["primary"]};
    padding-left: 11px;
    font-weight: 600;
}}
#toolList::item:hover:!selected {{
    background: {TOKENS["sidebar_hover"]};
}}

#contentShell {{
    background: {TOKENS["bg_page"]};
}}

#overviewCard {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 3px;
}}
#overviewBadge {{
    background: {TOKENS["sidebar_selected"]};
    color: {TOKENS["primary"]};
    border-radius: 2px;
    padding: 3px 9px;
    font-size: 11px;
    font-weight: 700;
}}
#overviewTitle {{
    color: {TOKENS["text_main"]};
    font-size: 16px;
    font-weight: 700;
}}
#overviewText {{
    color: {TOKENS["text_secondary"]};
    font-size: 11px;
}}
#overviewMeta {{
    color: {TOKENS["text_secondary"]};
    font-size: 11px;
}}

#miniStat {{
    background: {TOKENS["bg_page"]};
    border: 1px solid {TOKENS["border_weak"]};
    border-radius: 2px;
}}
#miniStatValue {{
    color: {TOKENS["text_main"]};
    font-size: 15px;
    font-weight: 700;
}}
#miniStatLabel {{
    color: {TOKENS["text_secondary"]};
    font-size: 11px;
    font-weight: 600;
}}

#surfaceCard {{
    background: transparent;
    border: none;
}}

#card {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 3px;
}}

#hintPanel {{
    background: {TOKENS["overview_bg"]};
    border: 1px solid {TOKENS["border_weak"]};
    border-radius: 3px;
}}

#stepSidebar {{
    background: transparent;
    border: none;
    border-radius: 0;
}}

#stepContentCard,
#configTableCard,
#logCard {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 3px;
}}

#configSection {{
    background: transparent;
    border: none;
    border-top: 1px solid {TOKENS["border_weak"]};
}}

#configRow {{
    background: transparent;
    border-bottom: 1px solid {TOKENS["border_weak"]};
}}

#configLabel {{
    color: {TOKENS["text_main"]};
    font-size: 14px;
    font-weight: 600;
}}

#fieldHint {{
    color: {TOKENS["text_secondary"]};
    font-size: 13px;
}}

QPushButton {{
    min-height: 30px;
    max-height: 30px;
    padding: 0 10px;
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 3px;
    color: {TOKENS["text_main"]};
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {TOKENS["sidebar_hover"]};
    border-color: #9FA6AC;
}}
QPushButton:pressed {{
    background: #DDE0E2;
}}
QPushButton:disabled {{
    background: {TOKENS["bg_page"]};
    color: {TOKENS["text_weak"]};
}}
QPushButton[primary="true"] {{
    background: {TOKENS["primary"]};
    border: 1px solid {TOKENS["primary"]};
    color: #FFFFFF;
}}
QPushButton[primary="true"]:hover {{
    background: #2C5D89;
    border-color: #2C5D89;
}}
QPushButton[primary="true"]:pressed {{
    background: #244E73;
    border-color: #244E73;
}}

QLineEdit,
QComboBox,
QSpinBox,
QDoubleSpinBox {{
    min-height: 30px;
    max-height: 30px;
    padding: 0 8px;
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 3px;
    selection-background-color: {TOKENS["primary"]};
}}
QLineEdit:focus,
QComboBox:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {{
    border-color: {TOKENS["primary"]};
}}

QPlainTextEdit {{
    background: {TOKENS["bg_page"]};
    border: 1px solid {TOKENS["border_weak"]};
    border-radius: 2px;
    padding: 6px;
    color: {TOKENS["text_secondary"]};
    font-family: "Cascadia Code", "Consolas", monospace;
    font-size: 11px;
}}

QListWidget {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 2px;
    outline: none;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollArea#workflowScroll,
QWidget#workflowScrollViewport,
QWidget#workflowScrollContent {{
    background: {TOKENS["bg_page"]};
}}
QScrollBar:horizontal,
QScrollBar:vertical {{
    background: transparent;
    border: none;
}}
QScrollBar::handle:horizontal,
QScrollBar::handle:vertical {{
    background: #B5BBC0;
    border-radius: 2px;
    min-width: 24px;
    min-height: 24px;
}}
QScrollBar::add-line,
QScrollBar::sub-line,
QScrollBar::add-page,
QScrollBar::sub-page {{
    background: transparent;
    border: none;
}}

QCheckBox,
QRadioButton {{
    color: {TOKENS["text_main"]};
    spacing: 8px;
}}

QGroupBox {{
    background: {TOKENS["bg_card"]};
    border: 1px solid {TOKENS["border_main"]};
    border-radius: 3px;
    margin-top: 12px;
    padding: 12px;
    color: {TOKENS["text_main"]};
    font-weight: 600;
}}
QGroupBox::title {{
    left: 12px;
    padding: 0 6px;
    color: {TOKENS["text_secondary"]};
}}

QStatusBar {{
    background: {TOKENS["bg_card"]};
    border-top: 1px solid {TOKENS["border_weak"]};
    color: {TOKENS["text_secondary"]};
}}

QMessageBox {{
    background: {TOKENS["bg_card"]};
}}
"""


def badge_style(kind: str) -> str:
    styles = {
        "success": (TOKENS["success_bg"], TOKENS["success"]),
        "warning": (TOKENS["warning_bg"], TOKENS["warning"]),
        "danger": (TOKENS["danger_bg"], TOKENS["danger"]),
        "neutral": (TOKENS["neutral_bg"], TOKENS["neutral_text"]),
        "info": (TOKENS["sidebar_selected"], TOKENS["primary"]),
    }
    bg, fg = styles.get(kind, styles["neutral"])
    return (
        f"background:{bg};color:{fg};border:1px solid {TOKENS['border_main']};"
        "border-radius:2px;padding:3px 7px;font-size:11px;font-weight:600;"
    )
