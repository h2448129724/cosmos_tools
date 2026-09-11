"""One visual language for the workbench and every embedded workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class StatusTone(StrEnum):
    NEUTRAL = "neutral"
    INFO = "info"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    canvas: str = "#f3f4f5"
    surface: str = "#ffffff"
    surface_subtle: str = "#f7f8f8"
    surface_muted: str = "#eceeef"
    border: str = "#cfd3d7"
    border_strong: str = "#bfc5ca"
    border_weak: str = "#dfe2e4"
    text: str = "#202428"
    text_secondary: str = "#697077"
    text_weak: str = "#7b8288"
    primary: str = "#356a9a"
    primary_hover: str = "#2c5d89"
    primary_selected: str = "#d8e3ee"
    hover: str = "#e7e9eb"
    success: str = "#356848"
    success_bg: str = "#e3eee7"
    warning: str = "#8a611f"
    warning_bg: str = "#f6ead8"
    danger: str = "#a83b32"
    danger_bg: str = "#f4e2e0"
    neutral: str = "#555d64"
    neutral_bg: str = "#e9ebed"
    canvas_dark: str = "#171c24"
    canvas_dark_text: str = "#b9c2ce"


DEFAULT_TOKENS = ThemeTokens()


def _legacy_tokens(tokens: ThemeTokens) -> Mapping[str, str]:
    values = {
        "bg_page": tokens.surface,
        "bg_card": tokens.surface,
        "border_main": tokens.border,
        "border_weak": tokens.border_weak,
        "primary": tokens.primary,
        "text_main": tokens.text,
        "text_secondary": tokens.text_secondary,
        "text_weak": tokens.text_weak,
        "success": tokens.success,
        "warning": tokens.warning,
        "danger": tokens.danger,
        "success_bg": tokens.success_bg,
        "warning_bg": tokens.warning_bg,
        "danger_bg": tokens.danger_bg,
        "neutral_bg": tokens.neutral_bg,
        "neutral_text": tokens.neutral,
        "sidebar_selected": tokens.primary_selected,
        "sidebar_hover": tokens.hover,
        "overview_bg": tokens.surface_subtle,
    }
    return MappingProxyType(values)


LEGACY_TOKENS = _legacy_tokens(DEFAULT_TOKENS)


def token_values(tokens: ThemeTokens = DEFAULT_TOKENS) -> Mapping[str, str]:
    return MappingProxyType(asdict(tokens))


def tone_colors(
    tone: StatusTone | str,
    tokens: ThemeTokens = DEFAULT_TOKENS,
) -> tuple[str, str, str]:
    try:
        resolved = StatusTone(tone)
    except ValueError:
        resolved = StatusTone.NEUTRAL
    values = {
        StatusTone.NEUTRAL: (tokens.neutral, tokens.neutral_bg, tokens.border),
        StatusTone.INFO: (tokens.primary, tokens.primary_selected, "#c5d4e1"),
        StatusTone.RUNNING: (tokens.primary, "#e5edf4", "#c5d4e1"),
        StatusTone.SUCCESS: (tokens.success, tokens.success_bg, "#c8d8ce"),
        StatusTone.WARNING: (tokens.warning, tokens.warning_bg, "#e2cfac"),
        StatusTone.DANGER: (tokens.danger, tokens.danger_bg, "#dec0bd"),
    }
    return values[resolved]


def status_badge_stylesheet(
    tone: StatusTone | str,
    tokens: ThemeTokens = DEFAULT_TOKENS,
) -> str:
    foreground, background, border = tone_colors(tone, tokens)
    return (
        f"color:{foreground};background:{background};border:1px solid {border};"
        "border-radius:2px;padding:3px 7px;font-size:11px;font-weight:600;"
    )


def _foundation_stylesheet(tokens: ThemeTokens) -> str:
    return f"""
QWidget {{
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
    color: {tokens.text};
}}
QToolTip {{
    color: {tokens.text}; background: {tokens.surface}; border: 1px solid {tokens.border}; padding: 4px;
}}
QPushButton {{
    min-height: 30px; padding: 2px 10px; background: {tokens.surface};
    border: 1px solid {tokens.border_strong}; border-radius: 3px;
    color: {tokens.text}; font-weight: 550;
}}
QPushButton:hover {{ background: {tokens.hover}; border-color: #9fa6ac; }}
QPushButton:pressed {{ background: #dde0e2; }}
QPushButton:disabled {{ background: {tokens.surface_subtle}; color: {tokens.text_weak}; border-color: {tokens.border}; }}
QPushButton[buttonRole="primary"], QPushButton[primary="true"] {{
    background: {tokens.primary}; border-color: {tokens.primary}; color: #ffffff;
}}
QPushButton[buttonRole="primary"]:hover, QPushButton[primary="true"]:hover {{
    background: {tokens.primary_hover}; border-color: {tokens.primary_hover};
}}
QPushButton[buttonRole="danger"] {{ background: {tokens.danger}; border-color: {tokens.danger}; color: #ffffff; }}
QPushButton[buttonRole="ghost"] {{ background: transparent; border-color: transparent; }}
QPushButton[compact="true"] {{ min-height: 26px; padding: 1px 7px; font-size: 11px; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit {{
    min-height: 30px; padding: 1px 8px; background: {tokens.surface};
    border: 1px solid {tokens.border_strong}; border-radius: 3px;
    selection-background-color: {tokens.primary};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {tokens.primary}; }}
QLineEdit[validationState="error"], QComboBox[validationState="error"] {{ border-color: {tokens.danger}; }}
QLineEdit[validationState="success"], QComboBox[validationState="success"] {{ border-color: {tokens.success}; }}
QPlainTextEdit, QTextEdit {{
    background: {tokens.surface}; border: 1px solid {tokens.border}; border-radius: 2px;
    padding: 6px; selection-background-color: {tokens.primary};
}}
QPlainTextEdit[uiRole="logViewer"], QTextEdit[uiRole="logViewer"], QTextEdit#logOutput {{
    background: {tokens.canvas_dark}; color: #e2e8f0;
    font-family: "Cascadia Mono", "Consolas", monospace; font-size: 11px;
}}
QListWidget, QTreeWidget, QTableWidget {{
    background: {tokens.surface}; border: 1px solid {tokens.border}; border-radius: 2px; outline: none;
    alternate-background-color: {tokens.surface_subtle};
}}
QListWidget::item, QTreeWidget::item {{ padding: 5px 7px; }}
QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {{
    background: {tokens.primary_selected}; color: {tokens.text};
}}
QHeaderView::section {{
    background: {tokens.surface_muted}; border: none; border-bottom: 1px solid {tokens.border};
    padding: 6px 7px; color: {tokens.text_secondary}; font-weight: 600;
}}
QGroupBox {{
    background: {tokens.surface}; border: 1px solid {tokens.border}; border-radius: 3px;
    margin-top: 11px; padding-top: 9px; font-weight: 600;
}}
QGroupBox::title {{ left: 10px; padding: 0 5px; color: {tokens.text_secondary}; }}
QCheckBox, QRadioButton {{ color: {tokens.text}; spacing: 7px; background: transparent; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 3px; }}
QCheckBox::indicator:unchecked {{ background: #ffffff; border: 1px solid #747d85; }}
QCheckBox::indicator:unchecked:hover {{ border: 2px solid {tokens.primary}; }}
QCheckBox::indicator:checked {{ background: {tokens.primary}; border: 1px solid {tokens.primary_hover}; }}
QCheckBox::indicator:disabled {{ background: #d7dadd; border: 1px solid {tokens.text_weak}; }}
QCheckBox:disabled {{ color: {tokens.text_weak}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 8px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; margin: 0; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: #b5bbc0; border-radius: 2px; min-height: 24px; min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; border: none; }}
QTabWidget::pane {{ border: 1px solid {tokens.border}; border-radius: 2px; background: {tokens.surface}; }}
QTabBar::tab {{
    background: {tokens.surface_muted}; border: 1px solid {tokens.border}; border-bottom: none;
    padding: 6px 10px; margin-right: 3px; border-top-left-radius: 3px; border-top-right-radius: 3px;
}}
QTabBar::tab:selected {{ background: {tokens.surface}; color: {tokens.text}; }}

QFrame[uiRole="pageHeader"] {{ background: transparent; border: none; }}
QLabel[uiRole="pageTitle"] {{ color: {tokens.text}; font-size: 19px; font-weight: 700; }}
QLabel[uiRole="pageDescription"], QLabel[uiRole="muted"] {{ color: {tokens.text_secondary}; }}
QFrame[uiRole="sectionSurface"], QFrame#card, QFrame#stepContentCard,
QFrame#configTableCard, QFrame#logCard {{
    background: {tokens.surface}; border: 1px solid {tokens.border}; border-radius: 3px;
}}
QFrame[uiRole="sectionSurface"][nested="true"] {{ background: transparent; border: none; border-top: 1px solid {tokens.border_weak}; border-radius: 0; }}
QLabel[uiRole="sectionTitle"] {{ color: {tokens.text}; font-size: 14px; font-weight: 650; }}
QLabel[uiRole="fieldLabel"] {{ color: {tokens.text_secondary}; font-size: 11px; font-weight: 600; }}
QLabel[uiRole="statusBadge"] {{
    color: {tokens.neutral}; background: {tokens.neutral_bg}; border: 1px solid {tokens.border};
    border-radius: 2px; padding: 3px 7px; font-size: 11px; font-weight: 650;
}}
QLabel[uiRole="statusBadge"][tone="info"], QLabel[uiRole="statusBadge"][tone="running"] {{
    color: {tokens.primary}; background: #e5edf4; border-color: #c5d4e1;
}}
QLabel[uiRole="statusBadge"][tone="success"] {{
    color: {tokens.success}; background: {tokens.success_bg}; border-color: #c8d8ce;
}}
QLabel[uiRole="statusBadge"][tone="warning"] {{
    color: {tokens.warning}; background: {tokens.warning_bg}; border-color: #e2cfac;
}}
QLabel[uiRole="statusBadge"][tone="danger"] {{
    color: {tokens.danger}; background: {tokens.danger_bg}; border-color: #dec0bd;
}}
QFrame[uiRole="hint"], QFrame#hintPanel {{ background: {tokens.surface_subtle}; border: 1px solid {tokens.border_weak}; border-radius: 3px; }}
QFrame[uiRole="statusBanner"] {{ border: 1px solid {tokens.border}; border-radius: 2px; background: {tokens.neutral_bg}; }}
QFrame[uiRole="statusBanner"][tone="info"], QFrame[uiRole="statusBanner"][tone="running"] {{ background: #e5edf4; border-color: #c5d4e1; }}
QFrame[uiRole="statusBanner"][tone="success"] {{ background: {tokens.success_bg}; border-color: #c8d8ce; }}
QFrame[uiRole="statusBanner"][tone="warning"] {{ background: {tokens.warning_bg}; border-color: #e2cfac; }}
QFrame[uiRole="statusBanner"][tone="danger"] {{ background: {tokens.danger_bg}; border-color: #dec0bd; }}
QFrame[uiRole="metric"] {{ background: {tokens.surface_subtle}; border: 1px solid {tokens.border_weak}; border-radius: 2px; }}
QLabel[uiRole="metricValue"] {{ color: {tokens.text}; font-size: 16px; font-weight: 700; }}
QLabel[uiRole="metricLabel"] {{ color: {tokens.text_secondary}; font-size: 11px; }}
QFrame[uiRole="emptyState"] {{ background: {tokens.surface}; border: 1px solid {tokens.border}; border-radius: 3px; }}
QLabel[uiRole="emptyStateTitle"] {{ color: {tokens.text}; font-size: 15px; font-weight: 650; }}
QLabel[uiRole="emptyStateDescription"] {{ color: {tokens.text_secondary}; }}
"""


def build_workspace_stylesheet(tokens: ThemeTokens = DEFAULT_TOKENS) -> str:
    return _foundation_stylesheet(tokens) + f"""
QMainWindow {{ background: {tokens.surface}; }}
QMenuBar {{ background: {tokens.surface}; border-bottom: 1px solid {tokens.border_weak}; padding: 3px 8px; }}
QMenuBar::item {{ padding: 5px 9px; border-radius: 2px; }}
QMenuBar::item:selected {{ background: {tokens.hover}; }}
QMenu {{ background: {tokens.surface}; border: 1px solid {tokens.border}; border-radius: 2px; padding: 3px; }}
QMenu::item {{ padding: 6px 12px; border-radius: 2px; }}
QMenu::item:selected {{ background: {tokens.primary_selected}; color: {tokens.primary}; }}
QStatusBar {{ background: {tokens.surface}; border-top: 1px solid {tokens.border_weak}; color: {tokens.text_secondary}; }}
QScrollArea#workflowScroll, QWidget#workflowScrollViewport, QWidget#workflowScrollContent {{ background: {tokens.surface}; }}
#sidebar {{ background: {tokens.surface}; border-right: 1px solid {tokens.border}; }}
#sidebarHeader, #sidebar QLabel, #sidebarFootnote {{ color: {tokens.text_secondary}; }}
#toolList {{ border: none; background: transparent; outline: none; padding: 2px; }}
#toolList::item {{ background: transparent; color: {tokens.text}; padding: 6px 9px; margin: 0; border-radius: 0; }}
#toolList::item:selected {{ background: {tokens.primary_selected}; color: {tokens.primary}; border-left: 3px solid {tokens.primary}; padding-left: 11px; font-weight: 600; }}
#toolList::item:hover:!selected {{ background: {tokens.hover}; }}
#contentShell, #surfaceCard, #stepSidebar {{ background: transparent; border: none; }}
#overviewCard {{ background: {tokens.surface}; border: 1px solid {tokens.border}; border-radius: 3px; }}
#overviewBadge {{ background: {tokens.primary_selected}; color: {tokens.primary}; border-radius: 2px; padding: 3px 9px; font-size: 11px; font-weight: 700; }}
#overviewTitle {{ color: {tokens.text}; font-size: 16px; font-weight: 700; }}
#overviewText, #overviewMeta, #fieldHint {{ color: {tokens.text_secondary}; }}
#miniStat {{ background: {tokens.surface_subtle}; border: 1px solid {tokens.border_weak}; border-radius: 2px; }}
#miniStatValue {{ color: {tokens.text}; font-size: 15px; font-weight: 700; }}
#miniStatLabel {{ color: {tokens.text_secondary}; font-size: 11px; font-weight: 600; }}
#configSection {{ background: transparent; border: none; border-top: 1px solid {tokens.border_weak}; }}
#configRow {{ background: transparent; border-bottom: 1px solid {tokens.border_weak}; }}
#configLabel {{ color: {tokens.text}; font-size: 14px; font-weight: 600; }}
"""


def build_labeling_stylesheet(tokens: ThemeTokens = DEFAULT_TOKENS) -> str:
    return build_workspace_stylesheet(tokens)


def build_trainer_stylesheet(tokens: ThemeTokens = DEFAULT_TOKENS) -> str:
    return build_workspace_stylesheet(tokens) + f"""
QFrame#statusCard, QFrame#navPanel {{ background: {tokens.surface}; border: 1px solid {tokens.border}; border-radius: 3px; }}
QListWidget#featureList {{ border: none; background: transparent; }}
QLabel#titleLabel {{ font-size: 20px; font-weight: 700; color: {tokens.text}; }}
QLabel#mutedLabel {{ color: {tokens.text_secondary}; }}
QLabel#statusBadge {{ font-weight: 700; padding: 4px 10px; border-radius: 2px; border: 1px solid {tokens.border}; }}
"""


def build_toolbox_stylesheet(tokens: ThemeTokens = DEFAULT_TOKENS) -> str:
    return _foundation_stylesheet(tokens) + f"""
QMainWindow#toolboxShell, QWidget#shellRoot {{ background: {tokens.canvas}; color: {tokens.text}; }}
QFrame#projectHeader {{ background: {tokens.surface}; border-bottom: 1px solid {tokens.border}; }}
QLabel#projectHeaderTitle {{ color: {tokens.text}; font-size: 17px; font-weight: 700; }}
QLabel#projectHeaderBreadcrumb {{ color: {tokens.text_secondary}; font-size: 11px; }}
QLabel#contextChip {{ color: #4f565d; background: transparent; border: none; padding: 4px 8px; }}
QFrame#navigationRail {{ background: {tokens.surface_muted}; border-right: 1px solid #c9cdd1; }}
QWidget#navigationContent {{ background: {tokens.surface_muted}; }}
QLabel#navigationBrand {{ color: {tokens.text}; font-size: 17px; font-weight: 700; letter-spacing: 1px; }}
QLabel#navigationSubtitle {{ color: #747b82; font-size: 11px; }}
QLabel#navigationSection {{ color: #767d84; font-size: 10px; font-weight: 700; padding: 10px 8px 4px 8px; }}
QPushButton#navigationItem {{ text-align: left; color: #30363b; background: transparent; border: none; border-left: 3px solid transparent; border-radius: 0; min-height: 30px; padding: 3px 9px; font-weight: 500; }}
QPushButton#navigationItem:hover {{ background: #e2e5e7; color: {tokens.text}; }}
QPushButton#navigationItem:checked {{ background: {tokens.primary_selected}; color: #183f66; border-left: 3px solid #356a9a; font-weight: 700; }}
QPushButton#navigationCollapse {{ border: none; background: transparent; text-align: right; color: {tokens.text_secondary}; }}
QLabel#runtimeLabel {{ color: #646b72; font-size: 10px; font-weight: 600; }}
QComboBox#runtimeSelector {{ color: #30363b; background: {tokens.surface}; border: 1px solid {tokens.border_strong}; border-radius: 3px; min-height: 26px; padding: 1px 6px; }}
QFrame#activityHost {{ background: {tokens.surface}; }}
QFrame#activityTitleBar {{ background: transparent; }}
QLabel#activityTitle {{ color: {tokens.text}; font-size: 20px; font-weight: 700; }}
QLabel#activityDescription {{ color: {tokens.text_secondary}; font-size: 12px; }}
QFrame#projectInspector {{ background: {tokens.surface}; border-left: 1px solid {tokens.border}; padding: 12px; }}
QLabel#inspectorTitle {{ color: {tokens.text}; font-size: 15px; font-weight: 700; }}
QLabel#inspectorSection {{ color: {tokens.text}; font-size: 12px; font-weight: 700; margin-top: 6px; }}
QLabel#fieldLabel {{ color: #646b72; font-size: 11px; font-weight: 600; }}
QPushButton#inspectorClose {{ min-width: 28px; max-width: 28px; padding: 0; }}
QListWidget#artifactList, QListWidget#artifactsActivityList {{ padding: 3px; }}
QFrame#taskStatusStrip {{ background: {tokens.surface}; border-top: 1px solid {tokens.border}; }}
QLabel#taskStatusTitle {{ color: {tokens.text}; font-weight: 650; }}
QLabel#taskStatusSummary {{ color: {tokens.text_secondary}; }}
QFrame#overviewHero, QFrame#pipelineBanner, QFrame#activityCard, QFrame#pipelineStep, QFrame#cabfCard {{ background: {tokens.surface}; border: 1px solid {tokens.border}; border-radius: 3px; }}
QLabel#overviewHeroTitle, QLabel#pipelineTitle {{ color: {tokens.text}; font-size: 18px; font-weight: 700; }}
QLabel#overviewHeroText, QLabel#pipelineSubtitle, QLabel#activityCardDescription, QLabel#pipelineStepDescription, QLabel#pipelineHint, QLabel#cabfMuted, QLabel#cabfCardSubtitle {{ color: {tokens.text_secondary}; }}
QLabel#readinessBadge, QLabel#pipelineStatus, QLabel#cabfBadge {{ color: #2f5b80; background: #e5edf4; border: 1px solid #c5d4e1; border-radius: 2px; padding: 4px 8px; font-weight: 700; }}
QLabel#sectionTitle, QLabel#activityCardTitle, QLabel#pipelineStepTitle, QLabel#cabfCardTitle, QLabel#cabfSectionTitle {{ color: {tokens.text}; font-size: 15px; font-weight: 700; }}
QLabel#activityCardNumber {{ color: {tokens.primary}; font-size: 11px; font-weight: 700; }}
QLabel#pipelineStepNumber {{ color: #ffffff; background: #4b6f90; border-radius: 2px; font-weight: 700; }}
QLabel#pipelineStepState {{ color: #72551d; background: #f5eedc; border-radius: 2px; padding: 3px 6px; }}
QLabel#pipelineStepState[complete="true"] {{ color: #2f623f; background: {tokens.success_bg}; }}
QLabel#overviewSummary, QLabel#cabfStatus {{ color: #525960; background: {tokens.neutral_bg}; border-radius: 2px; padding: 9px; }}
QLabel#cabfValidation {{ color: #525960; background: #eeeeee; border-radius: 2px; padding: 7px; }}
QLabel#cabfValidation[level="success"] {{ color: #2f623f; background: {tokens.success_bg}; }}
QLabel#cabfValidation[level="warning"] {{ color: #79521c; background: {tokens.warning_bg}; }}
QWidget#cabfCanvas {{ background: {tokens.canvas_dark}; border: 1px solid #4d5358; border-radius: 2px; }}
QFrame#embeddedPageHost {{ background: {tokens.surface}; }}
QFrame#embeddedPageHeader {{ background: {tokens.surface}; border-bottom: 1px solid {tokens.border}; }}
QLabel#embeddedPageTitle {{ font-size: 17px; font-weight: 700; color: {tokens.text}; }}
QLabel#embeddedPageSubtitle {{ font-size: 12px; color: {tokens.text_secondary}; }}
QPushButton#embeddedBackButton {{ padding: 5px 10px; }}
QScrollArea#workflowScroll, QWidget#workflowScrollViewport, QWidget#workflowScrollContent {{ background: {tokens.surface}; }}
"""
