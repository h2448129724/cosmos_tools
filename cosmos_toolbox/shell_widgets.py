from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .capabilities import CapabilityCatalog, STAGE_TITLES
from .project_context import ProjectState
from .project_session import ProjectArtifact
from .layout_presentation import LayoutPresentation
from shared.conda_runtime import CondaEnvInfo


class NavigationRail(QFrame):
    """Project-flow navigation projected from the capability catalog."""

    currentChanged = Signal(int)
    runtimeProfileChanged = Signal(str)

    def __init__(self, catalog: CapabilityCatalog, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.catalog = catalog
        self._keys: list[str] = []
        self._buttons: list[QPushButton] = []
        self._headings: list[QLabel] = []
        self._current_index = -1
        self._compact = False
        self.setObjectName("navigationRail")
        self.setMinimumWidth(188)
        self.setMaximumWidth(228)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 16, 12, 14)
        outer.setSpacing(8)
        self.brand = QLabel("COSMOS")
        self.brand.setObjectName("navigationBrand")
        outer.addWidget(self.brand)
        self.subtitle = QLabel("个人项目工作台")
        self.subtitle.setObjectName("navigationSubtitle")
        outer.addWidget(self.subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName("navigationContent")
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 12, 0, 0)
        self.content_layout.setSpacing(3)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        for stage, capabilities in catalog.grouped():
            heading = QLabel(STAGE_TITLES[stage])
            heading.setObjectName("navigationSection")
            self._headings.append(heading)
            self.content_layout.addWidget(heading)
            for capability in capabilities:
                self._add_capability(capability.key, capability.title)
            self.content_layout.addSpacing(8)
        self.content_layout.addStretch(1)

        self.runtime_label = QLabel("Conda 环境")
        self.runtime_label.setObjectName("runtimeLabel")
        outer.addWidget(self.runtime_label)
        self.runtime_combo = QComboBox()
        self.runtime_combo.setObjectName("runtimeSelector")
        self.runtime_combo.setToolTip("只影响之后启动的训练、推理和处理任务；不会重启当前界面")
        self.runtime_combo.currentIndexChanged.connect(self._emit_runtime_profile)
        outer.addWidget(self.runtime_combo)

    @property
    def is_compact(self) -> bool:
        return self._compact

    def set_compact(self, compact: bool) -> None:
        compact = bool(compact)
        if compact == self._compact:
            return
        self._compact = compact
        if compact:
            self.setMinimumWidth(64)
            self.setMaximumWidth(64)
            self.brand.setText("C")
            self.brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.subtitle.hide()
            self.runtime_label.hide()
            self.runtime_combo.hide()
        else:
            self.setMinimumWidth(188)
            self.setMaximumWidth(228)
            self.brand.setText("COSMOS")
            self.brand.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self.subtitle.show()
            self.runtime_label.show()
            self.runtime_combo.show()
        for heading in self._headings:
            heading.setVisible(not compact)
        for button in self._buttons:
            title = str(button.property("fullTitle") or button.text())
            button.setText(title[:2] if compact else title)
            button.setToolTip(title)
            button.setAccessibleName(title)

    def set_runtime_environments(self, environments: list[CondaEnvInfo], selected_name: str) -> None:
        blocked = self.runtime_combo.blockSignals(True)
        self.runtime_combo.clear()
        if environments:
            for environment in environments:
                label = environment.name
                if environment.is_active:
                    label += " · 当前"
                self.runtime_combo.addItem(label, environment.name)
                index = self.runtime_combo.count() - 1
                self.runtime_combo.setItemData(index, environment.prefix, Qt.ItemDataRole.ToolTipRole)
        else:
            self.runtime_combo.addItem("当前解释器", "")
            self.runtime_combo.setEnabled(False)
            self.runtime_combo.setToolTip("未发现 Conda；任务继续使用启动工具箱的 Python")
        self.runtime_combo.blockSignals(blocked)
        self.set_runtime_profile(selected_name)

    def set_runtime_profile(self, name: str) -> None:
        target = str(name or "").casefold()
        for index in range(self.runtime_combo.count()):
            if str(self.runtime_combo.itemData(index) or "").casefold() == target:
                if self.runtime_combo.currentIndex() != index:
                    blocked = self.runtime_combo.blockSignals(True)
                    self.runtime_combo.setCurrentIndex(index)
                    self.runtime_combo.blockSignals(blocked)
                return

    def _emit_runtime_profile(self, index: int) -> None:
        self.runtimeProfileChanged.emit(str(self.runtime_combo.itemData(index) or ""))

    def _add_capability(self, key: str, title: str) -> None:
        index = len(self._keys)
        button = QPushButton(title)
        button.setObjectName("navigationItem")
        button.setCheckable(True)
        button.setProperty("navItem", True)
        button.setProperty("fullTitle", title)
        button.setAccessibleName(title)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda checked=False, row=index: self.setCurrentIndex(row))
        self._keys.append(key)
        self._buttons.append(button)
        self.content_layout.addWidget(button)

    def count(self) -> int:
        return len(self._keys)

    def currentIndex(self) -> int:
        return self._current_index

    def tabData(self, index: int) -> str | None:
        return self._keys[index] if 0 <= index < len(self._keys) else None

    def index_for_key(self, key: str) -> int:
        try:
            return self._keys.index(key)
        except ValueError:
            return -1

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._keys):
            return
        previous = self._current_index
        self._current_index = index
        for row, button in enumerate(self._buttons):
            button.setChecked(row == index)
        if previous != index:
            self.currentChanged.emit(index)


class ProjectHeader(QFrame):
    settings_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectHeader")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 11, 16, 11)
        layout.setSpacing(12)

        text = QVBoxLayout()
        text.setSpacing(0)
        self.project_label = QLabel("未命名项目")
        self.project_label.setObjectName("projectHeaderTitle")
        self.activity_label = QLabel("项目 / 项目概览")
        self.activity_label.setObjectName("projectHeaderBreadcrumb")
        text.addWidget(self.project_label)
        text.addWidget(self.activity_label)
        layout.addLayout(text)
        layout.addStretch(1)

        self.dataset_chip = QLabel("数据集未选择")
        self.dataset_chip.setObjectName("contextChip")
        self.model_chip = QLabel("模型未选择")
        self.model_chip.setObjectName("contextChip")
        layout.addWidget(self.dataset_chip)
        layout.addWidget(self.model_chip)
        self.settings_button = QPushButton("项目设置")
        self.settings_button.setObjectName("headerSettingsButton")
        self.settings_button.clicked.connect(self.settings_requested)
        layout.addWidget(self.settings_button)

    def apply_state(self, state: ProjectState) -> None:
        self.project_label.setText(state.project_name or "未命名项目")
        self.dataset_chip.setText(Path(state.dataset_root).name if state.dataset_root else "数据集未选择")
        self.dataset_chip.setToolTip(state.dataset_root)
        self.model_chip.setText(Path(state.model_path).name if state.model_path else "模型未选择")
        self.model_chip.setToolTip(state.model_path)

    def set_activity(self, stage_title: str, activity_title: str) -> None:
        self.activity_label.setText(f"{stage_title}  /  {activity_title}")


class ActivityHost(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("activityHost")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 12)
        layout.setSpacing(10)

        self.header = QFrame()
        self.header.setObjectName("activityTitleBar")
        header_layout = QVBoxLayout(self.header)
        header_layout.setContentsMargins(2, 0, 2, 4)
        header_layout.setSpacing(2)
        self.title = QLabel()
        self.title.setObjectName("activityTitle")
        self.description = QLabel()
        self.description.setObjectName("activityDescription")
        self.description.setWordWrap(True)
        header_layout.addWidget(self.title)
        header_layout.addWidget(self.description)
        layout.addWidget(self.header)

        self.stack = QStackedWidget()
        self.stack.setObjectName("activityStack")
        layout.addWidget(self.stack, 1)

    def set_activity(self, title: str, description: str) -> None:
        self.title.setText(title)
        self.description.setText(description)

    def set_nested(self, nested: bool) -> None:
        self.header.setVisible(not nested)

    def apply_layout(self, presentation: LayoutPresentation) -> None:
        layout = self.layout()
        layout.setContentsMargins(
            presentation.page_margin,
            presentation.page_margin,
            presentation.page_margin,
            max(10, presentation.page_margin - 4),
        )
        layout.setSpacing(presentation.page_spacing)
        self.setProperty("density", presentation.density.value)


class ProjectInspector(QFrame):
    dataset_requested = Signal()
    model_requested = Signal()
    image_dir_requested = Signal()
    annotation_dir_requested = Signal()
    output_root_requested = Signal()
    close_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectInspector")
        self.setMinimumWidth(286)
        self.setMaximumWidth(360)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        title = QLabel("项目上下文")
        title.setObjectName("inspectorTitle")
        header.addWidget(title)
        header.addStretch(1)
        close_button = QPushButton("×")
        close_button.setObjectName("inspectorClose")
        close_button.setFixedSize(28, 28)
        close_button.clicked.connect(self.close_requested)
        header.addWidget(close_button)
        outer.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 8, 0, 8)
        content_layout.setSpacing(12)
        self.project_name_edit = self._field(content_layout, "项目名称")
        self.dataset_edit = self._path_field(content_layout, "数据集", self.dataset_requested)
        self.image_dir_edit = self._path_field(content_layout, "图片目录", self.image_dir_requested)
        self.annotation_dir_edit = self._path_field(content_layout, "标注目录", self.annotation_dir_requested)
        self.model_edit = self._path_field(content_layout, "当前模型", self.model_requested)
        self.output_root_edit = self._path_field(content_layout, "产物目录", self.output_root_requested)

        artifact_title = QLabel("最近产物")
        artifact_title.setObjectName("inspectorSection")
        content_layout.addWidget(artifact_title)
        self.artifact_list = QListWidget()
        self.artifact_list.setObjectName("artifactList")
        self.artifact_list.setMinimumHeight(150)
        content_layout.addWidget(self.artifact_list)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

    @staticmethod
    def _field(layout: QVBoxLayout, title: str) -> QLineEdit:
        label = QLabel(title)
        label.setObjectName("fieldLabel")
        layout.addWidget(label)
        edit = QLineEdit()
        layout.addWidget(edit)
        return edit

    def _path_field(self, layout: QVBoxLayout, title: str, signal: Signal) -> QLineEdit:
        label = QLabel(title)
        label.setObjectName("fieldLabel")
        layout.addWidget(label)
        row = QHBoxLayout()
        edit = QLineEdit()
        button = QPushButton("选择")
        button.setProperty("buttonRole", "secondary")
        button.clicked.connect(signal)
        row.addWidget(edit, 1)
        row.addWidget(button)
        layout.addLayout(row)
        return edit

    def apply_state(self, state: ProjectState) -> None:
        values = {
            self.project_name_edit: state.project_name,
            self.dataset_edit: state.dataset_root,
            self.image_dir_edit: state.image_dir,
            self.annotation_dir_edit: state.annotation_dir,
            self.model_edit: state.model_path,
            self.output_root_edit: state.output_root,
        }
        for edit, value in values.items():
            blocked = edit.blockSignals(True)
            edit.setText(value)
            edit.blockSignals(blocked)

    def apply_artifacts(self, artifacts: tuple[ProjectArtifact, ...]) -> None:
        self.artifact_list.clear()
        if not artifacts:
            item = QListWidgetItem("还没有登记产物")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.artifact_list.addItem(item)
            return
        for artifact in artifacts:
            item = QListWidgetItem(f"{artifact.name}\n{artifact.kind.value} · {Path(artifact.path).name}")
            item.setToolTip(artifact.path)
            self.artifact_list.addItem(item)

    def apply_layout(self, presentation: LayoutPresentation) -> None:
        mode = "overlay" if presentation.inspector_overlay else "docked"
        self.setProperty("presentationMode", mode)
        if presentation.inspector_overlay:
            self.setMinimumWidth(260)
            self.setMaximumWidth(300)
        else:
            self.setMinimumWidth(286)
            self.setMaximumWidth(360)


class TaskStatusStrip(QFrame):
    """Persistent compact projection of the shared task ledger."""

    open_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("taskStatusStrip")
        self.setAccessibleName("运行任务摘要")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 7, 16, 7)
        layout.setSpacing(10)
        self.title = QLabel("任务")
        self.title.setObjectName("taskStatusTitle")
        layout.addWidget(self.title)
        self.summary = QLabel()
        self.summary.setObjectName("taskStatusSummary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary, 1)
        self.open_button = QPushButton("查看任务")
        self.open_button.setProperty("buttonRole", "ghost")
        self.open_button.setAccessibleName("打开任务中心")
        self.open_button.clicked.connect(self.open_requested)
        layout.addWidget(self.open_button)
        self.hide()

    def apply_presentations(self, presentations: tuple[object, ...]) -> None:
        # ``TaskPresentation.active`` is the single status projection.  This
        # shell must not duplicate the ledger's status set or vocabulary.
        active = [item for item in presentations if bool(getattr(item, "active", False))]
        if not presentations:
            self.hide()
            return
        latest = active[0] if active else presentations[0]
        count = len(active)
        self.title.setText(f"运行任务 {count}" if active else "最近任务")
        self.summary.setText(str(getattr(latest, "summary", "")))
        self.setAccessibleDescription(self.summary.text())
        self.show()
