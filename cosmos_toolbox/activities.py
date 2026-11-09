from __future__ import annotations


from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .capabilities import CapabilityRuntime
from .project_context import ProjectState
from .project_session import ArtifactKind, ProjectArtifact
from .ui.primitives import EmptyState


class ProjectOverviewPage(QWidget):
    def __init__(self, runtime: CapabilityRuntime, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        hero = QFrame()
        hero.setObjectName("overviewHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(22, 20, 22, 20)
        hero_text = QVBoxLayout()
        title = QLabel("从原图到标注数据集，沿一条项目路径工作")
        title.setObjectName("overviewHeroTitle")
        intro = QLabel("选择一个项目目标，工作台会带入当前项目的图片、标注和输出目录。")
        intro.setObjectName("overviewHeroText")
        intro.setWordWrap(True)
        hero_text.addWidget(title)
        hero_text.addWidget(intro)
        hero_layout.addLayout(hero_text, 1)
        self.readiness = QLabel()
        self.readiness.setObjectName("readinessBadge")
        hero_layout.addWidget(self.readiness)
        layout.addWidget(hero)

        section = QLabel("项目流程")
        section.setObjectName("sectionTitle")
        layout.addWidget(section)
        cards = QGridLayout()
        cards.setSpacing(12)
        stages = (
            ("01", "CAB-F 标注流程", "筛选、点边标注、校验与数据集导出", "cabf.workflow"),
            ("02", "图片处理", "浏览、压缩、裁剪和数据整理", "image"),
            ("03", "现场 NG 数据", "从数据库或 Excel 筛选并提取 NG 原图", "data.ng_extract"),
            ("04", "查看产物", "集中查看处理任务和导出记录", "project.artifacts"),
        )
        for index, (number, title_text, description, key) in enumerate(stages):
            card = _ActivityCard(number, title_text, description, lambda target=key: runtime.open_capability(target))
            cards.addWidget(card, index // 2, index % 2)
        layout.addLayout(cards)

        self.summary = QLabel()
        self.summary.setObjectName("overviewSummary")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        layout.addStretch(1)

    def apply_project_context(self, state: ProjectState) -> None:
        ready = sum(bool(value) for value in (state.dataset_root, state.image_dir, state.annotation_dir))
        self.readiness.setText(f"项目就绪度 {ready}/3")
        self.summary.setText(
            f"数据集  {state.dataset_root or '尚未选择'}\n"
            f"图片目录  {state.image_dir or '尚未选择'}\n"
            f"标注集  {state.annotation_dir or '尚未选择'}"
        )


class _ActivityCard(QFrame):
    def __init__(self, number: str, title: str, description: str, open_action, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("activityCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        number_label = QLabel(number)
        number_label.setObjectName("activityCardNumber")
        layout.addWidget(number_label)
        title_label = QLabel(title)
        title_label.setObjectName("activityCardTitle")
        layout.addWidget(title_label)
        description_label = QLabel(description)
        description_label.setObjectName("activityCardDescription")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        layout.addStretch(1)
        button = QPushButton("打开")
        button.setProperty("buttonRole", "secondary")
        button.clicked.connect(open_action)
        layout.addWidget(button)


class ArtifactsActivity(QWidget):
    def __init__(self, runtime: CapabilityRuntime, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        intro = QLabel("处理和导出任务的结果会集中登记在这里，并保留来源任务。")
        intro.setObjectName("activityDescription")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.state_stack = QStackedWidget()
        self.state_stack.setObjectName("activityStateStack")
        self.empty_state = _EmptyState(
            "暂无项目产物",
            "完成数据处理或数据集导出后，输出目录会集中显示在这里。",
            "进入标注流程",
            lambda: runtime.open_capability("cabf.workflow"),
        )
        self.list = QListWidget()
        self.list.setObjectName("artifactsActivityList")
        self.state_stack.addWidget(self.empty_state)
        self.state_stack.addWidget(self.list)
        layout.addWidget(self.state_stack, 1)
        runtime.project_session.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        self.list.clear()
        artifacts: tuple[ProjectArtifact, ...] = self.runtime.project_session.recent(100)
        if not artifacts:
            self.state_stack.setCurrentWidget(self.empty_state)
            return
        self.state_stack.setCurrentWidget(self.list)
        labels = {
            ArtifactKind.DATASET: "数据集",
            ArtifactKind.ANNOTATIONS: "标注集",
            ArtifactKind.MODEL: "模型",
            ArtifactKind.RUN_OUTPUT: "运行产物",
            ArtifactKind.EXPORT: "导出",
        }
        for artifact in artifacts:
            item = QListWidgetItem(
                f"{artifact.name}\n{labels[artifact.kind]} · {artifact.created_at}\n{artifact.path}"
            )
            item.setToolTip(artifact.path)
            self.list.addItem(item)


class TaskCenterActivity(QWidget):
    def __init__(self, runtime: CapabilityRuntime, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.state_stack = QStackedWidget()
        self.state_stack.setObjectName("activityStateStack")
        self.empty_state = _EmptyState(
            "暂无运行任务",
            "启动压缩、NG 提取或导出等任务后，可以在这里查看进度、日志和最终状态。",
            "打开项目概览",
            lambda: runtime.open_capability("overview"),
        )
        self.content = QWidget()
        self.content.setObjectName("taskContent")
        content_layout = QHBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)
        self.list = QListWidget()
        self.list.setMinimumWidth(300)
        self.list.currentItemChanged.connect(self._show_log)
        self.log = QListWidget()
        log_area = QWidget()
        log_layout = QVBoxLayout(log_area)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_toolbar = QHBoxLayout()
        self.task_summary = QLabel("选择任务查看日志")
        self.task_summary.setObjectName("cabfMuted")
        log_toolbar.addWidget(self.task_summary, 1)
        self.stop_button = QPushButton("停止任务")
        self.stop_button.setProperty("buttonRole", "secondary")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._cancel_selected)
        log_toolbar.addWidget(self.stop_button)
        log_layout.addLayout(log_toolbar)
        log_layout.addWidget(self.log, 1)
        content_layout.addWidget(self.list, 1)
        content_layout.addWidget(log_area, 3)
        self.state_stack.addWidget(self.empty_state)
        self.state_stack.addWidget(self.content)
        layout.addWidget(self.state_stack)
        runtime.task_center.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        selected = self.list.currentItem().data(Qt.ItemDataRole.UserRole) if self.list.currentItem() else ""
        self.list.clear()
        tasks = self.runtime.task_center.tasks
        if not tasks:
            self.log.clear()
            self.state_stack.setCurrentWidget(self.empty_state)
            return
        self.state_stack.setCurrentWidget(self.content)
        for task in tasks:
            presentation = self.runtime.task_center.presentation(task.task_id)
            detail = presentation.summary if presentation is not None else task.title
            item = QListWidgetItem(detail)
            item.setData(Qt.ItemDataRole.UserRole, task.task_id)
            self.list.addItem(item)
            if task.task_id == selected:
                self.list.setCurrentItem(item)
        if self.list.count() and self.list.currentRow() < 0:
            self.list.setCurrentRow(0)

    def _show_log(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        self.log.clear()
        if current is None:
            self.task_summary.setText("选择任务查看日志")
            self.stop_button.setEnabled(False)
            return
        task = self.runtime.task_center.get(str(current.data(Qt.ItemDataRole.UserRole)))
        if task is None:
            self.stop_button.setEnabled(False)
            return
        presentation = self.runtime.task_center.presentation(task.task_id)
        self.task_summary.setText(presentation.summary if presentation is not None else task.title)
        self.stop_button.setEnabled(bool(presentation and presentation.cancellable))
        self.stop_button.setText(
            presentation.button_label if presentation and presentation.button_label else "停止任务"
        )
        for line in task.logs:
            self.log.addItem(line)

    def _cancel_selected(self) -> None:
        current = self.list.currentItem()
        if current is not None:
            self.runtime.task_center.cancel(str(current.data(Qt.ItemDataRole.UserRole)))


class _EmptyState(EmptyState):
    """Compatibility name for the shared actionable empty state."""

    def __init__(self, title: str, description: str, action_text: str, action, parent=None) -> None:
        super().__init__(title, description, action_text, parent)
        self.action_requested.connect(action)
