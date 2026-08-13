from __future__ import annotations

from pathlib import Path

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
from .task_center import TaskStatus


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
        title = QLabel("从数据到模型，沿一条项目路径工作")
        title.setObjectName("overviewHeroTitle")
        intro = QLabel("功能不再按旧工具拆分；选择一个项目目标，工作台会带入当前数据、标注、模型和产物。")
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
            ("01", "准备数据", "浏览、ROI、裁剪、增强和数据整理", "image"),
            ("02", "标注与复核", "CAB-F 点边标注、筛选和数据集导出", "cabf.workflow"),
            ("03", "训练模型", "直接进入当前项目的 Sew Point Connect 训练", "training.sew_point_conntect"),
            ("04", "查看产物", "集中查看训练输出、模型和导出记录", "project.artifacts"),
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
        ready = sum(bool(value) for value in (state.dataset_root, state.annotation_dir, state.model_path))
        self.readiness.setText(f"项目就绪度 {ready}/3")
        self.summary.setText(
            f"数据集  {state.dataset_root or '尚未选择'}\n"
            f"标注集  {state.annotation_dir or '尚未选择'}\n"
            f"当前模型  {state.model_path or '尚未选择'}"
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


class SewPointConnectActivity(QWidget):
    """Native project path that links CAB-F data work to model training."""

    def __init__(self, runtime: CapabilityRuntime, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        banner = QFrame()
        banner.setObjectName("pipelineBanner")
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(20, 16, 20, 16)
        text = QVBoxLayout()
        title = QLabel("Sew Point Connect 项目路径")
        title.setObjectName("pipelineTitle")
        subtitle = QLabel("沿同一数据集完成筛选、点边修正、训练和产物登记。")
        subtitle.setObjectName("pipelineSubtitle")
        text.addWidget(title)
        text.addWidget(subtitle)
        banner_layout.addLayout(text, 1)
        self.status_badge = QLabel()
        self.status_badge.setObjectName("pipelineStatus")
        banner_layout.addWidget(self.status_badge)
        layout.addWidget(banner)

        self.steps_layout = QGridLayout()
        self.steps_layout.setSpacing(12)
        self.step_cards: list[_PipelineStep] = []
        definitions = (
            ("1", "准备数据", "检查图片、ROI 与数据目录", "image", "打开素材工作区"),
            ("2", "筛选与标注", "进入 CAB-F 流程完成点边复核", "cabf.workflow", "打开 CAB-F 流程"),
            ("3", "训练连边模型", "自动选择 Sew Point Connect 训练能力", "training.sew_point_conntect", "配置训练"),
            ("4", "检查产物", "查看训练运行输出和登记产物", "project.artifacts", "查看产物"),
        )
        for index, (number, title_text, description, key, button_text) in enumerate(definitions):
            step = _PipelineStep(
                number,
                title_text,
                description,
                button_text,
                lambda target=key: runtime.open_capability(target),
            )
            self.step_cards.append(step)
            self.steps_layout.addWidget(step, index // 2, index % 2)
        layout.addLayout(self.steps_layout, 1)

        hint = QLabel("提示：通过左侧流程导航切换功能时，项目上下文会持续保留；训练完成后会自动进入统一任务中心。")
        hint.setObjectName("pipelineHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        context = runtime.project_context
        if hasattr(context, "changed"):
            context.changed.connect(self.apply_project_context)
        session = runtime.project_session
        if hasattr(session, "changed"):
            session.changed.connect(self._refresh_from_runtime)
        self.apply_project_context(context.state)

    def apply_project_context(self, state: ProjectState) -> None:
        has_data = bool(state.image_dir or state.dataset_root)
        has_annotations = bool(state.annotation_dir and Path(state.annotation_dir).exists())
        has_model = bool(state.model_path)
        has_artifacts = bool(getattr(self.runtime.project_session, "artifacts", ()))
        statuses = (has_data, has_annotations, has_model, has_artifacts)
        for card, complete in zip(self.step_cards, statuses):
            card.set_complete(complete)
        completed = sum(statuses)
        self.status_badge.setText(f"{completed}/4 已就绪")

    def _refresh_from_runtime(self) -> None:
        self.apply_project_context(self.runtime.project_context.state)


class _PipelineStep(QFrame):
    def __init__(self, number: str, title: str, description: str, button_text: str, callback, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pipelineStep")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        top = QHBoxLayout()
        badge = QLabel(number)
        badge.setObjectName("pipelineStepNumber")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(30, 30)
        top.addWidget(badge)
        self.state = QLabel("待配置")
        self.state.setObjectName("pipelineStepState")
        top.addStretch(1)
        top.addWidget(self.state)
        layout.addLayout(top)
        title_label = QLabel(title)
        title_label.setObjectName("pipelineStepTitle")
        layout.addWidget(title_label)
        description_label = QLabel(description)
        description_label.setObjectName("pipelineStepDescription")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        layout.addStretch(1)
        button = QPushButton(button_text)
        button.setProperty("buttonRole", "secondary")
        button.clicked.connect(callback)
        layout.addWidget(button)

    def set_complete(self, complete: bool) -> None:
        self.state.setText("已就绪" if complete else "待配置")
        self.state.setProperty("complete", complete)
        self.state.style().unpolish(self.state)
        self.state.style().polish(self.state)


class ArtifactsActivity(QWidget):
    def __init__(self, runtime: CapabilityRuntime, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.runtime = runtime
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        intro = QLabel("训练、推理和导出的结果会集中登记在这里，并保留来源任务。")
        intro.setObjectName("activityDescription")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.state_stack = QStackedWidget()
        self.state_stack.setObjectName("activityStateStack")
        self.empty_state = _EmptyState(
            "暂无项目产物",
            "完成训练、推理或数据集导出后，模型和输出目录会集中显示在这里。",
            "进入训练",
            lambda: runtime.open_capability("training.sew_point_conntect"),
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
            "从训练页面启动任务后，可以在这里查看进度、日志和最终状态。",
            "进入训练",
            lambda: runtime.open_capability("training.sew_point_conntect"),
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
        names = {
            TaskStatus.PENDING: "等待",
            TaskStatus.RUNNING: "运行中",
            TaskStatus.SUCCESS: "完成",
            TaskStatus.BUSINESS_NG: "业务 NG",
            TaskStatus.FAILED: "失败",
            TaskStatus.STOPPED: "停止",
        }
        tasks = self.runtime.task_center.tasks
        if not tasks:
            self.log.clear()
            self.state_stack.setCurrentWidget(self.empty_state)
            return
        self.state_stack.setCurrentWidget(self.content)
        for task in tasks:
            item = QListWidgetItem(f"{task.title}\n{names[task.status]} {task.progress_text}")
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
        self.task_summary.setText(f"{task.title} · {task.status.value}")
        self.stop_button.setEnabled(task.status == TaskStatus.RUNNING and task.cancellable)
        for line in task.logs:
            self.log.addItem(line)

    def _cancel_selected(self) -> None:
        current = self.list.currentItem()
        if current is not None:
            self.runtime.task_center.cancel(str(current.data(Qt.ItemDataRole.UserRole)))


class _EmptyState(QFrame):
    """Plain, actionable empty state shared by native project pages."""

    def __init__(self, title: str, description: str, action_text: str, action, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("emptyState")
        self.setStyleSheet(
            "QFrame#emptyState { background: #ffffff; border: 1px solid #cfd3d7; border-radius: 3px; }"
            "QLabel#emptyStateTitle { color: #202428; font-size: 15px; font-weight: 600; }"
            "QLabel#emptyStateDescription { color: #697077; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)
        layout.addStretch(1)
        title_label = QLabel(title)
        title_label.setObjectName("emptyStateTitle")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label = QLabel(description)
        description_label.setObjectName("emptyStateDescription")
        description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description_label.setWordWrap(True)
        button = QPushButton(action_text)
        button.setProperty("buttonRole", "secondary")
        button.setMaximumWidth(120)
        button.clicked.connect(action)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(button, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)
