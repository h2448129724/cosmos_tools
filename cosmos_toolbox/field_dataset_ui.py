"""Desktop entry point for CAB-F field dataset collection."""
from __future__ import annotations

import json
import hashlib
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QEvent, QThread, QUrl, Signal, Qt, QSize
from PySide6.QtGui import QDesktopServices, QImageReader, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
    QListWidget, QListWidgetItem, QTabWidget, QTreeView, QListView, QAbstractItemView, QSizePolicy,
    QTableWidget, QTableWidgetItem, QHeaderView,
)

from .paths import ensure_import_paths


def configured_product(path):
    import yaml
    document = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    inspection = document.get('inspection', {}) if isinstance(document, dict) else {}
    product = inspection.get('product') if isinstance(inspection, dict) else None
    if product not in {'D01-L', 'D01-R'}:
        raise ValueError('配置文件 inspection.product 必须是 D01-L 或 D01-R')
    return product


class DatasetWorker(QThread):
    progress = Signal(dict)
    result = Signal(dict)
    error = Signal(str)

    def __init__(self, options: dict, control, scan_only: bool = False, parent=None, environment_name=None):
        super().__init__(parent)
        self.options = options
        self.control = control
        self.scan_only = scan_only
        self.environment_name = environment_name

    def run(self):
        try:
            if not self.scan_only and ('jobs' in self.options or 'sources' in self.options):
                self._run_batch()
                return
            if 'sources' not in self.options:
                self.result.emit(self._run_one(self.options))
                return
            base = dict(self.options)
            sources = base.pop('sources')
            if base.get('product_config'):
                base['product'] = configured_product(base['product_config'])
            results = []
            errors = []
            for index, source in enumerate(sources):
                if self.control.stopped.is_set():
                    break
                options = {**base, 'source': source}
                # Stable per-source workspaces allow resume and avoid mixing folders.
                if len(sources) > 1:
                    suffix = hashlib.sha256(str(Path(source).resolve()).casefold().encode()).hexdigest()[:12]
                    options['output'] = str(Path(base['output']) / f'{Path(source).name}_{suffix}')
                self.progress.emit({'message': f'文件夹 {index + 1}/{len(sources)}：{source}'})
                try:
                    results.append({'source': source, **self._run_one(options)})
                except Exception as exc:
                    errors.append({'source': source, 'error': f'{type(exc).__name__}: {exc}'})
                    self.progress.emit({'message': f'文件夹失败：{source}：{exc}'})
            result = {'batch': True, 'results': results, 'errors': errors,
                      'stopped': self.control.stopped.is_set()}
            if self.scan_only:
                result.update(scan=True, count=sum(r['count'] for r in results),
                              bytes=sum(r['bytes'] for r in results))
            self.result.emit(result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

    def _run_batch(self):
        from .field_history import new_batch, workspace_for, config_hash, read_run, validate_options
        from .field_dataset import _json
        jobs = self.options.get('jobs')
        if jobs is None:
            base = dict(self.options)
            sources = base.pop('sources')
            if base.get('product_config'):
                base['product'] = configured_product(base['product_config'])
            jobs = []
            for source in sources:
                options = {**base, 'source': source}
                if len(sources) > 1:
                    suffix = hashlib.sha256(str(Path(source).resolve()).casefold().encode()).hexdigest()[:12]
                    options['output'] = str(Path(base['output']) / f'{Path(source).name}_{suffix}')
                jobs.append({'options': options, 'environment': self.environment_name})
        path, batch = new_batch(self.options['output'], jobs)
        self.progress.emit({'message': f'批次清单已保存（包含尚未开始的文件夹）：{path}'})
        results, errors = [], []
        for index, job in enumerate(batch['jobs']):
            while self.control.paused.is_set() and not self.control.stopped.wait(.1):
                pass
            if self.control.stopped.is_set():
                break
            options = dict(job['options'])
            source = options['source']
            self.progress.emit({'message': f'文件夹 {index + 1}/{len(batch["jobs"])}：{source}'})
            job['status'] = 'running'
            job['known_runs'] = [p.name for p in (workspace_for(options) / 'runs').glob('*.json')]
            _json(path, batch)
            try:
                if config_hash(options) != job['config_hash']:
                    raise ValueError('配置文件与批次保存时不同，已拒绝此任务，旧数据保留')
                if options.get('resume_from'):
                    validate_options(read_run(options['resume_from']), options)
                self.environment_name = job.get('environment')
                result = {'source': source, **self._run_one(options)}
                if result.get('stopped'):
                    self.control.stopped.set()
                results.append(result)
                job['result'] = result
                job['status'] = 'stopped' if result.get('stopped') or self.control.stopped.is_set() else 'complete'
                if result.get('failed_model_jobs') or result.get('export', {}).get('errors') or result.get('export', {}).get('preview_errors'):
                    job['status'] = 'failed'
            except Exception as exc:
                message = f'{type(exc).__name__}: {exc}'
                job.update(status='failed', error=message)
                errors.append({'source': source, 'error': message})
                self.progress.emit({'message': f'文件夹失败：{source}：{message}；其余任务按各自设置继续。'})
            _json(path, batch)
        self.result.emit({'batch': True, 'results': results, 'errors': errors,
                          'stopped': self.control.stopped.is_set(), 'batch_manifest': str(path)})

    def _run_one(self, options):
        options = dict(options)
        if not self.scan_only:
            options['task_environment'] = self.environment_name
        if self.environment_name:
            from .field_dataset_runtime import run_in_environment
            return run_in_environment(options, self.environment_name, self.control,
                                      self.progress.emit, self.scan_only)
        from .field_dataset import run, scan
        if self.scan_only:
            files = scan(options['source'], face=options['face'])
            return {'scan': True, 'count': len(files), 'bytes': sum(Path(p).stat().st_size for p in files)}
        return run(**options, control=self.control, on_progress=self.progress.emit)


class FieldDatasetPage(QWidget):
    """Independent model selection with a cooperative background runner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        ensure_import_paths()
        from .field_dataset import MODEL_IDS

        self.setProperty("ownsPageHeader", True)
        self.setWindowTitle("CAB-F 现场数据集生成")
        self.worker = None
        self.control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event(),
                                       export_on_stop=threading.Event())
        self._closing_window = None
        self._resume_record = None
        self._resume_jobs = None
        layout = QVBoxLayout(self)
        title = QLabel("CAB-F 现场数据集生成")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        layout.addWidget(title)
        help_text = QLabel("单独勾选任一模型，或组合生成多套数据；必要的定位步骤自动执行。自动标签需要人工复核。")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.settings = QGroupBox("输入与生成设置")
        form = QFormLayout(self.settings)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.source = QPlainTextEdit()
        self.source.setPlaceholderText('每行一个原图文件夹；可多选添加，也可粘贴多个路径')
        self.source.setMinimumHeight(85)
        self.source.setMaximumHeight(110)
        self.output = QLineEdit()
        source_row = QWidget()
        source_layout = QHBoxLayout(source_row)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.addWidget(self.source)
        add_sources = QPushButton('添加文件夹（多选）…')
        add_sources.clicked.connect(self._browse_sources)
        source_layout.addWidget(add_sources)
        form.addRow("原图文件夹列表", source_row)
        form.addRow("输出文件夹", self._folder_row(self.output))
        self.history_list = QListWidget()
        self.history_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.history_list.setMinimumHeight(100)
        self.history_list.setMaximumHeight(150)
        self.history_list.setToolTip('Ctrl / Shift 多选；同一工作区只能选择一条记录，整批与单项不可重复选择。')
        form.addRow('历史任务（多选）', self.history_list)
        history_actions = QWidget()
        from .ui.primitives import ActionBar
        history_bar = QVBoxLayout(history_actions)
        history_bar.setContentsMargins(0, 0, 0, 0)
        actions_bar = ActionBar()
        history_bar.addWidget(actions_bar)
        refresh_history = QPushButton('读取输出目录历史')
        restore_history = QPushButton('恢复选中任务')
        new_task = QPushButton('退出续跑 / 新任务')
        select_all = QPushButton('全选')
        for button in (refresh_history, restore_history, select_all, new_task):
            actions_bar.add_widget(button)
        refresh_history.clicked.connect(self._load_history)
        restore_history.clicked.connect(self._restore_history)
        select_all.clicked.connect(self.history_list.selectAll)
        new_task.clicked.connect(self._new_task)
        form.addRow(history_actions)
        self.environment = QComboBox()
        self.environment.setEditable(True)
        self.environment.addItem('onnx-gpu')
        self.environment.setToolTip('任务在所选 Conda 环境中运行；缺少依赖会报错，不会自动安装。')
        environment_row = QWidget()
        environment_layout = QHBoxLayout(environment_row)
        environment_layout.setContentsMargins(0, 0, 0, 0)
        environment_layout.addWidget(self.environment, 1)
        refresh_environments = QPushButton('加载环境列表')
        refresh_environments.clicked.connect(self._load_environments)
        environment_layout.addWidget(refresh_environments)
        form.addRow('执行 Conda 环境', environment_row)
        self.product = QLabel('款号由配置文件读取')
        self.face = QComboBox()
        for label, value in [("全部", "all"), ("Top", "top"), ("Bottom", "bottom")]:
            self.face.addItem(label, value)
        self.mode = QComboBox()
        self.mode.addItem("提取并自动标注（伪标签）", "auto")
        self.mode.addItem("仅提取图片（待标注）", "images")
        self.limit = QSpinBox()
        self.limit.setRange(0, 10000000)
        self.limit.setSpecialValueText("全部图片")
        self.limit.setToolTip("设置少量图片进行试运行；0 表示全部。按扫描顺序取样。")
        form.addRow("配置款号", self.product)
        self.product_config = QLineEdit()
        self.product_config.setPlaceholderText('请选择 YAML；自动读取 inspection.product，无需手选款号')
        self.product_config.textChanged.connect(self._update_config_product)
        self.product_config.setToolTip('可指定任意 YAML，包括 .local.yaml；不与默认文件合并。相对模板路径仍遵循 Cosmos 原有规则。')
        config_row = QWidget()
        config_layout = QHBoxLayout(config_row)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.addWidget(self.product_config)
        config_browse = QPushButton('选择 YAML…')
        config_browse.clicked.connect(self._browse_product_config)
        config_layout.addWidget(config_browse)
        form.addRow('产品配置文件', config_row)
        form.addRow("正反面", self.face)
        form.addRow("生成方式", self.mode)
        self.ng_only = QCheckBox('仅保存对应检查项为 NG 的数据')
        self.ng_only.setToolTip('默认关闭。使用配置中的正式检查判定；共用模型任一关联检查项 NG 即保存。异常、未执行不视为 NG。会增加检查耗时。')
        form.addRow('保存条件', self.ng_only)
        self.export_scope = QComboBox()
        self.export_scope.addItem('仅本次运行范围（推荐，含断点复用）', 'current')
        self.export_scope.addItem('全部历史结果（可能含旧版标签）', 'history')
        self.export_scope.setToolTip('历史模式会包含同一输出账本中不同来源和不同版本的数据；不会自动去重。')
        form.addRow('导出范围', self.export_scope)
        form.addRow("每个文件夹最多处理", self.limit)
        # Scroll the form instead of squeezing rows below their styled size hints.
        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)
        settings_layout.addWidget(self.settings)
        settings_layout.addStretch()
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setWidget(settings_content)
        self.tabs.addTab(self.settings_scroll, "输入设置")

        self.models_group = QGroupBox("选择要导出的模型数据集")
        models_layout = QVBoxLayout(self.models_group)
        shortcuts = QHBoxLayout()
        for label, selection in [("全选", "all"), ("清空", "none"), ("尾部全套", "tail"), ("耳片全套", "ear")]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, value=selection: self._select(value))
            shortcuts.addWidget(button)
        models_layout.addLayout(shortcuts)
        container = QWidget()
        self.model_grid = QGridLayout(container)
        self.model_grid.setVerticalSpacing(14)
        self.model_grid.setHorizontalSpacing(24)
        self.checks = {}
        for index, (model_id, label) in enumerate(MODEL_IDS.items()):
            check = QCheckBox(f"{label}\n{model_id}")
            check.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            check.setToolTip("仅导出勾选项；必要的前置定位模型由执行器自动补齐。")
            self.checks[model_id] = check
            check.toggled.connect(self._update_dependencies)
            self.model_grid.addWidget(check, index // 2, index % 2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setMinimumHeight(180)
        self.model_scroll = scroll
        self.model_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        models_layout.addWidget(scroll)
        self.dependencies = QLabel("自动依赖：无")
        self.dependencies.setWordWrap(True)
        models_layout.addWidget(self.dependencies)
        self.tabs.addTab(self.models_group, "模型选择（0）")

        actions = QGridLayout()
        self.scan_button = QPushButton("扫描图片")
        self.start_button = QPushButton("开始生成")
        self.preview_button = QPushButton("试跑前 2 张")
        self.pause_button = QPushButton("暂停")
        self.stop_button = QPushButton("安全停止")
        self.stop_export_button = QPushButton("停止并导出")
        self.stop_export_button.setToolTip('完成当前原图后停止后续处理，导出本次已完成的数据；不再启动后续文件夹。')
        self.open_button = QPushButton("打开输出目录")
        for index, button in enumerate((self.scan_button, self.preview_button, self.start_button, self.pause_button, self.stop_button, self.stop_export_button, self.open_button)):
            actions.addWidget(button, index // 3, index % 3)
        self.start_button.setProperty('buttonRole', 'primary')
        layout.addLayout(actions)
        self.scan_button.clicked.connect(lambda: self._start(scan_only=True))
        self.start_button.clicked.connect(lambda: self._start())
        self.preview_button.clicked.connect(self._preview_run)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.stop_button.clicked.connect(self._stop)
        self.stop_export_button.clicked.connect(self._stop_and_export)
        self.open_button.clicked.connect(self._open_output)
        self.status = QLabel("请选择文件夹和模型；可先限制处理数量进行试运行。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.tabs.addTab(self.log, "运行日志")
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        self.load_preview_button = QPushButton("加载输出目录中的样本")
        self.load_preview_button.clicked.connect(self._load_previews)
        preview_layout.addWidget(self.load_preview_button)
        preview_row = QHBoxLayout()
        self.sample_list = QListWidget()
        self.sample_list.setMaximumWidth(320)
        self.sample_list.currentItemChanged.connect(self._show_sample)
        preview_row.addWidget(self.sample_list)
        self.sample_image = QLabel("生成后可查看裁片与标签叠加；最多载入 500 个样本。")
        self.sample_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setWidget(self.sample_image)
        preview_row.addWidget(preview_scroll, 1)
        preview_layout.addLayout(preview_row)
        self.preview_page = preview
        self.tabs.addTab(preview, "样本预览")
        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        result_hint = QLabel('以下仅汇总本次运行范围（含有效断点复用），不混入其他历史版本。未启用不算失败，需复核不自动作为负样本。')
        result_hint.setWordWrap(True)
        result_layout.addWidget(result_hint)
        self.result_table = QTableWidget(0, 7)
        self.result_table.setHorizontalHeaderLabels(['模型', '完成任务', 'OK 过滤', '未启用', '需复核', '技术失败', '裁片数'])
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        result_layout.addWidget(self.result_table)
        self.result_page = result_page
        self.tabs.addTab(result_page, '结果汇总')
        self._set_busy(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self, 'model_grid'):
            return
        # Keep long model identifiers readable in the toolbox's narrow content area.
        cell_width = max(check.sizeHint().width() for check in self.checks.values())
        columns = 2 if self.width() >= 2 * cell_width + 90 else 1
        for check in self.checks.values():
            self.model_grid.removeWidget(check)
        for index, check in enumerate(self.checks.values()):
            self.model_grid.addWidget(check, index // columns, index % columns)

    def _update_config_product(self):
        try:
            self.product.setText(configured_product(self.product_config.text().strip()))
        except Exception:
            self.product.setText('请选择有效配置（inspection.product）')

    def _browse_sources(self):
        dialog = QFileDialog(self, '选择多个原图文件夹（Ctrl / Shift 多选）')
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        for view in dialog.findChildren(QTreeView) + dialog.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        if dialog.exec():
            paths = self.source.toPlainText().splitlines() + dialog.selectedFiles()
            self.source.setPlainText('\n'.join(dict.fromkeys(p.strip() for p in paths if p.strip())))

    def _browse_product_config(self):
        from .paths import COSMOS_ROOT
        path, _ = QFileDialog.getOpenFileName(self, '选择产品配置',
            self.product_config.text().strip() or str(COSMOS_ROOT / 'conf/cabf'),
            'YAML 配置 (*.yaml *.yml);;所有文件 (*)')
        if path:
            self.product_config.setText(path)

    def _load_environments(self):
        from shared.conda_runtime import CondaEnvManager
        current = self.environment.currentText().strip() or 'onnx-gpu'
        names = [item.name for item in CondaEnvManager().list_envs(refresh=True)]
        self.environment.clear()
        self.environment.addItems(list(dict.fromkeys([current, 'onnx-gpu', *names])))
        self.environment.setCurrentText(current)
        if not names:
            QMessageBox.warning(self, 'Conda 环境', '未读取到环境列表，请确认 Conda 可用；也可手动输入环境名称。')

    def _update_dependencies(self):
        from .field_dataset import MODEL_DEPENDENCIES, MODEL_IDS
        selected = {key for key, check in self.checks.items() if check.isChecked()}
        self.tabs.setTabText(self.tabs.indexOf(self.models_group), f"模型选择（{len(selected)}）")
        required = set()
        pending = list(selected)
        while pending:
            for dependency in MODEL_DEPENDENCIES.get(pending.pop(), []):
                if dependency not in required:
                    required.add(dependency)
                    pending.append(dependency)
        labels = [MODEL_IDS[key] for key in MODEL_IDS if key in required - selected]
        self.dependencies.setText("自动依赖（只运行，不额外导出）：" + ("、".join(labels) or "无"))

    def _preview_run(self):
        previous = self.limit.value()
        self.limit.setValue(2)
        self._start()
        self.limit.setValue(previous)

    def _load_previews(self):
        self.sample_list.clear()
        root = Path(self.output.text().strip()).resolve()
        databases = ([root / 'run.db'] if (root / 'run.db').is_file() else []) + sorted(root.glob('*/run.db')) + sorted(root.glob('*/ng_only/run.db'))
        if not databases:
            self.sample_image.setText("输出目录尚无 run.db，请先生成数据。")
            return
        try:
            count = 0
            for database in databases:
                connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
                try:
                    runs = sorted((database.parent / 'runs').glob('*.json'))
                    run_id = json.loads(runs[-1].read_text(encoding='utf-8')).get('run_id') if runs else None
                    if run_id:
                        rows = connection.execute("SELECT model,details FROM items JOIN run_items ON item_key=key "
                            "WHERE status='complete' AND run_id=? ORDER BY model,key", (run_id,))
                    else:
                        self.log.appendPlainText(f'旧账本预览（历史结果，尚无运行范围）：{database}')
                        rows = connection.execute("SELECT model,details FROM items WHERE status='complete' ORDER BY model,key")
                    for model, details in rows:
                        for record in json.loads(details):
                            directory = Path(record["directory"]).resolve()
                            if root not in directory.parents:
                                continue
                            item = QListWidgetItem(f"{model}\n{Path(record.get('source', '')).name} / {directory.name}")
                            item.setData(Qt.ItemDataRole.UserRole, str(directory))
                            self.sample_list.addItem(item)
                            count += 1
                            if count >= 500:
                                break
                        if count >= 500:
                            break
                finally:
                    connection.close()
                if count >= 500:
                    break
            self.tabs.setCurrentWidget(self.preview_page)
            if self.sample_list.count():
                self.sample_list.setCurrentRow(0)
            else:
                self.sample_image.setText("暂无已提交样本。")
        except Exception as exc:
            self.log.appendPlainText(f"样本预览读取失败：{exc}")

    def _show_sample(self, item, previous=None):
        if item is None:
            return
        try:
            directory = Path(item.data(Qt.ItemDataRole.UserRole))
            reader = QImageReader(str(directory / "image.png"))
            original = reader.size()
            if not original.isValid():
                raise ValueError("裁片无法读取")
            scaled = original.scaled(QSize(1200, 1200), Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(scaled)
            pixmap = QPixmap.fromImage(reader.read())
            if pixmap.isNull():
                raise ValueError(reader.errorString())
            annotation = json.loads((directory / "image.json").read_text(encoding="utf-8"))
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#ff6038"), 2))
            sx, sy = pixmap.width() / original.width(), pixmap.height() / original.height()
            for shape in annotation.get("shapes", []):
                points = [(int(x * sx), int(y * sy)) for x, y in shape.get("points", [])]
                if shape.get("shape_type") == "rectangle" and len(points) == 2:
                    (x1, y1), (x2, y2) = points
                    painter.drawRect(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1))
                elif shape.get("shape_type") == "point" and points:
                    painter.drawEllipse(points[0][0]-3, points[0][1]-3, 6, 6)
                else:
                    segments = list(zip(points, points[1:]))
                    if shape.get("shape_type") == "polygon" and len(points) > 2:
                        segments.append((points[-1], points[0]))
                    for start, end in segments:
                        painter.drawLine(*start, *end)
                if points:
                    painter.drawText(points[0][0], max(12, points[0][1]-4), str(shape.get("label", "")))
            painter.end()
            self.sample_image.setPixmap(pixmap)
        except Exception as exc:
            self.sample_image.setText(f"预览失败：{exc}")

    def _folder_row(self, edit):
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(edit)
        browse = QPushButton("浏览…")
        browse.clicked.connect(lambda: self._browse(edit))
        box.addWidget(browse)
        return row

    def _browse(self, edit):
        chosen = QFileDialog.getExistingDirectory(self, "选择文件夹", edit.text())
        if chosen:
            edit.setText(chosen)

    def _select(self, selection):
        tail = {"tail_roi_detector", "tail_placement_classifier", "tail_cloth_roi_detector", "tail_cloth_seam_classifier", "hook_detector"}
        ear = {"ear_placement_classifier", "knife_segment"}
        for key, check in self.checks.items():
            check.setChecked(selection == "all" or (selection == "tail" and key in tail) or (selection == "ear" and key in ear))

    def _start(self, scan_only=False):
        if self.worker and self.worker.isRunning():
            return
        if self._resume_jobs is not None:
            if scan_only:
                QMessageBox.information(self, '批量续跑', '批量续跑按各任务自己的参数执行，请直接点击“继续选中任务”。')
                return
            self.control.paused.clear()
            self.control.stopped.clear()
            self.control.export_on_stop.clear()
            self._launch_worker({'jobs': self._resume_jobs, 'output': self.output.text().strip(),
                                 'sources': [job['options']['source'] for job in self._resume_jobs]}, False, None)
            return
        sources = list(dict.fromkeys(str(Path(p.strip().strip('"')).resolve())
                                     for p in self.source.toPlainText().splitlines() if p.strip()))
        output = self.output.text().strip()
        selected = [key for key, check in self.checks.items() if check.isChecked()]
        if not sources or any(not Path(p).is_dir() for p in sources):
            QMessageBox.warning(self, "输入目录", "请选择存在的原图文件夹。")
            return
        if not scan_only and (not output or not selected):
            QMessageBox.warning(self, "生成设置", "请选择输出文件夹，并至少勾选一个模型。")
            return
        self.control.paused.clear()
        self.control.stopped.clear()
        self.control.export_on_stop.clear()
        config_path = self.product_config.text().strip()
        try:
            product = configured_product(config_path)
        except Exception as exc:
            QMessageBox.warning(self, '产品配置', f'请选择有效的产品配置文件：{exc}')
            return
        if output:
            target = Path(output).resolve()
            if any(target == Path(p) or target in Path(p).parents or Path(p) in target.parents for p in sources):
                QMessageBox.warning(self, '输出目录', '输入与输出目录不能互相包含。')
                return
        options = dict(sources=sources, output=output, product=product, selected=selected,
                       face=self.face.currentData(), mode=self.mode.currentData(), limit=self.limit.value() or None,
                       ng_only=self.ng_only.isChecked(), export_scope=self.export_scope.currentData())
        config_path = self.product_config.text().strip()
        if config_path:
            if not Path(config_path).is_file():
                QMessageBox.warning(self, '产品配置', '选择的配置文件不存在。')
                return
            options['product_config'] = str(Path(config_path).resolve())
        environment_name = self.environment.currentText().strip()
        if not environment_name:
            QMessageBox.warning(self, 'Conda 环境', '请选择或输入执行环境名称。')
            return
        if not scan_only and self._resume_record:
            from .field_history import validate_options
            try:
                if len(sources) != 1:
                    raise ValueError('历史任务按单个来源工作区续跑，请勿添加其他原图目录')
                validate_options(self._resume_record, {**options, 'source': sources[0]})
            except ValueError as exc:
                QMessageBox.warning(self, '未开始续跑', str(exc))
                return
            options['resume_from'] = self._resume_record['path']
        elif not scan_only and Path(output).is_dir():
            from .field_history import discover
            history, history_errors = discover(output)
            existing_ledger = any(Path(output).glob('run.db')) or any(Path(output).glob('*/run.db')) or any(Path(output).glob('*/*/run.db'))
            existing_batch = any((Path(output) / 'batches').glob('*.json'))
            if (history or history_errors or existing_ledger or existing_batch) and QMessageBox.question(self, '输出目录包含历史任务',
                    '当前未选择历史任务续跑。建议先读取历史并恢复设置。\n仍按当前设置启动任务吗？旧记录不会删除。',
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
                return
        self._launch_worker(options, scan_only, environment_name)

    def _launch_worker(self, options, scan_only, environment_name):
        self.worker = DatasetWorker(options, self.control, scan_only, self, environment_name=environment_name)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_result)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._finished)
        self._set_busy(True, scan_only)
        self.progress_bar.setRange(0, 0)
        self.status.setText("正在扫描图片…" if scan_only else "正在准备模型与输出…")
        self.log.appendPlainText(self.status.text())
        self.tabs.setCurrentWidget(self.log)
        from .worker_task_bridge import bind_worker_task
        bind_worker_task(self, '现场数据集：扫描' if scan_only else '现场数据集：生成', 'cabf.field_dataset')
        self.worker.start()

    def _load_history(self):
        from .field_history import discover, discover_batches
        if not self.output.text().strip():
            QMessageBox.warning(self, '历史任务', '请先选择上次的输出目录。')
            return
        self.history_list.clear()
        records, errors = discover(self.output.text().strip())
        batches, batch_errors = discover_batches(self.output.text().strip())
        for batch in batches:
            statuses = batch['statuses']
            label = (f"整批 {Path(batch['path']).stem} | {len(statuses)} 文件夹 | "
                     f"完成 {statuses.count('complete')} / 待开始 {statuses.count('pending')} / 停止或异常 {sum(s in ('running', 'failed', 'stopped') for s in statuses)}")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, batch)
            item.setToolTip(batch['path'])
            self.history_list.addItem(item)
        for record in records:
            data, counts = record['data'], record['counts']
            label = (f"{data['run_id']} | {Path(data['source']).name} | {data['product']} | "
                     f"{len(data['selected'])} 模型 | 完成 {counts.get('complete', 0)} / 失败 {counts.get('failed', 0)}")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, record)
            item.setToolTip(record['path'])
            self.history_list.addItem(item)
        self.status.setText(f'找到 {len(batches)} 个整批清单、{len(records)} 个单目录历史。可 Ctrl / Shift 多选；不要重复选择同一工作区。')
        for error in errors + batch_errors:
            self.log.appendPlainText('历史记录读取失败：' + error)

    def _restore_history(self):
        records = [item.data(Qt.ItemDataRole.UserRole) for item in self.history_list.selectedItems()]
        if not records:
            QMessageBox.warning(self, '历史任务', '请先读取历史并选择一个任务。')
            return
        if any(record.get('missing') for record in records) and QMessageBox.question(self, '旧版设置不完整',
                '旧任务未记录正反面或数量上限，无法自动还原。\n将暂填“全部正反面、数量不限”，请确认后按实际情况调整。\n'
                '其余模型/配置会恢复；开始时仍严格检查算法兼容性。是否恢复？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        if len(records) == 1 and records[0].get('kind') != 'batch':
            self._restore_record(records[0])
            return
        from .field_history import jobs_from_records
        try:
            jobs = jobs_from_records(records, self.environment.currentText().strip())
        except Exception as exc:
            QMessageBox.warning(self, '无法恢复选择', str(exc))
            return
        detail = '\n'.join(f"{job['options']['source']} | {job['options'].get('product')} | "
                           f"{job['options'].get('face')} | 上限 {job['options'].get('limit') or '不限'} | "
                           f"环境 {job.get('environment') or '当前进程'}" for job in jobs)
        if QMessageBox.question(self, '确认批量续跑',
                f'将按各任务保存的配置依次运行（不是套用界面的一份配置）：\n{detail}\n'
                '旧版缺失范围按已确认的全部/不限执行。完成任务会校验复用，不重新推理完整结果。是否恢复？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        self._resume_record = None
        self._resume_jobs = jobs
        self.source.setPlainText('\n'.join(job['options']['source'] for job in jobs))
        self.start_button.setText(f'继续选中任务（{len(jobs)}）')
        self._lock_batch_fields(True)
        self.status.setText(f'已恢复 {len(jobs)} 个任务；配置按任务分别保存，开始后逐个校验。退出续跑后可修改设置。')

    def _restore_record(self, record):
        self._resume_jobs = None
        self._lock_batch_fields(False)
        options = record['options']
        self.source.setPlainText(options['source'])
        self.output.setText(options['output'])
        self.product_config.setText(options.get('product_config') or '')
        for combo, key in ((self.face, 'face'), (self.mode, 'mode'), (self.export_scope, 'export_scope')):
            combo.setCurrentIndex(combo.findData(options[key]))
        self.limit.setValue(options.get('limit') or 0)
        self.ng_only.setChecked(options['ng_only'])
        for key, check in self.checks.items():
            check.setChecked(key in options['selected'])
        environment = record['data'].get('environment')
        if environment:
            self.environment.setCurrentText(environment)
        self._resume_record = record
        self.start_button.setText('继续此任务')
        self.status.setText('已恢复历史设置；点击“继续此任务”将先检查兼容性，再复用完整结果。')

    def _new_task(self):
        self._resume_record = None
        self._resume_jobs = None
        self._lock_batch_fields(False)
        self.start_button.setText('开始生成')
        self.status.setText('已退出续跑模式，当前设置保留；旧任务不会删除。')

    def _lock_batch_fields(self, locked):
        for widget in (self.source, self.output, self.environment, self.product_config):
            parent = widget.parentWidget()
            if parent is not None and parent is not self.settings:
                for button in parent.findChildren(QPushButton):
                    button.setEnabled(not locked)
        for widget in (self.source, self.output, self.environment, self.product_config, self.face,
                       self.mode, self.limit, self.ng_only, self.export_scope, self.models_group):
            widget.setEnabled(not locked)

    def _set_busy(self, busy, scan_only=False):
        self.settings.setEnabled(not busy)
        self.models_group.setEnabled(not busy)
        self.scan_button.setEnabled(not busy)
        self.start_button.setEnabled(not busy)
        self.preview_button.setEnabled(not busy)
        self.load_preview_button.setEnabled(not busy)
        self.pause_button.setEnabled(busy and not scan_only)
        self.stop_button.setEnabled(busy and not scan_only)
        self.stop_export_button.setEnabled(busy and not scan_only)
        if not busy and self._resume_jobs is not None:
            self._lock_batch_fields(True)

    def _toggle_pause(self):
        if self.control.paused.is_set():
            self.control.paused.clear()
            self.pause_button.setText("暂停")
            self.status.setText("继续处理中…")
        else:
            self.control.paused.set()
            self.pause_button.setText("继续")
            self.status.setText("已请求暂停，将在安全处理点等待。")
        if getattr(self, '_task_bridge', None):
            self._task_bridge.note(self.status.text())

    def _stop_and_export(self):
        self.control.export_on_stop.set()
        self._stop()

    def _stop(self):
        bridge = getattr(self, '_task_bridge', None)
        if bridge and bridge.center.cancel(bridge.task_id):
            return
        self.control.stopped.set()
        self.control.paused.clear()
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.stop_export_button.setEnabled(False)
        self.status.setText("已请求停止并导出：正在完成当前原图，随后导出本次已完成数据，请勿关闭程序…"
                            if self.control.export_on_stop.is_set() else
                            "已请求停止，正在完成当前安全处理点并保存进度…")

    def _on_progress(self, data):
        total = data.get("total", 0)
        done = data.get("current", data.get("done", data.get("processed", 0)))
        if isinstance(total, int) and total > 0 and isinstance(done, int):
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
        message = data.get("message")
        if not message and 'completed' in data:
            message = (f"原图 {done}/{total}；模型任务：完成 {data.get('completed', 0)}，"
                       f"OK 过滤 {data.get('filtered', 0)}，未启用 {data.get('not_applicable', 0)}，"
                       f"需复核 {data.get('review', 0)}，技术失败 {data.get('failed', 0)}，"
                       f"断点跳过 {data.get('skipped', 0)}")
        message = message or json.dumps(data, ensure_ascii=False, default=str)
        self.log.appendPlainText(str(message))
        if data.get('phase') == 'export' or (not self.control.paused.is_set() and not self.control.stopped.is_set()):
            self.status.setText(str(message))

    def _on_result(self, result):
        self.progress_bar.setRange(0, 100)
        if result.get("scan"):
            self.status.setText(f"扫描到 {result['count']} 张图片，合计 {result['bytes'] / 1024 ** 3:.2f} GiB。损坏图片在解码时检测。")
        else:
            self.status.setText("已停止，保留已完成结果。使用相同配置再次开始可续跑。" if self.control.stopped.is_set() else "生成结束，请查看各模型统计和输出报告。")
            training = result.get("export", {}).get("training_dataset", {}).get("directory")
            if result.get("export", {}).get("format") == "xanylabeling":
                training = result["export"]["directory"]
            if training and not self.control.stopped.is_set():
                self.status.setText(f"数据集已生成：{training}（datasets 为中间目录，无需用于训练）")
            self.progress_bar.setValue(0 if self.control.stopped.is_set() else 100)
        self.log.appendPlainText(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        if not result.get('scan'):
            runs = result.get('results', []) if result.get('batch') else [result]
            total = lambda key: sum(run.get(key, 0) for run in runs)
            failures = total('failed_model_jobs')
            folder_errors = len(result.get('errors', [])) if result.get('batch') else 0
            title = '已停止' if result.get('stopped') or self.control.stopped.is_set() else (
                '处理结束（有技术失败）' if failures or folder_errors else '处理结束')
            self.status.setText(
                f"{title}：本次模型任务完成 {total('completed_model_jobs')}，OK 过滤 {total('filtered_model_jobs')}，"
                f"未启用 {total('not_applicable_model_jobs')}，需复核 {total('review_model_jobs')}，"
                f"技术失败 {failures}，断点跳过 {total('skipped_model_jobs')}；文件夹异常 {folder_errors}。")
            exported = [run['export'] for run in runs if run.get('export')]
            if self.control.export_on_stop.is_set():
                export_errors = sum(len(item.get('errors') or []) + len(item.get('preview_errors') or []) for item in exported)
                detail = (f' 已生成 {len(exported)} 个导出目录，导出/预览错误 {export_errors}。' if exported else
                          ' 未生成导出目录：请检查是否尚无已完成样本或运行发生异常。')
                self.status.setText(self.status.text() + detail)
                for item in exported:
                    self.log.appendPlainText('导出目录：' + str(item.get('directory', '')))
            merged = {}
            for run in runs:
                for name, counts in run.get('models', {}).items():
                    row = merged.setdefault(name, {})
                    for key in ('complete', 'filtered', 'not_applicable', 'review', 'failed', 'samples'):
                        row[key] = row.get(key, 0) + counts.get(key, 0)
            self.result_table.setRowCount(len(merged))
            for index, (name, counts) in enumerate(sorted(merged.items())):
                values = [name, *[counts[key] for key in ('complete', 'filtered', 'not_applicable', 'review', 'failed', 'samples')]]
                for column, value in enumerate(values):
                    self.result_table.setItem(index, column, QTableWidgetItem(str(value)))
            self.tabs.setCurrentWidget(self.result_page)

    def _on_error(self, message):
        self.progress_bar.setRange(0, 100)
        self.status.setText("任务异常，请查看日志。")
        self.log.appendPlainText(message)

    def _finished(self):
        self._set_busy(False)
        self.pause_button.setText("暂停")
        if self.worker and not self.worker.scan_only:
            current_tab = self.tabs.currentWidget()
            self._load_previews()
            if current_tab is self.result_page:
                self.tabs.setCurrentWidget(current_tab)

    def _open_output(self):
        value = self.output.text().strip()
        if value and Path(value).is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(value).resolve())))

    def protect_window_close(self, window):
        self._closing_window = window
        window.installEventFilter(self)

    def eventFilter(self, watched, event):
        if watched is self._closing_window and event.type() == QEvent.Type.Close and self.worker and self.worker.isRunning():
            self._stop()
            event.ignore()
            self.log.appendPlainText("任务停止后请再次关闭窗口。")
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self._stop()
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    page = FieldDatasetPage()
    page.resize(1050, 900)
    page.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
