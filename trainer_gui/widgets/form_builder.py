from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QDoubleValidator, QFontMetrics, QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..models import FieldSchema

# 必填字段为空时的边框色
_REQUIRED_EMPTY_BORDER = "2px solid #ef4444"
_PATH_MISSING_BORDER = "2px solid #f59e0b"
_DEFAULT_BORDER = "1px solid #d8e1eb"


class ElidedLineEdit(QLineEdit):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._full_text = ""
        self.textChanged.connect(self._sync_full_text)

    def set_path_value(self, value: str) -> None:
        self._full_text = value or ""
        self._refresh_display()

    def path_value(self) -> str:
        return self._full_text if not self.hasFocus() else super().text().strip()

    def focusInEvent(self, event) -> None:  # noqa: N802
        super().focusInEvent(event)
        self._set_display_text(self._full_text)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        self._full_text = super().text().strip()
        super().focusOutEvent(event)
        self._refresh_display()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self.hasFocus():
            self._refresh_display()

    def _sync_full_text(self, text: str) -> None:
        if self.hasFocus():
            self._full_text = text.strip()
            self.setToolTip(self._full_text)

    def _refresh_display(self) -> None:
        if self.hasFocus():
            return
        metrics = QFontMetrics(self.font())
        width = max(self.width() - 18, 120)
        display = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideMiddle, width)
        self._set_display_text(display)
        self.setToolTip(self._full_text)

    def _set_display_text(self, text: str) -> None:
        blocked = self.blockSignals(True)
        super().setText(text)
        self.blockSignals(blocked)


class CollapsibleSection(QWidget):
    def __init__(self, title: str, collapsed: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.toggle_button = QToolButton(self)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(not collapsed)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if not collapsed else Qt.ArrowType.RightArrow)
        self.toggle_button.setText(title)
        self.toggle_button.clicked.connect(self._on_toggled)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle_button.setMinimumHeight(34)

        self.content = QWidget(self)
        self.content_layout = QFormLayout(self.content)
        self.content_layout.setContentsMargins(12, 4, 12, 10)
        self.content_layout.setHorizontalSpacing(12)
        self.content_layout.setVerticalSpacing(10)
        self.content_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.content.setVisible(not collapsed)
        self.content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        wrapper = QVBoxLayout(self)
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.setSpacing(2)
        wrapper.addWidget(self.toggle_button)
        wrapper.addWidget(self.content)
        self._sync_state()

    def _on_toggled(self, checked: bool) -> None:
        self._sync_state()

    def _sync_state(self) -> None:
        expanded = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.content.setVisible(expanded)
        self.content.setMaximumHeight(16777215 if expanded else 0)
        self.content.setMinimumHeight(0)
        self.updateGeometry()


class FormBuilder:
    GROUP_ORDER = ["基础配置", "数据配置", "训练参数", "模型参数", "高级参数", "其他参数"]
    TRAINING_FIELDS = {
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "warmup_epochs",
        "val_ratio",
        "aug_multiplier",
        "num_workers",
        "workers",
        "patience",
    }
    MODEL_FIELDS = {
        "img_size",
        "sigma",
        "hidden_dim",
        "num_layers",
        "dropout",
        "k_neighbors",
        "radius_multiplier",
        "threshold",
        "cluster_dist",
        "default_spacing",
        "patch_width",
        "patch_height",
        "opset",
        "stage1_threshold",
        "stage1_cluster_dist",
        "stage2_threshold",
        "stage2_max_degree",
        "stage2_max_small_cycle_length",
        "stage2_continuity_weight",
        "stage2_cycle_penalty",
    }
    ADVANCED_FIELDS = {
        "seed",
        "device",
        "cache",
        "amp",
        "dynamic",
        "half",
        "simplify",
        "overwrite_json",
        "no_compare_gt",
        "label",
        "output_format",
        "stage1_disable_tta",
        "perturb_jitter_std_ratio",
        "perturb_drop_prob",
        "perturb_spurious_prob",
    }
    DATA_KEYS = ("image", "img", "ann", "annotation", "input", "output", "json", "model", "dir", "path")
    DATA_FIELDS = {
        "point_source",
        "stage1_model_path",
        "stage2_model_path",
        "model",
        "custom_model",
        "data",
        "task",
        "model_series",
        "model_size",
    }

    def __init__(self, parent: QWidget | None = None):
        self.parent = parent
        self._widgets: dict[str, QWidget] = {}
        self._fields: list[FieldSchema] = []
        self._containers: list[QWidget] = []

    def build(self, layout: QFormLayout, fields: list[FieldSchema]) -> None:
        self.clear(layout)
        self._widgets.clear()
        self._fields = list(fields)

        visible_fields = [field for field in fields if not field.hidden]
        grouped_fields: dict[str, list[FieldSchema]] = {}
        for field in visible_fields:
            grouped_fields.setdefault(self._group_name(field), []).append(field)

        ordered_groups = sorted(
            grouped_fields.items(),
            key=lambda item: (self.GROUP_ORDER.index(item[0]) if item[0] in self.GROUP_ORDER else 99, item[0]),
        )
        for title, group_fields in ordered_groups:
            # 分组标题显示字段数量
            section_title = f"{title} ({len(group_fields)})"
            section = CollapsibleSection(section_title, collapsed=(title == "高级参数"), parent=self.parent)
            self._containers.append(section)
            for field in group_fields:
                label_text = field.label or field.name
                if field.required:
                    label_text = f"{label_text} *"
                label = QLabel(label_text)
                label.setToolTip(field.help or "")
                label.setMinimumWidth(132)
                label.setMinimumHeight(34)
                label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                widget = self._create_widget(field)
                widget.setToolTip(field.help or "")
                self._widgets[field.name] = widget
                self._connect_validation_trigger(field, widget)
                section.content_layout.addRow(label, widget)
            layout.addRow(section)

    def clear(self, layout: QFormLayout) -> None:
        while layout.rowCount():
            layout.removeRow(0)
        self._containers.clear()

    def values(self, fields: list[FieldSchema]) -> dict:
        payload = {}
        for field in fields:
            widget = self._widgets.get(field.name)
            if widget is None:
                continue
            payload[field.name] = self._read_value(field, widget)
        return payload

    def set_values(self, fields: list[FieldSchema], values: dict) -> None:
        for field in fields:
            if field.name not in values:
                continue
            widget = self._widgets.get(field.name)
            if widget is None:
                continue
            self._write_value(field, widget, values[field.name])

    def _create_widget(self, field: FieldSchema) -> QWidget:
        widget_name = field.widget or "text"
        if widget_name == "checkbox":
            widget = QCheckBox()
            widget.setChecked(bool(field.default))
            widget.setEnabled(not field.read_only)
            widget.setMinimumHeight(34)
            return widget
        if widget_name == "select":
            widget = QComboBox()
            for choice in field.choices or []:
                widget.addItem(str(choice), choice)
            if field.default is not None:
                index = widget.findData(field.default)
                if index >= 0:
                    widget.setCurrentIndex(index)
            widget.setEnabled(not field.read_only)
            widget.setMinimumHeight(38)
            return widget
        if widget_name == "number":
            if self._uses_nullable_numeric(field):
                return self._build_nullable_numeric_widget(field, "int")
            widget = QSpinBox()
            widget.setRange(-10_000_000, 10_000_000)
            widget.setValue(int(field.default) if self._is_int_like(field.default) else 0)
            widget.setEnabled(not field.read_only)
            widget.setMinimumHeight(38)
            return widget
        if widget_name == "float":
            if self._uses_nullable_numeric(field):
                return self._build_nullable_numeric_widget(field, "float")
            widget = QDoubleSpinBox()
            widget.setRange(-1e12, 1e12)
            widget.setDecimals(6)
            widget.setSingleStep(0.001)
            widget.setValue(float(field.default) if self._is_float_like(field.default) else 0.0)
            widget.setEnabled(not field.read_only)
            widget.setMinimumHeight(38)
            return widget
        if widget_name == "textarea":
            widget = QTextEdit()
            widget.setPlainText("" if field.default in (None, "None") else str(field.default))
            widget.setReadOnly(field.read_only)
            widget.setMinimumHeight(88)
            return widget
        if widget_name == "path":
            return self._build_path_widget(field)

        widget = QLineEdit()
        widget.setText("" if field.default in (None, "None") else str(field.default))
        if field.placeholder:
            widget.setPlaceholderText(field.placeholder)
        widget.setReadOnly(field.read_only)
        widget.setMinimumHeight(38)
        return widget

    def _build_nullable_numeric_widget(self, field: FieldSchema, kind: str) -> QLineEdit:
        widget = QLineEdit()
        widget.setProperty("nullable_numeric_kind", kind)
        widget.setText("" if field.default in (None, "None", "") else str(field.default))
        if kind == "int":
            widget.setValidator(QIntValidator(widget))
        else:
            widget.setValidator(QDoubleValidator(widget))
        widget.setPlaceholderText(field.placeholder or "留空表示使用脚本默认值")
        widget.setReadOnly(field.read_only)
        widget.setMinimumHeight(38)
        return widget

    def _build_path_widget(self, field: FieldSchema) -> QWidget:
        container = QFrame(self.parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        line_edit = ElidedLineEdit(container)
        line_edit.set_path_value("" if field.default in (None, "None") else str(field.default))
        if field.placeholder:
            line_edit.setPlaceholderText(field.placeholder)
        line_edit.setReadOnly(field.read_only)
        line_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        line_edit.setMinimumHeight(38)

        browse_button = QPushButton("选择")
        browse_button.setProperty("buttonRole", "ghost")
        browse_button.setEnabled(not field.read_only)
        browse_button.setMinimumHeight(36)
        browse_button.clicked.connect(lambda: self._browse_path(line_edit, field))

        open_button = QPushButton("打开")
        open_button.setProperty("buttonRole", "ghost")
        open_button.setMinimumHeight(36)
        open_button.clicked.connect(lambda: self._open_path(line_edit.path_value()))

        copy_button = QPushButton("复制")
        copy_button.setProperty("buttonRole", "ghost")
        copy_button.setMinimumHeight(36)
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(line_edit.path_value()))

        layout.addWidget(line_edit)
        layout.addWidget(browse_button)
        layout.addWidget(open_button)
        layout.addWidget(copy_button)

        container._value_widget = line_edit  # type: ignore[attr-defined]
        self._containers.append(container)
        return container

    def _browse_path(self, line_edit: ElidedLineEdit, field: FieldSchema) -> None:
        current = line_edit.path_value().strip() or str(Path.cwd())
        prefers_directory = self._is_directory_path_field(field)
        path = ""
        if prefers_directory:
            path = QFileDialog.getExistingDirectory(self.parent, f"选择 {field.label or field.name}", current)
        elif self._is_output_file_path_field(field):
            path, _ = QFileDialog.getSaveFileName(self.parent, f"选择 {field.label or field.name}", current)
        else:
            path, _ = QFileDialog.getOpenFileName(self.parent, f"选择 {field.label or field.name}", current)
            if not path:
                path = QFileDialog.getExistingDirectory(self.parent, f"选择 {field.label or field.name}", current)
        if path:
            line_edit.set_path_value(path)

    @staticmethod
    def _open_path(raw_path: str) -> None:
        path = (raw_path or "").strip()
        if not path:
            return
        resolved = Path(path).expanduser()
        target = resolved if resolved.exists() else resolved.parent
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _read_value(self, field: FieldSchema, widget: QWidget):
        widget_name = field.widget or "text"
        if widget_name == "checkbox":
            return widget.isChecked()  # type: ignore[union-attr]
        if widget_name == "select":
            return widget.currentData()  # type: ignore[union-attr]
        if widget_name == "number":
            if self._uses_nullable_numeric(field):
                return self._parse_nullable_numeric(widget.text().strip(), "int")  # type: ignore[union-attr]
            return widget.value()  # type: ignore[union-attr]
        if widget_name == "float":
            if self._uses_nullable_numeric(field):
                return self._parse_nullable_numeric(widget.text().strip(), "float")  # type: ignore[union-attr]
            return widget.value()  # type: ignore[union-attr]
        if widget_name == "textarea":
            return widget.toPlainText().strip()  # type: ignore[union-attr]
        if widget_name == "path":
            return widget._value_widget.path_value().strip()  # type: ignore[attr-defined]
        return widget.text().strip()  # type: ignore[union-attr]

    def _write_value(self, field: FieldSchema, widget: QWidget, value) -> None:
        widget_name = field.widget or "text"
        if widget_name == "checkbox":
            widget.setChecked(bool(value))  # type: ignore[union-attr]
            return
        if widget_name == "select":
            index = widget.findData(value)  # type: ignore[union-attr]
            if index >= 0:
                widget.setCurrentIndex(index)  # type: ignore[union-attr]
            return
        if widget_name in {"number", "float"}:
            if self._uses_nullable_numeric(field):
                widget.setText("" if value in (None, "None", "") else str(value))  # type: ignore[union-attr]
                return
            widget.setValue(value)  # type: ignore[union-attr]
            return
        if widget_name == "textarea":
            widget.setPlainText(str(value))  # type: ignore[union-attr]
            return
        if widget_name == "path":
            widget._value_widget.set_path_value(str(value))  # type: ignore[attr-defined]
            return
        widget.setText(str(value))  # type: ignore[union-attr]

    def _group_name(self, field: FieldSchema) -> str:
        if field.group:
            return field.group
        name = field.name
        if name in self.TRAINING_FIELDS:
            return "训练参数"
        if name in self.MODEL_FIELDS:
            return "模型参数"
        if name in self.ADVANCED_FIELDS:
            return "高级参数"
        if name in self.DATA_FIELDS or any(token in name for token in self.DATA_KEYS):
            return "数据配置"
        return "其他参数"

    @staticmethod
    def _is_int_like(value) -> bool:
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _is_float_like(value) -> bool:
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def _uses_nullable_numeric(field: FieldSchema) -> bool:
        return field.default is None and not field.required

    @staticmethod
    def _parse_nullable_numeric(value: str, kind: str):
        if not value:
            return None
        try:
            return int(value) if kind == "int" else float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _is_directory_path_field(field: FieldSchema) -> bool:
        if field.path_mode == "dir":
            return True
        if field.path_mode in {"save_file", "open_file"}:
            return False
        return field.name.endswith("_dir") or field.name in {"save_dir", "output_dir", "out_dir"}

    @classmethod
    def _is_output_file_path_field(cls, field: FieldSchema) -> bool:
        if field.path_mode == "save_file":
            return True
        if field.path_mode in {"dir", "open_file"}:
            return False
        return not cls._is_directory_path_field(field) and field.name in {"save_path", "output_path", "output"}

    # ------------------------------------------------------------------
    # 校验相关
    # ------------------------------------------------------------------

    def validate_visible(self) -> list[str]:
        """校验所有可见字段，返回错误信息列表。"""
        errors: list[str] = []
        for field in self._fields:
            if field.hidden:
                continue
            widget = self._widgets.get(field.name)
            if widget is None:
                continue
            error = self._validate_field(field, widget)
            if error:
                errors.append(error)
        return errors

    def _connect_validation_trigger(self, field: FieldSchema, widget: QWidget) -> None:
        """Validate only after an actual field interaction, never while building the form."""
        target = self._border_target(widget, field)
        if isinstance(target, QLineEdit):
            target.editingFinished.connect(
                lambda current_field=field, current_widget=widget: self._validate_field(
                    current_field, current_widget
                )
            )
        elif isinstance(target, QComboBox):
            target.activated.connect(
                lambda _index, current_field=field, current_widget=widget: self._validate_field(
                    current_field, current_widget
                )
            )
        elif isinstance(target, QTextEdit):
            target.textChanged.connect(
                lambda current_field=field, current_widget=widget: self._validate_field(
                    current_field, current_widget
                )
            )

    def _validate_field(self, field: FieldSchema, widget: QWidget) -> str | None:
        value = self._read_value(field, widget)
        if field.required and (value is None or str(value).strip() == ""):
            self._apply_border(widget, field, "required_empty")
            return f"{field.label or field.name}: 必填字段为空"
        self._apply_border(widget, field, "ok")
        if field.widget == "path" and value and str(value).strip():
            self._check_path_existence(widget, field, str(value).strip())
        return None

    def _apply_border(self, widget: QWidget, field: FieldSchema, state: str) -> None:
        """根据校验状态设置输入框边框。"""
        target = self._border_target(widget, field)
        if target is None:
            return
        if state == "required_empty":
            target.setStyleSheet(f"border: {_REQUIRED_EMPTY_BORDER}; border-radius: 3px;")
        elif state == "path_missing":
            target.setStyleSheet(f"border: {_PATH_MISSING_BORDER}; border-radius: 3px;")
        else:
            target.setStyleSheet(f"border: {_DEFAULT_BORDER}; border-radius: 3px;")

    @staticmethod
    def _border_target(widget: QWidget, field: FieldSchema) -> QWidget | None:
        """返回需要设置边框的 widget（path 类型是内嵌的 line_edit）。"""
        if field.widget == "path" and hasattr(widget, "_value_widget"):
            return widget._value_widget  # type: ignore[attr-defined]
        if isinstance(widget, (QLineEdit, QComboBox, QTextEdit)):
            return widget
        return None

    def _check_path_existence(self, widget: QWidget, field: FieldSchema, path_str: str) -> None:
        """路径字段存在性检查，不存在时显示黄色边框。"""
        p = Path(path_str).expanduser()
        if p.exists():
            self._apply_border(widget, field, "ok")
        else:
            self._apply_border(widget, field, "path_missing")
