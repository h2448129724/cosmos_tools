"""Readable inspection values, alongside the exact saved JSON (no re-evaluation)."""

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


LABELS = {
    "ear_distance": "相邻耳片间距",
    "ear_sewed": "耳片缝线距离",
    "ear": "耳片安放",
    "hook": "挂钩",
    "tail": "尾部",
    "reinforcement": "补强片",
    "decode_image": "二维码内容",
    "decode_boxes": "二维码检测框",
    "knife": "刀口",
    "glue": "胶条",
    "placement": "安放位置",
    "cloth": "尾布",
    "seam": "缝线",
    "count": "数量",
    "status": "保存的状态",
    "failed": "执行失败",
    "error": "错误",
    "message": "说明",
    "label": "类别",
    "score": "分数",
    "distance": "距离",
    "distance_mm": "距离（mm）",
    "unit": "单位",
    "detected_label": "检测类别",
    "expected_label": "目标类别",
    "target_number": "目标数量",
    "sew_point_count": "缝纫点数量",
    "text": "内容",
    "face": "检测面",
    "execution_time": "总耗时（秒）",
    "stage_timings": "阶段耗时（秒）",
}


def text_value(value):
    if value is None:
        return "未记录 / NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def quick_rows(result):
    """Project only explicit saved facts; never invent calibration or pass/fail."""
    rows = []
    distance = result.get("ear_distance")
    values = distance.get("distances") if isinstance(distance, dict) else None
    if isinstance(values, list) and values:
        for index, value in enumerate(values):
            rows.append(
                (
                    "相邻耳片间距",
                    f"{index + 1} → {index + 2}（从左到右）",
                    text_value(value),
                    text_value(distance.get("unit", "未记录单位")),
                    "未重新判定",
                )
            )
    else:
        rows.append(("相邻耳片间距", "—", "未记录测量值", "—", "未重新判定"))
    sewed = result.get("ear_sewed")
    if isinstance(sewed, list):
        for index, item in enumerate(sewed):
            if not isinstance(item, dict):
                rows.append(("耳片缝线距离", str(index + 1), text_value(item), "未记录单位", "未重新判定"))
                continue
            mm = item.get("distance_mm")
            value = mm if mm is not None else item.get("distance")
            position = item.get("cxcy")
            x = f" · x={position[0]}" if isinstance(position, list) and position else ""
            rows.append(
                (
                    "耳片缝线距离",
                    f"原始序号 {index + 1}{x}",
                    text_value(value),
                    "mm" if mm is not None else text_value(item.get("unit", "未记录单位")),
                    "未重新判定",
                )
            )
    if "decode_image" in result or "decode_boxes" in result:
        qr = result.get("decode_image")
        rows.append(
            (
                "二维码内容",
                "汇总",
                qr if isinstance(qr, str) and qr else (text_value(qr) if isinstance(qr, dict) else "未解码出内容"),
                "—",
                "未重新判定",
            )
        )
        boxes = result.get("decode_boxes")
        if isinstance(boxes, list):
            for index, box in enumerate(boxes):
                if isinstance(box, dict):
                    rows.append(
                        (
                            "二维码内容",
                            f"检测框 {index + 1}",
                            text_value(box.get("text")) if box.get("text") else "未解码出内容",
                            "—",
                            "未重新判定",
                        )
                    )
    for key, value in result.items():
        if key in ("ear_distance", "ear_sewed", "decode_image", "decode_boxes", "stage_timings"):
            continue
        status = "未记录判定"
        if isinstance(value, dict):
            if isinstance(value.get("status"), bool):
                status = "保存状态：通过" if value["status"] else "保存状态：不通过"
            if value.get("failed") is True:
                status = "保存状态：执行失败"
            facts = [
                f"{LABELS.get(k, k)}: {text_value(v)}" for k, v in value.items() if not isinstance(v, (dict, list))
            ]
            if isinstance(value.get("placement"), list):
                facts.append(f"安放结果 {len(value['placement'])} 项")
            summary = "；".join(facts) or "展开“全部字段”查看"
        elif isinstance(value, list):
            summary = f"{len(value)} 项，展开“全部字段”查看"
        else:
            summary = text_value(value)
        rows.append((LABELS.get(key, key), key, summary, "—", status))
    return rows


class InspectionDetailPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.documents = QComboBox()
        self.documents.setMinimumWidth(120)
        self.documents.currentIndexChanged.connect(self.render_document)
        bar.addWidget(self.documents, 1)
        paste = QPushButton("粘贴 JSON")
        paste.clicked.connect(self.open_input)
        bar.addWidget(paste)
        layout.addLayout(bar)
        self.notice = QLabel("选择数据库记录后点击“查看检查项”，或粘贴 JSON。只展示保存结果，不重新判定。")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self.tabs = QTabWidget()
        self.summary = QTableWidget()
        self.summary.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.summary.setWordWrap(False)
        self.summary.setColumnCount(5)
        self.summary.setHorizontalHeaderLabels(["检查项", "位置 / 序号", "结果 / 测量值", "单位", "判定来源"])
        self.tabs.addTab(self.summary, "检查项速览")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["全部字段", "保存值"])
        self.tree.itemExpanded.connect(self.expand_node)
        self.tabs.addTab(self.tree, "全部字段")
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.tabs.addTab(self.raw, "原始 JSON")
        input_page = QWidget()
        input_layout = QVBoxLayout(input_page)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("粘贴 result_detail JSON 对象（支持转义后的 JSON 字符串）。")
        input_layout.addWidget(self.input)
        parse = QPushButton("解析查看")
        parse.clicked.connect(lambda: self.set_documents([("粘贴内容", self.input.toPlainText())]))
        input_layout.addWidget(parse)
        self.tabs.addTab(input_page, "粘贴输入")
        layout.addWidget(self.tabs, 1)
        self.summary.cellDoubleClicked.connect(self.show_value)
        self.tree_values = {}
        self.next_token = 0

    def clear(self, message="请选择记录查看检查项。"):
        self.documents.clear()
        self.summary.setRowCount(0)
        self.tree.clear()
        self.tree_values = {}
        self.raw.clear()
        self.notice.setText(message)

    def open_input(self):
        self.tabs.setCurrentIndex(3)
        self.input.setFocus()

    def set_documents(self, documents):
        self.clear("")
        for name, raw in documents:
            self.documents.addItem(name, raw)
        if not documents:
            self.notice.setText("该记录没有保存检查项明细。")
        else:
            self.render_document()

    def render_document(self, *_):
        raw = self.documents.currentData()
        if self.documents.currentIndex() < 0:
            return
        self.summary.setRowCount(0)
        self.tree.clear()
        self.tree_values = {}
        self.raw.setPlainText(str(raw))
        try:
            result = raw
            for _ in range(3):
                if isinstance(result, str):
                    result = json.loads(result)
            if not isinstance(result, dict):
                raise ValueError("需要 JSON 对象")
        except (ValueError, TypeError, RecursionError) as exc:
            self.notice.setText(f"无法解析：{exc}")
            self.tabs.setCurrentIndex(2)
            return
        self.raw.setPlainText(json.dumps(result, ensure_ascii=False, indent=2))
        rows = quick_rows(result)
        self.summary.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.summary.setItem(row_index, column, item)
        for column, width in enumerate((130, 175, 300, 100, 145)):
            self.summary.setColumnWidth(column, width)
        for key, value in result.items():
            self.add_node(self.tree, key, value)
        self.tree.setColumnWidth(0, 280)
        face = {"top": "正面 / top", "bottom": "反面 / bottom"}.get(str(result.get("face")), "未记录检测面")
        self.notice.setText(f"{face} · 缝线距离保留原始序号；相邻间距按保存顺序。单位缺失时不推断，不重算合格性。")
        self.tabs.setCurrentIndex(0)

    def add_node(self, parent, key, value):
        name = f"{LABELS[key]} ({key})" if key in LABELS else str(key)
        container = isinstance(value, (dict, list))
        node = QTreeWidgetItem(parent, [name, f"{len(value)} 项" if container else text_value(value)])
        if container and value:
            token = self.next_token
            self.next_token += 1
            self.tree_values[token] = value
            node.setData(0, Qt.ItemDataRole.UserRole, token)
            QTreeWidgetItem(node, ["展开加载"])

    def expand_node(self, node):
        token = node.data(0, Qt.ItemDataRole.UserRole)
        if token is None:
            return
        value = self.tree_values.pop(token)
        node.setData(0, Qt.ItemDataRole.UserRole, None)
        node.takeChildren()
        entries = list(value.items()) if isinstance(value, dict) else [(str(i + 1), v) for i, v in enumerate(value)]
        for key, child in entries[:1000]:
            self.add_node(node, key, child)
        if len(entries) > 1000:
            QTreeWidgetItem(node, ["其余项目请在原始 JSON 中查看"])

    def show_value(self, row, column):
        # Copyable full value without opening a modal business window.
        self.input.setPlainText(self.summary.item(row, column).text())
        self.tabs.setCurrentIndex(3)
