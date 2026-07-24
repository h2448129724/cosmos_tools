"""关键字划分工具页面。"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from apps.data_tools.processing.keyword_split import classify_by_keywords
from .base import BaseToolPage, FuncWorker, make_card, make_log_box, make_log_card, make_page_header, set_primary


class KeywordSplitPage(BaseToolPage):
    tool_key = "keyword_split"
    tool_title = "关键字划分"
    tool_nav_title = "关键词划分"
    tool_icon = "◈"
    tool_summary = "根据文件名关键字批量归类图片，适合快速拆分 top、bottom 等目录。"
    tool_tags = ("批量整理", "文件名规则", "轻量处理")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        lay.addWidget(make_page_header("关键字划分", "根据文件名关键字将图片自动分类到子文件夹。"))

        settings_card = make_card()
        settings_lay = QVBoxLayout(settings_card)
        settings_lay.setContentsMargins(18, 16, 18, 16)
        settings_lay.setSpacing(10)

        self._in_entry, r1 = self._dir_row("输入图片目录")
        settings_lay.addLayout(r1)
        self._out_entry, r2 = self._dir_row("输出目录")
        settings_lay.addLayout(r2)

        mode_row = QHBoxLayout()
        mode_label = QLabel("处理方式")
        mode_label.setFixedWidth(92)
        mode_row.addWidget(mode_label)
        self._rb_copy = QRadioButton("复制")
        self._rb_copy.setChecked(True)
        self._rb_move = QRadioButton("移动")
        mode_row.addWidget(self._rb_copy)
        mode_row.addWidget(self._rb_move)
        mode_row.addStretch(1)
        settings_lay.addLayout(mode_row)
        lay.addWidget(settings_card)

        work_card = make_card()
        card_lay = QVBoxLayout(work_card)
        card_lay.setContentsMargins(18, 16, 18, 16)
        card_lay.setSpacing(10)
        title = QLabel("关键字与执行")
        title.setStyleSheet("color:#111827;font-size:15px;font-weight:700;")
        card_lay.addWidget(title)
        hint = QLabel("维护关键字列表后，可先预览再执行分类。")
        hint.setStyleSheet("color:#6B7280;font-size:12px;")
        card_lay.addWidget(hint)

        card_lay.addWidget(QLabel("关键字列表（不区分大小写）"))
        kw_row = QHBoxLayout()
        kw_row.setSpacing(10)
        self._kw_list = QListWidget()
        self._kw_list.setMaximumHeight(152)
        for kw in ("top", "bottom"):
            self._kw_list.addItem(kw)
        kw_row.addWidget(self._kw_list, 1)
        kw_btns = QVBoxLayout()
        kw_btns.setSpacing(8)
        b_add = QPushButton("添加")
        b_add.clicked.connect(self._add_kw)
        b_del = QPushButton("删除")
        b_del.clicked.connect(self._del_kw)
        kw_btns.addWidget(b_add)
        kw_btns.addWidget(b_del)
        kw_btns.addStretch(1)
        kw_row.addLayout(kw_btns)
        card_lay.addLayout(kw_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        b_preview = QPushButton("预览统计")
        b_preview.clicked.connect(self._preview)
        b_run = QPushButton("执行划分")
        set_primary(b_run)
        b_run.clicked.connect(self._run)
        btn_row.addWidget(b_preview)
        btn_row.addWidget(b_run)
        btn_row.addStretch(1)
        self._summary = QLabel("等待选择输入目录与关键字。")
        self._summary.setStyleSheet("color:#64748b;")
        card_lay.addWidget(self._summary)
        card_lay.addLayout(btn_row)
        lay.addWidget(work_card)

        self._log = make_log_box("运行日志...")
        lay.addWidget(make_log_card(self._log))

    def _dir_row(self, label: str) -> tuple:
        row = QHBoxLayout()
        row.setSpacing(10)
        lbl = QLabel(label)
        lbl.setFixedWidth(92)
        row.addWidget(lbl)
        entry = QLineEdit()
        row.addWidget(entry, 1)
        btn = QPushButton("浏览")
        btn.setFixedWidth(72)
        btn.clicked.connect(lambda: self._pick_dir(entry))
        row.addWidget(btn)
        return entry, row

    def _pick_dir(self, entry: QLineEdit):
        d = QFileDialog.getExistingDirectory(self._mw, "选择目录", entry.text() or ".")
        if d:
            entry.setText(d)

    def _add_kw(self):
        from PySide6.QtWidgets import QInputDialog

        kw, ok = QInputDialog.getText(self._mw, "添加关键字", "关键字：")
        if ok and kw.strip():
            self._kw_list.addItem(kw.strip())

    def _del_kw(self):
        row = self._kw_list.currentRow()
        if row >= 0:
            self._kw_list.takeItem(row)

    def _get_keywords(self) -> list[str]:
        return [self._kw_list.item(i).text() for i in range(self._kw_list.count())]

    def _preview(self):
        input_dir = self._in_entry.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self._mw, "提示", "请先选择有效的输入目录")
            return
        keywords = self._get_keywords()
        if not keywords:
            QMessageBox.warning(self._mw, "提示", "请至少添加一个关键字")
            return
        counts = classify_by_keywords(input_dir, keywords, "", dry_run=True)
        self._log.clear()
        lines = [f"{kw}: {cnt} 个文件" for kw, cnt in counts.items() if cnt > 0 or kw == "_unsorted"]
        self._log.setPlainText("\n".join(lines))
        assigned = sum(cnt for key, cnt in counts.items() if key != "_unsorted")
        self._summary.setText(f"预览完成：命中 {assigned} 个文件，未归类 {counts.get('_unsorted', 0)} 个文件。")

    def _run(self):
        input_dir = self._in_entry.text().strip()
        output_dir = self._out_entry.text().strip()
        keywords = self._get_keywords()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self._mw, "提示", "请先选择有效的输入目录")
            return
        if not output_dir:
            QMessageBox.warning(self._mw, "提示", "请先选择输出目录")
            return
        if not keywords:
            QMessageBox.warning(self._mw, "提示", "请至少添加一个关键字")
            return
        mode = "move" if self._rb_move.isChecked() else "copy"
        self._log.clear()
        self._log.appendPlainText(f"开始{mode}：{keywords}")
        self._worker = FuncWorker(classify_by_keywords, input_dir, keywords, output_dir, mode)
        self._worker.finished.connect(self._on_done)
        self._worker.start()

    def _on_done(self, result):
        if isinstance(result, Exception):
            self._log.appendPlainText(f"错误: {result}")
            self._summary.setText("执行失败，请检查输入目录、输出目录和关键字配置。")
        else:
            lines = [f"{kw}: {cnt} 个文件" for kw, cnt in result.items()]
            self._log.appendPlainText("完成！\n" + "\n".join(lines))
            assigned = sum(cnt for key, cnt in result.items() if key != "_unsorted")
            self._summary.setText(
                f"处理完成：已归类 {assigned} 个文件，仍有 {result.get('_unsorted', 0)} 个文件进入未分类目录。"
            )
        self._mw.show_status("关键字划分完成")
