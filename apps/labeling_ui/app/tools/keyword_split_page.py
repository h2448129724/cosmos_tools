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
from cosmos_toolbox.ui import ActionBar, PageScaffold, PathField, SectionSurface, StatusBanner
from .base import BaseToolPage, make_log_box, make_log_card, set_primary


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
        scaffold = PageScaffold("关键字划分", "根据文件名关键字将图片自动分类到子文件夹。")
        QVBoxLayout(self).addWidget(scaffold)
        lay = scaffold.content_layout

        settings_card = SectionSurface("输入与输出", "选择目录和处理方式。")
        settings_lay = settings_card.body_layout
        input_field = PathField("输入图片目录", placeholder="选择输入目录", browse_text="浏览")
        output_field = PathField("输出目录", placeholder="选择输出目录", browse_text="浏览")
        input_field.browse_requested.connect(lambda: self._pick_dir(input_field.line_edit))
        output_field.browse_requested.connect(lambda: self._pick_dir(output_field.line_edit))
        self._in_entry, self._out_entry = input_field.line_edit, output_field.line_edit
        settings_lay.addWidget(input_field)
        settings_lay.addWidget(output_field)

        mode_row = QHBoxLayout()
        mode_label = QLabel("处理方式")
        mode_label.setAccessibleName("处理方式")
        mode_row.addWidget(mode_label)
        self._rb_copy = QRadioButton("复制")
        self._rb_copy.setAccessibleName("复制模式")
        self._rb_copy.setChecked(True)
        self._rb_move = QRadioButton("移动")
        self._rb_move.setAccessibleName("移动模式")
        mode_row.addWidget(self._rb_copy)
        mode_row.addWidget(self._rb_move)
        mode_row.addStretch(1)
        settings_lay.addLayout(mode_row)
        lay.addWidget(settings_card)

        work_card = SectionSurface("关键字与执行", "维护关键字列表后，可先预览再执行分类。")
        card_lay = work_card.body_layout

        card_lay.addWidget(QLabel("关键字列表（不区分大小写）"))
        kw_row = QHBoxLayout()
        kw_row.setSpacing(10)
        self._kw_list = QListWidget()
        self._kw_list.setAccessibleName("关键字列表")
        self._kw_list.setMaximumHeight(152)
        for kw in ("top", "bottom"):
            self._kw_list.addItem(kw)
        kw_row.addWidget(self._kw_list, 1)
        kw_btns = ActionBar()
        b_add = QPushButton("添加")
        b_add.clicked.connect(self._add_kw)
        b_del = QPushButton("删除")
        b_del.clicked.connect(self._del_kw)
        kw_btns.add_widget(b_add)
        kw_btns.add_widget(b_del)
        kw_row.addWidget(kw_btns)
        card_lay.addLayout(kw_row)

        btn_row = ActionBar()
        b_preview = QPushButton("预览统计")
        b_preview.clicked.connect(self._preview)
        b_run = QPushButton("执行划分")
        set_primary(b_run)
        b_run.clicked.connect(self._run)
        btn_row.add_widget(b_preview)
        btn_row.add_widget(b_run)
        self._status_banner = StatusBanner("等待选择输入目录与关键字。")
        self._summary = self._status_banner.label
        card_lay.addWidget(self._status_banner)
        card_lay.addWidget(btn_row)
        lay.addWidget(work_card)

        self._log = make_log_box("运行日志...")
        lay.addWidget(make_log_card(self._log))

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
        if self._worker is not None and self._worker.isRunning():
            return
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
        self.run_background(
            classify_by_keywords,
            input_dir,
            keywords,
            output_dir,
            mode,
            on_result=self._on_done,
        )

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
