"""数据筛选工具页面 — 逐张浏览图片，快捷键筛选保留或淘汰。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QLabel, QMessageBox, QPushButton, QSplitter,
    QSizePolicy, QVBoxLayout,
)

from apps.data_tools.processing.image_io import read_image
from ..preview_widget import ZoomableLabel, cv2_to_qpixmap
from cosmos_toolbox.ui.primitives import ActionBar, PathField, SectionSurface, set_ui_role
from .base import (
    BaseToolPage, make_log_box, make_log_card,
    make_page_header, set_primary,
)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

class ImageFilterPage(BaseToolPage):
    tool_key = "image_filter"
    tool_title = "数据筛选"
    tool_nav_title = "数据筛选"
    tool_icon = "⊘"
    tool_summary = "逐张浏览图片，快捷键筛选保留或淘汰，安全可恢复。"
    tool_tags = ("数据筛选", "质量控制", "图片浏览")

    def __init__(self, main_window, parent=None):
        super().__init__(main_window, parent)
        self._files: list[str] = []
        self._all_files: list[str] = []
        self._current_index: int = -1
        self._source_dir: str = ""
        self._rejected_dir: str = ""
        self._kept_dir: str = ""
        self._rejected_count: int = 0
        self._kept_count: int = 0
        self._last_rejected: tuple[str, str] | None = None  # (原路径, rejected中的实际路径)
        self._build_ui()

    # ------------------------------------------------------------------- ui
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        lay.addWidget(make_page_header(
            "数据筛选", "逐张浏览图片，用键盘快速筛选。保留的移入 kept/，淘汰的移入 rejected/，均可恢复。"
        ))

        # ---- left workspace ----
        left = SectionSurface("预览与筛选", "使用键盘或按钮处理当前图片。")
        left_lay = left.body_layout
        left_lay.setSpacing(10)

        self._current_name = QLabel("当前图片：未加载")
        set_ui_role(self._current_name, "sectionTitle")
        self._current_name.setAccessibleName("当前图片")
        self._current_name.setWordWrap(True)
        self._nav_hint = QLabel("A 前一张 / D 后一张 / W 淘汰 / S 保留 / Z 撤销淘汰")
        set_ui_role(self._nav_hint, "muted")
        left_lay.addWidget(self._current_name)
        left_lay.addWidget(self._nav_hint)

        self._preview_meta = QLabel("加载图片目录后可开始筛选。")
        set_ui_role(self._preview_meta, "muted")
        self._preview_meta.setAccessibleName("预览信息")
        left_lay.addWidget(self._preview_meta)

        self._preview = ZoomableLabel()
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview.setAccessibleName("图片预览画布")
        left_lay.addWidget(self._preview, 1)

        nav_btns = ActionBar()
        b_prev = QPushButton("上一张(A)")
        b_prev.setAccessibleName("上一张图片")
        b_prev.clicked.connect(lambda: self._show_offset(-1))
        b_next = QPushButton("下一张(D)")
        b_next.setAccessibleName("下一张图片")
        b_next.clicked.connect(lambda: self._show_offset(+1))
        nav_btns.add_widget(b_prev)
        nav_btns.add_widget(b_next)
        left_lay.addWidget(nav_btns)

        action_btns = ActionBar()
        b_reject = QPushButton("淘汰(W)")
        b_reject.setProperty("buttonRole", "danger")
        b_reject.setAccessibleName("淘汰当前图片")
        b_reject.clicked.connect(self._reject_current)
        b_keep = QPushButton("保留(S)")
        b_keep.setAccessibleName("保留当前图片")
        set_primary(b_keep)
        b_keep.clicked.connect(self._keep_current)
        action_btns.add_widget(b_reject)
        action_btns.add_widget(b_keep)
        b_undo = QPushButton("撤销淘汰(Z)")
        b_undo.setAccessibleName("撤销上次淘汰")
        b_undo.clicked.connect(self._undo_last_reject)
        action_btns.add_widget(b_undo)
        left_lay.addWidget(action_btns)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.addWidget(left)

        # ---- right card: status panel ----
        right = SectionSurface("筛选统计", "目录、处理进度与恢复操作。")
        right_lay = right.body_layout
        right_lay.setSpacing(10)
        right.setMinimumWidth(250)
        right.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        directory_field = PathField("图片目录", placeholder="选择图片目录")
        directory_field.browse_requested.connect(self._pick_dir)
        self._dir_entry = directory_field.line_edit
        right_lay.addWidget(directory_field)

        b_load = QPushButton("加载目录")
        b_load.setAccessibleName("加载图片目录")
        set_primary(b_load)
        b_load.clicked.connect(self._load_folder)
        right_lay.addWidget(b_load)

        self._stat_total = QLabel("总图片数：0")
        self._stat_progress = QLabel("已处理：0 / 0")
        self._stat_kept = QLabel("保留：0")
        self._stat_rejected = QLabel("淘汰：0")
        self._stat_remaining = QLabel("剩余：0")
        for lbl in (self._stat_total, self._stat_progress, self._stat_kept,
                     self._stat_rejected, self._stat_remaining):
            set_ui_role(lbl, "muted")
            right_lay.addWidget(lbl)

        self._stat_state = QLabel("当前状态：待加载")
        set_ui_role(self._stat_state, "muted")
        self._stat_state.setAccessibleName("筛选状态")
        right_lay.addWidget(self._stat_state)

        right_lay.addStretch(1)

        b_recover_kept = QPushButton("恢复已保留（kept）")
        b_recover_kept.setAccessibleName("恢复已保留图片")
        b_recover_kept.clicked.connect(lambda: self._recover_from(self._kept_dir, "保留"))
        right_lay.addWidget(b_recover_kept)

        b_recover_rejected = QPushButton("恢复已淘汰（rejected）")
        b_recover_rejected.setAccessibleName("恢复已淘汰图片")
        b_recover_rejected.clicked.connect(lambda: self._recover_from(self._rejected_dir, "淘汰"))
        right_lay.addWidget(b_recover_rejected)

        main_splitter.addWidget(right)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([760, 300])
        lay.addWidget(main_splitter, 1)

        self._log = make_log_box("运行日志...")
        lay.addWidget(make_log_card(self._log))

    # -------------------------------------------------------------- dir ops
    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self._mw, "选择图片目录", self._dir_entry.text() or ".")
        if d:
            self._dir_entry.setText(d)

    # ----------------------------------------------------------- load / show
    def _load_folder(self):
        input_dir = self._dir_entry.text().strip()
        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(self._mw, "提示", "请先选择有效的图片目录")
            return
        # 只扫描顶层文件（不递归），避免加载 rejected 子文件夹的内容
        new_files = sorted([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
            and Path(f).suffix.lower() in _IMG_EXTS
        ])
        if not new_files:
            QMessageBox.warning(self._mw, "提示", "目录中没有图片文件")
            return

        same_dir = (self._source_dir == input_dir)

        self._files = new_files
        self._source_dir = input_dir
        self._rejected_dir = os.path.join(input_dir, "rejected")
        self._kept_dir = os.path.join(input_dir, "kept")
        self._current_index = -1

        if not same_dir:
            # 新目录：重置所有统计
            self._all_files = list(new_files)
            self._rejected_count = 0
            self._kept_count = 0
            self._last_rejected = None
            self._kept_count = 0
        # 同目录重新加载：保留 _all_files（原始总数）和累计计数，仅刷新文件列表

        self._log.clear()
        self._log.appendPlainText(f"已加载 {len(self._files)} 张图片")
        self._show_file_at(0)
        self._update_stats()

    def _show_file_at(self, index: int):
        if not self._files:
            self._show_empty_state()
            return
        if index < 0 or index >= len(self._files):
            return

        path = self._files[index]
        img = read_image(path)
        if img is None:
            self._log.appendPlainText(f"无法读取，跳过: {os.path.basename(path)}")
            self._files.pop(index)
            if not self._files:
                self._show_empty_state()
                return
            idx = min(index, len(self._files) - 1)
            self._show_file_at(idx)
            return

        self._current_index = index
        h, w = img.shape[:2]
        self._preview.set_pixmap(cv2_to_qpixmap(img))
        self._current_name.setText(f"当前图片：{os.path.basename(path)}")
        self._preview_meta.setText(
            f"当前第 {index + 1} / {len(self._files)} 张，尺寸 {w} × {h}"
        )
        self._update_stats()

    def _show_offset(self, delta: int):
        if not self._files:
            return
        next_index = 0 if self._current_index < 0 else self._current_index + delta
        next_index = max(0, min(len(self._files) - 1, next_index))
        if next_index != self._current_index:
            self._show_file_at(next_index)

    # -------------------------------------------------------- keep / reject
    def _reject_current(self):
        if not self._files or self._current_index < 0:
            return
        path = self._files[self._current_index]
        basename = os.path.basename(path)

        os.makedirs(self._rejected_dir, exist_ok=True)
        target = os.path.join(self._rejected_dir, basename)
        # 处理目标路径冲突
        if os.path.exists(target):
            stem, ext = os.path.splitext(basename)
            counter = 1
            while os.path.exists(target):
                target = os.path.join(self._rejected_dir, f"{stem}_rejected{counter}{ext}")
                counter += 1

        shutil.move(path, target)
        self._files.pop(self._current_index)
        self._rejected_count += 1
        self._last_rejected = (path, target)
        self._log.appendPlainText(f"淘汰: {basename} -> rejected/")

        if not self._files:
            self._show_empty_state()
        else:
            idx = min(self._current_index, len(self._files) - 1)
            self._show_file_at(idx)
        self._update_stats()

    def _undo_last_reject(self):
        if self._last_rejected is None:
            return
        original_path, rejected_path = self._last_rejected
        if not os.path.isfile(rejected_path):
            self._last_rejected = None
            return
        shutil.move(rejected_path, original_path)
        self._rejected_count -= 1
        self._last_rejected = None
        self._log.appendPlainText(f"撤销淘汰: {os.path.basename(original_path)}")
        # 重新加载以刷新文件列表
        self._load_folder()

    def _keep_current(self):
        if not self._files or self._current_index < 0:
            return
        path = self._files[self._current_index]
        basename = os.path.basename(path)

        os.makedirs(self._kept_dir, exist_ok=True)
        target = os.path.join(self._kept_dir, basename)
        if os.path.exists(target):
            stem, ext = os.path.splitext(basename)
            counter = 1
            while os.path.exists(target):
                target = os.path.join(self._kept_dir, f"{stem}_kept{counter}{ext}")
                counter += 1

        shutil.move(path, target)
        self._files.pop(self._current_index)
        self._kept_count += 1
        self._last_rejected = None  # keep 之后无法再撤销上一条 reject
        self._log.appendPlainText(f"保留: {basename} -> kept/")

        if not self._files:
            self._show_empty_state()
        else:
            idx = min(self._current_index, len(self._files) - 1)
            self._show_file_at(idx)
        self._update_stats()

    # --------------------------------------------------------------- recover
    def _recover_from(self, sub_dir: str, label: str):
        if not self._source_dir or not os.path.isdir(self._source_dir):
            QMessageBox.information(self._mw, "提示", "请先加载图片目录")
            return
        if not sub_dir or not os.path.isdir(sub_dir):
            QMessageBox.information(self._mw, "提示", f"没有已{label}的图片可恢复")
            return
        recovered = []
        for f in os.listdir(sub_dir):
            src = os.path.join(sub_dir, f)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(self._source_dir, f)
            if os.path.exists(dst):
                stem, ext = os.path.splitext(f)
                counter = 1
                while os.path.exists(dst):
                    dst = os.path.join(self._source_dir, f"{stem}_recovered{counter}{ext}")
                    counter += 1
            shutil.move(src, dst)
            recovered.append(f)
        if not recovered:
            QMessageBox.information(self._mw, "提示", f"没有已{label}的图片可恢复")
            return
        count = len(recovered)
        # 扣减对应计数
        if sub_dir == self._kept_dir:
            self._kept_count = max(0, self._kept_count - count)
        elif sub_dir == self._rejected_dir:
            self._rejected_count = max(0, self._rejected_count - count)
        self._load_folder()
        self._log.appendPlainText(f"已恢复 {count} 张已{label}图片")

    # ------------------------------------------------------------ empty state
    def _show_empty_state(self):
        self._current_index = -1
        self._preview.set_pixmap(cv2_to_qpixmap(None))
        self._current_name.setText("筛选完成")
        total = len(self._all_files)
        if total == 0:
            self._preview_meta.setText("请加载图片目录开始筛选。")
        else:
            self._preview_meta.setText(
                f"全部处理完毕 — 共 {total} 张，保留 {self._kept_count} 张，淘汰 {self._rejected_count} 张。"
            )
        self._update_stats()

    # ----------------------------------------------------------- stats panel
    def _update_stats(self):
        total = len(self._all_files)
        processed = self._kept_count + self._rejected_count
        remaining = len(self._files)
        self._stat_total.setText(f"总图片数：{total}")
        self._stat_progress.setText(f"已处理：{processed} / {total}")
        self._stat_kept.setText(f"保留：{self._kept_count}")
        self._stat_rejected.setText(f"淘汰：{self._rejected_count}")
        self._stat_remaining.setText(f"剩余：{remaining}")

        if total == 0:
            state = "当前状态：待加载"
        elif remaining == 0:
            state = "当前状态：全部筛选完成 ✓"
        else:
            state = "当前状态：筛选中…"
        self._stat_state.setText(state)

    # -------------------------------------------------------- lifecycle / kb
    def on_activated(self):
        self.setFocus()

    def on_deactivated(self):
        pass

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self._show_offset(-1)
            event.accept()
            return
        if event.key() == Qt.Key_D:
            self._show_offset(+1)
            event.accept()
            return
        if event.key() == Qt.Key_W:
            self._reject_current()
            event.accept()
            return
        if event.key() == Qt.Key_S:
            self._keep_current()
            event.accept()
            return
        if event.key() == Qt.Key_Z:
            self._undo_last_reject()
            event.accept()
            return
        super().keyPressEvent(event)
