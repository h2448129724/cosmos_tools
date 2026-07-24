"""数据筛选工具页面 — 逐张浏览图片，快捷键筛选保留或淘汰。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout,
)

from apps.data_tools.processing.image_io import read_image
from ..preview_widget import ZoomableLabel, cv2_to_qpixmap
from .base import (
    BaseToolPage, make_card, make_log_box, make_log_card,
    make_page_header, set_primary,
)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

_REJECT_BTN_STYLE = (
    "QPushButton{background:#DC2626;color:#fff;border:none;border-radius:6px;"
    "padding:6px 16px;font-weight:600;}"
    "QPushButton:hover{background:#B91C1C;}"
    "QPushButton:pressed{background:#991B1B;}"
)


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

        main_row = QHBoxLayout()
        main_row.setSpacing(12)

        # ---- left card: preview workspace ----
        left = make_card()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(18, 18, 18, 18)
        left_lay.setSpacing(10)

        info_row = QHBoxLayout()
        self._current_name = QLabel("当前图片：未加载")
        self._current_name.setStyleSheet("color:#0f172a;font-size:15px;font-weight:700;")
        self._current_name.setWordWrap(True)
        info_row.addWidget(self._current_name, 1)
        self._nav_hint = QLabel("A 前一张 / D 后一张 / W 淘汰 / S 保留 / Z 撤销淘汰")
        self._nav_hint.setStyleSheet("color:#64748b;font-size:12px;")
        info_row.addWidget(self._nav_hint)
        left_lay.addLayout(info_row)

        self._preview_meta = QLabel("加载图片目录后可开始筛选。")
        self._preview_meta.setStyleSheet("color:#64748b;")
        left_lay.addWidget(self._preview_meta)

        self._preview = ZoomableLabel()
        self._preview.setMinimumHeight(420)
        left_lay.addWidget(self._preview, 1)

        nav_btns = QHBoxLayout()
        nav_btns.setSpacing(8)
        b_prev = QPushButton("上一张(A)")
        b_prev.clicked.connect(lambda: self._show_offset(-1))
        b_next = QPushButton("下一张(D)")
        b_next.clicked.connect(lambda: self._show_offset(+1))
        nav_btns.addWidget(b_prev)
        nav_btns.addWidget(b_next)
        nav_btns.addStretch(1)
        left_lay.addLayout(nav_btns)

        action_btns = QHBoxLayout()
        action_btns.setSpacing(8)
        b_reject = QPushButton("淘汰(W)")
        b_reject.setStyleSheet(_REJECT_BTN_STYLE)
        b_reject.clicked.connect(self._reject_current)
        b_keep = QPushButton("保留(S)")
        set_primary(b_keep)
        b_keep.clicked.connect(self._keep_current)
        action_btns.addWidget(b_reject)
        action_btns.addWidget(b_keep)
        action_btns.addStretch(1)
        b_undo = QPushButton("撤销淘汰(Z)")
        b_undo.clicked.connect(self._undo_last_reject)
        action_btns.addWidget(b_undo)
        left_lay.addLayout(action_btns)

        main_row.addWidget(left, 1)

        # ---- right card: status panel ----
        right = make_card()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(16, 16, 16, 16)
        right_lay.setSpacing(10)
        right.setFixedWidth(280)

        dir_row = QHBoxLayout()
        dir_row.setSpacing(10)
        dir_label = QLabel("图片目录")
        dir_label.setFixedWidth(72)
        dir_row.addWidget(dir_label)
        self._dir_entry = QLineEdit()
        dir_row.addWidget(self._dir_entry, 1)
        btn_browse = QPushButton("浏览")
        btn_browse.setFixedWidth(60)
        btn_browse.clicked.connect(self._pick_dir)
        dir_row.addWidget(btn_browse)
        right_lay.addLayout(dir_row)

        b_load = QPushButton("加载目录")
        set_primary(b_load)
        b_load.clicked.connect(self._load_folder)
        right_lay.addWidget(b_load)

        # divider
        div = QLabel("")
        div.setFixedHeight(1)
        div.setStyleSheet("background:#E5E7EB;")
        right_lay.addWidget(div)

        stat_title = QLabel("筛选统计")
        stat_title.setStyleSheet("color:#111827;font-size:15px;font-weight:700;")
        right_lay.addWidget(stat_title)

        self._stat_total = QLabel("总图片数：0")
        self._stat_progress = QLabel("已处理：0 / 0")
        self._stat_kept = QLabel("保留：0")
        self._stat_rejected = QLabel("淘汰：0")
        self._stat_remaining = QLabel("剩余：0")
        for lbl in (self._stat_total, self._stat_progress, self._stat_kept,
                     self._stat_rejected, self._stat_remaining):
            lbl.setStyleSheet("color:#475569;")
            right_lay.addWidget(lbl)

        self._stat_state = QLabel("当前状态：待加载")
        self._stat_state.setStyleSheet("color:#64748b;font-size:12px;")
        right_lay.addWidget(self._stat_state)

        right_lay.addStretch(1)

        b_recover_kept = QPushButton("恢复已保留（kept）")
        b_recover_kept.clicked.connect(lambda: self._recover_from(self._kept_dir, "保留"))
        right_lay.addWidget(b_recover_kept)

        b_recover_rejected = QPushButton("恢复已淘汰（rejected）")
        b_recover_rejected.clicked.connect(lambda: self._recover_from(self._rejected_dir, "淘汰"))
        right_lay.addWidget(b_recover_rejected)

        main_row.addWidget(right, 0)
        lay.addLayout(main_row, 1)

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
