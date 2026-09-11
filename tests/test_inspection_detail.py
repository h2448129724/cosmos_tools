import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from cosmos_toolbox.database_ui import DatabasePage
from cosmos_toolbox.inspection_detail_ui import InspectionDetailPanel, quick_rows


class InspectionDetailTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_distances_are_distinct_and_units_not_invented(self):
        rows = quick_rows(
            {
                "ear_distance": {"distances": [123.4, None]},
                "ear_sewed": [{"distance_mm": 19.11, "cxcy": [4534, 6566]}],
                "decode_image": None,
                "decode_boxes": [{"text": None}],
            }
        )
        self.assertEqual(rows[0][2:4], ("123.4", "未记录单位"))
        self.assertIn("NULL", rows[1][2])
        self.assertEqual(rows[2][0], "耳片缝线距离")
        self.assertEqual(rows[2][2:4], ("19.11", "mm"))
        self.assertEqual(rows[3][2], "未解码出内容")

    def test_qr_explicit_status_and_lazy_tree(self):
        panel = InspectionDetailPanel()
        result = {"decode_image": "ABC123", "hook": {"status": False, "box": [1, 2]}, "tail": {"cloth": {"count": 2}}}
        panel.set_documents([("escaped", json.dumps(json.dumps(result)))])
        displayed = {
            panel.summary.item(row, 0).text(): [panel.summary.item(row, column).text() for column in range(5)]
            for row in range(panel.summary.rowCount())
        }
        self.assertEqual(displayed['二维码内容'][2], "ABC123")
        self.assertEqual(displayed['挂钩'][4], "保存状态：不通过")
        hook, tail = panel.tree.topLevelItem(1), panel.tree.topLevelItem(2)
        hook.setExpanded(True)
        tail.setExpanded(True)
        self.assertEqual(tail.child(0).text(0), "尾布 (cloth)")
        panel.set_documents([("bad", "{invalid")])
        self.assertEqual(panel.summary.rowCount(), 0)
        self.assertIn("无法解析", panel.notice.text())
        panel.set_documents([("null", None)])
        self.assertIn("无法解析", panel.notice.text())
        panel.close()

    def test_result_record_loads_all_matching_faces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.db"
            connection = sqlite3.connect(path)
            connection.executescript(
                "CREATE TABLE inspection_result (id INTEGER PRIMARY KEY);"
                "INSERT INTO inspection_result VALUES (1);"
                "CREATE TABLE inspection_metadata (id INTEGER PRIMARY KEY, result_id INTEGER, result_detail TEXT);"
            )
            connection.executemany(
                "INSERT INTO inspection_metadata VALUES (?, ?, ?)",
                [
                    (1, 1, json.dumps({"face": "top", "decode_image": "TOP"})),
                    (2, 1, json.dumps({"face": "bottom", "decode_image": "BOTTOM"})),
                    (3, 2, json.dumps({"decode_image": "OTHER"})),
                ],
            )
            connection.commit()
            connection.close()
            before = path.read_bytes()
            page = DatabasePage()
            page.path.setText(str(path))

            def wait():
                deadline = time.monotonic() + 10
                while page.worker is not None and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.01)
                self.assertIsNone(page.worker)

            page.load(reset=True)
            wait()
            page.data.setCurrentCell(0, 0)
            page.inspect_selected()
            wait()
            self.assertEqual(page.inspector.documents.count(), 2)
            self.assertIn("top", page.inspector.notice.text())
            page.inspector.documents.setCurrentIndex(1)
            self.assertIn("bottom", page.inspector.notice.text())
            page.close()
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
