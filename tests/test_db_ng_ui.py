"""Offscreen contract checks for the NG extraction page (no image copying)."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from cosmos_toolbox.db_ng_ui import DbNgPage, DbNgWorker


class DbNgUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_defaults_and_kinds(self):
        page = DbNgPage()
        self.assertTrue(page.database.text().endswith("cosmos.db"))
        self.assertEqual(page.project.text(), "CAB-F")
        self.assertTrue(page.raw.isChecked())
        self.assertFalse(page.result.isChecked())
        self.assertEqual(page._options()["kinds"], ("raw",))
        page.result.setChecked(True)
        self.assertEqual(page._options()["kinds"], ("raw", "result"))
        page.close()

    def test_scan_worker_passes_read_only_filter_contract(self):
        control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
        options = dict(
            database="fixture.db", project="CAB-F", product="D01-R", start="2026-01-01",
            end="2026-01-02", exclude_misjudged=True, kinds=("raw", "result"),
            old_root="old", new_root="new",
        )
        worker = DbNgWorker(options, control, scan_only=True)
        received = []
        worker.result.connect(received.append)
        plan = {"summary": {"ng_records": 1}}
        with patch("cosmos_toolbox.db_ng_export.scan", return_value=plan) as scanner:
            worker.run()
        scanner.assert_called_once_with(**options)
        self.assertEqual(received, [plan])

    def test_copy_worker_uses_same_plan_and_control(self):
        control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
        plan = {"files": [{"source": "a", "status": "ready"}]}
        worker = DbNgWorker({"plan": plan, "output": "out"}, control, scan_only=False)
        received = []
        worker.result.connect(received.append)
        report = {"copied": 1, "stopped": False}
        with patch("cosmos_toolbox.db_ng_export.copy_plan", return_value=report) as copier:
            worker.run()
        self.assertIs(copier.call_args.args[0], plan)
        self.assertEqual(copier.call_args.args[1], "out")
        self.assertIs(copier.call_args.kwargs["control"], control)
        self.assertTrue(callable(copier.call_args.kwargs["on_progress"]))
        self.assertEqual(received, [report])

    def test_plan_is_invalidated_when_filter_or_output_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sample.db"
            database.touch()
            page = DbNgPage()
            page.database.setText(str(database))
            plan = {"summary": {"ng_records": 2, "image_records": 3, "ready_files": 3, "missing_files": 0, "bytes": 4}}
            page._on_scan_result(plan)
            self.assertIs(page.plan, plan)
            self.assertTrue(page.copy_button.isEnabled())
            page.product.setText("D01-R")
            self.assertIsNone(page.plan)
            page._on_scan_result(plan)
            page.output.setText(str(Path(directory) / "out"))
            self.assertIsNone(page.plan)
            page.close()


if __name__ == "__main__":
    unittest.main()
