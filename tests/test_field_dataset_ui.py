"""Offscreen selection and worker-contract regression checks (no inference)."""
import os
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from cosmos_toolbox.field_dataset import MODEL_IDS
from cosmos_toolbox.field_dataset_ui import DatasetWorker, FieldDatasetPage
from cosmos_toolbox.capability_catalog import plan_default_capabilities


class FieldDatasetUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_independent_selection_and_presets(self):
        page = FieldDatasetPage()
        self.assertEqual(set(page.checks), set(MODEL_IDS))
        self.assertEqual(len(page.checks), 14)
        page._select("all")
        self.assertTrue(all(c.isChecked() for c in page.checks.values()))
        page._select("none")
        page.checks["tail_cloth_roi_detector"].setChecked(True)
        self.assertEqual([k for k, c in page.checks.items() if c.isChecked()], ["tail_cloth_roi_detector"])
        self.assertIn("公共区域", page.dependencies.text())
        self.assertIn("尾部区域", page.dependencies.text())
        page._select("tail")
        self.assertEqual(sum(c.isChecked() for c in page.checks.values()), 5)
        self.assertEqual([page.face.itemData(i) for i in range(3)], ["all", "top", "bottom"])
        page.close()

    def test_worker_passes_selected_mode_and_control(self):
        control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
        options = dict(source="source", output="output", product="D01-L", selected=["knife_segment"],
                       face="bottom", mode="images", limit=2)
        worker = DatasetWorker(options, control)
        received = []
        worker.result.connect(received.append)
        with patch("cosmos_toolbox.field_dataset.run", return_value={"completed": 2}) as runner:
            worker.run()
            self.assertEqual({k: runner.call_args.kwargs[k] for k in options}, options)
            self.assertIs(runner.call_args.kwargs["control"], control)
            self.assertTrue(callable(runner.call_args.kwargs["on_progress"]))
        self.assertEqual(received, [{"completed": 2}])

    def test_capability_registered(self):
        planned = plan_default_capabilities([], include_project_capabilities=True)
        item = next(p for p in planned if p.spec.key == "cabf.field_dataset")
        self.assertEqual(item.page_factory_key, "cabf_field_dataset")


if __name__ == "__main__":
    unittest.main()
