"""Offscreen selection and worker-contract regression checks (no inference)."""
import os
import sys
import threading
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from cosmos_toolbox.field_dataset import MODEL_IDS
from cosmos_toolbox.field_dataset_ui import DatasetWorker, FieldDatasetPage, configured_product
from cosmos_toolbox.capability_catalog import plan_default_capabilities


class FieldDatasetUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_independent_selection_and_presets(self):
        page = FieldDatasetPage()
        self.assertEqual(set(page.checks), set(MODEL_IDS))
        self.assertEqual(len(page.checks), 14)
        self.assertEqual(page.environment.currentText(), 'onnx-gpu')
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

    def test_config_product_and_no_manual_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'arbitrary_name.yaml'
            config.write_text('inspection:\n  product: D01-L\n', encoding='utf-8')
            self.assertEqual(configured_product(config), 'D01-L')
            page = FieldDatasetPage()
            page.product_config.setText(str(config))
            self.assertEqual(page.product.text(), 'D01-L')
            self.assertFalse(hasattr(page.product, 'currentText'))
            page.close()
            config.write_text('inspection: {}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'inspection.product'):
                configured_product(config)

    def test_batch_same_name_folders_keep_independent_outputs_and_config_product(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text('inspection:\n  product: D01-L\n', encoding='utf-8')
            sources = [str(Path(directory) / branch / '0907') for branch in ('a', 'b')]
            control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
            options = dict(sources=sources, output=str(Path(directory) / 'out'),
                           product='D01-R', product_config=str(config), selected=['hook_detector'])
            worker = DatasetWorker(options, control, environment_name='onnx-gpu')
            received = []
            worker.result.connect(received.append)
            with patch('cosmos_toolbox.field_dataset_runtime.run_in_environment', return_value={'completed_model_jobs': 1}) as runner:
                worker.run()
                calls = [call.args[0] for call in runner.call_args_list]
            self.assertEqual([call['source'] for call in calls], sources)
            self.assertTrue(all(call['product'] == 'D01-L' for call in calls))
            self.assertEqual(len({call['output'] for call in calls}), 2)
            self.assertEqual(len(received[0]['results']), 2)
            self.assertFalse(received[0]['errors'])

    def test_batch_stops_before_next_folder(self):
        control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
        worker = DatasetWorker(dict(sources=['a', 'b'], output='out'), control)
        def run_one(options):
            control.stopped.set()
            return {'stopped': True}
        received = []
        worker.result.connect(received.append)
        with patch.object(worker, '_run_one', side_effect=run_one) as runner:
            worker.run()
        self.assertEqual(runner.call_count, 1)
        self.assertTrue(received[0]['stopped'])

    def test_batch_scan_aggregates_and_keeps_going_after_folder_error(self):
        control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
        worker = DatasetWorker(dict(sources=['a', 'b', 'c'], output='out'), control, scan_only=True)
        received = []
        worker.result.connect(received.append)
        with patch.object(worker, '_run_one', side_effect=[
                {'count': 2, 'bytes': 10}, ValueError('broken folder'), {'count': 3, 'bytes': 20}]):
            worker.run()
        self.assertEqual(received[0]['count'], 5)
        self.assertEqual(received[0]['bytes'], 30)
        self.assertEqual(len(received[0]['errors']), 1)

    def test_capability_registered(self):
        planned = plan_default_capabilities([], include_project_capabilities=True)
        item = next(p for p in planned if p.spec.key == "cabf.field_dataset")
        self.assertEqual(item.page_factory_key, "cabf_field_dataset")

    def test_selected_environment_routes_to_child_runner(self):
        control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
        options = {'source': 'fixture', 'product_config': 'D:/configs/custom.yaml'}
        worker = DatasetWorker(options, control, environment_name='custom-gpu')
        received = []
        worker.result.connect(received.append)
        with patch('cosmos_toolbox.field_dataset_runtime.run_in_environment', return_value={'samples': 1}) as runner:
            worker.run()
        self.assertEqual(runner.call_args.args[:3], (options, 'custom-gpu', control))
        self.assertEqual(received, [{'samples': 1}])

    def test_missing_environment_does_not_fall_back(self):
        from cosmos_toolbox.field_dataset_runtime import run_in_environment
        control = SimpleNamespace(paused=threading.Event(), stopped=threading.Event())
        with patch('shared.conda_runtime.CondaEnvManager.find', return_value=None):
            with self.assertRaisesRegex(ValueError, 'missing-env'):
                run_in_environment({}, 'missing-env', control, lambda value: None)

    def test_torch_preload_reports_version(self):
        from cosmos_toolbox.field_dataset_runtime import preload_torch
        messages = []
        torch = SimpleNamespace(__version__='test', version=SimpleNamespace(cuda='12.4'))
        with patch('cosmos_toolbox.field_dataset_runtime.importlib.import_module', return_value=torch) as importer:
            preload_torch(messages.append)
        importer.assert_called_once_with('torch')
        self.assertIn('12.4', messages[-1]['message'])

    def test_torch_preload_failure_is_reported_without_installation(self):
        from cosmos_toolbox.field_dataset_runtime import preload_torch
        messages = []
        with patch('cosmos_toolbox.field_dataset_runtime.importlib.import_module', side_effect=ImportError('missing torch')):
            preload_torch(messages.append)
        self.assertIn('missing torch', messages[-1]['message'])
        self.assertIn('不会自动安装', messages[-1]['message'])


if __name__ == "__main__":
    unittest.main()
