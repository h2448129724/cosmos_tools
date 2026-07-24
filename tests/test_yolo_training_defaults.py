from __future__ import annotations

from pathlib import Path

from trainer_gui.feature_scanner import FeatureScanner
from trainer_gui.history_manager import HistoryManager
from trainer_gui.run_manager import RunManager


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_yolo_train_defaults_export_yolo11_1024_batch4():
    scanner = FeatureScanner(REPO_ROOT)
    yolo_feature = next(feature for feature in scanner.scan() if feature.feature_name == "yolo")
    train_action = next(action for action in yolo_feature.actions if action.action_name == "train")

    from trainer_gui.argparse_parser import ArgparseSchemaParser

    ArgparseSchemaParser().parse_action(train_action)
    params = {field.name: field.default for field in train_action.schema}
    params["data"] = "D:/project/changrui/CAB-F/error/yolo_dataset/data.yaml"

    command = RunManager(REPO_ROOT, HistoryManager(REPO_ROOT)).build_export_command_text(
        yolo_feature,
        train_action,
        params,
        conda_env=None,
        target_platform="windows",
    )

    assert "model=yolo11n.pt" in command
    assert "imgsz=1024" in command
    assert "batch=4" in command
