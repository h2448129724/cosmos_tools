import json

from img_tools.core.tasking import TaskSummary, write_task_report


def test_task_report_writes_json_and_csv(tmp_path):
    json_path, csv_path = write_task_report(tmp_path, TaskSummary("crop", 3, 2, ("a.png: bad",), True), parameters={"mode": "scaled"})

    assert json.loads(json_path.read_text(encoding="utf-8"))["cancelled"] is True
    assert "a.png: bad" in csv_path.read_text(encoding="utf-8-sig")
