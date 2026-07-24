from img_tools.core.state import add_task_history, load_state, save_state


def test_state_round_trip_and_history_limit(tmp_path):
    path = tmp_path / "state.json"
    state = load_state(path)
    for number in range(55):
        add_task_history(state, task_type="crop", output_dir=f"out-{number}", summary="ok")
    save_state(state, path)

    restored = load_state(path)
    assert restored["version"] == 1
    assert len(restored["task_history"]) == 50
    assert restored["task_history"][0]["output_dir"] == "out-54"
