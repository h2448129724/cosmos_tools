from img_tools.core.output import resolve_output_path


def test_output_collision_policies(tmp_path):
    target = tmp_path / "output.png"
    target.write_bytes(b"old")

    assert resolve_output_path(target, "skip") is None
    assert resolve_output_path(target, "overwrite") == target
    assert resolve_output_path(target, "rename") == tmp_path / "output_1.png"
