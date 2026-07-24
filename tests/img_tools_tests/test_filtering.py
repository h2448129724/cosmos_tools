from img_tools.core.filtering import move_for_review, undo_review_move


def test_review_move_can_be_undone(tmp_path):
    source = tmp_path / "source" / "a.png"
    source.parent.mkdir()
    source.write_bytes(b"image")

    move = move_for_review(source, tmp_path / "keep", "keep")
    assert move is not None and move.target.exists()
    restored = undo_review_move(move)
    assert restored == source and restored.exists()
