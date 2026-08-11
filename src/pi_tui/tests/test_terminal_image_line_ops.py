"""Terminal Image 行级操作测试。"""

from __future__ import annotations

from pi_tui.terminal_image import (
    allocate_image_id,
    crop_kitty_image_line,
    get_kitty_image_metadata,
    is_image_line,
    register_kitty_image_metadata,
)


def test_is_image_line() -> None:
    assert is_image_line("\x1b_Ga=T;abc\x1b\\")
    assert is_image_line("\x1b]1337;File=inline=1:abc\x07")
    assert is_image_line("prefix \x1b_Ga=p;abc\x1b\\ suffix")
    assert not is_image_line("plain")


def test_allocate_image_id_is_in_range() -> None:
    for _ in range(20):
        assert 1 <= allocate_image_id() <= 0xFFFFFFFF


def test_register_and_get_metadata() -> None:
    register_kitty_image_metadata(
        {"imageId": 7, "columns": 4, "rows": 2, "widthPx": 400, "heightPx": 200}
    )
    metadata = get_kitty_image_metadata("\x1b_Ga=p,i=7;payload\x1b\\")
    assert metadata is not None
    assert metadata["rows"] == 2


def test_crop_kitty_image_line() -> None:
    register_kitty_image_metadata(
        {"imageId": 9, "columns": 4, "rows": 2, "widthPx": 400, "heightPx": 200}
    )
    line = "\x1b_Ga=p,i=9,x=0;payload\x1b\\"
    cropped = crop_kitty_image_line(line, 1, 1)
    assert "y=100" in cropped
    assert "h=100" in cropped
    assert "r=1" in cropped
    assert "x=0" in cropped


def test_crop_unknown_line_unchanged() -> None:
    line = "\x1b_Ga=p,i=999;payload\x1b\\"
    assert crop_kitty_image_line(line, 1, 1) == line
