"""OSC8 链接工具测试。"""

from __future__ import annotations

from pi_tui.engine.cells import line_from_text
from pi_tui.links import linkify_lines, linkify_paths, osc8_hyperlink_supported


def test_osc8_supported_detection(monkeypatch) -> None:
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.delenv("TMUX", raising=False)
    assert osc8_hyperlink_supported() is False
    monkeypatch.setenv("WT_SESSION", "1")
    assert osc8_hyperlink_supported() is True
    monkeypatch.delenv("WT_SESSION")
    monkeypatch.setenv("TMUX", "tmux")
    assert osc8_hyperlink_supported() is False


def test_linkify_paths_wraps_existing_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "wezterm")
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    out = linkify_paths(f"see {target} now")
    assert f"[link={target.resolve().as_uri()}]" in out
    assert "[/link]" in out


def test_linkify_paths_skips_missing_and_plain_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "wezterm")
    missing = tmp_path / "missing.txt"
    out = linkify_paths(f"see {missing} now")
    assert "[link=" not in out
    assert linkify_paths("no paths here") == "no paths here"


def test_linkify_paths_disabled_terminal(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    monkeypatch.delenv("WT_SESSION", raising=False)
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    out = linkify_paths(f"see {target} now")
    assert "[link=" not in out


def test_linkify_lines_sets_cell_links(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "wezterm")
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    line = line_from_text(f"see {target} now", 120)
    linkify_lines([line])
    linked = [cell.link for cell in line.cells if cell.link]
    assert linked and linked[0] == target.resolve().as_uri()
