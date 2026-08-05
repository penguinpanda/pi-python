"""终端图像协议序列生成测试。"""

from __future__ import annotations

import base64

from pi_tui.terminal_image import (
    TerminalImage,
    detect_capabilities,
    encode_iterm2_image,
    encode_kitty_image,
)


def test_detect_capabilities(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "")
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert detect_capabilities() == ("iterm2",)
    monkeypatch.setenv("TERM", "xterm-kitty")
    monkeypatch.setenv("TERM_PROGRAM", "")
    assert detect_capabilities() == ("kitty",)
    monkeypatch.setenv("TERM", "xterm-256color")
    assert detect_capabilities() == ()


def test_encode_kitty_image_single_chunk() -> None:
    out = encode_kitty_image(b"abc")
    assert out.startswith("\x1b_Ga=T,f=100,m=0;")
    assert out.endswith("\x1b\\")
    payload = out.split(";", 1)[1].rstrip("\x1b\\")
    assert base64.b64decode(payload) == b"abc"


def test_encode_kitty_image_chunked() -> None:
    out = encode_kitty_image(b"abcdef", chunk_size=2)
    assert "m=1;" in out
    assert "m=0;" in out
    assert out.startswith("\x1b_Ga=T,f=100,m=1;")


def test_encode_kitty_image_with_size() -> None:
    out = encode_kitty_image(b"abc", width=100, height=50)
    assert "s=100" in out
    assert "v=50" in out


def test_encode_iterm2_image() -> None:
    out = encode_iterm2_image(b"abc", name="pic.png")
    assert out.startswith("\x1b]1337;File=name=pic.png;inline=1:")
    assert out.endswith("\x07")
    payload = out.split(":", 1)[1].rstrip("\x07")
    assert base64.b64decode(payload) == b"abc"


def test_terminal_image_fallback_without_capabilities(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "")
    image = TerminalImage(b"abc", name="pic.png")
    assert image.render() == "[image: pic.png]"


def test_terminal_image_missing_file_fallback(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "")
    image = TerminalImage("/nonexistent/pic.png")
    assert image.render() == "[image: /nonexistent/pic.png]"
