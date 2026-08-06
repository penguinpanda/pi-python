"""render_utils 展示层工具测试（对齐 TS core/tools/render-utils.ts）。"""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.tools.render_utils import (
    get_image_dimensions,
    get_text_output,
    hyperlink,
    image_fallback,
    invalid_arg_text,
    link_path,
    normalize_display_text,
    render_tool_path,
    replace_tabs,
    shorten_path,
    str_value,
    strip_ansi,
)


class _FakeTheme:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fg(self, name: str, text: str = "") -> str:
        self.calls.append(name)
        return f"<{name}:{text}>"


def test_shorten_path_home_prefix() -> None:
    path = str(Path.home() / "work" / "a.py")
    shortened = shorten_path(path)
    assert shortened.startswith("~")
    assert shortened.endswith("work/a.py") or shortened.endswith("work\\a.py")


def test_shorten_path_non_string_and_other() -> None:
    assert shorten_path(123) == ""
    assert shorten_path("/etc/hosts") == "/etc/hosts"


def test_str_value() -> None:
    assert str_value("x") == "x"
    assert str_value(None) == ""
    assert str_value(42) is None


def test_text_normalization() -> None:
    assert replace_tabs("a\tb") == "a   b"
    assert normalize_display_text("a\r\nb") == "a\nb"


def test_strip_ansi_removes_csi_and_osc() -> None:
    assert strip_ansi("\x1b[31mred\x1b[0m") == "red"
    assert strip_ansi("\x1b]8;;https://example.com\x1b\\link\x1b]8;;\x1b\\") == "link"
    assert strip_ansi("plain") == "plain"


def test_hyperlink_wraps_osc8() -> None:
    assert hyperlink("text", "file:///tmp/a.py") == (
        "\x1b]8;;file:///tmp/a.py\x1b\\text\x1b]8;;\x1b\\"
    )


def test_link_path_supported(monkeypatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    result = link_path("styled", "src/a.py", "/tmp/proj")
    assert result.startswith("\x1b]8;;file:///")
    assert result.endswith("\\styled\x1b]8;;\x1b\\")
    assert "/tmp/proj" in result or "src/a.py" in result


def test_link_path_unsupported_returns_text(monkeypatch) -> None:
    monkeypatch.setenv("TERM", "screen")
    assert link_path("styled", "src/a.py", "/tmp/proj") == "styled"


def test_get_image_dimensions_formats() -> None:
    png = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + (800).to_bytes(4, "big")
        + (600).to_bytes(4, "big")
    )
    assert get_image_dimensions(png, "image/png") == (800, 600)

    gif = b"GIF89a" + (16).to_bytes(2, "little") + (9).to_bytes(2, "little")
    assert get_image_dimensions(gif, "image/gif") == (16, 9)

    jpeg = (
        b"\xff\xd8\xff\xc0\x00\x11\x08"
        + (20).to_bytes(2, "big")
        + (30).to_bytes(2, "big")
        + b"\x11\x22\x33"
    )
    assert get_image_dimensions(jpeg, "image/jpeg") == (30, 20)

    webp = (
        b"RIFF"
        + b"\x00" * 4
        + b"WEBP"
        + b"VP8 "
        + b"\x00" * 10
        + (40).to_bytes(2, "little")
        + (25).to_bytes(2, "little")
    )
    assert get_image_dimensions(webp, "image/webp") == (40, 25)

    assert get_image_dimensions(b"nope", "image/png") is None


def test_image_fallback_plain() -> None:
    assert image_fallback("image/png") == "[Image: [image/png]]"
    assert image_fallback("image/png", (800, 600)) == "[Image: [image/png] 800x600]"


def test_image_fallback_with_filename(monkeypatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    filename = str(Path.cwd() / "pic.png")
    result = image_fallback("image/png", (1, 1), filename)
    assert result.startswith("[Image: \x1b]8;;file:///")
    assert result.endswith(" 1x1]")


def test_get_text_output_text_blocks() -> None:
    result = {
        "content": [
            {"type": "text", "text": "first\r"},
            {"type": "text", "text": "\x1b[31msecond\x1b[0m"},
        ]
    }
    assert get_text_output(result, show_images=True) == "first\nsecond"


def test_get_text_output_image_fallback(monkeypatch) -> None:
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + (2).to_bytes(4, "big")
        + (3).to_bytes(4, "big")
    )
    import base64

    result = {
        "content": [
            {"type": "text", "text": "shown"},
            {
                "type": "image",
                "data": base64.b64encode(png).decode("ascii"),
                "mimeType": "image/png",
            },
        ]
    }
    output = get_text_output(result, show_images=False)
    assert output == "shown\n[Image: [image/png] 2x3]"

    # 终端支持图像且 showImages=True 时不输出回退。
    monkeypatch.setenv("TERM_PROGRAM", "kitty")
    assert get_text_output(result, show_images=True) == "shown"


def test_invalid_arg_text() -> None:
    theme = _FakeTheme()
    assert invalid_arg_text(theme) == "<error:[invalid arg]>"
    assert theme.calls == ["error"]


def test_render_tool_path_none_and_empty(monkeypatch) -> None:
    monkeypatch.setattr("pi_coding_agent.tools.render_utils._hyperlink_supported", lambda: False)
    theme = _FakeTheme()
    assert render_tool_path(None, theme, "/tmp") == "<error:[invalid arg]>"
    assert render_tool_path("", theme, "/tmp") == "<toolOutput:...>"
    assert render_tool_path("", theme, "/tmp", {"emptyFallback": "fallback"}) == (
        "<accent:fallback>"
    )


def test_render_tool_path_linked(monkeypatch) -> None:
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    theme = _FakeTheme()
    result = render_tool_path("/tmp/proj/a.py", theme, "/tmp/proj")
    assert result.startswith("\x1b]8;;file:///")
    assert "<accent:" in result
    assert theme.calls == ["accent"]
