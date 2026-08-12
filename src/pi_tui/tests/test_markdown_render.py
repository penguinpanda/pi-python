"""TS 对齐 Markdown 渲染器测试：LaTeX / 主题回调 / 流式 fence / OSC8。"""

from __future__ import annotations

from rich.style import Style

from pi_tui.engine.cells import Line
from pi_tui.engine.latex import render_latex
from pi_tui.engine.markdown_render import (
    DefaultTextStyle,
    MarkdownOptions,
    MarkdownTheme,
    render_markdown_lines,
)
from pi_tui.engine.text import render_markdown, strip_ansi


def _plain(lines: list[Line]) -> list[str]:
    return [strip_ansi(line.text()).rstrip() for line in lines]


def test_render_latex_unicode() -> None:
    assert render_latex(r"x^2 + y_1") == "x²+y₁"
    assert render_latex(r"\frac{1}{2}") == "1/2"
    assert render_latex(r"\sqrt{x}") == "√x"
    assert render_latex(r"\alpha \leq \beta") == "α ≤ β"
    assert render_latex(r"\mathbb{R}") == "ℝ"
    assert render_latex(r"\sum_{i=1}^{n} i", display=True) is not None
    assert render_latex(r"\unknowncommand{x}") is None


def test_markdown_renders_inline_and_block_latex() -> None:
    lines = render_markdown("Math $x^2$ and $$\\frac{1}{2}$$", 40)
    plain = _plain(lines)
    assert any("x²" in line for line in plain)
    assert any("1/2" in line for line in plain)


def test_strikethrough_uses_theme_style() -> None:
    lines = render_markdown("~~gone~~ and **bold**", 40)
    assert any("gone" in line.text() for line in lines)
    strike_cells = [
        cell
        for line in lines
        for cell in line.cells
        if cell.char in "gone" and cell.style is not None and cell.style.strike
    ]
    assert strike_cells
    bold_cells = [
        cell
        for line in lines
        for cell in line.cells
        if cell.char in "bold" and cell.style is not None and cell.style.bold
    ]
    assert bold_cells


def test_streaming_partial_fence_is_trimmed() -> None:
    lines = render_markdown("```py\nprint(1)\n``", 40)
    plain = _plain(lines)
    assert any("print(1)" in line for line in plain)
    assert not any(line.strip() == "``" for line in plain)


class _RedTheme(MarkdownTheme):
    def heading(self, base: Style | None, level: int) -> Style:
        return (base or Style()) + Style(color="red")


def test_theme_callbacks_style_headings() -> None:
    lines = render_markdown_lines("# Title", 40, _RedTheme())
    heading_cells = [
        cell for line in lines for cell in line.cells if cell.style is not None and cell.style.color
    ]
    assert heading_cells
    assert "red" in str(heading_cells[0].style.color)


def test_transform_hook() -> None:
    options = MarkdownOptions(transform=lambda text, width: f"{text}!")
    lines = render_markdown_lines("hello", 40, MarkdownTheme(), options=options)
    assert any("hello!" in line.text() for line in lines)


def test_preserve_ordered_list_markers() -> None:
    options = MarkdownOptions(preserve_ordered_list_markers=True)
    lines = render_markdown_lines("3. three\n4. four", 40, MarkdownTheme(), options=options)
    plain = _plain(lines)
    assert any(line.startswith("3. ") for line in plain)
    assert any(line.startswith("4. ") for line in plain)


def test_links_carry_osc8_cell_links() -> None:
    lines = render_markdown("[pi](https://example.com)", 40)
    links = {cell.link for line in lines for cell in line.cells if cell.link}
    assert links == {"https://example.com"}


def test_default_text_style_applied() -> None:
    style = DefaultTextStyle(color="red", bold=True)
    lines = render_markdown_lines("plain", 40, MarkdownTheme(), default_style=style)
    cells = [cell for line in lines for cell in line.cells if cell.char == "p"]
    assert cells
    assert cells[0].style is not None
    assert cells[0].style.bold
    assert "red" in str(cells[0].style.color)


def test_table_renders_with_cells() -> None:
    lines = render_markdown("| a | b |\n|---|---|\n| 1 | 2 |", 40)
    plain = _plain(lines)
    assert any("│ a │ b │" in line for line in plain)
    assert any("│ 1 │ 2 │" in line for line in plain)


def test_table_narrow_falls_back_to_raw_markdown() -> None:
    lines = render_markdown("| Provider | 模型 ID |\n|---|---|\n| OpenAI | gpt-5 |", 8)
    plain = _plain(lines)
    assert any("Provider" in line for line in plain)
    assert any("OpenAI" in line for line in plain)


def test_table_wrapped_cells_keep_column_alignment() -> None:
    lines = render_markdown(
        "| 脚本 | 功能 |\n|---|---|\n| generate_models.py | 自动生成模型元数据 |",
        40,
    )
    raw = ["".join(cell.char for cell in line.cells) for line in lines]
    first = next(line for line in raw if line.startswith("│ generate_models.p"))
    second = next(line for line in raw if line.startswith("│ y"))
    assert first.index("│", 2) == second.index("│", 2)
    assert second[second.index("│", 2) - 1] == " "
