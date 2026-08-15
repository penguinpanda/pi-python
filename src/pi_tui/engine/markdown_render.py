"""TS 对齐的 Markdown 渲染器。

基于 markdown-it-py 分词（Rich 的依赖，无需新增包），按
packages/tui/src/components/markdown.ts 的语义渲染：
- MarkdownTheme 按元素回调（heading/link/code/quote/hr/list/bold/italic/...）
- DefaultTextStyle 全局应用
- LaTeX 数学（$..$ / $$..$$ / \\(..\\) / \\[..\\]）→ Unicode
- 流式 fence 部分闭合裁剪（避免闪烁）
- 保留有序列表原始序号 + 任务列表
- transform(markdown, width) 钩子
- OSC 8 链接（Cell.link）
- 表格按列宽换行
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from markdown_it import MarkdownIt
from rich.cells import cell_len
from rich.style import Style

from .cells import Cell, Line, blank_line
from .latex import render_latex

INLINE_LATEX_START = "\uf0002"
INLINE_LATEX_END = "\uf0003"
BLOCK_LATEX_START = "\uf0004"
BLOCK_LATEX_END = "\uf0005"
PLACEHOLDER_PATTERN = re.compile(
    rf"{INLINE_LATEX_START}(\d+){INLINE_LATEX_END}|{BLOCK_LATEX_START}(\d+){BLOCK_LATEX_END}"
)


@dataclass
class DefaultTextStyle:
    """全局文本样式（对齐 TS DefaultTextStyle）。"""

    color: str | None = None
    bg_color: str | None = None
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False


@dataclass
class MarkdownOptions:
    """渲染选项（对齐 TS MarkdownOptions）。"""

    preserve_ordered_list_markers: bool = False
    preserve_backslash_escapes: bool = False
    transform: Callable[[str, int], str] | None = None
    render_latex: bool = True


class MarkdownTheme:
    """元素样式回调（对齐 TS MarkdownTheme，返回 Rich Style 叠加层）。"""

    code_block_indent: str = "  "

    def heading(self, base: Style | None, level: int) -> Style:
        return (base or Style()) + Style(bold=True)

    def link(self, base: Style | None) -> Style:
        return (base or Style()) + Style(underline=True)

    def link_url(self, base: Style | None) -> Style:
        return base or Style()

    def code(self, base: Style | None) -> Style:
        return (base or Style()) + Style()

    def code_block(self, base: Style | None) -> Style:
        return self.code(base)

    def code_block_border(self, base: Style | None) -> Style:
        return base or Style()

    def quote(self, base: Style | None) -> Style:
        return (base or Style()) + Style(italic=True)

    def quote_border(self, base: Style | None) -> Style:
        return base or Style()

    def hr(self, base: Style | None) -> Style:
        return base or Style()

    def list_bullet(self, base: Style | None) -> Style:
        return base or Style()

    def bold(self, base: Style | None) -> Style:
        return (base or Style()) + Style(bold=True)

    def italic(self, base: Style | None) -> Style:
        return (base or Style()) + Style(italic=True)

    def strikethrough(self, base: Style | None) -> Style:
        return (base or Style()) + Style(strike=True)

    def underline(self, base: Style | None) -> Style:
        return (base or Style()) + Style(underline=True)

    def highlight_code(self, code: str, lang: str | None) -> list[Line] | None:
        return None


def _style(color: Any, **attributes: Any) -> Style:
    from rich.color import Color

    if isinstance(color, tuple):
        color = Color.from_rgb(*color)
    bgcolor = attributes.pop("bgcolor", None)
    if isinstance(bgcolor, tuple):
        bgcolor = Color.from_rgb(*bgcolor)
    if color is not None:
        attributes["color"] = color
    if bgcolor is not None:
        attributes["bgcolor"] = bgcolor
    return Style(**attributes)


class ThemeMarkdownTheme(MarkdownTheme):
    """按 pi 主题颜色表构造的默认主题。"""

    def __init__(
        self,
        colors: dict[str, str] | None = None,
        *,
        code_theme: str = "monokai",
    ) -> None:
        self.colors = dict(colors or {})
        self.code_theme = code_theme

    def _color(self, *keys: str) -> str | None:
        for key in keys:
            value = self.colors.get(key)
            if value:
                return value
        return None

    def heading(self, base: Style | None, level: int) -> Style:
        return (base or Style()) + _style(
            self._color("mdHeading", "markdownHeading", "heading"),
            bold=True,
            underline=level == 1,
        )

    def link(self, base: Style | None) -> Style:
        return (base or Style()) + _style(
            self._color("mdLink", "markdownLink", "link"),
            underline=True,
        )

    def link_url(self, base: Style | None) -> Style:
        return (base or Style()) + _style(
            self._color("mdLinkUrl", "markdownLinkUrl", "textDim", "dim")
        )

    def code(self, base: Style | None) -> Style:
        return (base or Style()) + _style(
            self._color("mdCode", "code_fg", "text"),
            bgcolor=self._color("code_bg", "bgPanel"),
        )

    def code_block(self, base: Style | None) -> Style:
        return (base or Style()) + _style(self._color("mdCodeBlock", "code_fg", "text"))

    def code_block_border(self, base: Style | None) -> Style:
        return (base or Style()) + _style(self._color("mdCodeBlockBorder", "textDim", "dim"))

    def quote(self, base: Style | None) -> Style:
        return (base or Style()) + _style(
            self._color("mdQuote", "textDim", "dim"),
            italic=True,
        )

    def quote_border(self, base: Style | None) -> Style:
        return (base or Style()) + _style(self._color("mdQuoteBorder", "textDim", "dim"))

    def hr(self, base: Style | None) -> Style:
        return (base or Style()) + _style(self._color("mdHr", "textDim", "dim"))

    def list_bullet(self, base: Style | None) -> Style:
        return (base or Style()) + _style(self._color("mdListBullet", "accent", "markdownHeading"))

    def highlight_code(self, code: str, lang: str | None) -> list[Line] | None:
        # 延迟导入避免 text.py ↔ markdown_render.py 循环依赖。
        from .text import render_renderable

        from rich.syntax import Syntax

        if not lang:
            return None
        try:
            renderable = Syntax(code, lang, theme=self.code_theme)
        except Exception:
            return None
        try:
            return render_renderable(renderable, 10_000)
        except Exception:
            return None


def _visible_width(value: str) -> int:
    return cell_len(value)


def _is_escaped(source: str, index: int) -> bool:
    backslashes = 0
    position = index - 1
    while position >= 0 and source[position] == "\\":
        backslashes += 1
        position -= 1
    return backslashes % 2 == 1


def _find_closing(source: str, closing: str, start: int) -> int:
    index = source.find(closing, start)
    while index >= 0 and _is_escaped(source, index):
        index = source.find(closing, index + len(closing))
    return index


def _looks_like_pending_math(source: str) -> bool:
    return re.search(r"\\[A-Za-z]+|[_^=+*/<>()[\]|±≤≥≠≈∈→⇒∞∫∑√-]", source) is not None


def _preprocess_strikethrough(source: str) -> str:
    """把 ~~text~~ 转成 <del>text</del>，跳过围栏代码块。"""
    lines = source.split("\n")
    in_fence = False
    output: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        elif not in_fence:
            line = re.sub(r"~~(?=[^\s~])(.+?)(?<=[^\s~\\])~~", r"<del>\1</del>", line)
        output.append(line)
    return "\n".join(output)


def _preprocess_latex(
    source: str,
) -> tuple[str, dict[int, str], set[int]]:
    """提取 LaTeX 数学为占位符；渲染失败或 pending 时保留原文。"""
    # 围栏保护：围栏代码块内的 $ 先以哨兵替换，处理后按行号还原，
    # 避免代码里的 shell 变量被 LaTeX 占位符改写。
    lines = source.split("\n")
    protected: list[tuple[int, str]] = []
    in_fence = False
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            if "$" in line:
                protected.append((index, line))
                lines[index] = line.replace("$", "\u0001")
            continue
        if in_fence and "$" in line:
            protected.append((index, line))
            lines[index] = line.replace("$", "\u0001")
    source = "\n".join(lines)

    latex_map: dict[int, str] = {}
    block_ids: set[int] = set()

    def add(text: str, block: bool) -> str | None:
        rendered = render_latex(text, display=block)
        if rendered is None:
            return None
        identifier = len(latex_map)
        latex_map[identifier] = rendered
        if block:
            block_ids.add(identifier)
            return f"{BLOCK_LATEX_START}{identifier}{BLOCK_LATEX_END}"
        return f"{INLINE_LATEX_START}{identifier}{INLINE_LATEX_END}"

    # Block: $$...$$ 与 \[...\]（行首，最多 3 空格缩进）。
    block_patterns = [
        re.compile(r"^ {0,3}\$\$[ \t]*(?:\n)?([\s\S]*?)\$\$[ \t]*(?:\n|$)", re.MULTILINE),
        re.compile(r"^ {0,3}\\\[[ \t]*(?:\n)?([\s\S]*?)\\\][ \t]*(?:\n|$)", re.MULTILINE),
    ]
    for pattern in block_patterns:
        result: list[str] = []
        position = 0
        for match in pattern.finditer(source):
            result.append(source[position : match.start()])
            placeholder = add(match.group(1).strip(), True)
            if placeholder is not None:
                # 保持行数不变，围栏还原的行号索引才稳定。
                result.append(placeholder + "\n" * match.group(0).count("\n"))
            else:
                result.append(match.group(0))
            position = match.end()
        result.append(source[position:])
        source = "".join(result)

    # Inline: $..$ / $$..$$ / \(..\) / \[..\]（转义感知）。
    output: list[str] = []
    position = 0
    while position < len(source):
        dollar = source.find("$", position)
        paren = source.find("\\(", position)
        bracket = source.find("\\[", position)
        candidates = [index for index in (dollar, paren, bracket) if index >= 0]
        if not candidates:
            output.append(source[position:])
            break
        index = min(candidates)
        output.append(source[position:index])
        if index == paren:
            opening, closing = "\\(", "\\)"
        elif index == bracket:
            opening, closing = "\\[", "\\]"
        elif source.startswith("$$", index):
            opening, closing = "$$", "$$"
        else:
            opening, closing = "$", "$"
        closing_index = _find_closing(source, closing, index + len(opening))
        if closing_index < 0:
            output.append(source[index:])
            break
        text = source[index + len(opening) : closing_index]
        if not text or "\n" in text:
            output.append(source[index : closing_index + len(closing)])
            position = closing_index + len(closing)
            continue
        raw = source[index : closing_index + len(closing)]
        placeholder = add(text, False)
        output.append(placeholder if placeholder is not None else raw)
        position = closing_index + len(closing)
    source = "".join(output)

    # 还原围栏行。
    if protected:
        lines = source.split("\n")
        for index, original in protected:
            if index < len(lines):
                lines[index] = original
        source = "\n".join(lines)
    return source, latex_map, block_ids


_parser: MarkdownIt | None = None


def _get_parser() -> MarkdownIt:
    global _parser
    if _parser is None:
        parser = MarkdownIt("commonmark", {"html": True})
        parser.enable("table")
        _parser = parser
    return _parser


def _trim_partial_closing_fences(tokens: list[Any]) -> None:
    """裁剪流式到达的部分闭合围栏，避免代码块高度闪烁（仅最后一个 fence）。"""
    fence = next((token for token in reversed(tokens) if token.type == "fence"), None)
    if fence is None:
        return
    marker = fence.markup or ""
    last_line = fence.content.split("\n")[-1] if fence.content else ""
    if (
        marker
        and last_line
        and len(last_line) < len(marker)
        and last_line == marker[0] * len(last_line)
    ):
        fence.content = fence.content[: -len(last_line)].rstrip("\n")


@dataclass
class _Node:
    token_type: str
    token: Any
    children: list["_Node"] = field(default_factory=list)


def _build_tree(tokens: list[Any]) -> list[_Node]:
    root = _Node("root", None)
    stack = [root]
    for token in tokens:
        if token.nesting == 1:
            node = _Node(token.type, token)
            stack[-1].children.append(node)
            stack.append(node)
        elif token.nesting == -1:
            if len(stack) > 1:
                stack.pop()
        else:
            stack[-1].children.append(_Node(token.type, token))
    return root.children


class _Renderer:
    def __init__(
        self,
        theme: MarkdownTheme,
        default_style: DefaultTextStyle | None = None,
        options: MarkdownOptions | None = None,
    ) -> None:
        self.theme = theme
        self.default_style = default_style or DefaultTextStyle()
        self.options = options or MarkdownOptions()
        self.latex_map: dict[int, str] = {}
        self.block_ids: set[int] = set()
        self.source = ""

    def render(self, text: str, width: int) -> list[Line]:
        safe_width = max(1, int(width))
        source = (
            self.options.transform(text, safe_width) if self.options.transform is not None else text
        )
        if not source.strip():
            return []
        self.source = source
        source = source.replace("\t", "   ")
        source = _preprocess_strikethrough(source)
        if self.options.render_latex:
            source, self.latex_map, self.block_ids = _preprocess_latex(source)
        tokens = _get_parser().parse(source)
        _trim_partial_closing_fences(tokens)
        tree = _build_tree(tokens)
        default_style = self._default_style_style()
        logical = self._render_blocks(tree, safe_width, default_style)
        return self._wrap_and_pad(logical, safe_width)

    def _default_style_style(self) -> Style | None:
        style = self.default_style
        return Style(
            color=style.color,
            bgcolor=style.bg_color,
            bold=style.bold,
            italic=style.italic,
            strike=style.strikethrough,
            underline=style.underline,
        )

    # ------------------------------------------------------------------
    # Block 渲染
    # ------------------------------------------------------------------

    def _render_blocks(
        self,
        nodes: list[_Node],
        width: int,
        base_style: Style | None,
    ) -> list[list[Cell]]:
        lines: list[list[Cell]] = []
        for index, node in enumerate(nodes):
            next_type = nodes[index + 1].token_type if index + 1 < len(nodes) else None
            lines.extend(self._render_block(node, width, base_style, next_type))
        return lines

    def _render_block(
        self,
        node: _Node,
        width: int,
        base_style: Style | None,
        next_type: str | None,
    ) -> list[list[Cell]]:
        token_type = node.token_type
        if token_type in ("heading_open",):
            level = int((node.token.tag or "h1")[1:])
            heading_style = self.theme.heading(base_style, level)
            if level == 1:
                heading_style = self.theme.bold(self.theme.underline(heading_style))
            else:
                heading_style = self.theme.bold(heading_style)
            rendered = self._render_children_inline(node.children, heading_style)
            if level >= 3:
                prefix = _cells_from_text("#" * level + " ", heading_style)
                if rendered:
                    rendered[0] = prefix + rendered[0]
                else:
                    rendered = [prefix]
            if next_type and next_type != "space":
                rendered.append([])
            return rendered

        if token_type == "paragraph_open":
            rendered = self._render_children_inline(node.children, base_style)
            if next_type and next_type not in ("list", "space"):
                rendered.append([])
            return rendered

        if token_type == "inline":
            return self._render_inline_children(node.token.children or [], base_style)

        if token_type in ("fence", "code_block"):
            return self._render_code(node, width, base_style, next_type)

        if token_type in ("bullet_list_open", "ordered_list_open"):
            return self._render_list(node, 0, width, base_style)

        if token_type == "blockquote_open":
            return self._render_blockquote(node, width, base_style, next_type)

        if token_type == "hr":
            line = _cells_from_text("─" * min(width, 80), self.theme.hr(base_style))
            result = [line]
            if next_type and next_type != "space":
                result.append([])
            return result

        if token_type == "table_open":
            return self._render_table(node, width, base_style, next_type)

        if token_type == "html_block":
            raw = (node.token.content or "").strip()
            return [list(_cells_from_text(line, base_style)) for line in raw.split("\n") if line]

        if token_type == "space":
            return [[]]

        if token_type in (
            "paragraph_close",
            "heading_close",
            "blockquote_close",
            "list_item_close",
            "list_close",
        ):
            return []

        content = getattr(node.token, "content", None)
        if isinstance(content, str):
            return [list(_cells_from_text(content, base_style))]
        return []

    def _render_children_inline(
        self,
        nodes: list[_Node],
        base_style: Style | None,
    ) -> list[list[Cell]]:
        result: list[list[Cell]] = []
        for node in nodes:
            if node.token_type == "inline":
                result.extend(self._render_inline_children(node.token.children or [], base_style))
            else:
                result.extend(self._render_block(node, 10_000, base_style, None))
        return result

    def _render_code(
        self,
        node: _Node,
        width: int,
        base_style: Style | None,
        next_type: str | None,
    ) -> list[list[Cell]]:
        lang = (node.token.info or "").strip() if node.token_type == "fence" else ""
        border_style = self.theme.code_block_border(base_style)
        lines: list[list[Cell]] = [
            list(_cells_from_text(f"```{lang}", border_style)),
        ]
        code = (node.token.content or "").rstrip("\n")
        indent = self.theme.code_block_indent
        highlighted = self.theme.highlight_code(code, lang or None)
        if highlighted is not None:
            for hl_line in highlighted:
                cells = list(_cells_from_text(indent, base_style))
                cells.extend(hl_line.cells)
                lines.append(cells)
        else:
            code_style = self.theme.code_block(base_style)
            for code_line in code.split("\n"):
                cells = list(_cells_from_text(indent, code_style))
                cells.extend(_cells_from_text(code_line, code_style))
                lines.append(cells)
        lines.append(list(_cells_from_text("```", border_style)))
        if next_type and next_type != "space":
            lines.append([])
        return lines

    def _render_list(
        self,
        node: _Node,
        depth: int,
        width: int,
        base_style: Style | None,
    ) -> list[list[Cell]]:
        lines: list[list[Cell]] = []
        ordered = node.token_type == "ordered_list_open"
        try:
            start = int(node.token.attrs.get("start", 1))
        except (TypeError, ValueError):
            start = 1
        indent = "    " * depth
        for index, item in enumerate(node.children):
            marker = self._list_marker(item.token, ordered, start + index, index)
            marker_cells = list(_cells_from_text(marker, self.theme.list_bullet(base_style)))
            first_prefix = list(_cells_from_text(indent, base_style)) + marker_cells
            continuation_prefix = list(_cells_from_text(indent, base_style)) + list(
                _cells_from_text(" " * _visible_width(marker), base_style)
            )
            item_width = max(1, width - _visible_width(indent + marker))
            item_lines: list[list[Cell]] = []
            for child in item.children:
                if child.token_type in ("bullet_list_open", "ordered_list_open"):
                    item_lines.extend(self._render_list(child, depth + 1, width, base_style))
                else:
                    item_lines.extend(self._render_block(child, item_width, base_style, None))
            rendered_any = False
            for item_line in item_lines:
                prefix = continuation_prefix if rendered_any else first_prefix
                lines.append(prefix + item_line)
                rendered_any = True
            if not rendered_any:
                lines.append(first_prefix)
            if getattr(node.token, "loose", False) and index < len(node.children) - 1:
                lines.append([])
        return lines

    def _list_marker(self, token: Any, ordered: bool, number: int, index: int) -> str:
        if self.options.preserve_ordered_list_markers:
            if ordered:
                info = token.info or ""
                markup = token.markup or "."
                if info:
                    return f"{info}{markup} "
            else:
                markup = token.markup or "-"
                if markup:
                    return f"{markup} "
        if ordered:
            return f"{number}. "
        return "- "

    def _render_blockquote(
        self,
        node: _Node,
        width: int,
        base_style: Style | None,
        next_type: str | None,
    ) -> list[list[Cell]]:
        quote_style = self.theme.quote(base_style)
        content_width = max(1, width - 2)
        inner = self._render_blocks(node.children, content_width, quote_style)
        while inner and not inner[-1]:
            inner.pop()
        border = self.theme.quote_border(base_style)
        result = [list(_cells_from_text("│ ", border)) + line for line in inner]
        if next_type and next_type != "space":
            result.append([])
        return result

    def _render_table(
        self,
        node: _Node,
        width: int,
        base_style: Style | None,
        next_type: str | None,
    ) -> list[list[Cell]]:
        header: list[list[Cell]] = []
        rows: list[list[list[Cell]]] = []

        def render_cell(cell_node: _Node) -> list[Cell]:
            for child in cell_node.children:
                if child.token_type == "inline":
                    rendered = self._render_inline_children(child.token.children or [], base_style)
                    return rendered[0] if rendered else []
            return []

        for section in node.children:
            for row_node in section.children:
                cells = [render_cell(cell_node) for cell_node in row_node.children]
                if section.token_type == "thead_open":
                    header = cells
                else:
                    rows.append(cells)
        num_cols = len(header)
        if num_cols == 0:
            return []
        border_overhead = 3 * num_cols + 1
        available_for_cells = width - border_overhead
        if available_for_cells < num_cols:
            start, end = node.token.map or (0, len(self.source.splitlines()))
            source_lines = self.source.splitlines()
            raw = "\n".join(source_lines[start:end]).strip() or self.source.strip()
            fallback_lines = [list(_cells_from_text(line, base_style)) for line in raw.split("\n")]
            if next_type and next_type != "space":
                fallback_lines.append([])
            return fallback_lines

        def visible(cells: list[Cell]) -> int:
            return sum(_visible_width(cell.char) for cell in cells)

        def longest_word(cells: list[Cell], max_width: int = 30) -> int:
            text = "".join(cell.char for cell in cells)
            longest = 0
            for word in re.split(r"\s+", text):
                longest = max(longest, _visible_width(word))
            return min(longest, max_width)

        natural_widths = [visible(cell) for cell in header]
        min_word_widths = [max(1, longest_word(cell)) for cell in header]
        for row in rows:
            for index in range(min(num_cols, len(row))):
                natural_widths[index] = max(natural_widths[index], visible(row[index]))
                min_word_widths[index] = max(min_word_widths[index], longest_word(row[index]))

        min_column_widths = list(min_word_widths)
        min_cells_width = sum(min_column_widths)
        if min_cells_width > available_for_cells:
            min_column_widths = [1] * num_cols
            remaining = available_for_cells - num_cols
            if remaining > 0:
                total_weight = sum(max(0, w - 1) for w in min_word_widths)
                if total_weight > 0:
                    growth = [(max(0, w - 1) * remaining) // total_weight for w in min_word_widths]
                    for index in range(num_cols):
                        min_column_widths[index] += growth[index]
                    leftover = remaining - sum(growth)
                    for index in range(num_cols):
                        if leftover <= 0:
                            break
                        min_column_widths[index] += 1
                        leftover -= 1
            min_cells_width = sum(min_column_widths)

        total_natural = sum(natural_widths) + border_overhead
        if total_natural <= width:
            column_widths = [
                max(natural_widths[index], min_column_widths[index]) for index in range(num_cols)
            ]
        else:
            extra = max(0, available_for_cells - min_cells_width)
            grow_potential = sum(
                max(0, natural_widths[index] - min_column_widths[index])
                for index in range(num_cols)
            )
            column_widths = list(min_column_widths)
            if grow_potential > 0:
                for index in range(num_cols):
                    delta = max(0, natural_widths[index] - min_column_widths[index])
                    column_widths[index] += (delta * extra) // grow_potential
                allocated = sum(column_widths)
                remaining = available_for_cells - allocated
                while remaining > 0:
                    grew = False
                    for index in range(num_cols):
                        if remaining <= 0:
                            break
                        if column_widths[index] < natural_widths[index]:
                            column_widths[index] += 1
                            remaining -= 1
                            grew = True
                    if not grew:
                        break

        def wrap_cells(cells: list[Cell], column_width: int) -> list[list[Cell]]:
            logical = self._wrap_logical([cells], column_width)
            return [line.cells for line in logical]

        def render_row(cells_list: list[list[Cell]], bold: bool) -> list[list[Cell]]:
            cell_lines = [
                wrap_cells(cells, column_widths[index]) for index, cells in enumerate(cells_list)
            ]
            line_count = max((len(c) for c in cell_lines), default=0)
            result: list[list[Cell]] = []
            for line_index in range(line_count):
                parts: list[list[Cell]] = []
                for col_index in range(num_cols):
                    cells = (
                        cell_lines[col_index][line_index]
                        if line_index < len(cell_lines[col_index])
                        else []
                    )
                    padded = list(cells) + list(
                        _cells_from_text(
                            " " * max(0, column_widths[col_index] - visible(cells)), base_style
                        )
                    )
                    if bold:
                        padded = [_apply_style(cell, Style(bold=True)) for cell in padded]
                    parts.append(padded)
                row_cells: list[Cell] = []
                row_cells.extend(_cells_from_text("│ ", base_style))
                for index, part in enumerate(parts):
                    if index > 0:
                        row_cells.extend(_cells_from_text(" │ ", base_style))
                    row_cells.extend(part)
                row_cells.extend(_cells_from_text(" │", base_style))
                result.append(row_cells)
            return result

        top_border = "┌─" + "─┬─".join("─" * w for w in column_widths) + "─┐"
        separator = "├─" + "─┼─".join("─" * w for w in column_widths) + "─┤"
        bottom_border = "└─" + "─┴─".join("─" * w for w in column_widths) + "─┘"
        result: list[list[Cell]] = [list(_cells_from_text(top_border, base_style))]
        result.extend(render_row(header, bold=True))
        result.append(list(_cells_from_text(separator, base_style)))
        for row_index, row in enumerate(rows):
            result.extend(render_row(row, bold=False))
            if row_index < len(rows) - 1:
                result.append(list(_cells_from_text(separator, base_style)))
        result.append(list(_cells_from_text(bottom_border, base_style)))
        if next_type and next_type != "space":
            result.append([])
        return result

    # ------------------------------------------------------------------
    # Inline 渲染
    # ------------------------------------------------------------------

    def _render_inline_children(
        self,
        children: list[Any],
        base_style: Style | None,
    ) -> list[list[Cell]]:
        lines: list[list[Cell]] = [[]]
        style_stack: list[Style] = []
        link_stack: list[str] = []

        def current_style() -> Style | None:
            style = base_style
            for extra in style_stack:
                style = (style or Style()) + extra
            return style

        for token in children:
            token_type = token.type
            if token_type in ("text", "text_special"):
                self._append_text_with_placeholders(
                    lines, token.content, current_style(), link_stack[-1] if link_stack else None
                )
            elif token_type == "code_inline":
                lines[-1].extend(
                    _cells_from_text(
                        token.content,
                        self.theme.code(current_style()),
                        link_stack[-1] if link_stack else None,
                    )
                )
            elif token_type in ("em_open", "em_close"):
                if token_type == "em_open":
                    style_stack.append(Style(italic=True))
                else:
                    style_stack.pop()
            elif token_type in ("strong_open", "strong_close"):
                if token_type == "strong_open":
                    style_stack.append(Style(bold=True))
                else:
                    style_stack.pop()
            elif token_type in ("s_open", "s_close"):
                if token_type == "s_open":
                    style_stack.append(Style(strike=True))
                else:
                    style_stack.pop()
            elif token_type == "link_open":
                attrs = dict(token.attrs or [])
                href = attrs.get("href", "")
                link_stack.append(href)
                style_stack.append(self.theme.link(current_style()))
            elif token_type == "link_close":
                if style_stack:
                    style_stack.pop()
                if link_stack:
                    link_stack.pop()
            elif token_type == "html_inline":
                content = token.content or ""
                if content in ("<del>", "</del>", "<s>", "</s>"):
                    if content.startswith("</"):
                        style_stack.pop()
                    else:
                        style_stack.append(Style(strike=True))
                elif content in ("<br>", "<br/>", "<br />"):
                    lines.append([])
                else:
                    self._append_text_with_placeholders(
                        lines, content, current_style(), link_stack[-1] if link_stack else None
                    )
            elif token_type in ("softbreak", "hardbreak", "br"):
                lines.append([])
            elif token_type == "escape":
                content = (
                    token.content
                    if not self.options.preserve_backslash_escapes
                    else (token.markup or "") + (token.content or "")
                )
                self._append_text_with_placeholders(
                    lines, content, current_style(), link_stack[-1] if link_stack else None
                )
            elif token_type == "image":
                self._append_text_with_placeholders(
                    lines,
                    token.content or "",
                    current_style(),
                    link_stack[-1] if link_stack else None,
                )
            elif token_type == "latex":
                self._append_text_with_placeholders(
                    lines,
                    token.content or "",
                    current_style(),
                    link_stack[-1] if link_stack else None,
                )
            else:
                content = getattr(token, "content", None)
                if isinstance(content, str):
                    self._append_text_with_placeholders(
                        lines, content, current_style(), link_stack[-1] if link_stack else None
                    )
        return lines

    def _append_text_with_placeholders(
        self,
        lines: list[list[Cell]],
        text: str,
        style: Style | None,
        link: str | None,
    ) -> None:
        position = 0
        for match in PLACEHOLDER_PATTERN.finditer(text):
            if match.start() > position:
                lines[-1].extend(_cells_from_text(text[position : match.start()], style, link))
            identifier = int(match.group(1) or match.group(2))
            rendered = self.latex_map.get(identifier)
            if rendered is not None:
                parts = rendered.split("\n")
                for index, part in enumerate(parts):
                    if index > 0:
                        lines.append([])
                    lines[-1].extend(_cells_from_text(part, style, link))
            position = match.end()
        if position < len(text):
            lines[-1].extend(_cells_from_text(text[position:], style, link))

    # ------------------------------------------------------------------
    # 换行 / 定宽
    # ------------------------------------------------------------------

    def _wrap_logical(self, logical: list[list[Cell]], width: int) -> list[Line]:
        wrapped: list[Line] = []
        for cells in logical:
            if not cells:
                wrapped.append(blank_line(width, self._default_style_style()))
                continue
            current: list[Cell] = []
            current_width = 0
            word: list[Cell] = []
            word_width = 0

            def flush_word(current: list[Cell], current_width: int) -> int:
                nonlocal word, word_width
                if not word:
                    return current_width
                space_width = 1 if current and current[-1].char != " " else 0
                if current and current_width + space_width + word_width > width:
                    wrapped.append(_pad_line(current, width))
                    current.clear()
                    current_width = 0
                elif current and current[-1].char != " ":
                    current.extend(_cells_from_text(" ", self._default_style_style()))
                    current_width += 1
                current.extend(word)
                current_width += word_width
                word = []
                word_width = 0
                return current_width

            for cell in cells:
                if cell.char == " ":
                    current_width = flush_word(current, current_width)
                    current.extend([Cell(" ", cell.style, cell.link)])
                    current_width += 1
                else:
                    word.append(cell)
                    word_width += _visible_width(cell.char)
                    if word_width > width:
                        oversized = word
                        word = []
                        word_width = 0
                        for char in oversized:
                            if current_width >= width:
                                wrapped.append(_pad_line(current, width))
                                current.clear()
                                current_width = 0
                            current.append(char)
                            current_width += _visible_width(char.char)
            current_width = flush_word(current, current_width)
            if current or not wrapped:
                wrapped.append(_pad_line(current, width))
        return wrapped

    def _wrap_and_pad(self, logical: list[list[Cell]], width: int) -> list[Line]:
        return self._wrap_logical(logical, width)


def _pad_line(cells: list[Cell], width: int) -> Line:
    style = cells[-1].style if cells else None
    padded = list(cells)
    padded.extend(
        Cell(" ", style) for _ in range(max(0, width - sum(_visible_width(c.char) for c in cells)))
    )
    return Line(padded[:width])


def _cells_from_text(text: str, style: Style | None, link: str | None = None) -> list[Cell]:
    return [Cell(char, style, link) for char in text]


def _apply_style(cell: Cell, style: Style) -> Cell:
    return Cell(cell.char, (cell.style or Style()) + style, cell.link)


def render_markdown_lines(
    text: str,
    width: int,
    theme: MarkdownTheme,
    default_style: DefaultTextStyle | None = None,
    options: MarkdownOptions | None = None,
) -> list[Line]:
    """用 TS 对齐渲染器把 Markdown 渲染为定宽 Line 列表。"""
    return _Renderer(theme, default_style, options).render(text, width)


__all__ = [
    "DefaultTextStyle",
    "MarkdownOptions",
    "MarkdownTheme",
    "ThemeMarkdownTheme",
    "render_markdown_lines",
]
