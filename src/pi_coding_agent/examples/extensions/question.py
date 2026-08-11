"""Question Tool - single question with options.

Python port of question.ts（完整对齐）：编号选项 + 可选描述 +
"Type something." 内联输入；↑↓/Enter/Esc；编辑态 Esc 返回选项。
"""

from __future__ import annotations

from rich.style import Style

from pi_coding_agent import ExtensionAPI, ToolDefinition
from pi_tui.engine.cells import Cell, Line, blank_line, line_from_text
from pi_tui.engine.widgets import Input, Widget


class QuestionDialog(Widget):
    """选项列表 + 自定义输入对话框。"""

    def __init__(self, question: str, options: list[dict], done) -> None:
        super().__init__(focusable=True)
        self.question = question
        self.options = options
        self.all_options = [*options, {"label": "Type something.", "description": None}]
        self.option_index = 0
        self.edit_mode = False
        self.input = Input(value="", placeholder="Your answer")
        self.done = done

    def handle_key(self, key) -> bool:
        if self.edit_mode:
            if key.name == "escape":
                self.edit_mode = False
                self.input.value = ""
                self.input.cursor = 0
                self.refresh()
                return True
            if key.name == "enter":
                value = self.input.value.strip()
                if value:
                    self.done({"answer": value, "wasCustom": True})
                else:
                    self.edit_mode = False
                    self.input.value = ""
                    self.input.cursor = 0
                    self.refresh()
                return True
            if self.input.handle_key(key):
                self.refresh()
                return True
            return True
        if key.name == "up":
            self.option_index = max(0, self.option_index - 1)
            self.refresh()
            return True
        if key.name == "down":
            self.option_index = min(len(self.all_options) - 1, self.option_index + 1)
            self.refresh()
            return True
        if key.name == "enter":
            selected = self.all_options[self.option_index]
            if selected["label"] == "Type something.":
                self.edit_mode = True
                self.input.value = ""
                self.input.cursor = 0
                self.refresh()
            else:
                self.done(
                    {
                        "answer": selected["label"],
                        "wasCustom": False,
                        "index": self.option_index + 1,
                    }
                )
            return True
        if key.name == "escape":
            self.done(None)
            return True
        return False

    def render(self, width: int, height: int) -> list[Line]:
        lines: list[Line] = [line_from_text(self.question, width, Style(bold=True))]
        lines.append(blank_line(width))
        for index, option in enumerate(self.all_options):
            selected = index == self.option_index
            prefix = "> " if selected else "  "
            label = f"{prefix}{index + 1}. {option['label']}"
            style = Style(reverse=True) if selected else None
            lines.append(line_from_text(label, width, style))
            description = option.get("description")
            if description:
                lines.append(line_from_text(f"     {description}", width, Style(dim=True)))
        if self.edit_mode:
            lines.append(blank_line(width))
            lines.append(line_from_text("Your answer:", width, Style(dim=True)))
            input_line = self.input.render(max(0, width - 2), 1)[0]
            cells = [Cell(" "), Cell(" ")] + list(input_line.cells)
            cells = cells[:width]
            while len(cells) < width:
                cells.append(Cell(" "))
            lines.append(Line(cells))
        lines.append(blank_line(width))
        hint = (
            "Enter to submit • Esc to go back"
            if self.edit_mode
            else "↑↓ navigate • Enter to select • Esc to cancel"
        )
        lines.append(line_from_text(hint, width, Style(dim=True)))
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (max(1, len(self.question)), len(self.all_options) + 5)


def _render_call(args, theme, _context) -> str:
    question = str(args.get("question", ""))
    labels = [str(option.get("label", option)) for option in args.get("options") or []]
    numbered = [f"{index + 1}. {label}" for index, label in enumerate([*labels, "Type something."])]
    text = theme.fg("accent", "question ") + theme.fg("textAlt", question)
    if numbered:
        text += "\n" + theme.fg("dim", f"  Options: {', '.join(numbered)}")
    return text


def _render_result(result, _options, theme, _context) -> str:
    details = (result or {}).get("details") or {}
    answer = details.get("answer")
    if answer is None:
        return theme.fg("warning", "Cancelled")
    if details.get("wasCustom"):
        return (
            theme.fg("success", "✓ ")
            + theme.fg("textAlt", "(wrote) ")
            + theme.fg("accent", str(answer))
        )
    options = details.get("options") or []
    index = (options.index(answer) + 1) if answer in options else 0
    display = f"{index}. {answer}" if index > 0 else str(answer)
    return theme.fg("success", "✓ ") + theme.fg("accent", display)


def create_extension(pi: ExtensionAPI):
    async def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        question = str(params.get("question", ""))
        raw_options = params.get("options") or []
        options: list[dict] = []
        for option in raw_options:
            if isinstance(option, dict):
                description = option.get("description")
                options.append(
                    {
                        "label": str(option.get("label", option)),
                        "description": str(description) if description else None,
                    }
                )
            else:
                options.append({"label": str(option), "description": None})
        simple_options = [option["label"] for option in options]
        details = {"question": question, "options": simple_options, "answer": None}
        if ctx is None or not ctx.has_ui:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: UI not available (running in non-interactive mode)",
                    }
                ],
                "details": details,
            }
        if not options:
            return {
                "content": [{"type": "text", "text": "Error: No options provided"}],
                "details": details,
            }
        result = await ctx.ui.custom(
            lambda tui, theme, keybindings, done: QuestionDialog(question, options, done)
        )
        if result is None:
            return {
                "content": [{"type": "text", "text": "User cancelled the selection"}],
                "details": details,
            }
        if result.get("wasCustom"):
            return {
                "content": [{"type": "text", "text": f"User wrote: {result['answer']}"}],
                "details": {
                    "question": question,
                    "options": simple_options,
                    "answer": result["answer"],
                    "wasCustom": True,
                },
            }
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"User selected: {result['index']}. {result['answer']}",
                }
            ],
            "details": {
                "question": question,
                "options": simple_options,
                "answer": result["answer"],
                "wasCustom": False,
            },
        }

    pi.register_tool(
        ToolDefinition(
            name="question",
            label="Question",
            description=(
                "Ask the user a question and let them pick from options. "
                "Use when you need user input to proceed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user",
                    },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "Display label for the option",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Optional description shown below label",
                                },
                            },
                            "required": ["label"],
                        },
                        "description": "Options for the user to choose from",
                    },
                },
                "required": ["question", "options"],
            },
            execution_mode="sequential",
            render_call=_render_call,
            render_result=_render_result,
            execute=execute,
        )
    )
