"""Questionnaire Tool - ask one or more questions with tab navigation.

Python port of questionnaire.ts（完整对齐）：单题走简单选项列表，
多题走 tab bar（含 Submit tab）；每题支持选项/自定义输入。
"""

from __future__ import annotations

from rich.style import Style

from pi_coding_agent import ExtensionAPI, ToolDefinition
from pi_tui.engine.cells import Cell, Line, blank_line, line_from_text
from pi_tui.engine.widgets import Input, Widget


def _normalize_questions(raw_questions: list) -> list[dict]:
    questions: list[dict] = []
    for index, raw in enumerate(raw_questions):
        if not isinstance(raw, dict):
            continue
        options: list[dict] = []
        for option in raw.get("options") or []:
            if isinstance(option, dict):
                description = option.get("description")
                options.append(
                    {
                        "value": str(option.get("value", option.get("label", ""))),
                        "label": str(option.get("label", option.get("value", ""))),
                        "description": str(description) if description else None,
                    }
                )
            else:
                value = str(option)
                options.append({"value": value, "label": value, "description": None})
        questions.append(
            {
                "id": str(raw.get("id", f"q{index + 1}")),
                "label": str(raw.get("label") or f"Q{index + 1}"),
                "prompt": str(raw.get("prompt", "")),
                "options": options,
                "allowOther": raw.get("allowOther", True) is not False,
            }
        )
    return questions


class QuestionnaireDialog(Widget):
    """多问题问卷对话框（单题自动退化为简单选项列表）。"""

    def __init__(self, questions: list[dict], done) -> None:
        super().__init__(focusable=True)
        self.questions = questions
        self.done = done
        self.is_multi = len(questions) > 1
        self.total_tabs = len(questions) + 1 if self.is_multi else 1
        self.current_tab = 0
        self.option_index = 0
        self.input_mode = False
        self.input_question_id: str | None = None
        self.input = Input(value="", placeholder="Your answer")
        self.answers: dict[str, dict] = {}

    def _current_question(self) -> dict | None:
        if not self.is_multi or self.current_tab < len(self.questions):
            return self.questions[self.current_tab]
        return None

    def _current_options(self) -> list[dict]:
        question = self._current_question()
        if question is None:
            return []
        options = [dict(option) for option in question["options"]]
        if question["allowOther"]:
            options.append({"value": "__other__", "label": "Type something.", "description": None})
        return options

    def _all_answered(self) -> bool:
        return all(question["id"] in self.answers for question in self.questions)

    def _advance(self) -> None:
        if not self.is_multi:
            self.done(self._result(False))
            return
        if self.current_tab < len(self.questions) - 1:
            self.current_tab += 1
        else:
            self.current_tab = len(self.questions)
        self.option_index = 0
        self.refresh()

    def _save_answer(
        self,
        question_id: str,
        value: str,
        label: str,
        was_custom: bool,
        index: int | None = None,
    ) -> None:
        answer: dict = {"id": question_id, "value": value, "label": label, "wasCustom": was_custom}
        if index is not None:
            answer["index"] = index
        self.answers[question_id] = answer

    def _result(self, cancelled: bool) -> dict:
        return {
            "questions": self.questions,
            "answers": list(self.answers.values()),
            "cancelled": cancelled,
        }

    def handle_key(self, key) -> bool:
        if self.input_mode:
            if key.name == "escape":
                self.input_mode = False
                self.input_question_id = None
                self.input.value = ""
                self.input.cursor = 0
                self.refresh()
                return True
            if key.name == "enter":
                question_id = self.input_question_id
                if question_id is not None:
                    value = self.input.value.strip() or "(no response)"
                    self._save_answer(question_id, value, value, True)
                    self.input_mode = False
                    self.input_question_id = None
                    self.input.value = ""
                    self.input.cursor = 0
                    self._advance()
                return True
            if self.input.handle_key(key):
                self.refresh()
                return True
            return True

        if self.is_multi:
            if key.name in ("tab", "right"):
                self.current_tab = (self.current_tab + 1) % self.total_tabs
                self.option_index = 0
                self.refresh()
                return True
            if key.name in ("shift+tab", "left"):
                self.current_tab = (self.current_tab - 1 + self.total_tabs) % self.total_tabs
                self.option_index = 0
                self.refresh()
                return True

        if self.is_multi and self.current_tab == len(self.questions):
            if key.name == "enter" and self._all_answered():
                self.done(self._result(False))
            elif key.name == "escape":
                self.done(self._result(True))
            return True

        options = self._current_options()
        if key.name == "up":
            self.option_index = max(0, self.option_index - 1)
            self.refresh()
            return True
        if key.name == "down":
            self.option_index = min(len(options) - 1, self.option_index + 1)
            self.refresh()
            return True
        if key.name == "enter":
            question = self._current_question()
            if question is None or not options:
                return True
            option = options[self.option_index]
            if option["value"] == "__other__":
                self.input_mode = True
                self.input_question_id = question["id"]
                self.input.value = ""
                self.input.cursor = 0
                self.refresh()
            else:
                self._save_answer(
                    question["id"],
                    option["value"],
                    option["label"],
                    False,
                    self.option_index + 1,
                )
                self._advance()
            return True
        if key.name == "escape":
            self.done(self._result(True))
            return True
        return False

    def render(self, width: int, height: int) -> list[Line]:
        lines: list[Line] = []
        if self.is_multi:
            tabs = [question["label"] for question in self.questions] + ["Submit"]
            tab_texts: list[str] = []
            for index, label in enumerate(tabs):
                marker = "> " if index == self.current_tab else "  "
                tab_texts.append(f"{marker}{label}")
            lines.append(line_from_text("  ".join(tab_texts), width, Style(dim=True)))
            lines.append(blank_line(width))
        question = self._current_question()
        if question is not None:
            lines.append(line_from_text(question["prompt"], width, Style(bold=True)))
            lines.append(blank_line(width))
            options = self._current_options()
            for index, option in enumerate(options):
                selected = index == self.option_index
                prefix = "> " if selected else "  "
                label = f"{prefix}{index + 1}. {option['label']}"
                style = Style(reverse=True) if selected else None
                lines.append(line_from_text(label, width, style))
                description = option.get("description")
                if description:
                    lines.append(line_from_text(f"     {description}", width, Style(dim=True)))
            if self.input_mode:
                lines.append(blank_line(width))
                lines.append(line_from_text("Your answer:", width, Style(dim=True)))
                input_line = self.input.render(max(0, width - 2), 1)[0]
                cells = [Cell(" "), Cell(" ")] + list(input_line.cells)
                cells = cells[:width]
                while len(cells) < width:
                    cells.append(Cell(" "))
                lines.append(Line(cells))
        else:
            answered = sum(1 for q in self.questions if q["id"] in self.answers)
            lines.append(
                line_from_text(
                    f"Submit ({answered}/{len(self.questions)} answered)",
                    width,
                    Style(bold=True),
                )
            )
        lines.append(blank_line(width))
        if self.input_mode:
            hint = "Enter to submit • Esc to go back"
        elif self.is_multi and self.current_tab == len(self.questions):
            hint = "Enter to submit • Esc to cancel"
        elif self.is_multi:
            hint = "Tab/←→ switch • ↑↓ navigate • Enter to select • Esc to cancel"
        else:
            hint = "↑↓ navigate • Enter to select • Esc to cancel"
        lines.append(line_from_text(hint, width, Style(dim=True)))
        while len(lines) < height:
            lines.append(blank_line(width))
        return lines[:height]

    def content_size(self) -> tuple[int, int]:
        return (max(1, len(self.questions[0]["prompt"])), len(self.questions) + 6)


def _render_call(args, theme, _context) -> str:
    raw_questions = args.get("questions") or []
    count = len(raw_questions)
    labels = ", ".join(str(q.get("label") or q.get("id")) for q in raw_questions)
    text = theme.fg("accent", "questionnaire ")
    text += theme.fg("textAlt", f"{count} question{'s' if count != 1 else ''}")
    if labels:
        text += theme.fg("dim", f" ({labels})")
    return text


def _render_result(result, _options, theme, _context) -> str:
    details = (result or {}).get("details") or {}
    if details.get("cancelled"):
        return theme.fg("warning", "Cancelled")
    lines: list[str] = []
    for answer in details.get("answers") or []:
        if answer.get("wasCustom"):
            lines.append(
                theme.fg("success", "✓ ")
                + theme.fg("accent", str(answer.get("id", "")))
                + ": "
                + theme.fg("textAlt", "(wrote) ")
                + str(answer.get("label", ""))
            )
        else:
            display = (
                f"{answer['index']}. {answer['label']}"
                if answer.get("index")
                else answer.get("label", "")
            )
            lines.append(
                theme.fg("success", "✓ ")
                + theme.fg("accent", str(answer.get("id", "")))
                + f": {display}"
            )
    return "\n".join(lines)


def create_extension(pi: ExtensionAPI):
    async def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        questions = _normalize_questions(params.get("questions") or [])
        if ctx is None or not ctx.has_ui:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Error: UI not available (running in non-interactive mode)",
                    }
                ],
                "details": {"questions": questions, "answers": [], "cancelled": True},
            }
        if not questions:
            return {
                "content": [{"type": "text", "text": "Error: No questions provided"}],
                "details": {"questions": [], "answers": [], "cancelled": True},
            }
        result = await ctx.ui.custom(
            lambda tui, theme, keybindings, done: QuestionnaireDialog(questions, done)
        )
        cancelled = bool(result.get("cancelled")) if isinstance(result, dict) else True
        return {
            "content": [
                {
                    "type": "text",
                    "text": "Cancelled" if cancelled else "Answers collected",
                }
            ],
            "details": result
            if isinstance(result, dict)
            else {
                "questions": questions,
                "answers": [],
                "cancelled": True,
            },
        }

    pi.register_tool(
        ToolDefinition(
            name="questionnaire",
            label="Questionnaire",
            description=(
                "Ask the user one or more questions. Use for clarifying requirements, "
                "getting preferences, or confirming decisions. For single questions, "
                "shows a simple option list. For multiple questions, shows a tab-based interface."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique identifier for this question",
                                },
                                "label": {
                                    "type": "string",
                                    "description": (
                                        "Short contextual label for tab bar, "
                                        "e.g. 'Scope', 'Priority' (defaults to Q1, Q2)"
                                    ),
                                },
                                "prompt": {
                                    "type": "string",
                                    "description": "The full question text to display",
                                },
                                "options": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "value": {
                                                "type": "string",
                                                "description": "The value returned when selected",
                                            },
                                            "label": {
                                                "type": "string",
                                                "description": "Display label for the option",
                                            },
                                            "description": {
                                                "type": "string",
                                                "description": (
                                                    "Optional description shown below label"
                                                ),
                                            },
                                        },
                                        "required": ["value", "label"],
                                    },
                                    "description": "Available options to choose from",
                                },
                                "allowOther": {
                                    "type": "boolean",
                                    "description": "Allow 'Type something' option (default: true)",
                                },
                            },
                            "required": ["id", "prompt", "options"],
                        },
                        "description": "Questions to ask the user",
                    }
                },
                "required": ["questions"],
            },
            render_call=_render_call,
            render_result=_render_result,
            execute=execute,
        )
    )
