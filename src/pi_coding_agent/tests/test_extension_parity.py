"""question / questionnaire / plan-mode 示例与 TS 对齐测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pi_coding_agent.extensions.loader import ExtensionLoader

EXAMPLES_EXTENSIONS = Path(__file__).resolve().parents[1] / "examples" / "extensions"
sys.path.insert(0, str(EXAMPLES_EXTENSIONS))

from pi_tui.engine.keys import Key  # noqa: E402
from question import QuestionDialog  # noqa: E402
from questionnaire import QuestionnaireDialog  # noqa: E402


def _key(name: str, char: str | None = None) -> Key:
    return Key(name=name, char=char)


class _FakeTheme:
    def fg(self, _name: str, text: str = "") -> str:
        return text

    def strikethrough(self, text: str = "") -> str:
        return text


class _FakeUI:
    def __init__(self, custom_result=None) -> None:
        self.custom_result = custom_result
        self.status: dict[str, str | None] = {}
        self.widgets: dict[str, list[str] | None] = {}

    @property
    def theme(self) -> _FakeTheme:
        return _FakeTheme()

    async def custom(self, factory, **kwargs):
        return self.custom_result

    def set_status(self, key: str, text: str | None) -> None:
        self.status[key] = text

    def set_widget(self, key: str, lines=None, options=None) -> None:
        self.widgets[key] = lines

    def notify(self, message: str, notify_type: str | None = None) -> None:
        pass


class _FakeCtx:
    def __init__(self, custom_result=None, has_ui: bool = True) -> None:
        self.ui = _FakeUI(custom_result)
        self.has_ui = has_ui
        self.session_manager = None


async def _load_examples(tmp_path) -> tuple[dict[str, object], object]:
    loader = ExtensionLoader(global_dir=tmp_path / "none")
    result = await loader.load(
        explicit_paths=[
            str(EXAMPLES_EXTENSIONS / "question.py"),
            str(EXAMPLES_EXTENSIONS / "questionnaire.py"),
            str(EXAMPLES_EXTENSIONS / "plan_mode.py"),
        ]
    )
    assert not result.errors, [error.error for error in result.errors]
    # Windows 上 extension.path 为反斜杠路径,用 Path.name 取基名。
    extensions = {Path(extension.path).name: extension for extension in result.extensions}
    return extensions, result.runtime


@pytest.mark.asyncio
async def test_question_definition_matches_ts(tmp_path):
    extensions, _runtime = await _load_examples(tmp_path)
    question = extensions["question.py"]
    tool = question.tools["question"]
    assert "Use when you need user input to proceed." in tool.description
    assert tool.execution_mode == "sequential"
    assert tool.render_call is not None
    assert tool.render_result is not None
    option_schema = tool.parameters["properties"]["options"]["items"]["properties"]
    assert "description" in option_schema
    assert option_schema["label"]["type"] == "string"


@pytest.mark.asyncio
async def test_question_execute_returns_ts_text(tmp_path):
    extensions, _runtime = await _load_examples(tmp_path)
    tool = extensions["question.py"].tools["question"]
    params = {"question": "Pick?", "options": [{"label": "A"}]}

    no_ui = await tool.execute("1", params, None, None, _FakeCtx(has_ui=False))
    assert "UI not available (running in non-interactive mode)" in no_ui["content"][0]["text"]

    cancelled = await tool.execute("2", params, None, None, _FakeCtx(custom_result=None))
    assert cancelled["content"][0]["text"] == "User cancelled the selection"
    assert cancelled["details"]["answer"] is None

    custom = await tool.execute(
        "3",
        params,
        None,
        None,
        _FakeCtx(custom_result={"answer": "x", "wasCustom": True}),
    )
    assert custom["content"][0]["text"] == "User wrote: x"
    assert custom["details"]["wasCustom"] is True

    selected = await tool.execute(
        "4",
        params,
        None,
        None,
        _FakeCtx(custom_result={"answer": "A", "wasCustom": False, "index": 1}),
    )
    assert selected["content"][0]["text"] == "User selected: 1. A"
    assert selected["details"]["wasCustom"] is False


@pytest.mark.asyncio
async def test_questionnaire_definition_and_errors(tmp_path):
    extensions, _runtime = await _load_examples(tmp_path)
    tool = extensions["questionnaire.py"].tools["questionnaire"]
    assert "shows a tab-based interface" in tool.description
    question_schema = tool.parameters["properties"]["questions"]["items"]["properties"]
    assert set(question_schema) >= {"id", "label", "prompt", "options", "allowOther"}
    option_schema = question_schema["options"]["items"]["properties"]
    assert set(option_schema) >= {"value", "label", "description"}

    no_ui = await tool.execute(
        "1",
        {"questions": [{"id": "q1", "prompt": "?", "options": []}]},
        None,
        None,
        _FakeCtx(has_ui=False),
    )
    assert "UI not available (running in non-interactive mode)" in no_ui["content"][0]["text"]

    empty = await tool.execute("2", {"questions": []}, None, None, _FakeCtx())
    assert empty["content"][0]["text"] == "Error: No questions provided"

    cancelled = await tool.execute(
        "3",
        {"questions": [{"id": "q1", "prompt": "?", "options": []}]},
        None,
        None,
        _FakeCtx(custom_result={"questions": [], "answers": [], "cancelled": True}),
    )
    assert cancelled["content"][0]["text"] == "Cancelled"


@pytest.mark.asyncio
async def test_plan_mode_utils():
    from pi_coding_agent.examples.extensions.plan_mode_utils import (
        extract_todo_items,
        is_safe_command,
        mark_completed_steps,
    )

    assert is_safe_command("cat README.md") is True
    assert is_safe_command("git status") is True
    assert is_safe_command("rm -rf /") is False
    assert is_safe_command("git commit -m x") is False
    assert is_safe_command("pip install x") is False

    items = extract_todo_items("Plan:\n1. First step\n2. Second step\n")
    assert [item["text"] for item in items] == ["First step", "Second step"]
    assert mark_completed_steps("Done [DONE:1]", items) == 1
    assert items[0]["completed"] is True
    assert items[1]["completed"] is False


@pytest.mark.asyncio
async def test_plan_mode_tool_call_block_and_context_filter(tmp_path):
    extensions, runtime = await _load_examples(tmp_path)
    plan = extensions["plan_mode.py"]
    ctx = _FakeCtx()

    tool_call = plan.handlers["tool_call"][0]
    context = plan.handlers["context"][0]
    before_agent = plan.handlers["before_agent_start"][0]

    # 默认未启用：context 过滤 plan-mode 消息。
    filtered = context(
        {
            "messages": [
                {"role": "user", "customType": "plan-mode-context", "content": "x"},
                {"role": "user", "content": "keep"},
                {"role": "user", "content": "[PLAN MODE ACTIVE]"},
            ]
        },
        ctx,
    )
    assert [m["content"] for m in filtered["messages"]] == ["keep"]

    # 启用 plan 模式后：bash 白名单拦截 + before_agent_start 注入。
    # 通过 --plan flag 值驱动 session_start（无需真实 session 动作）。
    runtime.set_action("get_active_tools", lambda: [])
    runtime.set_action("set_active_tools", lambda names: None)
    runtime.set_action("append_entry", lambda *args, **kwargs: None)
    runtime.flag_values["plan"] = True
    plan.handlers["session_start"][0]({}, ctx)
    blocked = tool_call(
        {"type": "tool_call", "toolName": "bash", "input": {"command": "rm -rf /"}},
        ctx,
    )
    assert blocked and blocked.get("block") is True
    allowed = tool_call(
        {"type": "tool_call", "toolName": "bash", "input": {"command": "cat README.md"}},
        ctx,
    )
    assert allowed is None
    injected = before_agent({}, ctx)
    assert injected["message"]["customType"] == "plan-mode-context"
    assert "[PLAN MODE ACTIVE]" in injected["message"]["content"]


def test_question_dialog_options_and_cancel() -> None:
    results: list = []
    dialog = QuestionDialog(
        "Pick?",
        [{"label": "A", "description": None}, {"label": "B", "description": "second"}],
        results.append,
    )
    dialog.handle_key(_key("down"))
    dialog.handle_key(_key("enter"))
    assert results == [{"answer": "B", "wasCustom": False, "index": 2}]

    dialog2 = QuestionDialog("Pick?", [{"label": "A", "description": None}], results.append)
    dialog2.handle_key(_key("escape"))
    assert results[-1] is None


def test_question_dialog_custom_input_esc_back() -> None:
    results: list = []
    dialog = QuestionDialog("Pick?", [{"label": "A", "description": None}], results.append)
    dialog.handle_key(_key("down"))
    dialog.handle_key(_key("enter"))
    assert dialog.edit_mode is True
    dialog.handle_key(_key("escape"))
    assert dialog.edit_mode is False
    dialog.handle_key(_key("enter"))  # Type something again
    dialog.handle_key(_key("h", "h"))
    dialog.handle_key(_key("i", "i"))
    dialog.handle_key(_key("enter"))
    assert results == [{"answer": "hi", "wasCustom": True}]


def test_questionnaire_dialog_tab_flow() -> None:
    questions = [
        {
            "id": "q1",
            "label": "Q1",
            "prompt": "P1",
            "options": [{"value": "a", "label": "A", "description": None}],
            "allowOther": False,
        },
        {
            "id": "q2",
            "label": "Q2",
            "prompt": "P2",
            "options": [{"value": "b", "label": "B", "description": None}],
            "allowOther": False,
        },
    ]
    results: list = []
    dialog = QuestionnaireDialog(questions, results.append)
    dialog.handle_key(_key("enter"))
    assert dialog.current_tab == 1
    dialog.handle_key(_key("enter"))
    assert dialog.current_tab == 2
    dialog.handle_key(_key("enter"))
    assert results and results[0]["cancelled"] is False
    answers = {answer["id"]: answer for answer in results[0]["answers"]}
    assert answers["q1"]["value"] == "a"
    assert answers["q2"]["value"] == "b"


def test_questionnaire_single_question_submits_after_select() -> None:
    questions = [
        {
            "id": "q1",
            "label": "Q1",
            "prompt": "P1",
            "options": [{"value": "a", "label": "A", "description": None}],
            "allowOther": False,
        }
    ]
    results: list = []
    dialog = QuestionnaireDialog(questions, results.append)
    dialog.handle_key(_key("enter"))
    assert results and results[0]["cancelled"] is False
    assert results[0]["answers"][0]["value"] == "a"
