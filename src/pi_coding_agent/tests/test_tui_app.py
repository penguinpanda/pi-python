"""引擎版 PiTuiApp 集成测试（FakeTerminal 驱动，无真实 TTY）。"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from pi_coding_agent.modes.interactive.app import PiTuiApp
from pi_tui.components import MessageEntry, ToolExecutionEntry
from pi_tui.engine import FakeTerminal
from pi_tui.selectors import ChoiceSelector


def _make_session() -> MagicMock:
    session = MagicMock()
    session.cwd = "C:\\tmp"
    session.get_messages.return_value = []
    session.subscribe.return_value = lambda: None
    session.model = None
    session.thinking_level = "off"
    session.session_name = "test"
    session.session_id = "sess-1"
    session.is_bash_running = False
    session.auto_compaction_enabled = False
    session.scoped_models = []
    session.extension_runner = None
    session.skill_loader = None
    session.template_loader = None
    session.session_manager = MagicMock()
    session.session_manager.get_tree.return_value = []
    session.session_manager.get_leaf_id.return_value = None
    session.session_manager.get_entries.return_value = []
    return session


def _make_app(
    term: FakeTerminal,
    session: MagicMock | None = None,
    *,
    ui_mode: str = "regular",
) -> PiTuiApp:
    return PiTuiApp(
        session or _make_session(),
        MagicMock(),
        terminal=term,
        theme_name="dark",
        no_context_files=True,
        ui_mode=ui_mode,
    )


async def _run(app: PiTuiApp, term: FakeTerminal, actions=None) -> None:
    """启动应用，执行 actions(term, app)，然后退出。"""
    task = asyncio.create_task(app.run_async())
    await asyncio.sleep(0.1)
    if actions is not None:
        await actions(term, app)
    app.exit()
    await asyncio.sleep(0.1)
    await task


@pytest.mark.asyncio
async def test_app_mounts_and_renders() -> None:
    term = FakeTerminal(size=(100, 30))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.15)
        output = term.output_text
        assert "pi" in output
        assert "model:" in output
        assert _app._header is not None
        assert _app._editor is not None
        assert _app.focused is _app._editor

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_typing_and_submit_calls_prompt() -> None:
    term = FakeTerminal(size=(100, 30))
    session = _make_session()
    app = _make_app(term, session)

    async def actions(_term, _app) -> None:
        term.feed_text("hello engine")
        await asyncio.sleep(0.1)
        term.feed(b"\r")
        await asyncio.sleep(0.2)
        session.prompt.assert_called_once_with("hello engine")

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_choice_selector_flow() -> None:
    term = FakeTerminal(size=(100, 30))
    app = _make_app(term)
    results: list[str | None] = []

    async def actions(_term, _app) -> None:
        app.push_screen(
            ChoiceSelector("Pick", ["alpha", "beta"]),
            callback=results.append,
        )
        await asyncio.sleep(0.1)
        term.feed_text("b")
        await asyncio.sleep(0.1)
        term.feed(b"\r")
        await asyncio.sleep(0.2)
        assert results == ["beta"]
        assert len(app._overlays) == 0

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_streaming_message_events() -> None:
    term = FakeTerminal(size=(100, 30))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        app._on_session_event({"type": "message_start"})
        await asyncio.sleep(0.05)
        app._on_session_event(
            {"type": "message_update", "message": {"content": [{"type": "text", "text": "part"}]}}
        )
        await asyncio.sleep(0.05)
        app._on_session_event(
            {
                "type": "message_end",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "final"}]},
            }
        )
        await asyncio.sleep(0.1)
        entries = app._chat.query(MessageEntry)
        assert any("final" in entry.entry_text for entry in entries)

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_layout_matches_ts_chat_above_editor() -> None:
    """regular 模式（对齐 TS TuiMainScreen）：聊天内容在编辑器上方，dock 跟随文档末尾。"""
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.15)
        assert app._chat.rect[0] == 1
        assert app._editor.rect[0] == app._status.rect[0] + 1
        assert app._editor.rect[2:] == (100, 6)
        assert app._footer.rect == (app._editor.rect[0] + 6, 0, 100, 1)

        # 一轮完整 user/assistant 消息后，文档变长，dock 随内容下移（非粘性）。
        app._on_session_event(
            {"type": "message_start", "message": {"role": "user", "content": "hi"}}
        )
        app._on_session_event({"type": "message_end", "message": {"role": "user", "content": "hi"}})
        app._on_session_event(
            {"type": "message_start", "message": {"role": "assistant", "content": []}}
        )
        app._on_session_event(
            {
                "type": "message_update",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "reply"}],
                },
            }
        )
        app._on_session_event(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "reply"}],
                },
            }
        )
        await asyncio.sleep(0.1)
        assert app._editor.rect[0] > 2
        assert app._footer.rect[0] == app._editor.rect[0] + 6
        entries = app._chat.query(MessageEntry)
        assert [entry.label for entry in entries] == ["User", "Assistant"]
        assert not any(entry.speaking for entry in entries)

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_message_start_user_does_not_create_stream_placeholder() -> None:
    term = FakeTerminal(size=(100, 30))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.15)
        app._on_session_event(
            {"type": "message_start", "message": {"role": "user", "content": "hi"}}
        )
        await asyncio.sleep(0.05)
        assert app._stream_entry is None
        assert app._chat.query(MessageEntry) == []

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_exit_writes_main_screen_document() -> None:
    """退出时对齐 TS：把最后一帧文档写入主屏并落在新行（fullscreen 退出路径）。"""
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term, ui_mode="fullscreen")
    task = asyncio.create_task(app.run_async())
    await asyncio.sleep(0.15)
    term.reset_output()
    app.exit()
    await asyncio.sleep(0.2)
    await task
    output = term.output_text
    assert "\r\x1b[2K" in output
    assert output.endswith("\x1b[0m\r\n\x1b[?25h")
    assert "thinking:" in output


@pytest.mark.asyncio
async def test_slash_notify_adds_system_message() -> None:
    term = FakeTerminal(size=(100, 30))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        app._slash_notify("hello system")
        await asyncio.sleep(0.1)
        entries = app._chat.query(MessageEntry)
        assert any("hello system" in entry.entry_text for entry in entries)
        assert app._status is not None and "hello system" in app._status.content

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_slash_completion_renders_below_editor() -> None:
    term = FakeTerminal(size=(100, 30))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        app._completion_items = [
            {"value": "/help ", "label": "Show help"},
            {"value": "/model ", "label": "Switch model"},
        ]
        app._completion_index = 1
        app._render_slash_completion()
        await asyncio.sleep(0.1)
        # 对齐 TS：补全渲染在编辑器内部（底部边框下方），不再使用上方 overlay。
        assert app._overlay_manager.get("slash-completion") is None
        editor = app._editor
        assert editor.completion_active is True
        assert [value for value, _label in editor.completion_items] == ["/help", "/model"]
        assert editor.rect[3] > 6
        rendered = "\n".join(line.text() for line in editor.render(60, editor.rect[3]))
        assert "/model" in rendered
        assert "→" in rendered
        app._hide_slash_completion()
        await asyncio.sleep(0.1)
        assert app._editor.completion_active is False
        assert app._editor.rect[3] == 6

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_exec_bash_calls_session() -> None:
    term = FakeTerminal(size=(100, 30))
    session = _make_session()
    result = MagicMock()
    result.exit_code = 0
    result.cancelled = False
    result.truncated = False
    result.full_output_path = None
    session.execute_bash.return_value = result
    app = _make_app(term, session)

    async def actions(_term, _app) -> None:
        term.feed_text("!echo hi")
        await asyncio.sleep(0.1)
        term.feed(b"\r")
        await asyncio.sleep(0.2)
        session.execute_bash.assert_called_once()
        args = session.execute_bash.call_args
        assert args.kwargs["exclude_from_context"] is False

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_toggle_tools_rerenders_chat() -> None:
    term = FakeTerminal(size=(100, 30))
    session = _make_session()
    session.get_messages.return_value = [{"role": "user", "content": "hello"}]
    app = _make_app(term, session)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.15)
        assert app._show_tools is True
        app.action_toggle_tools()
        await asyncio.sleep(0.1)
        assert app._show_tools is False
        entries = app._chat.query(MessageEntry)
        assert any("hello" in entry.entry_text for entry in entries)

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_theme_switch() -> None:
    term = FakeTerminal(size=(100, 30))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        app._set_theme("light")
        assert app._theme.name == "light"
        await asyncio.sleep(0.1)

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_ctrl_d_exits_when_editor_empty() -> None:
    term = FakeTerminal(size=(100, 30))
    app = _make_app(term)
    task = asyncio.create_task(app.run_async())
    await asyncio.sleep(0.15)
    term.feed(b"\x04")  # ctrl+d
    await asyncio.sleep(0.3)
    assert app._running is False
    await task


@pytest.mark.asyncio
async def test_queue_update_renders_pending_messages() -> None:
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.1)
        app._on_session_event(
            {
                "type": "queue_update",
                "steer": [],
                "follow_up": [{"content": "hello queue"}],
                "next_turn": [],
            }
        )
        await asyncio.sleep(0.05)
        assert "hello queue" in app._pending_messages.content

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_follow_up_queues_pending_message() -> None:
    term = FakeTerminal(size=(100, 24))
    session = _make_session()
    app = _make_app(term, session)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.1)
        app._editor.text = "queued follow-up"
        app.action_follow_up()
        assert "queued follow-up" in app._pending_messages.content
        session.follow_up.assert_called_once_with("queued follow-up")

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_agent_start_sets_terminal_progress() -> None:
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.1)
        assert term.progress is False
        app._show_terminal_progress = True
        app._on_session_event({"type": "agent_start"})
        assert term.progress is True
        app._on_session_event({"type": "agent_settled"})
        assert term.progress is False

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_startup_resources_in_separate_container() -> None:
    term = FakeTerminal(size=(100, 24))
    app = PiTuiApp(
        _make_session(),
        MagicMock(),
        terminal=term,
        theme_name="dark",
        no_context_files=True,
        startup_resources={"skills": [{"name": "demo-skill"}]},
    )

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.15)
        assert "demo-skill" in app._resources.content
        assert not any("demo-skill" in entry.entry_text for entry in app._chat.query(MessageEntry))

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_toggle_tools_expands_header() -> None:
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.1)
        assert app._header._expanded is False
        app.action_toggle_tools()
        assert app._header._expanded is True
        assert app._show_tools is False

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_editor_border_renders_in_app() -> None:
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.1)
        assert app._editor.border is True
        # 正文为 muted textAlt（#a6adc8），光标保持反色块。
        from rich.color import Color

        assert app._editor.base_style is not None
        assert app._editor.base_style.color == Color.from_rgb(166, 173, 200)
        # 对齐 TS：输入框不涂底色。
        assert app._editor.base_style.bgcolor is None
        assert app._editor.border_style is not None
        assert app._editor.border_style.bgcolor is None
        lines = app._editor.render(20, 6)
        assert lines[0].text().strip() == "─" * 20
        assert lines[-1].text().strip() == "─" * 20

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_toolbar_widgets_have_no_background() -> None:
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.1)
        for widget in (app._header, app._status, app._editor, app._footer, app.screen):
            assert widget.base_style is not None
            assert widget.base_style.bgcolor is None, type(widget).__name__
        # header 内容与渲染帧都包含快捷键提示。
        assert "cycle model" in app._header.content
        frame_text = "\n".join(line.text() for line in app._last_frame_lines or [])
        assert "cycle model" in frame_text
        assert "Idle" in frame_text

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_streaming_message_wraps_and_grows_document() -> None:
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.1)
        app._on_session_event(
            {"type": "message_start", "message": {"role": "assistant", "content": []}}
        )
        await asyncio.sleep(0.05)
        text = "你好！有什么可以帮你的吗？我可以帮你读写文件、执行命令或搜索代码。还可以帮你分析问题、生成测试、修复 bug。 你只需要告诉我目标。"
        app._on_session_event(
            {
                "type": "message_update",
                "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            }
        )
        await asyncio.sleep(0.1)
        entries = app._chat.query(MessageEntry)
        assert entries and "Assistant" in entries[0].label
        doc = "\n".join(line.text() for line in app._last_frame_lines or [])
        assert "你好！" in doc
        assert "你只需要告诉我目标" in doc  # 长消息不截断
        assert doc.count("🤖 Assistant") == 1  # 标签只渲染一次，不叠行
        assert app._chat.rect[3] >= 3

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_tool_call_creates_expandable_entry() -> None:
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.1)
        app._on_session_event(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "done"},
                        {
                            "type": "toolCall",
                            "id": "call-1",
                            "name": "read",
                            "arguments": {"path": "a.txt"},
                            "result": {"content": "file body", "isError": False},
                        },
                    ],
                },
            }
        )
        await asyncio.sleep(0.05)
        entries = app._chat.query(ToolExecutionEntry)
        assert len(entries) == 1
        assert entries[0].tool_name == "read"
        assert entries[0].status == "success"
        assert not any("read(" in entry.entry_text for entry in app._chat.query(MessageEntry))
        entries[0].set_expanded(True)
        assert entries[0].content_size()[1] > 1

    await _run(app, term, actions)


@pytest.mark.asyncio
async def test_scroll_to_prompt_navigates_user_messages() -> None:
    term = FakeTerminal(size=(100, 24))
    app = _make_app(term)

    async def actions(_term, _app) -> None:
        await asyncio.sleep(0.1)
        app._chat.add_message_agent({"role": "user", "content": "first"})
        app._chat.add_message_agent(
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]}
        )
        app._chat.add_message_agent({"role": "user", "content": "second"})
        app._chat.scroll_offset = 0
        app.action_next_prompt()
        assert app._chat.scroll_offset > 0
        current = app._chat.scroll_offset
        app.action_previous_prompt()
        assert app._chat.scroll_offset < current

    await _run(app, term, actions)


def test_user_message_has_osc133_marker() -> None:
    from pi_tui.components import MessageEntry

    entry = MessageEntry("User", "hello")
    entry.prompt_marker = True
    lines = entry.render(40, 3)
    assert "\x1b]133;A\x07" in lines[0].passthrough
    plain = MessageEntry("Assistant", "hello")
    plain_lines = plain.render(40, 3)
    assert "\x1b]133;A\x07" not in plain_lines[0].passthrough
