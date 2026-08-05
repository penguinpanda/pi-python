"""TUI 主应用测试（Textual 无头 run_test）。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widget import Widget
from textual.widgets import Label, Static
from pi_agent import Agent, AgentOptions
from pi_ai import Model, Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime
from pi_coding_agent.modes.interactive.app import PiTuiApp
from pi_tui.components import BashExecutionEntry, MessageEntry, PiEditor
from pi_tui.selectors import (
    ChoiceSelector,
    ExtensionSelector,
    ModelSelector,
    OAuthSelector,
    ScopedModelsSelector,
    SessionPicker,
    SettingsSelector,
    ThinkingSelector,
    TreeSelector,
)


class _FocusablePanel(Widget):
    """组件树 overlay 测试用：可聚焦的简单面板。"""

    can_focus = True

    def render(self):
        return "panel"


def _make_runtime(
    model_count: int = 2,
    responses: list | None = None,
) -> ModelRuntime:
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    models_list = [
        Model(
            id=f"faux-{index}",
            provider="faux",
            api="openai-completions",
            name=f"Faux {index}",
        )
        for index in range(1, model_count + 1)
    ]
    core = faux_provider(models=models_list)
    if responses:
        core.set_responses(responses)
    models.add_provider(core.provider)
    runtime = ModelRuntime(models, store)
    return runtime


def _make_session(runtime: ModelRuntime, tmp_path: Path) -> AgentSession:
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=runtime.stream,
        )
    )
    return AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=model,
        model_runtime=runtime,
    )


def _make_session_with_manager(runtime: ModelRuntime, manager, tmp_path: Path) -> AgentSession:
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=runtime.stream,
        )
    )
    return AgentSession(
        agent=agent,
        session_manager=manager,
        cwd=str(tmp_path),
        model=model,
        model_runtime=runtime,
    )


async def _wait_until(
    condition,
    timeout: float = 5.0,
    pilot=None,
    message: str = "Condition",
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError(f"{message} not met within timeout")
        if pilot is not None:
            await pilot.pause()
        else:
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_app_mounts_and_footer(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test():
        assert app._editor.has_focus
        footer_text = app._footer.content
        assert "faux/faux-1" in str(footer_text)
        assert "messages: 0" in str(footer_text)


@pytest.mark.asyncio
async def test_set_editor_component_replaces_editor(tmp_path):
    from pi_tui import PiEditor

    class CustomEditor(PiEditor):
        pass

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        custom = CustomEditor()
        app._replace_editor(custom)
        await pilot.pause()
        assert app._editor is custom
        assert app._editor.has_focus
        await pilot.press("h", "i")
        await pilot.pause()
        assert app._editor.text == "hi"


@pytest.mark.asyncio
async def test_set_widget_above_and_below_editor(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._set_widget("w1", ["line1", "line2"])
        await pilot.pause()
        above = app.query_one("#pi-widgets-above", Static)
        assert "line1" in str(above.content)
        assert "line2" in str(above.content)

        app._set_widget("w2", ["below line"], {"placement": "belowEditor"})
        await pilot.pause()
        below = app.query_one("#pi-widgets-below", Static)
        assert "below line" in str(below.content)
        assert "line1" not in str(below.content)

        # 清空（空列表）后移除该 key。
        app._set_widget("w1", [])
        await pilot.pause()
        assert "line1" not in str(app.query_one("#pi-widgets-above", Static).content)


@pytest.mark.asyncio
async def test_set_overlay_anchor_and_clear(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._set_overlay("ov1", ["overlay text"], {"anchor": "top-right", "margin": 2})
        await pilot.pause()
        widget = app.query_one("#pi-overlay-ov1", Static)
        assert "overlay text" in str(widget.content)
        assert widget.styles.layer == "overlay"
        assert widget.styles.position == "absolute"
        assert widget.styles.offset is not None

        app._set_overlay(
            "ov1",
            ["bordered overlay"],
            {"anchor": "center", "border": "round", "border_color": "blue", "title": "demo"},
        )
        await pilot.pause()
        assert widget.styles.border is not None
        assert widget.border_title == "demo"

        app._set_overlay("ov1", [])
        await pilot.pause()
        assert app.query("#pi-overlay-ov1").nodes == []


@pytest.mark.asyncio
async def test_set_overlay_animation(tmp_path):
    import time

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._set_overlay(
            "ov2",
            ["animated overlay"],
            {"anchor": "center", "animate": True, "duration": 0.2},
        )
        await pilot.pause()
        widget = app.query_one("#pi-overlay-ov2", Static)
        deadline = time.monotonic() + 3
        last = widget.styles.offset
        while time.monotonic() < deadline:
            await pilot.pause()
            current = widget.styles.offset
            if current == last:
                break
            last = current
        assert widget.styles.offset is not None


@pytest.mark.asyncio
async def test_set_overlay_focus_and_unfocus_restores_editor(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        assert app.screen.focused is app._editor
        handle = app._set_overlay("ov1", ["capture"], {"anchor": "top-left"})
        await pilot.pause()
        assert handle is not None
        assert handle.is_focused()
        assert app.screen.focused is not app._editor
        handle.unfocus()
        await pilot.pause()
        assert app.screen.focused is app._editor
        assert handle.is_focused() is False


@pytest.mark.asyncio
async def test_set_overlay_non_capturing_keeps_editor_focus(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._set_overlay(
            "toast",
            ["notification"],
            {"anchor": "top-right", "nonCapturing": True},
        )
        await pilot.pause()
        assert app.screen.focused is app._editor
        assert app.query("#pi-overlay-toast").nodes


@pytest.mark.asyncio
async def test_set_overlay_handle_hide_show(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        handle = app._set_overlay("ov1", ["x"], {"anchor": "top-left"})
        await pilot.pause()
        widget = app.query_one("#pi-overlay-ov1", Static)
        assert widget.display is True
        assert handle is not None
        handle.set_hidden(True)
        await pilot.pause()
        assert handle.is_hidden()
        assert widget.display is False
        handle.set_hidden(False)
        await pilot.pause()
        assert widget.display is True
        assert not handle.is_hidden()
        handle.hide()
        await pilot.pause()
        assert app.query("#pi-overlay-ov1").nodes == []


@pytest.mark.asyncio
async def test_set_overlay_component_mounts_and_focuses(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        panel = _FocusablePanel()
        handle = app._set_overlay_component("c1", panel, {"anchor": "top-left"})
        await pilot.pause()
        await pilot.pause()
        assert handle is not None
        assert handle.is_focused()
        assert app.query("#pi-overlay-c1").nodes
        assert panel.is_attached
        assert app.screen.focused is panel
        handle.unfocus()
        await pilot.pause()
        assert app.screen.focused is app._editor
        assert handle.is_focused() is False
        app._set_overlay_component("c1", None, {})
        await pilot.pause()
        assert app.query("#pi-overlay-c1").nodes == []


@pytest.mark.asyncio
async def test_set_overlay_component_replaces_lines_overlay(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._set_overlay("c1", ["lines"], {"anchor": "top-left"})
        await pilot.pause()
        assert app.query("#pi-overlay-c1").nodes
        panel = _FocusablePanel()
        app._set_overlay_component("c1", panel, {"anchor": "top-left"})
        await pilot.pause()
        await pilot.pause()
        assert app.query_one("#pi-overlay-c1", Widget) is not None
        assert panel.is_attached
        assert app.screen.focused is panel


@pytest.mark.asyncio
async def test_overlay_mounts_on_screen_without_covering_base(tmp_path):
    """回归：overlay 直接挂 Screen overlay 层，不遮挡聊天/编辑器/底栏。"""
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query("#pi-overlay-layer").nodes == []
        assert app._editor.display is True
        app._set_overlay("ov1", ["overlay text"], {"anchor": "top-left"})
        await pilot.pause()
        widget = app.query_one("#pi-overlay-ov1", Widget)
        assert widget.parent is app.screen
        assert app._editor.display is True


@pytest.mark.asyncio
async def test_overlay_dialog_focus_restores_overlay(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        handle = app._set_overlay("ov1", ["x"], {"anchor": "top-left"})
        await pilot.pause()
        overlay_widget = app.query_one("#pi-overlay-ov1", Widget)
        assert app.screen.focused is overlay_widget
        app.push_screen(ChoiceSelector("Pick", ["a", "b"]))
        await pilot.pause()
        await pilot.pause()
        assert app.query(ChoiceSelector).nodes
        assert app.screen.focused is not overlay_widget
        app.query_one(ChoiceSelector).dismiss("a")
        await pilot.pause()
        await pilot.pause()
        assert app.screen.focused is overlay_widget
        assert handle is not None
        assert handle.is_focused()


@pytest.mark.asyncio
async def test_overlay_dialog_focus_restores_editor(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        assert app.screen.focused is app._editor
        app.push_screen(ChoiceSelector("Pick", ["a", "b"]))
        await pilot.pause()
        await pilot.pause()
        assert app.screen.focused is not app._editor
        app.query_one(ChoiceSelector).dismiss(None)
        await pilot.pause()
        await pilot.pause()
        assert app.screen.focused is app._editor


@pytest.mark.asyncio
async def test_hidden_thinking_label_in_chat(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._set_hidden_thinking_label("Pondering...")
        app._chat.add_message_agent(
            {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": "reasoning"}],
            }
        )
        await pilot.pause()
        labels = [entry.label for entry in app.query(MessageEntry)]
        assert "Pondering..." in labels


@pytest.mark.asyncio
async def test_working_message_and_theme(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._set_working_message("Working... (custom)")
        assert app._working_message == "Working... (custom)"
        app._set_theme("light")
        await pilot.pause()
        assert app._theme.name == "light"


@pytest.mark.asyncio
async def test_autocomplete_inserts_value(tmp_path):
    from pi_coding_agent.extensions.runner import ExtensionRunner
    from pi_coding_agent.extensions.types import Extension
    from pi_tui.selectors import ChoiceSelector

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    extension = Extension(path="<inline>", resolved_path="<inline>")

    def provider(text):
        return [{"value": "#123", "label": "#123 Fix bug"}] if "#" in text else None

    extension.autocomplete.append(provider)
    session.set_extension_runner(ExtensionRunner([extension], cwd=str(tmp_path)))
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._editor.text = "fix #"
        app._editor.focus()
        await pilot.press("tab")
        await pilot.pause()
        assert app.query(ChoiceSelector).nodes
        app.query_one(ChoiceSelector).action_select()
        await pilot.pause()
        assert "#123" in app._editor.text


@pytest.mark.asyncio
async def test_autocomplete_async_provider(tmp_path):
    from pi_coding_agent.extensions.runner import ExtensionRunner
    from pi_coding_agent.extensions.types import Extension
    from pi_tui.selectors import ChoiceSelector

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    extension = Extension(path="<inline>", resolved_path="<inline>")

    async def provider(text):
        return [{"value": "#456", "label": "#456 Fix async"}] if "#" in text else None

    extension.autocomplete.append(provider)
    session.set_extension_runner(ExtensionRunner([extension], cwd=str(tmp_path)))
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._editor.text = "fix #"
        app._editor.focus()
        await pilot.press("tab")
        await pilot.pause()
        await pilot.pause()
        assert app.query(ChoiceSelector).nodes
        app.query_one(ChoiceSelector).action_select()
        await pilot.pause()
        assert "#456" in app._editor.text


@pytest.mark.asyncio
async def test_session_streaming_updates_partial_entry(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._on_session_event(
            {"type": "message_start", "message": {"role": "assistant", "content": []}}
        )
        await pilot.pause()
        app._on_session_event(
            {
                "type": "message_update",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hel"}],
                },
            }
        )
        await pilot.pause()
        entries = app._chat.query(MessageEntry)
        assert len(entries) == 1
        assert entries[0].label == "Assistant"
        assert "Hel" in entries[0].entry_text
        app._on_session_event(
            {
                "type": "message_update",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello"}],
                },
            }
        )
        await pilot.pause()
        assert "Hello" in app._chat.query(MessageEntry)[0].entry_text
        app._on_session_event(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello world"}],
                },
            }
        )
        await _wait_until(
            lambda: len(app._chat.query(MessageEntry)) == 1,
            pilot=pilot,
            message="stream placeholder replaced by final message",
        )
        assert "Hello world" in app._chat.query(MessageEntry)[0].entry_text


@pytest.mark.asyncio
async def test_session_streaming_thinking_and_toolcall(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._on_session_event(
            {
                "type": "message_update",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "pondering"},
                        {"type": "toolCall", "name": "read", "arguments": {"path": "a"}},
                    ],
                },
            }
        )
        await pilot.pause()
        entries = app._chat.query(MessageEntry)
        assert len(entries) == 1
        text = entries[0].entry_text
        assert "pondering" in text
        assert "read(" in text
        app._on_session_event({"type": "agent_settled"})
        await _wait_until(
            lambda: len(app._chat.query(MessageEntry)) == 0,
            pilot=pilot,
            message="stream placeholder cleaned on settle",
        )


@pytest.mark.asyncio
async def test_bindings_wired(tmp_path):
    """快捷键 manager → Textual BINDINGS 的映射正确。"""
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        assert app._keybindings.resolve("ctrl+p") == "app.model.cycleForward"
        assert app._keybindings.resolve("ctrl+d") == "app.exit"
        binding_actions = {binding.action for binding in app.BINDINGS}
        assert "cycle_model_forward" in binding_actions
        assert "select_model" in binding_actions
        assert "exit" in binding_actions
        # 绑定真实可分发：ctrl+p 切换模型。
        await pilot.press("ctrl+p")
        await _wait_until(
            lambda: session.model is not None and session.model.id == "faux-2",
            pilot=pilot,
            message="binding dispatched",
        )


@pytest.mark.asyncio
async def test_prompt_renders_user_and_assistant(tmp_path):
    runtime = _make_runtime(responses=[faux_assistant_message("tui assistant reply")])
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test():
        app._editor.text = "hello tui"
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "hello tui"))
        await _wait_until(
            lambda: any(
                entry.label == "User" and "hello tui" in entry.entry_text
                for entry in app._chat.query(MessageEntry)
            )
        )
        await _wait_until(
            lambda: any(
                entry.label == "Assistant" and "tui assistant reply" in entry.entry_text
                for entry in app._chat.query(MessageEntry)
            )
        )
        assert "messages: 2" in str(app._footer.content)


@pytest.mark.asyncio
async def test_slash_command_name(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test():
        app._editor.text = "/name tui-session"
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/name tui-session"))
        await _wait_until(lambda: session.session_name == "tui-session")
        await _wait_until(lambda: "tui-session" in str(app._footer.content))


@pytest.mark.asyncio
async def test_cycle_model_shortcut(tmp_path):
    runtime = _make_runtime(model_count=2)
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test():
        app.action_cycle_model_forward()
        await _wait_until(lambda: session.model is not None and session.model.id == "faux-2")
        assert "faux/faux-2" in str(app._footer.content)


@pytest.mark.asyncio
async def test_model_selector(tmp_path):
    runtime = _make_runtime(model_count=2)
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.action_select_model()
        await _wait_until(
            lambda: app.query(ModelSelector).nodes,
            pilot=pilot,
            message="model selector screen",
        )
        await _wait_until(
            lambda: len(app.query_one(ModelSelector).query_one("#model-list").children) == 2,
            pilot=pilot,
            message="model list populated",
        )
        list_view = app.query_one(ModelSelector).query_one("#model-list")
        list_view.index = 1
        app.query_one(ModelSelector).action_select()
        await _wait_until(
            lambda: session.model is not None and session.model.id == "faux-2",
            pilot=pilot,
            message="model selection applied",
        )


@pytest.mark.asyncio
async def test_model_selector_with_colon_model_ids():
    """回归：模型 id 含冒号（如 ollama/qwen3:30b）时选择器不应崩溃。"""
    from textual.app import App

    models = [
        Model(id="faux-1", provider="faux", api="openai-completions", name="Faux 1"),
        Model(
            id="qwen3:30b",
            provider="ollama",
            api="openai-completions",
            name="Qwen3 30B",
        ),
        Model(
            id="deepseek-r1:14b",
            provider="ollama",
            api="openai-completions",
            name="DeepSeek R1 14B",
        ),
    ]

    class Host(App):
        def compose(self):
            yield ModelSelector(models, current=models[1])

        def on_mount(self) -> None:
            self.query_one(ModelSelector).query_one("#model-list").focus()

    app = Host()
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: app.query(ModelSelector).nodes,
            pilot=pilot,
            message="model selector screen",
        )
        await _wait_until(
            lambda: len(app.query_one(ModelSelector).query_one("#model-list").children) == 3,
            pilot=pilot,
            message="model list populated",
        )


@pytest.mark.asyncio
async def test_model_selector_keyboard_navigation():
    """回归：搜索框持焦时 ↑↓/Enter/Esc 应操作列表，而不是被输入框吞掉。"""
    from textual.app import App
    from textual.widgets import Input

    models = [
        Model(id=f"faux-{index}", provider="faux", api="openai-completions") for index in (1, 2, 3)
    ]

    class Host(App):
        def __init__(self) -> None:
            super().__init__()
            self.dismissed: list[Any] = []

        def compose(self):
            yield ModelSelector(models)

        def on_mount(self) -> None:
            self.query_one(Input).focus()

        def _close_overlay_dialog(self, component, value=None) -> None:
            self.dismissed.append(value)

    app = Host()
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: app.query(ModelSelector).nodes,
            pilot=pilot,
            message="model selector screen",
        )
        list_view = app.query_one(ModelSelector).query_one("#model-list")

        # 焦点在搜索框时，方向键应移动列表选择
        base = list_view.index if list_view.index is not None else 0
        await pilot.press("down")
        assert list_view.index == (base + 1) % len(models)
        await pilot.press("down")
        assert list_view.index == (base + 2) % len(models)
        await pilot.press("up")
        assert list_view.index == (base + 1) % len(models)

        # Enter 应选中并关闭选择器
        await pilot.press("enter")
        assert app.dismissed and app.dismissed[-1].id == models[(base + 1) % len(models)].id


@pytest.mark.asyncio
async def test_editor_enter_submits_and_shift_enter_newlines():
    """回归：Enter 应提交消息而非插入换行；Shift+Enter 插入换行。"""
    from textual.app import App

    submitted: list[str] = []

    class Host(App):
        def compose(self):
            yield PiEditor()

        def on_pi_editor_submitted(self, message: PiEditor.Submitted) -> None:
            submitted.append(message.text)

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(PiEditor)
        editor.focus()
        await pilot.press("h", "i")
        await pilot.press("shift+enter")
        await pilot.press("t")
        assert editor.text == "hi\nt"
        await pilot.press("enter")
        assert editor.text == ""
        assert submitted == ["hi\nt"]


@pytest.mark.asyncio
async def test_editor_ctrl_d_bubbles_when_empty():
    """回归：编辑器为空时 ctrl+d 应冒泡到应用（退出），非空时保留删除行为。"""
    from textual.app import App

    exited: list[bool] = []

    class Host(App):
        def compose(self):
            yield PiEditor()

        def on_pi_editor_exit_requested(self, _message) -> None:
            exited.append(True)

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(PiEditor)
        editor.focus()

        # 空编辑器：ctrl+d 冒泡到应用
        await pilot.press("ctrl+d")
        assert exited == [True]

        # 非空：保留 TextArea 默认（删除右侧字符）
        editor.text = "ab"
        await pilot.press("ctrl+d")
        assert editor.text != "ab"
        assert exited == [True]


@pytest.mark.asyncio
async def test_editor_ctrl_x_posts_copy_requested():
    """回归：ctrl+x 应触发复制最后一条消息（对齐 TS），而不是 TextArea 剪切。"""
    from textual.app import App

    copied: list[bool] = []

    class Host(App):
        def compose(self):
            yield PiEditor()

        def on_pi_editor_copy_requested(self, _message) -> None:
            copied.append(True)

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(PiEditor)
        editor.focus()
        editor.text = "hello"
        await pilot.press("ctrl+x")
        assert copied == [True]
        assert editor.text == "hello"  # 没有被剪切


@pytest.mark.asyncio
async def test_editor_shift_tab_posts_cycle_thinking_requested():
    """回归：shift+tab 应触发循环 thinking（对齐 TS），而不是 Textual 焦点切换。"""
    from textual.app import App

    cycled: list[bool] = []

    class Host(App):
        def compose(self):
            yield PiEditor()

        def on_pi_editor_cycle_thinking_requested(self, _message) -> None:
            cycled.append(True)

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(PiEditor)
        editor.focus()
        await pilot.press("shift+tab")
        assert cycled == [True]


@pytest.mark.asyncio
async def test_session_picker_mounts_with_sessions():
    """回归：SessionPicker 挂载时填充列表不应抛 MountError（Ctrl+R 崩溃）。"""
    from textual.app import App

    sessions = [
        {
            "path": "/home/pi/.pi/agent/sessions/aaa.jsonl",
            "session_id": "aaa",
            "modified": 1785839617,
        },
        {
            "path": "/home/pi/.pi/agent/sessions/bbb.jsonl",
            "session_id": "bbb",
            "modified": 1785839618,
        },
    ]
    picked: list[str] = []

    class Host(App):
        def compose(self):
            yield SessionPicker(sessions)

        def on_mount(self) -> None:
            self.query_one("#session-list").focus()

        def _close_overlay_dialog(self, component, value=None) -> None:
            picked.append(value)

    app = Host()
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: app.query(SessionPicker).nodes,
            pilot=pilot,
            message="session picker screen",
        )
        await _wait_until(
            lambda: len(app.query_one(SessionPicker).query_one("#session-list").children) == 2,
            pilot=pilot,
            message="session list populated",
        )
        # 初始选中第一项；不按方向键直接 Enter 也应选中并关闭选择器
        assert app.query_one(SessionPicker).query_one("#session-list").index == 0
        await pilot.press("enter")
        assert picked == [sessions[0]["path"]]


@pytest.mark.asyncio
async def test_exit_with_empty_editor(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test():
        app.exit()
    assert app._exit is not None


@pytest.mark.asyncio
async def test_follow_up_and_clear(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test():
        app._editor.text = "queued follow-up"
        app.action_follow_up()
        await _wait_until(lambda: session.pending_message_count == 1)
        assert "queued follow-up" not in app._editor.text

        app.action_dequeue()
        await _wait_until(lambda: session.pending_message_count == 0)


@pytest.mark.asyncio
async def test_new_session_factory(tmp_path):
    runtime = _make_runtime()
    original = _make_session(runtime, tmp_path)

    def factory():
        return _make_session(runtime, tmp_path)

    app = PiTuiApp(original, runtime, session_factory=factory)
    async with app.run_test():
        app.action_new_session()
        await _wait_until(lambda: app._session is not original)
        assert app._session.session_id != original.session_id


@pytest.mark.asyncio
async def test_slash_tree_in_app(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    from pi_ai._types import UserMessage

    entry_id = await session.session_manager.append_message(UserMessage(role="user", content="hi"))
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/tree"))
        await _wait_until(
            lambda: app.query(TreeSelector).nodes,
            pilot=pilot,
            message="tree selector opened",
        )
        await _wait_until(
            lambda: (
                len(app.query_one(TreeSelector).query_one("#tree-list").children) == 1
                and app.query_one(TreeSelector)
                .query_one("#tree-list")
                .children[0]
                .query(Label)
                .nodes
            ),
            pilot=pilot,
            message="tree list populated",
        )
        list_view = app.query_one(TreeSelector).query_one("#tree-list")
        label = list_view.children[0].query_one("Label")
        assert entry_id[:8] in str(label.content)


@pytest.mark.asyncio
async def test_slash_model_no_args_opens_selector(tmp_path):
    """回归：/model 无参应打开模型选择器（协程未被 await 的 bug）。"""
    runtime = _make_runtime(model_count=2)
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/model"))
        await _wait_until(
            lambda: app.query(ModelSelector).nodes,
            pilot=pilot,
            message="model selector opened",
        )


@pytest.mark.asyncio
async def test_slash_settings_opens_selector(tmp_path):
    """/settings 无参在 TUI 中打开菜单式设置选择器。"""
    import json

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    initial_auto_compact = session.auto_compaction_enabled
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/settings"))
        await _wait_until(
            lambda: app.query(SettingsSelector).nodes,
            pilot=pilot,
            message="settings selector opened",
        )
        await _wait_until(
            lambda: len(app.query_one(SettingsSelector).query_one("#settings-list").children) == 5,
            pilot=pilot,
            message="settings list populated",
        )
        # 第一项 autoCompaction：选择后切换并落盘到项目 .pi/settings.json。
        # （headless 下 Enter 键路由偶发时序抖动，直接调用 action_select 保持确定。）
        await pilot.pause()
        app.query_one(SettingsSelector).action_select()
        await _wait_until(
            lambda: app._settings.get("autoCompaction") is (not initial_auto_compact),
            pilot=pilot,
            message="autoCompaction toggled",
        )
        project_path = tmp_path / ".pi" / "settings.json"
        assert project_path.exists()
        data = json.loads(project_path.read_text(encoding="utf-8"))
        assert data["autoCompaction"] is (not initial_auto_compact)


@pytest.mark.asyncio
async def test_slash_thinking_opens_selector(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/thinking"))
        await _wait_until(
            lambda: app.query(ThinkingSelector).nodes,
            pilot=pilot,
            message="thinking selector opened",
        )
        await _wait_until(
            lambda: len(app.query_one(ThinkingSelector).query_one("#thinking-list").children) > 0,
            pilot=pilot,
            message="thinking list populated",
        )
        levels = app.query_one(ThinkingSelector)._levels
        app.query_one(ThinkingSelector).action_select()
        await _wait_until(
            lambda: session.thinking_level == levels[0],
            pilot=pilot,
            message="thinking level applied",
        )


@pytest.mark.asyncio
async def test_slash_scoped_models_opens_selector(tmp_path):
    runtime = _make_runtime(model_count=2)
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/scoped-models"))
        await _wait_until(
            lambda: app.query(ScopedModelsSelector).nodes,
            pilot=pilot,
            message="scoped models selector opened",
        )
        await _wait_until(
            lambda: (
                len(app.query_one(ScopedModelsSelector).query_one("#scoped-list").children) == 2
            ),
            pilot=pilot,
            message="scoped list populated",
        )
        app.query_one(ScopedModelsSelector).action_toggle_scoped()
        app.query_one(ScopedModelsSelector).action_cancel()
        await _wait_until(
            lambda: len(session.scoped_models) == 1,
            pilot=pilot,
            message="scoped models applied",
        )


@pytest.mark.asyncio
async def test_slash_oauth_opens_selector(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/oauth"))
        await _wait_until(
            lambda: app.query(OAuthSelector).nodes,
            pilot=pilot,
            message="oauth selector opened",
        )
        await _wait_until(
            lambda: len(app.query_one(OAuthSelector).query_one("#oauth-list").children) > 0,
            pilot=pilot,
            message="oauth list populated",
        )


@pytest.mark.asyncio
async def test_slash_extensions_opens_selector(tmp_path):
    from pi_coding_agent.extensions import ExtensionRunner
    from pi_coding_agent.extensions.types import Extension

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    extension = Extension(path="/tmp/ext.py", resolved_path="/tmp/ext.py")
    session.set_extension_runner(ExtensionRunner([extension], cwd=str(tmp_path)))
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/extensions"))
        await _wait_until(
            lambda: app.query(ExtensionSelector).nodes,
            pilot=pilot,
            message="extension selector opened",
        )
        await _wait_until(
            lambda: (
                len(app.query_one(ExtensionSelector).query_one("#extension-list").children) == 1
            ),
            pilot=pilot,
            message="extension list populated",
        )


@pytest.mark.asyncio
async def test_skill_invocation_event_renders_entry(tmp_path):
    from pi_tui.components import MessageEntry

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app._on_session_event({"type": "skill_invocation", "skill": "docs"})
        await _wait_until(
            lambda: any(
                entry.label == "Skill" and "docs" in entry.entry_text
                for entry in app.query(MessageEntry)
            ),
            pilot=pilot,
            message="skill invocation entry rendered",
        )


@pytest.mark.asyncio
async def test_branch_summary_rendered_from_entries(tmp_path):
    from pi_tui.components import MessageEntry

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    from pi_ai._types import UserMessage

    await session.session_manager.append_message(UserMessage(role="user", content="a"))
    await session.session_manager.append_branch_summary(
        session.session_manager.get_leaf_id(), "summarized branch"
    )
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: any(
                entry.label == "Branch summary" and "summarized branch" in entry.entry_text
                for entry in app.query(MessageEntry)
            ),
            pilot=pilot,
            message="branch summary entry rendered",
        )


@pytest.mark.asyncio
async def test_slash_scoped_models_comma_separated(tmp_path):
    """回归：/scoped-models 支持逗号分隔多个模型。"""
    runtime = _make_runtime(model_count=3)
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/scoped-models faux-1,faux-2"))
        await _wait_until(
            lambda: len(session.scoped_models) == 2,
            pilot=pilot,
            message="scoped models set",
        )


@pytest.mark.asyncio
async def test_slash_login_unknown_provider(tmp_path):
    """回归：TUI /login 不应阻塞事件循环，未知 provider 直接提示。"""
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/login bogus"))
        await _wait_until(
            lambda: any(
                "Unknown provider" in str(entry.entry_text) for entry in app.query(MessageEntry)
            ),
            pilot=pilot,
            message="unknown provider message",
        )


@pytest.mark.asyncio
async def test_tui_auth_interaction_notify_routes_to_chat(tmp_path):
    """回归：TUI OAuth 的 URL/设备码应发到聊天区，而非阻塞的 print/input。"""
    from pi_coding_agent.modes.interactive.app import _TuiAuthInteraction

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        _TuiAuthInteraction(app).notify(
            {
                "type": "device_code",
                "verificationUri": "https://example.com/device",
                "userCode": "ABCD-1234",
            }
        )
        await _wait_until(
            lambda: any("ABCD-1234" in str(entry.entry_text) for entry in app.query(MessageEntry)),
            pilot=pilot,
            message="device code shown in chat",
        )
        assert any("剪贴板" in str(entry.entry_text) for entry in app.query(MessageEntry))


@pytest.mark.asyncio
async def test_text_input_dialog_submits():
    """回归：TextInputDialog 输入后 Enter 返回结果。"""
    from textual.app import App
    from textual.widgets import Input

    from pi_tui.selectors import TextInputDialog

    result: list[str] = []

    class Host(App):
        def compose(self):
            yield TextInputDialog("paste url")

        def on_mount(self) -> None:
            self.query_one(Input).focus()

        def on_input_submitted(self, event) -> None:
            result.append(event.value)

    app = Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("h", "i")
        await pilot.press("enter")
    assert result == ["hi"]


@pytest.mark.asyncio
async def test_slash_fork_in_app(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    from pi_ai._types import UserMessage

    e1 = await session.session_manager.append_message(UserMessage(role="user", content="a"))
    await session.session_manager.append_message(UserMessage(role="user", content="b"))

    def rebuilder(manager):
        return _make_session_with_manager(runtime, manager, tmp_path)

    app = PiTuiApp(session, runtime, session_rebuilder=rebuilder)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, f"/fork {e1}"))
        await _wait_until(
            lambda: app._session is not session,
            pilot=pilot,
            message="session replaced after fork",
        )
        assert app._session.session_manager.get_leaf_id() == e1


@pytest.mark.asyncio
async def test_slash_fork_no_args_opens_selector(tmp_path):
    """回归：/fork 无参应打开树选择器，选中后直接 fork（无需手填 entryId）。"""
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    from pi_ai._types import UserMessage

    await session.session_manager.append_message(UserMessage(role="user", content="a"))
    await session.session_manager.append_message(UserMessage(role="user", content="b"))

    def rebuilder(manager):
        return _make_session_with_manager(runtime, manager, tmp_path)

    app = PiTuiApp(session, runtime, session_rebuilder=rebuilder)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/fork"))
        await _wait_until(
            lambda: app.query(TreeSelector).nodes,
            pilot=pilot,
            message="fork selector opened",
        )
        await _wait_until(
            lambda: app.screen.focused is app.query_one(TreeSelector).query_one("#tree-list"),
            pilot=pilot,
            message="tree list focused",
        )
        # 初始选中第一项，Enter 直接 fork
        await pilot.press("enter")
        await _wait_until(
            lambda: app._session is not session,
            pilot=pilot,
            message="session replaced after fork",
        )
        assert app._slash_context.session is app._session


@pytest.mark.asyncio
async def test_slash_input_merges_and_continues(tmp_path):
    """/input：合并文本进历史 user 消息 → 重建会话 → 自动 continue。"""
    from pi_ai._types import UserMessage

    runtime = _make_runtime(responses=[faux_assistant_message("edited reply")])
    session = _make_session(runtime, tmp_path)
    e1 = await session.session_manager.append_message(
        UserMessage(role="user", content="old instruction")
    )
    await session.session_manager.append_message(UserMessage(role="user", content="later msg"))

    def rebuilder(manager):
        return _make_session_with_manager(runtime, manager, tmp_path)

    app = PiTuiApp(session, runtime, session_rebuilder=rebuilder)
    # 选中即复制：避免测试把内容写进真实剪贴板。
    app._copy_to_clipboard = lambda text: None
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/input 请改用Python"))
        await _wait_until(
            lambda: app.query(TreeSelector).nodes,
            pilot=pilot,
            message="input selector opened",
        )
        tree = app.query_one(TreeSelector)
        await _wait_until(
            lambda: len(tree.query_one("#tree-list").children) == 2,
            pilot=pilot,
            message="user messages listed",
        )
        await _wait_until(
            lambda: app.screen.focused is tree.query_one("#tree-list"),
            pilot=pilot,
            message="tree list focused",
        )
        # 最新在前：index 0 = later msg，index 1 = e1。
        tree.query_one("#tree-list").index = 1
        await pilot.press("enter")
        await _wait_until(
            lambda: app._session is not session,
            pilot=pilot,
            message="session replaced after input",
        )
        await _wait_until(
            lambda: (
                len(app._session.get_messages()) >= 2
                and app._session.get_messages()[-1].get("role") == "assistant"
            ),
            pilot=pilot,
            message="edited branch continued",
        )

        messages = app._session.get_messages()
        assert messages[0]["content"] == "old instruction\n\n请改用Python"
        assert app._session.session_manager.get_leaf_id() != e1  # continue 后 leaf 前进
        assert "edited reply" in (app._session.get_last_assistant_text() or "")


@pytest.mark.asyncio
async def test_input_selector_copy_selected(tmp_path):
    """对话框选中即复制：/input 选择器 Enter 后复制完整消息文本。"""
    from pi_ai._types import UserMessage

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    await session.session_manager.append_message(
        UserMessage(role="user", content="full message text")
    )

    def rebuilder(manager):
        return _make_session_with_manager(runtime, manager, tmp_path)

    app = PiTuiApp(session, runtime, session_rebuilder=rebuilder)
    copied: list[str] = []
    app._copy_to_clipboard = lambda text: copied.append(text)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/input"))
        await _wait_until(
            lambda: app.query(TreeSelector).nodes,
            pilot=pilot,
            message="input selector opened",
        )
        await _wait_until(
            lambda: app.screen.focused is app.query_one(TreeSelector).query_one("#tree-list"),
            pilot=pilot,
            message="tree list focused",
        )
        await pilot.press("enter")
        await _wait_until(
            lambda: copied,
            pilot=pilot,
            message="copied selected message",
        )
    assert copied == ["full message text"]


@pytest.mark.asyncio
async def test_slash_reload_reloads_resources(tmp_path):
    import json as _json

    from pi_coding_agent.extensions import ExtensionLoader, ExtensionRunner
    from pi_coding_agent.prompt_templates import PromptTemplateLoader
    from pi_coding_agent.skills import SkillLoader
    from pi_tui.keybindings import KeybindingsManager
    from pi_tui.theme import BUILTIN_THEMES, ThemeLoader

    skills_dir = tmp_path / "skills"
    prompts_dir = tmp_path / "prompts"
    extensions_dir = tmp_path / "extensions"
    themes_dir = tmp_path / "themes"
    for directory in (skills_dir, prompts_dir, extensions_dir, themes_dir):
        directory.mkdir(parents=True)

    theme_colors = dict(BUILTIN_THEMES["dark"])
    theme_colors["accent"] = "#111111"
    (themes_dir / "custom.json").write_text(_json.dumps(theme_colors), encoding="utf-8")

    runtime = _make_runtime()
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=runtime.stream,
        )
    )
    skill_loader = SkillLoader(global_dir=skills_dir)
    template_loader = PromptTemplateLoader(global_dir=prompts_dir)
    session = AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=model,
        model_runtime=runtime,
        skill_loader=skill_loader,
        template_loader=template_loader,
        extension_runner=ExtensionRunner([], cwd=str(tmp_path), model_runtime=runtime),
    )
    extension_loader = ExtensionLoader(global_dir=extensions_dir, cwd=str(tmp_path))
    settings: dict = {}
    app = PiTuiApp(
        session,
        runtime,
        keybindings_manager=KeybindingsManager(),
        theme_loader=ThemeLoader(themes_dir),
        theme_name="custom",
        settings=settings,
        extension_loader=extension_loader,
    )

    async with app.run_test() as pilot:
        (skills_dir / "alpha").mkdir(parents=True, exist_ok=True)
        (skills_dir / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Alpha skill\n---\nBody",
            encoding="utf-8",
        )
        (prompts_dir / "review.md").write_text(
            "---\ndescription: Review\n---\nReview {{0}}",
            encoding="utf-8",
        )
        (extensions_dir / "hello.py").write_text(
            "def create_extension(api):\n"
            '    api.register_command("hello", '
            '{"description": "Hello", "handler": lambda ctx, args: "hi"})\n',
            encoding="utf-8",
        )
        (extensions_dir / "bad.py").write_text("def create_extension(api\n", encoding="utf-8")
        theme_colors["accent"] = "#abcdef"
        (themes_dir / "custom.json").write_text(_json.dumps(theme_colors), encoding="utf-8")
        settings["keybindings"] = {"app.model.select": "ctrl+m"}

        app._editor.text = "/reload"
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/reload"))
        await _wait_until(
            lambda: (
                session.skill_loader is not None
                and session.skill_loader.get("alpha") is not None
                and session.template_loader is not None
                and session.template_loader.get("review") is not None
                and session.extension_runner is not None
                and len(session.extension_runner.extensions) == 1
            ),
            pilot=pilot,
            message="reload applied to loaders",
        )
        await _wait_until(
            lambda: "Reloaded:" in str(app._status.content),
            pilot=pilot,
            message="reload status shown",
        )
        assert "extension error" in str(app._status.content)

        assert app._theme.colors["accent"] == "#abcdef"
        assert app._keybindings.get_action_key("app.model.select") == "ctrl+m"
        assert app._slash_registry.get("hello") is not None
        assert any(
            binding.key == "ctrl+m" and binding.action == "select_model" for binding in app.BINDINGS
        )

        # 重载后的快捷键真实可分发（打开模型选择器）。
        await pilot.press("ctrl+m")
        await _wait_until(
            lambda: app.query(ModelSelector).nodes,
            pilot=pilot,
            message="reloaded keybinding dispatches",
        )


@pytest.mark.asyncio
async def test_bang_command_runs_and_records_bash_execution(tmp_path):
    """`!cmd` 本地执行、流式展示并写入 bashExecution 消息。"""
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test():
        app._editor.text = "!echo hello-bash"
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "!echo hello-bash"))
        await _wait_until(
            lambda: any(
                entry.label == "Bash" and "hello-bash" in entry.output
                for entry in app._chat.query(BashExecutionEntry)
            ),
            message="bash entry shows output",
        )
        await _wait_until(
            lambda: (
                not session.is_bash_running
                and session.get_messages()
                and session.get_messages()[-1].get("role") == "bashExecution"
            ),
            message="bash result recorded",
        )
        bash = session.get_messages()[-1]
        assert bash["command"] == "echo hello-bash"
        assert "hello-bash" in str(bash["output"])
        assert bash["exitCode"] == 0
        assert bash.get("excludeFromContext") is False


@pytest.mark.asyncio
async def test_double_bang_command_excludes_from_context(tmp_path):
    """`!!cmd` 本地执行，但 bashExecution 不进入 LLM 上下文。"""
    from pi_agent._agent import _default_convert_to_llm

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test():
        app._editor.text = "!!echo hidden-bash"
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "!!echo hidden-bash"))
        await _wait_until(
            lambda: (
                not session.is_bash_running
                and session.get_messages()
                and session.get_messages()[-1].get("role") == "bashExecution"
            ),
            message="double-bang bash result recorded",
        )
        bash = session.get_messages()[-1]
        assert bash.get("excludeFromContext") is True
        llm_messages = _default_convert_to_llm(session.get_messages())
        assert all("hidden-bash" not in str(message.get("content", "")) for message in llm_messages)


@pytest.mark.asyncio
async def test_bang_command_failure_renders_error(tmp_path):
    """命令失败时条目显示 exit code，会话仍可继续。"""
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test():
        app._editor.text = "!exit 7"
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "!exit 7"))
        await _wait_until(
            lambda: any(
                entry.label == "Bash" and entry.status == "(exit 7)"
                for entry in app._chat.query(BashExecutionEntry)
            ),
            message="bash failure status shown",
        )
        bash = session.get_messages()[-1]
        assert bash["exitCode"] == 7


def test_user_message_nodes_filters_user_messages():
    """/input 选择器：只列出 user 消息，最新在前，label 显示内容片段。"""
    from pi_ai._types import UserMessage

    from pi_coding_agent.modes.interactive.app import _user_message_nodes

    mgr = SessionManager.in_memory(cwd="/tmp")
    e1 = asyncio.run(
        mgr.append_message(UserMessage(role="user", content="first line\nsecond line"))
    )
    asyncio.run(mgr.append_message(faux_assistant_message("hi")))
    e3 = asyncio.run(mgr.append_message(UserMessage(role="user", content="latest [x]")))

    nodes = _user_message_nodes(mgr)

    assert [node.id for node in nodes] == [e3, e1]
    assert "first line" in nodes[1].label
    assert r"\[x\]" in nodes[0].label  # 方括号已转义，避免被 Textual 当 Rich markup。


def test_format_context_path():
    from pi_coding_agent.modes.interactive.app import _format_context_path

    assert _format_context_path(str(Path.home() / "work" / "AGENTS.md"), "/tmp/proj") == (
        "~/work/AGENTS.md"
    )
    cwd = Path.cwd()
    assert _format_context_path(str(cwd / "AGENTS.md"), str(cwd)) == "AGENTS.md"
    assert _format_context_path(str(cwd / "docs" / "AGENTS.md"), str(cwd)) == "docs/AGENTS.md"


@pytest.mark.asyncio
async def test_startup_context_hint_shows_agents(tmp_path):
    """启动提示：加载到 AGENTS.md 时聊天区出现 [Context] System 消息。"""
    (tmp_path / "AGENTS.md").write_text("project rules", encoding="utf-8")
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: any(
                entry.label == "System" and "[Context]" in entry.entry_text
                for entry in app._chat.query(MessageEntry)
            ),
            pilot=pilot,
            message="startup context hint shown",
        )
        hint = next(
            entry for entry in app._chat.query(MessageEntry) if "[Context]" in entry.entry_text
        )
        assert "AGENTS.md" in hint.entry_text


@pytest.mark.asyncio
async def test_startup_context_hint_suppressed_by_flag(tmp_path):
    """--no-context-files：启动不显示 Context 提示。"""
    (tmp_path / "AGENTS.md").write_text("project rules", encoding="utf-8")
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime, no_context_files=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        assert not any("[Context]" in entry.entry_text for entry in app._chat.query(MessageEntry))


@pytest.mark.asyncio
async def test_startup_resources_hint_shows_all_sections(tmp_path):
    """启动提示：Context / Skills / Prompts / Extensions / Themes 全汇总。"""
    import json as _json

    from pi_tui.theme import BUILTIN_THEMES as _BUILTIN_THEMES
    from pi_tui.theme import ThemeLoader as _ThemeLoader

    themes_dir = tmp_path / "themes"
    themes_dir.mkdir()
    theme_colors = dict(_BUILTIN_THEMES["dark"])
    theme_colors["accent"] = "#123456"
    (themes_dir / "custom.json").write_text(_json.dumps(theme_colors), encoding="utf-8")

    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(
        session,
        runtime,
        theme_loader=_ThemeLoader(themes_dir),
        startup_resources={
            "context_files": [{"path": str(tmp_path / "AGENTS.md"), "content": "rules"}],
            "skills": [{"name": "skill-a", "path": "/x/SKILL.md"}],
            "prompts": [{"name": "review", "path": "/x/review.md"}],
            "extensions": [{"name": "ext-a", "path": "/x/ext-a.py"}],
        },
    )
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: any("[Context]" in entry.entry_text for entry in app._chat.query(MessageEntry)),
            pilot=pilot,
            message="resources hint shown",
        )
        hint = next(
            entry for entry in app._chat.query(MessageEntry) if "[Context]" in entry.entry_text
        )
        for marker in ("[Context]", "[Skills]", "[Prompts]", "[Extensions]", "[Themes]"):
            assert marker in hint.entry_text
        assert "skill-a" in hint.entry_text
        assert "review" in hint.entry_text
        assert "ext-a" in hint.entry_text
        assert "custom" in hint.entry_text


@pytest.mark.asyncio
async def test_startup_resources_hint_respects_no_context_files(tmp_path):
    """--no-context-files：只隐藏 Context 段，Skills 等仍显示。"""
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(
        session,
        runtime,
        no_context_files=True,
        startup_resources={
            "context_files": [{"path": str(tmp_path / "AGENTS.md"), "content": "rules"}],
            "skills": [{"name": "skill-a", "path": "/x/SKILL.md"}],
        },
    )
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: any("[Skills]" in entry.entry_text for entry in app._chat.query(MessageEntry)),
            pilot=pilot,
            message="skills section shown",
        )
        hint = next(
            entry for entry in app._chat.query(MessageEntry) if "[Skills]" in entry.entry_text
        )
        assert "[Context]" not in hint.entry_text
        assert "skill-a" in hint.entry_text


@pytest.mark.asyncio
async def test_editor_ctrl_c_copies_selection_or_clears():
    """输入框：ctrl+c 有选区复制，无选区清空（对齐 TS）。"""
    from textual.app import App
    from textual.selection import Selection

    from pi_tui.components import PiEditor

    copied: list[str] = []

    class Host(App):
        def compose(self):
            yield PiEditor()

    app = Host()
    async with app.run_test() as pilot:
        app.copy_to_clipboard = lambda text: copied.append(text)
        editor = app.query_one(PiEditor)
        editor.focus()
        editor.text = "hello world"
        editor.selection = Selection(start=(0, 0), end=(0, 5))
        await pilot.press("ctrl+c")
        assert copied == ["hello"]
        assert editor.text == "hello world"
        editor.selection = Selection(start=(0, 0), end=(0, 0))
        await pilot.press("ctrl+c")
        assert editor.text == ""
    assert copied == ["hello"]


@pytest.mark.asyncio
async def test_message_entry_click_copies_text():
    """输出框：点击消息复制整条文本（Textual 无鼠标选词，只能整条复制）。"""
    from textual.app import App

    from pi_tui.components import MessageEntry

    copied: list[str] = []

    class Host(App):
        def compose(self):
            yield MessageEntry("User", "hello **world**")

        def on_copy_requested(self, message):
            copied.append(message.text)

    app = Host()
    async with app.run_test() as pilot:
        await pilot.click(MessageEntry)
        await pilot.pause()
    assert copied == ["hello **world**"]


@pytest.mark.asyncio
async def test_bash_entry_click_copies_command_and_output():
    """输出框：点击 bash 条目复制命令 + 输出。"""
    from textual.app import App

    from pi_tui.components import BashExecutionEntry

    copied: list[str] = []

    class Host(App):
        def compose(self):
            yield BashExecutionEntry("echo hi")

        def on_copy_requested(self, message):
            copied.append(message.text)

    app = Host()
    async with app.run_test() as pilot:
        entry = app.query_one(BashExecutionEntry)
        entry.output = "hi\n"
        entry.set_complete(0)
        await pilot.pause()
        await pilot.click(BashExecutionEntry)
        await pilot.pause()
    assert copied == ["$ echo hi\nhi\n"]
