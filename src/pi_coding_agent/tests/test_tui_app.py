"""TUI 主应用测试（Textual 无头 run_test）。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions
from pi_ai import Model, Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime
from pi_coding_agent.modes.interactive.app import PiTuiApp
from pi_tui.components import MessageEntry, PiEditor
from pi_tui.selectors import ModelSelector, SessionPicker, TreeSelector


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
    agent = Agent(AgentOptions(
        system_prompt="You are a helpful coding assistant.",
        model=model,
        stream_fn=runtime.stream,
    ))
    return AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=model,
        model_runtime=runtime,
    )


def _make_session_with_manager(
    runtime: ModelRuntime, manager, tmp_path: Path
) -> AgentSession:
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(AgentOptions(
        system_prompt="You are a helpful coding assistant.",
        model=model,
        stream_fn=runtime.stream,
    ))
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
    async with app.run_test() as pilot:
        assert app._editor.has_focus
        footer_text = app._footer.content
        assert "faux/faux-1" in str(footer_text)
        assert "messages: 0" in str(footer_text)


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
    runtime = _make_runtime(
        responses=[faux_assistant_message("tui assistant reply")]
    )
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
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
    async with app.run_test() as pilot:
        app._editor.text = "/name tui-session"
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/name tui-session"))
        await _wait_until(
            lambda: session.session_name == "tui-session"
        )
        await _wait_until(
            lambda: "tui-session" in str(app._footer.content)
        )


@pytest.mark.asyncio
async def test_cycle_model_shortcut(tmp_path):
    runtime = _make_runtime(model_count=2)
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.action_cycle_model_forward()
        await _wait_until(
            lambda: session.model is not None and session.model.id == "faux-2"
        )
        assert "faux/faux-2" in str(app._footer.content)


@pytest.mark.asyncio
async def test_model_selector(tmp_path):
    runtime = _make_runtime(model_count=2)
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.action_select_model()
        await _wait_until(
            lambda: isinstance(app.screen, ModelSelector),
            pilot=pilot,
            message="model selector screen",
        )
        await _wait_until(
            lambda: len(app.screen.query_one("#model-list").children) == 2,
            pilot=pilot,
            message="model list populated",
        )
        list_view = app.screen.query_one("#model-list")
        list_view.index = 1
        app.screen.action_select()
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
        def on_mount(self) -> None:
            self.push_screen(ModelSelector(models, current=models[1]))

    app = Host()
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: isinstance(app.screen, ModelSelector),
            pilot=pilot,
            message="model selector screen",
        )
        await _wait_until(
            lambda: len(app.screen.query_one("#model-list").children) == 3,
            pilot=pilot,
            message="model list populated",
        )


@pytest.mark.asyncio
async def test_model_selector_keyboard_navigation():
    """回归：搜索框持焦时 ↑↓/Enter/Esc 应操作列表，而不是被输入框吞掉。"""
    from textual.app import App

    models = [
        Model(id=f"faux-{index}", provider="faux", api="openai-completions")
        for index in (1, 2, 3)
    ]
    dismissed: list[Any] = []

    class Host(App):
        def on_mount(self) -> None:
            self.push_screen(ModelSelector(models), callback=dismissed.append)

    app = Host()
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: isinstance(app.screen, ModelSelector),
            pilot=pilot,
            message="model selector screen",
        )
        list_view = app.screen.query_one("#model-list")

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
        assert dismissed and dismissed[-1].id == models[(base + 1) % len(models)].id


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
        {"path": "/home/pi/.pi/agent/sessions/aaa.jsonl", "session_id": "aaa", "modified": 1785839617},
        {"path": "/home/pi/.pi/agent/sessions/bbb.jsonl", "session_id": "bbb", "modified": 1785839618},
    ]
    picked: list[str] = []

    class Host(App):
        def on_mount(self) -> None:
            self.push_screen(SessionPicker(sessions), callback=picked.append)

    app = Host()
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: isinstance(app.screen, SessionPicker),
            pilot=pilot,
            message="session picker screen",
        )
        await _wait_until(
            lambda: len(app.screen.query_one("#session-list").children) == 2,
            pilot=pilot,
            message="session list populated",
        )
        # 初始选中第一项；不按方向键直接 Enter 也应选中并关闭选择器
        assert app.screen.query_one("#session-list").index == 0
        await pilot.press("enter")
        assert picked == [sessions[0]["path"]]


@pytest.mark.asyncio
async def test_exit_with_empty_editor(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.exit()
    assert app._exit is not None


@pytest.mark.asyncio
async def test_follow_up_and_clear(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
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
    async with app.run_test() as pilot:
        app.action_new_session()
        await _wait_until(lambda: app._session is not original)
        assert app._session.session_id != original.session_id


@pytest.mark.asyncio
async def test_slash_tree_in_app(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    from pi_ai._types import UserMessage

    entry_id = await session.session_manager.append_message(
        UserMessage(role="user", content="hi")
    )
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/tree"))
        await _wait_until(
            lambda: isinstance(app.screen, TreeSelector),
            pilot=pilot,
            message="tree selector opened",
        )
        await _wait_until(
            lambda: len(app.screen.query_one("#tree-list").children) == 1,
            pilot=pilot,
            message="tree list populated",
        )
        list_view = app.screen.query_one("#tree-list")
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
            lambda: isinstance(app.screen, ModelSelector),
            pilot=pilot,
            message="model selector opened",
        )


@pytest.mark.asyncio
async def test_slash_scoped_models_comma_separated(tmp_path):
    """回归：/scoped-models 支持逗号分隔多个模型。"""
    runtime = _make_runtime(model_count=3)
    session = _make_session(runtime, tmp_path)
    app = PiTuiApp(session, runtime)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(
            PiEditor.Submitted(app._editor, "/scoped-models faux-1,faux-2")
        )
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
                "Unknown provider" in str(entry.entry_text)
                for entry in app.query(MessageEntry)
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
        _TuiAuthInteraction(app).notify({
            "type": "device_code",
            "verificationUri": "https://example.com/device",
            "userCode": "ABCD-1234",
        })
        await _wait_until(
            lambda: any(
                "ABCD-1234" in str(entry.entry_text)
                for entry in app.query(MessageEntry)
            ),
            pilot=pilot,
            message="device code shown in chat",
        )
        assert any(
            "剪贴板" in str(entry.entry_text) for entry in app.query(MessageEntry)
        )


@pytest.mark.asyncio
async def test_text_input_dialog_submits():
    """回归：TextInputDialog 输入后 Enter 返回结果。"""
    from textual.app import App
    from textual.widgets import Input

    from pi_tui.selectors import TextInputDialog

    result: list[str] = []

    class Host(App):
        def on_mount(self) -> None:
            self.push_screen(TextInputDialog("paste url"), callback=result.append)

    app = Host()
    async with app.run_test() as pilot:
        await _wait_until(
            lambda: isinstance(app.screen, TextInputDialog),
            pilot=pilot,
            message="input dialog opened",
        )
        inp = app.screen.query_one(Input)
        inp.focus()
        await pilot.press("h", "i")
        await pilot.press("enter")
    assert result == ["hi"]


@pytest.mark.asyncio
async def test_slash_fork_in_app(tmp_path):
    runtime = _make_runtime()
    session = _make_session(runtime, tmp_path)
    from pi_ai._types import UserMessage

    e1 = await session.session_manager.append_message(
        UserMessage(role="user", content="a")
    )
    await session.session_manager.append_message(
        UserMessage(role="user", content="b")
    )

    def rebuilder(manager):
        return _make_session_with_manager(runtime, manager, tmp_path)

    app = PiTuiApp(session, runtime, session_rebuilder=rebuilder)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(
            PiEditor.Submitted(app._editor, f"/fork {e1}")
        )
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

    await session.session_manager.append_message(
        UserMessage(role="user", content="a")
    )
    await session.session_manager.append_message(
        UserMessage(role="user", content="b")
    )

    def rebuilder(manager):
        return _make_session_with_manager(runtime, manager, tmp_path)

    app = PiTuiApp(session, runtime, session_rebuilder=rebuilder)
    async with app.run_test() as pilot:
        app.on_pi_editor_submitted(PiEditor.Submitted(app._editor, "/fork"))
        await _wait_until(
            lambda: isinstance(app.screen, TreeSelector),
            pilot=pilot,
            message="fork selector opened",
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
    (themes_dir / "custom.json").write_text(
        _json.dumps(theme_colors), encoding="utf-8"
    )

    runtime = _make_runtime()
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(AgentOptions(
        system_prompt="You are a helpful coding assistant.",
        model=model,
        stream_fn=runtime.stream,
    ))
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
        extension_runner=ExtensionRunner(
            [], cwd=str(tmp_path), model_runtime=runtime
        ),
    )
    extension_loader = ExtensionLoader(
        global_dir=extensions_dir, cwd=str(tmp_path)
    )
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
        (extensions_dir / "bad.py").write_text(
            "def create_extension(api\n", encoding="utf-8"
        )
        theme_colors["accent"] = "#abcdef"
        (themes_dir / "custom.json").write_text(
            _json.dumps(theme_colors), encoding="utf-8"
        )
        settings["keybindings"] = {"app.model.select": "ctrl+m"}

        app._editor.text = "/reload"
        app.on_pi_editor_submitted(
            PiEditor.Submitted(app._editor, "/reload")
        )
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
            binding.key == "ctrl+m" and binding.action == "select_model"
            for binding in app.BINDINGS
        )

        # 重载后的快捷键真实可分发（打开模型选择器）。
        await pilot.press("ctrl+m")
        await _wait_until(
            lambda: isinstance(app.screen, ModelSelector),
            pilot=pilot,
            message="reloaded keybinding dispatches",
        )
