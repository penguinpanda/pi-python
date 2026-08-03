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
from pi_tui.selectors import ModelSelector


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
    async with app.run_test():
        assert app._keybindings.resolve("ctrl+p") == "app.model.cycleForward"
        assert app._keybindings.resolve("ctrl+d") == "app.exit"
        binding_actions = {binding.action for binding in app.BINDINGS}
        assert "cycle_model_forward" in binding_actions
        assert "select_model" in binding_actions
        assert "exit" in binding_actions


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
