"""Slash 命令系统测试。"""

from __future__ import annotations

import pytest

from pi_coding_agent.modes.interactive.slash_commands import (
    SlashCommandRegistry,
    SlashContext,
    register_builtin_commands,
)


def _make_registry() -> SlashCommandRegistry:
    registry = SlashCommandRegistry()
    register_builtin_commands(registry)
    return registry


class TestParse:
    def test_simple(self):
        name, args = SlashCommandRegistry.parse("/name hello")
        assert name == "name"
        assert args == "hello"

    def test_quoted_args(self):
        name, args = SlashCommandRegistry.parse('/name "hello world"')
        assert name == "name"
        assert args == "hello world"

    def test_not_slash(self):
        name, args = SlashCommandRegistry.parse("hello")
        assert name is None
        assert args == ""

    def test_bare_slash(self):
        name, args = SlashCommandRegistry.parse("/")
        assert name == ""


class TestRegistry:
    def test_register_and_list(self):
        registry = SlashCommandRegistry()

        async def handler(_ctx, _args):
            return "ok"

        registry.register("ping", handler, description="Ping")
        assert registry.get("ping") is not None
        names = [command.name for command in registry.list()]
        assert "ping" in names

    def test_builtin_commands_registered(self):
        registry = _make_registry()
        names = {command.name for command in registry.list()}
        for expected in (
            "model",
            "name",
            "compact",
            "new",
            "quit",
            "help",
            "hotkeys",
            "session",
            "reload",
            "export",
            "tree",
            "settings",
        ):
            assert expected in names

    async def test_execute_unknown(self):
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(notify=notifications.append)
        handled = await registry.execute("/nope", context)
        assert handled is True
        assert "Unknown command" in notifications[0]

    async def test_execute_non_slash_returns_false(self):
        registry = _make_registry()
        context = SlashContext()
        assert await registry.execute("hello", context) is False

    async def test_execute_quit(self):
        registry = _make_registry()
        exited = []
        context = SlashContext(exit_app=lambda: exited.append(True))
        assert await registry.execute("/quit", context) is True
        assert exited == [True]

    async def test_execute_help(self):
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(
            slash_registry=registry,
            notify=notifications.append,
        )
        await registry.execute("/help", context)
        assert "/model" in notifications[0]
        assert "/quit" in notifications[0]

    async def test_execute_not_implemented(self):
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(notify=notifications.append)
        await registry.execute("/tree", context)
        assert "not implemented" in notifications[0]


class FakeSession:
    """极简 session 桩。"""

    def __init__(self) -> None:
        self._name = None
        self._model = None
        self._compacted = False

    def set_session_name(self, name):
        self._name = name

    @property
    def session_name(self):
        return self._name

    async def set_model(self, model):
        self._model = model

    async def compact(self, _instructions=None):
        self._compacted = True
        return object()

    def get_session_stats(self):
        return {
            "sessionId": "s1",
            "userMessages": 1,
            "assistantMessages": 1,
            "totalMessages": 2,
            "cost": 0.0,
        }

    def get_last_assistant_text(self):
        return "hello"


class TestBuiltinHandlers:
    async def test_name(self):
        registry = _make_registry()
        session = FakeSession()
        context = SlashContext(session=session)
        await registry.execute("/name my-task", context)
        assert session.session_name == "my-task"

    async def test_compact(self):
        registry = _make_registry()
        session = FakeSession()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)
        await registry.execute("/compact", context)
        assert session._compacted is True
        assert "compacted" in notifications[0]

    async def test_model_with_runtime(self):
        from pi_ai import Model, Models
        from pi_coding_agent.auth_storage import AuthStorage
        from pi_coding_agent.model_runtime import ModelRuntime
        from pi_ai.providers.faux import faux_provider

        store = AuthStorage.in_memory()
        models = Models(credentials=store)
        core = faux_provider()
        models.add_provider(core.provider)
        runtime = ModelRuntime(models, store)

        session = FakeSession()
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(
            session=session,
            model_runtime=runtime,
            notify=notifications.append,
        )
        await registry.execute("/model faux/faux-1", context)
        assert session._model is not None
        assert session._model.id == "faux-1"
        assert "Switched to" in notifications[0]
