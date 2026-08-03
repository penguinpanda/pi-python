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

class FakeSession:
    """极简 session 桩（Phase 7 命令用）。"""

    def __init__(self, cwd: str = "/tmp", manager=None) -> None:
        from pi_coding_agent._session_manager import SessionManager

        self._manager = manager or SessionManager.in_memory(cwd=cwd)
        self._cwd = cwd
        self._name = None
        self._model = None
        self._compacted = False
        self._scoped: list = []

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

    @property
    def session_manager(self):
        return self._manager

    @property
    def session_id(self):
        return self._manager.session_id

    @property
    def cwd(self):
        return self._cwd

    def set_scoped_models(self, scoped):
        self._scoped = list(scoped)

    @property
    def scoped_models(self):
        return list(self._scoped)

    async def navigate_to(self, entry_id, *, summarize=True, custom_instructions=None):
        await self._manager.move_to(entry_id, None)
        return True

    def get_messages(self):
        return self._manager.build_context()


class TestPhase7Commands:
    async def _registry(self, **kwargs):
        return _make_registry()

    async def test_tree_render_and_navigate(self, tmp_path):
        from pi_ai._types import UserMessage

        from pi_coding_agent._session_manager import SessionManager

        manager = SessionManager.in_memory(cwd=str(tmp_path))
        e1 = await manager.append_message(UserMessage(role="user", content="a"))
        e2 = await manager.append_message(UserMessage(role="user", content="b"))
        session = FakeSession(cwd=str(tmp_path), manager=manager)
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        await registry.execute("/tree", context)
        assert e1[:8] in notifications[0]
        assert e2[:8] in notifications[0]

        await registry.execute(f"/tree {e1}", context)
        assert manager.get_leaf_id() == e1

    async def test_fork_and_clone_rebuild(self, tmp_path):
        from pi_ai._types import UserMessage

        from pi_coding_agent._session_manager import SessionManager

        manager = SessionManager.in_memory(cwd=str(tmp_path))
        e1 = await manager.append_message(UserMessage(role="user", content="a"))
        await manager.append_message(UserMessage(role="user", content="b"))
        rebuilt: list = []

        def rebuild(sm):
            rebuilt.append(sm)
            return FakeSession(cwd=str(tmp_path), manager=sm)

        session = FakeSession(cwd=str(tmp_path), manager=manager)
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(
            session=session,
            notify=notifications.append,
            rebuild_session=rebuild,
        )

        await registry.execute(f"/fork {e1}", context)
        assert len(rebuilt) == 1
        assert rebuilt[0].get_leaf_id() == e1
        assert "Forked" in notifications[-1]

        await registry.execute("/clone", context)
        assert len(rebuilt) == 2
        assert rebuilt[1].get_leaf_id() == rebuilt[0].get_leaf_id()

    async def test_settings_set_and_show(self, tmp_path):
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        await registry.execute("/settings defaultModel=deepseek-chat", context)
        assert "Saved defaultModel" in notifications[-1]
        project_path = tmp_path / ".pi" / "settings.json"
        assert project_path.exists()

        await registry.execute("/settings", context)
        assert "defaultModel" in notifications[-1]
        assert "deepseek-chat" in notifications[-1]

    async def test_scoped_models(self, tmp_path):
        from pi_ai import Model, Models
        from pi_ai.providers.faux import faux_provider

        from pi_coding_agent.auth_storage import AuthStorage
        from pi_coding_agent.model_runtime import ModelRuntime

        models = Models(credentials=AuthStorage.in_memory())
        core = faux_provider(models=[
            Model(id="faux-1", provider="faux", api="openai-completions"),
            Model(id="faux-2", provider="faux", api="openai-completions"),
        ])
        models.add_provider(core.provider)
        runtime = ModelRuntime(models, AuthStorage.in_memory())

        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(
            session=session,
            model_runtime=runtime,
            notify=notifications.append,
        )

        await registry.execute("/scoped-models faux/faux-1", context)
        assert len(session.scoped_models) == 1
        assert session.scoped_models[0].model.id == "faux-1"

        await registry.execute("/scoped-models", context)
        assert "faux/faux-1" in notifications[-1]

        await registry.execute("/scoped-models clear", context)
        assert session.scoped_models == []

    async def test_login_and_logout(self, tmp_path, monkeypatch):
        from pi_ai import Models
        from pi_ai.providers.faux import faux_provider

        from pi_coding_agent.auth_storage import AuthStorage
        from pi_coding_agent.model_runtime import ModelRuntime

        class _FakeFlow:
            name = "Fake OAuth"

            async def login(self, interaction):
                return {"type": "oauth", "access": "a", "refresh": "r", "expires": 9999999999999}

        def _fake_providers():
            return [("fake", "Fake OAuth", _FakeFlow())]

        monkeypatch.setattr("pi_ai.auth.oauth.builtin_oauth_providers", _fake_providers)

        store = AuthStorage.in_memory()
        models = Models(credentials=store)
        models.add_provider(faux_provider().provider)
        runtime = ModelRuntime(models, store)
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(
            session=session,
            model_runtime=runtime,
            notify=notifications.append,
        )

        await registry.execute("/login fake", context)
        assert "Logged in" in notifications[-1]
        credential = await store.read("fake")
        assert credential is not None

        await registry.execute("/logout fake", context)
        assert await store.read("fake") is None
        assert "Logged out" in notifications[-1]

    async def test_share_gist(self, tmp_path, monkeypatch):
        from pi_ai._types import UserMessage

        from pi_coding_agent._session_manager import SessionManager

        monkeypatch.setenv("GITHUB_TOKEN", "test-token")
        manager = SessionManager.in_memory(cwd=str(tmp_path))
        await manager.append_message(UserMessage(role="user", content="share me"))
        session = FakeSession(cwd=str(tmp_path), manager=manager)
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        import httpx

        class _FakeResponse:
            status_code = 201

            def json(self):
                return {"html_url": "https://gist.github.com/abc"}

        class _FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, *args, **kwargs):
                return _FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())
        await registry.execute("/share", context)
        assert "gist.github.com/abc" in notifications[-1]

    async def test_import_and_resume(self, tmp_path):
        from pi_ai._types import UserMessage

        from pi_coding_agent._session_manager import SessionManager

        saved = SessionManager.create(
            cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"), session_id="saved1"
        )
        await saved.append_message(UserMessage(role="user", content="saved msg"))

        rebuilt: list = []

        def rebuild(sm):
            rebuilt.append(sm)
            return FakeSession(cwd=str(tmp_path), manager=sm)

        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(
            session=session,
            notify=notifications.append,
            rebuild_session=rebuild,
        )

        await registry.execute(f"/resume {saved.session_path}", context)
        assert len(rebuilt) == 1
        assert rebuilt[0].session_id == "saved1"

        await registry.execute(f"/import {saved.session_path}", context)
        assert len(rebuilt) == 2
        assert rebuilt[1].session_id == "saved1"

        await registry.execute("/resume", context)
        assert "Saved sessions" in notifications[-1]


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
