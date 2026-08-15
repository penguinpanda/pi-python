"""Slash 命令系统测试。"""

from __future__ import annotations


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


class TestInputCommand:
    """/input：把输入合并进历史 user 消息并继续。"""

    async def test_opens_selector_with_text(self):
        registry = _make_registry()
        opened: list[str | None] = []
        context = SlashContext(open_input_selector=opened.append)
        assert await registry.execute("/input 请改用Python", context) is True
        assert opened == ["请改用Python"]

    async def test_opens_selector_without_text(self):
        registry = _make_registry()
        opened: list[str | None] = []
        context = SlashContext(open_input_selector=opened.append)
        assert await registry.execute("/input", context) is True
        assert opened == [None]

    async def test_not_available_outside_tui(self):
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(notify=notifications.append)
        assert await registry.execute("/input 文本", context) is True
        assert "TUI" in notifications[0]


class FakeSession:
    """极简 session 桩（Phase 7 命令用）。"""

    def __init__(self, cwd: str = "/tmp", manager=None) -> None:
        from pi_coding_agent._session_manager import SessionManager

        self._manager = manager or SessionManager.in_memory(cwd=cwd)
        self._cwd = cwd
        self._name = None
        self._model = None
        self._thinking_level = "off"
        self._compacted = False
        self._scoped: list = []
        self.extension_runner = None

    def set_session_name(self, name):
        self._name = name

    @property
    def session_name(self):
        return self._name

    async def set_model(self, model):
        self._model = model

    def set_thinking_level(self, level):
        self._thinking_level = level

    @property
    def thinking_level(self):
        return self._thinking_level

    def get_available_thinking_levels(self):
        return ["off", "low", "medium", "high"]

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

    async def test_settings_no_args_opens_selector_in_tui(self, tmp_path):
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        opened: list[bool] = []
        context = SlashContext(
            session=session,
            notify=lambda _message: None,
            open_settings_selector=lambda: opened.append(True),
        )

        await registry.execute("/settings", context)
        assert opened == [True]

    async def test_settings_preserves_existing_keys(self, tmp_path):
        """回归（CF-04）：/settings 顶层合并，写入新键不覆盖已有键。"""
        import json

        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        project_path = tmp_path / ".pi" / "settings.json"
        project_path.parent.mkdir(parents=True)
        project_path.write_text(
            json.dumps({"keybindings": {"app.model.select": "ctrl+0"}}),
            encoding="utf-8",
        )

        await registry.execute("/settings defaultModel=qwen-plus", context)

        data = json.loads(project_path.read_text(encoding="utf-8"))
        assert data["defaultModel"] == "qwen-plus"
        assert data["keybindings"] == {"app.model.select": "ctrl+0"}

    async def test_trust_save_clear_and_status(self, tmp_path):
        from pi_coding_agent.trust import TrustManager

        manager = TrustManager(tmp_path / "trust.json")
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(
            session=session,
            notify=notifications.append,
            trust_manager=manager,
        )

        await registry.execute("/trust trust", context)
        assert manager.is_trusted(str(tmp_path)) is True

        await registry.execute("/trust", context)
        assert "Project trust: trusted" in notifications[-1]

        await registry.execute("/trust block", context)
        assert manager.is_trusted(str(tmp_path)) is False

        await registry.execute("/trust unset", context)
        assert manager.is_trusted(str(tmp_path)) is None

    async def test_trust_opens_selector_in_tui(self, tmp_path):
        from pi_coding_agent.trust import TrustManager

        manager = TrustManager(tmp_path / "trust.json")
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        opened: list[str] = []
        context = SlashContext(
            session=session,
            notify=notifications.append,
            trust_manager=manager,
            open_trust_selector=lambda: opened.append("opened"),
        )

        await registry.execute("/trust", context)
        assert opened == ["opened"]

    async def test_trust_unavailable_without_manager(self, tmp_path):
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        await registry.execute("/trust trust", context)
        assert "not available" in notifications[-1]

    async def test_trust_usage(self, tmp_path):
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        await registry.execute("/trust bogus", context)
        assert "Usage: /trust" in notifications[-1]

    async def test_thinking_command(self, tmp_path):
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        await registry.execute("/thinking low", context)
        assert session.thinking_level == "low"
        assert "Thinking level: low" in notifications[-1]

    async def test_thinking_opens_selector_in_tui(self, tmp_path):
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        opened: list[bool] = []
        context = SlashContext(
            session=session,
            notify=lambda _message: None,
            open_thinking_selector=lambda: opened.append(True),
        )
        await registry.execute("/thinking", context)
        assert opened == [True]

    async def test_extensions_lists_runner(self, tmp_path):
        from pi_coding_agent.extensions.types import Extension

        session = FakeSession(cwd=str(tmp_path))
        extension = Extension(path="/tmp/ext.py", resolved_path="/tmp/ext.py")
        session.extension_runner = type("Runner", (), {"extensions": [extension]})()
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        await registry.execute("/extensions", context)
        assert "ext.py" in notifications[-1]

    async def test_extensions_opens_selector_in_tui(self, tmp_path):
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        opened: list[bool] = []
        context = SlashContext(
            session=session,
            notify=lambda _message: None,
            open_extensions_selector=lambda: opened.append(True),
        )
        await registry.execute("/extensions", context)
        assert opened == [True]

    async def test_changelog_renders_entries(self, tmp_path):
        (tmp_path / "CHANGELOG.md").write_text(
            "## [0.2.0]\n\n### Added\n\n- thing\n\n## [0.1.0]\n\n### Fixed\n\n- bug\n",
            encoding="utf-8",
        )
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        await registry.execute("/changelog", context)
        assert "0.2.0" in notifications[-1]
        assert "thing" in notifications[-1]

    async def test_changelog_version_filter(self, tmp_path):
        (tmp_path / "CHANGELOG.md").write_text(
            "## [0.2.0]\n\n### Added\n\n- thing\n\n## [0.1.0]\n\n### Fixed\n\n- bug\n",
            encoding="utf-8",
        )
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        await registry.execute("/changelog 0.1.0", context)
        assert "0.2.0" in notifications[-1]
        assert "bug" not in notifications[-1]

    async def test_changelog_missing(self, tmp_path):
        session = FakeSession(cwd=str(tmp_path))
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(session=session, notify=notifications.append)

        await registry.execute("/changelog", context)
        assert "No changelog found" in notifications[-1]

    async def test_login_unknown_provider_friendly_error(self):
        """回归（P13）：/login 未知 provider 返回可读错误，不抛 traceback。"""
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(notify=notifications.append)

        handled = await registry.execute("/login bogus", context)

        assert handled is True
        assert "Unknown provider: bogus" in notifications[0]
        assert "Traceback" not in notifications[0]

    async def test_scoped_models(self, tmp_path):
        from pi_ai import Model, Models
        from pi_ai.providers.faux import faux_provider

        from pi_coding_agent.auth_storage import AuthStorage
        from pi_coding_agent.model_runtime import ModelRuntime

        models = Models(credentials=AuthStorage.in_memory())
        core = faux_provider(
            models=[
                Model(id="faux-1", provider="faux", api="openai-completions"),
                Model(id="faux-2", provider="faux", api="openai-completions"),
            ]
        )
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

    async def test_import_and_resume(self, tmp_path, monkeypatch):
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

        # /resume 使用默认会话目录；测试目录并非默认目录，因此指向 saved
        # 所在目录，保证列表断言确定性。
        import pi_coding_agent._config as config

        monkeypatch.setattr(config, "get_sessions_dir", lambda: str(tmp_path / "sessions"))
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
        from pi_ai import Models
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


class TestReloadCommand:
    async def test_reload_invokes_host_callback(self):
        registry = _make_registry()
        notifications: list[str] = []
        reloaded: list[str] = []

        async def reload_all() -> str:
            reloaded.append("called")
            return "Reloaded: 1 skills; keybindings refreshed"

        context = SlashContext(
            notify=notifications.append,
            reload_all=reload_all,
        )
        await registry.execute("/reload", context)
        assert reloaded == ["called"]
        assert notifications[-1] == "Reloaded: 1 skills; keybindings refreshed"

    async def test_reload_unavailable_without_host(self):
        registry = _make_registry()
        notifications: list[str] = []
        context = SlashContext(notify=notifications.append)
        await registry.execute("/reload", context)
        assert "not available" in notifications[-1]


def test_format_tree_entry_renderer():
    from pi_coding_agent._session_manager import SessionTreeNode
    from pi_coding_agent.modes.interactive.slash_commands import _format_tree

    node = SessionTreeNode(
        id="abc123",
        parent_id=None,
        entry={"type": "custom", "customType": "status-card", "data": {}},
    )
    lines = _format_tree(
        [node],
        None,
        entry_renderer=lambda custom_type, entry, state: f"CARD:{custom_type}",
    )
    assert "CARD:status-card" in lines[0]
    # 无渲染器时回退默认 id/type 行。
    default_lines = _format_tree([node], None)
    assert "abc123" in default_lines[0]


class TestAutocompleteOptions:
    def test_menu_matches_ts_builtins(self):
        from pi_coding_agent.modes.interactive.autocomplete import (
            create_slash_command_provider,
        )

        provider = create_slash_command_provider(_make_registry())
        items = provider("/")
        names = {str(item["value"]).strip().lstrip("/") for item in items}

        # TS BUILTIN_SLASH_COMMANDS 全量出现。
        for name in (
            "settings",
            "model",
            "scoped-models",
            "export",
            "import",
            "share",
            "copy",
            "name",
            "session",
            "changelog",
            "hotkeys",
            "fork",
            "clone",
            "tree",
            "trust",
            "login",
            "logout",
            "new",
            "compact",
            "resume",
            "reload",
            "quit",
        ):
            assert name in names, name

        # Python 独有命令不进入补全菜单（仍可手动输入执行）。
        for name in (
            "thinking",
            "oauth",
            "extensions",
            "help",
            "input",
            "debug",
            "arminsayshi",
            "dementedelves",
        ):
            assert name not in names, name

    def test_provider_filters_by_prefix(self):
        from pi_coding_agent.modes.interactive.autocomplete import (
            create_slash_command_provider,
        )

        provider = create_slash_command_provider(_make_registry())
        items = provider("/mo")
        names = [str(item["value"]).strip().lstrip("/") for item in items]
        assert names == ["model"]
