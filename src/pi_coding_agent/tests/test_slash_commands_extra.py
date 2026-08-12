"""Slash 命令补充测试：未覆盖处理器分支。"""

from __future__ import annotations

from types import SimpleNamespace

from pi_coding_agent.modes.interactive.slash_commands import (
    SlashCommandRegistry,
    SlashContext,
    register_builtin_commands,
)


def _registry() -> SlashCommandRegistry:
    registry = SlashCommandRegistry()
    register_builtin_commands(registry)
    return registry


class _Session:
    def __init__(self, cwd: str = "/tmp") -> None:
        from pi_coding_agent._session_manager import SessionManager

        self._manager = SessionManager.in_memory(cwd=cwd)
        self._cwd = cwd
        self._copy_text = "hello"
        self._stats = {"sessionId": "s1", "userMessages": 1}
        self._scoped = []
        self.extension_runner = None
        self._model = None

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def session_manager(self):
        return self._manager

    @property
    def session_id(self) -> str:
        return self._manager.session_id

    def get_last_assistant_text(self) -> str:
        return self._copy_text

    def get_session_stats(self) -> dict:
        return self._stats

    def set_scoped_models(self, scoped) -> None:
        self._scoped = list(scoped)

    @property
    def scoped_models(self):
        return list(self._scoped)

    async def set_model(self, model) -> None:
        self._model = model


async def _run(command: str, context: SlashContext) -> str | None:
    notifications: list[str] = []
    context._notify = notifications.append
    await _registry().execute(command, context)
    return notifications[-1] if notifications else ""


async def test_model_no_args_opens_selector() -> None:
    opened: list[bool] = []
    context = SlashContext(session=_Session(), open_model_selector=lambda: opened.append(True))
    assert await _run("/model", context) == ""
    assert opened == [True]


async def test_name_empty_usage() -> None:
    message = await _run("/name", SlashContext(session=_Session()))
    assert message == "Usage: /name <name>"


async def test_session_stats_with_timings_and_cache() -> None:
    session = _Session()
    session._stats = {
        "sessionId": "s1",
        "userMessages": 2,
        "assistantMessages": 3,
        "totalMessages": 5,
        "cost": 0.5,
        "turnTimings": {"turnCount": 2, "lastMs": 3, "averageMs": 4},
        "cacheStats": {"missCount": 1, "missedTokens": 20, "missedCost": 0.1},
    }
    message = await _run("/session", SlashContext(session=session))
    assert message is not None
    assert "Turns: 2" in message
    assert "Cache misses: 1" in message


async def test_copy_no_last_text() -> None:
    session = _Session()
    session._copy_text = ""
    message = await _run("/copy", SlashContext(session=session))
    assert message == "No assistant message to copy"


async def test_hotkeys_lists_bindings() -> None:
    manager = SimpleNamespace(
        all_bindings=lambda: [
            SimpleNamespace(key="ctrl+k", description="Copy"),
        ]
    )
    message = await _run("/hotkeys", SlashContext(keybindings_manager=manager))
    assert message is not None
    assert "ctrl+k" in message
    assert "Copy" in message


async def test_oauth_opens_selector_and_usage() -> None:
    opened: list[str] = []
    context = SlashContext(
        session=_Session(),
        open_oauth_selector=lambda mode: opened.append(mode),
    )
    assert await _run("/oauth", context) == ""
    assert opened == ["login"]

    message = await _run("/oauth", SlashContext(session=_Session()))
    assert message == "Usage: /oauth [login|logout]"


async def test_settings_invalid_usage() -> None:
    message = await _run("/settings nokey", SlashContext(session=_Session()))
    assert message == "Usage: /settings [key=value]"


async def test_scoped_models_opens_selector() -> None:
    opened: list[bool] = []
    context = SlashContext(
        session=_Session(),
        open_scoped_models_selector=lambda: opened.append(True),
    )
    assert await _run("/scoped-models", context) == ""
    assert opened == [True]


async def test_login_and_logout_usage() -> None:
    login = await _run("/login", SlashContext())
    assert login is not None
    assert login.startswith("Usage: /login <provider>. Available: ")
    assert await _run("/logout", SlashContext()) == "Usage: /logout <provider>"


async def test_tree_and_fork_missing_target(tmp_path) -> None:
    session = _Session(cwd=str(tmp_path))
    assert await _run("/tree missing", SlashContext(session=session)) == "Entry not found: missing"
    assert await _run("/fork missing", SlashContext(session=session)) == "Entry not found: missing"


async def test_fork_opens_selector() -> None:
    opened: list[bool] = []
    context = SlashContext(
        session=_Session(),
        open_fork_selector=lambda: opened.append(True),
    )
    assert await _run("/fork", context) == ""
    assert opened == [True]


async def test_clone_no_current_entry() -> None:
    message = await _run("/clone", SlashContext(session=_Session()))
    assert message == "No current entry to clone"


async def test_export_creates_html(tmp_path) -> None:
    from pi_ai.types import UserMessage

    session = _Session(cwd=str(tmp_path))
    await session.session_manager.append_message(UserMessage(role="user", content="export me"))
    output = tmp_path / "out.html"
    message = await _run(
        f"/export {output}",
        SlashContext(session=session),
    )
    assert message is not None
    assert output.exists()


async def test_share_missing_token(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    message = await _run("/share", SlashContext(session=_Session(cwd=str(tmp_path))))
    assert message == "No GITHUB_TOKEN environment variable set"


async def test_share_http_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "test")
    import httpx

    class _Response:
        status_code = 500

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _Client())
    message = await _run("/share", SlashContext(session=_Session(cwd=str(tmp_path))))
    assert message is not None
    assert "HTTP 500" in message
