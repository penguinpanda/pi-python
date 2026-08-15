"""llama.cpp 内置扩展测试。"""

from __future__ import annotations

import inspect

import pytest

from pi_coding_agent.extensions import Extension, ExtensionAPI, ExtensionRuntime
from pi_coding_agent.extensions.builtin_llama import LLAMA_PROVIDER_ID, create_extension


def _make_builtin() -> Extension:
    runtime = ExtensionRuntime()
    extension = Extension(
        path="<builtin>/llama",
        resolved_path="<builtin>/llama",
        source="builtin",
        hidden=True,
    )
    create_extension(ExtensionAPI(extension, runtime, cwd="."))
    return extension


def test_builtin_llama_registers_provider_and_command() -> None:
    extension = _make_builtin()

    assert extension.providers[0][0] == LLAMA_PROVIDER_ID
    provider_config = extension.providers[0][1]
    assert provider_config["api"] == "openai-completions"
    assert provider_config["models"][0]["id"] == "llama3"
    assert "llama" in extension.commands


@pytest.mark.asyncio
async def test_builtin_llama_command_handler_argument_order(monkeypatch) -> None:
    """回归：/llama handler 必须按 (ctx, args) 顺序接收参数。"""
    extension = _make_builtin()
    handler = extension.commands["llama"].handler
    assert handler is not None

    notifications: list[tuple[str, str]] = []

    class _UI:
        def notify(self, message, notify_type=None):
            notifications.append((message, notify_type))

    class _Ctx:
        mode = "tui"
        ui = _UI()

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "m1"}, {"id": "m2"}]}

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeClient)
    result = handler(_Ctx(), "")
    if inspect.isawaitable(result):
        await result
    assert notifications
    assert "m1" in notifications[0][0] and "m2" in notifications[0][0]
