"""自动补全 provider 栈测试。"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from types import SimpleNamespace

import pytest

from pi_tui.autocomplete import CombinedAutocompleteProvider
from pi_tui.autocomplete import (
    _build_completion_value,
    _extract_at_prefix,
    _walk_directory_with_fd,
)


@pytest.mark.asyncio
async def test_merge_and_dedupe_by_value() -> None:
    provider_a = lambda text: [  # noqa: E731
        {"value": "a", "label": "A"},
        {"value": "b"},
    ]
    provider_b = lambda text: [  # noqa: E731
        {"value": "b", "label": "B"},
        {"value": "c"},
    ]
    items = await CombinedAutocompleteProvider([provider_a, provider_b]).collect("")
    assert [item["value"] for item in items] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_async_provider() -> None:
    async def provider(text: str):
        await asyncio.sleep(0)
        return [{"value": "x", "label": "X"}]

    items = await CombinedAutocompleteProvider([provider]).collect("")
    assert items == [{"value": "x", "label": "X"}]


@pytest.mark.asyncio
async def test_exception_skipped() -> None:
    def bad_provider(text: str):
        raise RuntimeError("boom")

    good_provider = lambda text: [{"value": "ok"}]  # noqa: E731
    items = await CombinedAutocompleteProvider([bad_provider, good_provider]).collect("")
    assert items == [{"value": "ok"}]


@pytest.mark.asyncio
async def test_concurrent_collect_keeps_provider_order() -> None:
    order: list[str] = []

    async def provider_a(text: str):
        await asyncio.sleep(0.05)
        order.append("a")
        return [{"value": "a"}]

    async def provider_b(text: str):
        order.append("b")
        return [{"value": "b"}]

    items = await CombinedAutocompleteProvider([provider_a, provider_b]).collect("")
    assert order == ["b", "a"]
    assert [item["value"] for item in items] == ["a", "b"]


@pytest.mark.asyncio
async def test_empty_and_invalid_results() -> None:
    items = await CombinedAutocompleteProvider(
        [lambda text: None, lambda text: "not-a-list"]  # noqa: E731
    ).collect("")
    assert items == []


@pytest.mark.asyncio
async def test_empty_providers() -> None:
    assert await CombinedAutocompleteProvider().collect("x") == []


@pytest.mark.asyncio
async def test_command_suggestions_fuzzy() -> None:
    provider = CombinedAutocompleteProvider(
        commands=[
            SimpleNamespace(
                name="model",
                description="Select model",
                argument_hint="<provider/model>",
            ),
            SimpleNamespace(
                name="new",
                description="Start a new session",
                argument_hint=None,
            ),
        ],
        base_path="/tmp",
    )
    suggestions = await provider.get_suggestions("/mo")
    assert suggestions is not None
    assert suggestions.kind == "command"
    assert [item.value for item in suggestions.items] == ["model"]
    assert suggestions.items[0].description == "<provider/model> — Select model"


@pytest.mark.asyncio
async def test_argument_suggestions() -> None:
    provider = CombinedAutocompleteProvider(
        commands=[
            SimpleNamespace(
                name="model",
                description="Select model",
                argument_hint="<provider/model>",
                get_argument_completions=lambda prefix: [
                    {"value": "faux/faux-1", "label": "faux-1", "description": "faux"}
                ],
            )
        ],
        base_path="/tmp",
    )
    suggestions = await provider.get_suggestions("/model faux", force=True)
    assert suggestions is not None
    assert suggestions.kind == "argument"
    assert suggestions.items[0].value == "faux/faux-1"


@pytest.mark.asyncio
async def test_path_completion_directories_first(tmp_path) -> None:
    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
    (tmp_path / "beta").mkdir()
    provider = CombinedAutocompleteProvider(base_path=str(tmp_path))
    suggestions = await provider.get_suggestions("", force=True)
    assert suggestions is not None
    assert suggestions.kind == "path"
    assert suggestions.items[0].value.endswith("/")
    assert {item.label for item in suggestions.items} == {"beta/", "alpha.txt"}


@pytest.mark.asyncio
async def test_path_completion_quotes_spaces(tmp_path) -> None:
    (tmp_path / "my docs").mkdir()
    provider = CombinedAutocompleteProvider(base_path=str(tmp_path))
    suggestions = await provider.get_suggestions("my", force=True)
    assert suggestions is not None
    assert suggestions.items[0].value == '"my docs/"'


@pytest.mark.asyncio
async def test_apply_completion() -> None:
    provider = CombinedAutocompleteProvider(
        commands=[SimpleNamespace(name="model", description="", argument_hint=None)],
        base_path="/tmp",
    )
    suggestions = await provider.get_suggestions("/mo")
    assert suggestions is not None
    new_text, cursor = provider.apply_completion(
        "/mo",
        suggestions.items[0],
        suggestions.prefix,
    )
    assert new_text == "/model "
    assert cursor == len("/model ")


@pytest.mark.asyncio
async def test_extension_provider_fallback(tmp_path) -> None:
    provider = CombinedAutocompleteProvider(
        [lambda text: [{"value": "ext:item", "label": "Ext"}]],
        base_path=str(tmp_path),
    )
    suggestions = await provider.get_suggestions("hello")
    assert suggestions is not None
    assert suggestions.items[0].value == "ext:item"


def test_path_prefix_helpers() -> None:
    assert _extract_at_prefix("hello @foo") == "@foo"
    assert _extract_at_prefix('@"foo bar') == '@"foo bar'
    assert _extract_at_prefix("plain") is None
    provider = CombinedAutocompleteProvider(base_path="/tmp")
    assert provider._extract_path_prefix("src/foo", force=False) == "src/foo"
    assert provider._extract_path_prefix("plain", force=True) == "plain"
    assert provider._extract_path_prefix("plain", force=False) is None


def test_build_completion_value_quoting() -> None:
    assert (
        _build_completion_value(
            "a/b", is_directory=True, is_at_prefix=False, is_quoted_prefix=False
        )
        == "a/b"
    )
    assert (
        _build_completion_value(
            "my docs", is_directory=False, is_at_prefix=True, is_quoted_prefix=False
        )
        == '@"my docs"'
    )
    assert (
        _build_completion_value("a", is_directory=False, is_at_prefix=True, is_quoted_prefix=True)
        == '@"a"'
    )


@pytest.mark.asyncio
async def test_fd_walker_failure_returns_empty(monkeypatch) -> None:
    async def _fail(*args, **kwargs):
        raise OSError("fd missing")

    monkeypatch.setattr(
        "pi_tui.autocomplete.asyncio.create_subprocess_exec",
        _fail,
    )
    assert await _walk_directory_with_fd("/tmp", "fd", "x") == []


@pytest.mark.skipif(sys.platform == "win32", reason="posix shell script only")
@pytest.mark.asyncio
async def test_fd_walker_timeout_kills_subprocess(tmp_path) -> None:
    """卡住的 fd：超时后终止子进程并返回空结果，不冻结调用方。"""
    pidfile = tmp_path / "fd.pid"
    script = tmp_path / "slow_fd.sh"
    script.write_text(f"#!/bin/sh\necho $$ > {pidfile}\nsleep 30\n", encoding="utf-8")
    script.chmod(0o755)

    started = time.monotonic()
    entries = await _walk_directory_with_fd(str(tmp_path), str(script), "x", timeout=0.3)
    elapsed = time.monotonic() - started
    assert entries == []
    assert elapsed < 5.0, f"fd walker did not time out ({elapsed:.1f}s)"
    for _ in range(50):
        if pidfile.exists():
            break
        await asyncio.sleep(0.05)
    assert pidfile.exists()
    pid = int(pidfile.read_text().strip())
    await _wait_pid_dead(pid)


@pytest.mark.skipif(sys.platform == "win32", reason="posix shell script only")
@pytest.mark.asyncio
async def test_fd_walker_cancellation_kills_subprocess(tmp_path) -> None:
    """上层取消（abort 旧请求）时 fd 子进程被终止，CancelledError 向上传播。"""
    pidfile = tmp_path / "fd.pid"
    script = tmp_path / "slow_fd.sh"
    script.write_text(f"#!/bin/sh\necho $$ > {pidfile}\nsleep 30\n", encoding="utf-8")
    script.chmod(0o755)

    task = asyncio.create_task(_walk_directory_with_fd(str(tmp_path), str(script), "x"))
    for _ in range(50):
        if pidfile.exists():
            break
        await asyncio.sleep(0.05)
    assert pidfile.exists()
    pid = int(pidfile.read_text().strip())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _wait_pid_dead(pid)


async def _wait_pid_dead(pid: int, timeout: float = 5.0) -> None:
    """轮询等待 pid 退出（含 init 收割僵尸的时间）。仅 POSIX。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"process {pid} still alive")
