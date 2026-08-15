"""Regression tests for TS-parity fixes discovered during parity review."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pi_ai import Model
from pi_coding_agent._json_event import to_json_event
from pi_coding_agent._session_manager_v4 import in_memory_session_manager
from pi_coding_agent.file_processor import process_file_arguments
from pi_coding_agent.model_resolver import find_initial_model
from pi_coding_agent.rpc.rpc_mode import RpcMessageHandler


class _FakeRuntime:
    def __init__(self) -> None:
        self.models = [
            Model(id="default", provider="foo", api="openai-completions"),
            Model(id="requested", provider="bar", api="openai-completions"),
        ]

    def get_models(self):
        return list(self.models)

    def get_model(self, provider: str, model_id: str):
        return next(
            (m for m in self.models if m.provider == provider and m.id == model_id),
            None,
        )

    def has_configured_auth(self, _provider: str) -> bool:
        return True

    async def get_available(self):
        return list(self.models)

    def get_available_snapshot(self):
        return list(self.models)


@pytest.mark.asyncio
async def test_initial_model_resolves_without_provider() -> None:
    """--model <id> must work without --provider (TS buildSessionOptions)."""
    result = await find_initial_model(
        cli_provider=None,
        cli_model="requested",
        scoped_models=[],
        is_continuing=False,
        model_runtime=_FakeRuntime(),
    )
    assert result.model is not None
    assert (result.model.provider, result.model.id) == ("bar", "requested")


def test_json_event_uses_ts_camel_case_wire_keys() -> None:
    event = to_json_event(
        {
            "type": "tool_execution_start",
            "tool_call_id": "call-1",
            "tool_name": "bash",
            "args": {},
        }
    )
    assert event == {
        "type": "tool_execution_start",
        "toolCallId": "call-1",
        "toolName": "bash",
        "args": {},
    }


def test_json_message_update_strips_partial_snapshot() -> None:
    event = to_json_event(
        {
            "type": "message_update",
            "message": {"role": "assistant"},
            "assistant_message_event": {
                "type": "text_delta",
                "partial": {"huge": "snapshot"},
                "text": "delta",
            },
        }
    )
    assistant = event["assistantMessageEvent"]
    assert "partial" not in assistant
    assert assistant["text"] == "delta"


@pytest.mark.asyncio
async def test_synthesized_session_header_matches_ts_v3_contract() -> None:
    manager = await in_memory_session_manager("/tmp/pi-parity-test")
    header = manager.get_header()
    assert header is not None
    assert header["type"] == "session"
    assert header["version"] == 3
    assert header["cwd"] == "/tmp/pi-parity-test"
    assert isinstance(header["timestamp"], str)
    assert header["timestamp"].endswith("Z")


@pytest.mark.asyncio
async def test_process_file_arguments_matches_ts_wrapping_and_missing_errors(
    tmp_path: Path,
) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    text, images = await process_file_arguments(["note.txt"], str(tmp_path))
    assert text == f'<file name="{tmp_path / "note.txt"}">\nhello\n</file>\n'
    assert images == []

    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    text, images = await process_file_arguments(["empty.txt"], str(tmp_path))
    assert text == ""
    assert images == []

    with pytest.raises(FileNotFoundError):
        await process_file_arguments(["missing.txt"], str(tmp_path))


@pytest.mark.asyncio
async def test_rpc_prompt_passes_streaming_behavior_while_streaming() -> None:
    class StreamingSession:
        is_streaming = True
        captured: dict = {}

        async def prompt(self, *args, **kwargs):
            self.captured = kwargs

    session = StreamingSession()
    handler = RpcMessageHandler(session, object())
    response = await handler.handle_command(
        {"id": "1", "type": "prompt", "message": "hi", "streamingBehavior": "steer"}
    )
    assert response is None
    for _ in range(100):
        if session.captured:
            break
        await asyncio.sleep(0)
    assert session.captured["streaming_behavior"] == "steer"
    assert session.captured["source"] == "rpc"


@pytest.mark.asyncio
async def test_build_initial_message_merges_stdin_file_and_first_message(
    tmp_path: Path,
) -> None:
    """print 模式按 TS buildInitialMessage 合并 stdin/@file/首条消息。"""
    from types import SimpleNamespace

    from pi_coding_agent._cli import _build_initial_message

    (tmp_path / "ctx.txt").write_text("file context\n", encoding="utf-8")
    parsed = SimpleNamespace(message=["@ctx.txt", "Explain it", "Second message"])
    initial, images, remaining = await _build_initial_message(
        parsed, str(tmp_path), stdin_text="stdin\n"
    )
    assert initial == (
        f'stdin\n<file name="{tmp_path / "ctx.txt"}">\nfile context\n\n</file>\nExplain it'
    )
    assert images is None
    assert remaining == ["Second message"]


def test_parse_args_exposes_unknown_flags_and_file_args() -> None:
    from pi_coding_agent import parseArgs

    parsed = parseArgs(["--plan", "-p", "hi", "@ctx.txt"])
    assert parsed.unknown_flags == {"plan": True}
    assert parsed.messages == ["hi"]
    assert parsed.file_args == ["ctx.txt"]


def test_tool_definition_and_truncation_exports(tmp_path: Path) -> None:
    from pi_coding_agent import (
        create_bash_tool_definition,
        format_size,
        truncate_head,
    )

    definition = create_bash_tool_definition(str(tmp_path))
    assert definition["name"] == "bash"
    assert callable(definition["execute"])
    assert format_size(2048) == "2.0KB"
    result = truncate_head("a\nb\nc", max_lines=1, max_bytes=1024)
    assert result.truncated is True
    assert result.content == "a"


@pytest.mark.asyncio
async def test_resume_falls_back_to_global_sessions(monkeypatch, tmp_path: Path) -> None:
    from pi_coding_agent import _cli
    from pi_coding_agent._session_manager import SessionInfo

    info = SessionInfo(
        path=str(tmp_path / "other.jsonl"),
        session_id="other-1",
        cwd="/other/project",
        modified=1.0,
    )
    calls = {"cwd": None}

    async def fake_list(directory, cwd=None):
        calls["cwd"] = cwd
        return [] if cwd is not None else [info]

    monkeypatch.setattr(_cli, "list_sessions", fake_list)
    monkeypatch.setattr(_cli, "get_sessions_dir", lambda: tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt: "1")

    opened = {}

    async def fake_open(path, cwd_override=None):
        opened["path"] = path
        opened["cwd_override"] = cwd_override
        return object()

    monkeypatch.setattr(_cli, "open_session_manager", fake_open)
    manager = await _cli._pick_session_to_resume(str(tmp_path), None)
    assert manager is not None
    assert opened["path"] == info.path
