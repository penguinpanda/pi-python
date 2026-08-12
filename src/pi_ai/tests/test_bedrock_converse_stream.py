"""AWS Bedrock ConverseStream 测试。"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timezone

import httpx
import pytest

from pi_ai.api import bedrock_converse_stream
from pi_ai.api.api_provider_registry import get_api_provider
from pi_ai.api.bedrock_converse_stream import (
    _aws_sigv4_headers,
    _resolve_bedrock_credentials,
    parse_eventstream_messages,
)
from pi_ai._types import Context, Model


def _model() -> Model:
    return Model(
        id="anthropic.claude-sonnet-4-20250514",
        provider="amazon-bedrock",
        api="bedrock-converse-stream",
    )


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "hi"}])


def _frame(payload: dict, event_type: str) -> bytes:
    body = json.dumps(payload).encode()
    name = b":event-type"
    header = (
        bytes([len(name)])
        + name
        + bytes([7])
        + struct.pack(">H", len(event_type))
        + event_type.encode()
    )
    total = 16 + len(header) + len(body)
    return struct.pack(">II", total, len(header)) + b"\0\0\0\0" + header + body + b"\0\0\0\0"


def test_bedrock_api_registered() -> None:
    assert get_api_provider("bedrock-converse-stream") is not None


def test_parse_eventstream_messages() -> None:
    frame = _frame({"type": "contentBlockDelta"}, "contentBlockDelta")
    messages, remainder = parse_eventstream_messages(frame)
    assert remainder == b""
    assert messages[0]["_event_type"] == "contentBlockDelta"
    assert messages[0]["type"] == "contentBlockDelta"


def test_aws_sigv4_headers() -> None:
    headers = _aws_sigv4_headers(
        method="POST",
        url="https://bedrock-runtime.us-east-1.amazonaws.com/model/m/converse-stream",
        payload=b"{}",
        region="us-east-1",
        access_key="AKID",
        secret_key="SECRET",
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert headers["x-amz-content-sha256"] == (
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


def test_resolve_bedrock_credentials_from_shared_file(tmp_path) -> None:
    credentials = tmp_path / "credentials"
    credentials.write_text(
        "[default]\n"
        "aws_access_key_id = AKID\n"
        "aws_secret_access_key = SECRET\n"
        "aws_session_token = TOKEN\n",
        encoding="utf-8",
    )
    access_key, secret_key, session_token = _resolve_bedrock_credentials(
        {"aws_shared_credentials_file": str(credentials)}
    )
    assert access_key == "AKID"
    assert secret_key == "SECRET"
    assert session_token == "TOKEN"


@pytest.mark.asyncio
async def test_bedrock_text_stream(monkeypatch) -> None:
    events = [
        _frame({"type": "messageStart"}, "messageStart"),
        _frame(
            {"type": "contentBlockDelta", "delta": {"text": "Hello"}},
            "contentBlockDelta",
        ),
        _frame({"type": "contentBlockStop", "contentBlockIndex": 0}, "contentBlockStop"),
        _frame({"type": "messageStop", "stopReason": "end_turn"}, "messageStop"),
        _frame(
            {"type": "metadata", "usage": {"inputTokens": 5, "outputTokens": 2}},
            "metadata",
        ),
    ]
    body = b"".join(events)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    monkeypatch.setattr(
        bedrock_converse_stream,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    stream = bedrock_converse_stream.bedrock_converse_stream(
        _model(), _context(), "token", "", {"region": "us-west-2"}
    )
    events_out = [event async for event in stream]
    message = events_out[-1]["message"]
    assert message["content"][0]["type"] == "text"
    assert message["content"][0]["text"] == "Hello"
    assert message["stop_reason"] == "stop"
    assert message["usage"]["input"] == 5


@pytest.mark.asyncio
async def test_bedrock_tool_stream(monkeypatch) -> None:
    events = [
        _frame({"type": "messageStart"}, "messageStart"),
        _frame(
            {
                "type": "contentBlockStart",
                "start": {"toolUse": {"toolUseId": "t1", "name": "read"}},
            },
            "contentBlockStart",
        ),
        _frame(
            {"type": "contentBlockDelta", "delta": {"toolUse": {"input": '{"path":"a"}'}}},
            "contentBlockDelta",
        ),
        _frame({"type": "contentBlockStop", "contentBlockIndex": 0}, "contentBlockStop"),
        _frame({"type": "messageStop", "stopReason": "tool_use"}, "messageStop"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"".join(events))

    monkeypatch.setattr(
        bedrock_converse_stream,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    stream = bedrock_converse_stream.bedrock_converse_stream(
        _model(), _context(), "token", "", {"region": "us-east-1"}
    )
    events_out = [event async for event in stream]
    message = events_out[-1]["message"]
    tool = message["content"][0]
    assert tool["type"] == "toolCall"
    assert tool["name"] == "read"
    assert tool["arguments"] == {"path": "a"}
    assert message["stop_reason"] == "tool_call"
