"""AWS Bedrock ConverseStream 测试。"""

from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from typing import Any, cast

import httpx
import pytest

from pi_ai.api import bedrock_converse_stream
from pi_ai.api.api_provider_registry import get_api_provider
from pi_ai.api.bedrock_converse_stream import (
    _aws_sigv4_headers,
    _encode_json_payload,
    _resolve_bedrock_credentials,
    _to_bedrock_messages,
    _build_thinking_fields,
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


def test_tool_results_merged_and_images_converted() -> None:
    """连续 toolResult 合并为单条 user 消息；image 内容转换为 Bedrock 格式。"""
    context = Context(
        messages=cast(
            Any,
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc-1",
                            "name": "bash",
                            "arguments": {"command": "ls"},
                        }
                    ],
                },
                {
                    "role": "toolResult",
                    "tool_call_id": "tc-1",
                    "content": [{"type": "text", "text": "ok"}],
                },
                {
                    "role": "toolResult",
                    "tool_call_id": "tc-1",
                    "content": [
                        {
                            "type": "image",
                            "mime_type": "image/png",
                            "data": b"\x89PNG",
                        }
                    ],
                },
            ],
        ),
    )
    messages = _to_bedrock_messages(context)
    assert len(messages) == 2
    tool_message = messages[1]
    assert tool_message["role"] == "user"
    blocks = tool_message["content"]
    assert len(blocks) == 2
    assert blocks[0]["toolResult"]["content"] == [{"text": "ok"}]
    assert blocks[1]["toolResult"]["content"][0]["image"] == {
        "format": "png",
        "source": {"bytes": b"\x89PNG"},
    }


def test_empty_tool_result_uses_placeholder() -> None:
    context = Context(
        messages=cast(
            Any,
            [
                {"role": "toolResult", "tool_call_id": "tc-1", "content": []},
            ],
        )
    )
    messages = _to_bedrock_messages(context)
    assert messages[0]["content"][0]["toolResult"]["content"] == [{"text": "<empty>"}]


def test_supports_prompt_caching() -> None:
    from pi_ai.api.bedrock_converse_stream import _supports_prompt_caching

    claude4 = Model(
        id="anthropic.claude-sonnet-4-20250514",
        provider="amazon-bedrock",
        api="bedrock-converse-stream",
    )
    assert _supports_prompt_caching(claude4, None)
    claude37 = Model(id="anthropic.claude-3-7-sonnet", provider="p", api="a")
    assert _supports_prompt_caching(claude37, None)
    other = Model(id="arn:aws:bedrock:us-east-1:123:inference-profile/x", provider="p", api="a")
    assert not _supports_prompt_caching(other, None)
    assert _supports_prompt_caching(other, {"AWS_BEDROCK_FORCE_CACHE": "1"})


@pytest.mark.asyncio
async def test_cache_point_added_to_last_user_message(monkeypatch) -> None:
    """cacheRetention 非 none 且模型支持时，最后一条 user 消息追加 cachePoint。"""
    events = [
        _frame({"type": "messageStart"}, "messageStart"),
        _frame(
            {"type": "contentBlockDelta", "delta": {"text": "ok"}},
            "contentBlockDelta",
        ),
        _frame({"type": "contentBlockStop", "contentBlockIndex": 0}, "contentBlockStop"),
        _frame({"type": "messageStop", "stopReason": "end_turn"}, "messageStop"),
    ]
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, content=b"".join(events), headers={"content-type": "application/json"}
        )

    monkeypatch.setattr(
        bedrock_converse_stream,
        "_AsyncClient",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs),
    )
    stream = bedrock_converse_stream.bedrock_converse_stream(
        _model(),
        _context(),
        "token",
        "",
        cast(Any, {"region": "us-west-2", "cache_retention": "long"}),
    )
    async for _event in stream:
        pass
    blocks = captured["payload"]["messages"][-1]["content"]
    assert blocks[-1]["cachePoint"] == {"type": "default", "ttl": 3600}


def test_thinking_fields() -> None:
    model = Model(
        id="anthropic.claude-sonnet-4-20250514",
        provider="amazon-bedrock",
        api="bedrock-converse-stream",
        reasoning=True,
    )
    assert _build_thinking_fields(model, {"reasoning": "high"}) == {
        "thinking": {"type": "enabled", "budget_tokens": 16384}
    }
    assert _build_thinking_fields(model, {"reasoning": "max"}) == {
        "thinking": {"type": "enabled", "budget_tokens": 16384}
    }
    assert (
        _build_thinking_fields(
            model, {"reasoning": "medium", "thinking_budgets": {"medium": 4000}}
        )["thinking"]["budget_tokens"]
        == 4000
    )
    assert (
        _build_thinking_fields(
            Model(id="m", provider="p", api="a", reasoning=False), {"reasoning": "high"}
        )
        is None
    )
    # reasoning="off" 必须禁用 thinking，不得生成 budget_tokens=0 的非法配置。
    assert _build_thinking_fields(model, {"reasoning": "off"}) is None


def test_bedrock_api_registered() -> None:
    assert get_api_provider("bedrock-converse-stream") is not None


def test_parse_eventstream_messages() -> None:
    frame = _frame({"type": "contentBlockDelta"}, "contentBlockDelta")
    messages, remainder = parse_eventstream_messages(frame)
    assert remainder == b""
    assert messages[0]["_event_type"] == "contentBlockDelta"
    assert messages[0]["type"] == "contentBlockDelta"


def test_encode_json_payload_matches_httpx_wire_format() -> None:
    """SigV4 的 payload 字节必须与 httpx json= 的序列化完全一致。"""
    import json as _json
    from httpx import _content

    payload = {"modelId": "x", "messages": [{"role": "user", "content": [{"text": "中文"}]}]}
    encoded = _encode_json_payload(payload)
    _, stream = _content.encode_json(payload)
    assert encoded == b"".join(stream)
    assert encoded != _json.dumps(payload).encode("utf-8")


def test_bedrock_provider_accepts_ambient_aws_credentials(monkeypatch) -> None:
    from pi_ai.providers.amazon_bedrock import amazon_bedrock_provider

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET")
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    provider = amazon_bedrock_provider()
    resolved = provider.auth.resolve(None)
    assert resolved is not None
    assert resolved.api_key == ""
    assert resolved.source == "AWS access keys"


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


@pytest.mark.asyncio
async def test_bedrock_parallel_tool_stream(monkeypatch) -> None:
    """并行 toolUse 按 contentBlockIndex 独立累积参数，交错增量不串线。"""
    events = [
        _frame({"type": "messageStart"}, "messageStart"),
        _frame(
            {
                "type": "contentBlockStart",
                "contentBlockIndex": 0,
                "start": {"toolUse": {"toolUseId": "t1", "name": "read"}},
            },
            "contentBlockStart",
        ),
        _frame(
            {
                "type": "contentBlockStart",
                "contentBlockIndex": 1,
                "start": {"toolUse": {"toolUseId": "t2", "name": "search"}},
            },
            "contentBlockStart",
        ),
        _frame(
            {
                "type": "contentBlockDelta",
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": '{"path":"'}},
            },
            "contentBlockDelta",
        ),
        _frame(
            {
                "type": "contentBlockDelta",
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"q":"x"'}},
            },
            "contentBlockDelta",
        ),
        _frame(
            {
                "type": "contentBlockDelta",
                "contentBlockIndex": 0,
                "delta": {"toolUse": {"input": 'a"}'}},
            },
            "contentBlockDelta",
        ),
        _frame(
            {
                "type": "contentBlockDelta",
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "}"}},
            },
            "contentBlockDelta",
        ),
        _frame({"type": "contentBlockStop", "contentBlockIndex": 0}, "contentBlockStop"),
        _frame({"type": "contentBlockStop", "contentBlockIndex": 1}, "contentBlockStop"),
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
    blocks = {b["id"]: b for b in message["content"] if b["type"] == "toolCall"}
    assert blocks["t1"]["raw_arguments"] == '{"path":"a"}'
    assert blocks["t1"]["arguments"] == {"path": "a"}
    assert blocks["t2"]["raw_arguments"] == '{"q":"x"}'
    assert blocks["t2"]["arguments"] == {"q": "x"}
