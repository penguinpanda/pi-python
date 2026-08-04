"""
12 事件协议测试。

验证（对应 TS AssistantMessageEventStream 协议）：
1. AssistantMessageEvent 联合恰好包含 12 种判别事件。
2. 每个增量事件都携带 partial 快照（done/error 除外）。
3. 流式过程中 partial 反映当前累积状态（快照一致性）。
4. toolcall_end 携带已解析 arguments 的 ToolCall。
"""

import typing
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pi_ai._types import AssistantMessageEvent, Context, Model
from pi_ai.api.completions import chat_completions_stream


# ---------------------------------------------------------------------------
# 事件联合结构
# ---------------------------------------------------------------------------

EXPECTED_EVENT_TYPES = {
    "start",
    "text_start",
    "text_delta",
    "text_end",
    "thinking_start",
    "thinking_delta",
    "thinking_end",
    "toolcall_start",
    "toolcall_delta",
    "toolcall_end",
    "done",
    "error",
}


def test_twelve_event_types():
    """AssistantMessageEvent 联合恰好 12 种事件。"""
    members = typing.get_args(AssistantMessageEvent)
    assert len(members) == 12

    actual: set[str] = set()
    for m in members:
        type_ann = m.__annotations__["type"]
        actual.update(typing.get_args(type_ann))
    assert actual == EXPECTED_EVENT_TYPES


def test_incremental_events_carry_partial():
    """除 done/error 外，每种事件都携带 partial: AssistantMessage。"""
    from pi_ai._types import DoneEvent, ErrorEvent

    for m in typing.get_args(AssistantMessageEvent):
        if m in (DoneEvent, ErrorEvent):
            assert "partial" not in m.__annotations__
        else:
            assert "partial" in m.__annotations__


# ---------------------------------------------------------------------------
# completions 状态机（mock chunks）
# ---------------------------------------------------------------------------


def _make_model() -> Model:
    return Model(
        id="m",
        provider="p",
        api="openai-completions",
        name="m",
        input=["text"],
        output=["text"],
    )


def _chunk(content=None, tool_calls=None, finish_reason=None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _tool_call(
    index: int, id_: str | None, name: str | None, arguments: str | None
) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        id=id_,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


async def _collect(chunks: list[SimpleNamespace]) -> list:
    async def _gen():
        for c in chunks:
            yield c

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_gen())
    with patch("pi_ai.api.completions._create_client", return_value=client):
        stream = await chat_completions_stream(
            _make_model(), Context(messages=[{"role": "user", "content": "Hi"}]), "sk", "https://x"
        )
        return [e async for e in stream]


@pytest.mark.asyncio
async def test_text_stream_partial_snapshot_consistency():
    """text_delta 的 partial 反映当前累积文本（快照一致性）。"""
    events = await _collect(
        [
            _chunk(content="Hel"),
            _chunk(content="lo", finish_reason="stop"),
        ]
    )

    types = [e["type"] for e in events]
    assert types == ["start", "text_start", "text_delta", "text_delta", "text_end", "done"]

    deltas = [e for e in events if e["type"] == "text_delta"]
    assert deltas[0]["delta"] == "Hel"
    assert deltas[0]["partial"]["content"] == [{"type": "text", "text": "Hel"}]
    assert deltas[1]["delta"] == "lo"
    assert deltas[1]["partial"]["content"] == [{"type": "text", "text": "Hello"}]

    # 增量事件的 partial 与最终 done message 一致（无 toolCall 时）。
    text_end = events[-2]
    assert text_end["type"] == "text_end"
    assert text_end["content"] == "Hello"
    assert events[-1]["message"]["content"] == [{"type": "text", "text": "Hello"}]


@pytest.mark.asyncio
async def test_toolcall_end_carries_parsed_toolcall():
    """toolcall_end 携带已解析 arguments 的 ToolCall。"""
    events = await _collect(
        [
            _chunk(
                tool_calls=[_tool_call(0, "call_1", "get_weather", '{"city":')],
                finish_reason=None,
            ),
            _chunk(
                tool_calls=[_tool_call(0, None, None, '"Beijing"}')],
                finish_reason="tool_calls",
            ),
        ]
    )

    types = [e["type"] for e in events]
    assert types == [
        "start",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_delta",
        "toolcall_end",
        "done",
    ]

    end = [e for e in events if e["type"] == "toolcall_end"][0]
    assert end["tool_call"] == {
        "type": "toolCall",
        "id": "call_1",
        "name": "get_weather",
        "raw_arguments": '{"city":"Beijing"}',
        "arguments": {"city": "Beijing"},
    }
    # partial 中的 toolCall 块也已完成解析。
    assert end["partial"]["content"] == [
        {
            "type": "toolCall",
            "id": "call_1",
            "name": "get_weather",
            "raw_arguments": '{"city":"Beijing"}',
            "arguments": {"city": "Beijing"},
        }
    ]

    # 最终 done message 的 content 与 toolcall_end 一致。
    assert events[-1]["message"]["content"] == end["partial"]["content"]


@pytest.mark.asyncio
async def test_invalid_tool_arguments_error_placeholder():
    """arguments JSON 解析失败时使用错误占位（不崩溃）。"""
    events = await _collect(
        [
            _chunk(
                tool_calls=[_tool_call(0, "call_1", "fn", '{"bad":')],
                finish_reason="tool_calls",
            ),
        ]
    )

    end = [e for e in events if e["type"] == "toolcall_end"][0]
    assert end["tool_call"]["arguments"] == {"_error": "Invalid JSON arguments"}
