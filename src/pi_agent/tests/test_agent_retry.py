"""Agent 内部调用重试测试（Phase 3）。

验证 _stream_assistant_response 通过 retry_assistant_call 包裹 LLM 调用：

- 可重试错误 → 自动重试并发射 auto_retry_start / auto_retry_end
- 失败尝试不污染状态（最终 messages 只有最终结果）
- 重试耗尽 → 返回最终错误
- 关闭重试 / 不可重试 / 中止 → 不重试
"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai.types import (
    AssistantMessage,
    Model,
)
from pi_ai.providers.faux import FauxCore, faux_assistant_message, faux_provider
from pi_ai.utils.retry import RetryPolicy

from pi_agent._agent_loop import run_agent_loop
from pi_agent._types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    StreamFn,
)

USER_MSG: AgentMessage = {"role": "user", "content": "hi", "timestamp": 0}


def _make_model() -> Model:
    return Model(
        id="test-model",
        provider="test",
        api="openai-completions",
        name="Test",
    )


def _make_faux(responses: list[AssistantMessage]) -> FauxCore:
    core = faux_provider()
    core.set_responses(responses)
    return core


def _llm_error(text: str) -> AssistantMessage:
    return faux_assistant_message([], stop_reason="error", error_message=text)


def _llm_ok(text: str = "ok") -> AssistantMessage:
    return faux_assistant_message(text)


def _make_config(retry_policy: RetryPolicy | None = None) -> AgentLoopConfig:
    return AgentLoopConfig(
        model=_make_model(),
        convert_to_llm=lambda messages: messages,  # type: ignore[arg-type]
        retry_policy=retry_policy,
    )


async def _run(
    stream_fn: StreamFn,
    config: AgentLoopConfig,
    *,
    signal: asyncio.Event | None = None,
) -> tuple[list[AgentMessage], list[AgentEvent]]:
    events: list[AgentEvent] = []

    async def _emit(evt: AgentEvent) -> None:
        events.append(evt)

    result = await run_agent_loop(
        prompts=[USER_MSG],
        context=AgentContext(system_prompt="", messages=[], tools=[]),
        config=config,
        emit=_emit,
        signal=signal,
        stream_fn=stream_fn,
    )
    return result, events


def _events_of(events: list[AgentEvent], event_type: str) -> list[AgentEvent]:
    return [e for e in events if e["type"] == event_type]


# ============================================================================
# 可重试错误 → 自动重试
# ============================================================================


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt() -> None:
    core = _make_faux(
        [
            _llm_error("500 Internal Server Error"),
            _llm_ok("retried ok"),
        ]
    )
    config = _make_config(RetryPolicy(max_retries=3, base_delay_ms=1, jitter=False))

    result, events = await _run(core.stream, config)

    # 初始调用 + 1 次重试
    assert core.call_count == 2
    # 最终消息为成功
    assert result[-1]["stop_reason"] == "stop"

    # 每次尝试各发射一次 assistant message_start；
    # message_end 只对最终结果发射一次（失败尝试未提交）
    assistant_starts = [
        e
        for e in events
        if e["type"] == "message_start" and e["message"].get("role") == "assistant"
    ]
    assert len(assistant_starts) == 2
    ends = [
        e for e in events if e["type"] == "message_end" and e["message"].get("role") == "assistant"
    ]
    assert len(ends) == 1
    assert ends[0]["message"]["stop_reason"] == "stop"

    # 重试事件序列
    retries = _events_of(events, "auto_retry_start")
    assert len(retries) == 1
    assert retries[0]["attempt"] == 1
    assert retries[0]["max_attempts"] == 3
    assert retries[0]["error_message"] == "500 Internal Server Error"
    retry_ends = _events_of(events, "auto_retry_end")
    assert len(retry_ends) == 1
    assert retry_ends[0]["success"] is True
    assert retry_ends[0]["final_error"] is None

    # 状态无污染：最终 transcript 只有一条 assistant 消息（失败尝试未提交）
    assistant_msgs = [m for m in result if m.get("role") == "assistant"]
    assert len(assistant_msgs) == 1


@pytest.mark.asyncio
async def test_default_policy_disabled() -> None:
    """retry_policy=None → 默认不重试（对齐 TS agent-loop）。"""
    core = _make_faux(
        [
            _llm_error("503 Service Unavailable"),
            _llm_ok("ok"),
        ]
    )
    config = _make_config()  # retry_policy=None

    result, events = await _run(core.stream, config)

    assert core.call_count == 1
    assert result[-1]["stop_reason"] == "error"
    assert not _events_of(events, "auto_retry_start")


@pytest.mark.asyncio
async def test_retry_exhausted_returns_error() -> None:
    """重试预算耗尽 → 返回最终错误，且只提交一条错误消息。"""
    core = _make_faux(
        [
            _llm_error("503 Service Unavailable"),
            _llm_error("503 Service Unavailable"),
            _llm_error("503 Service Unavailable"),
        ]
    )
    config = _make_config(RetryPolicy(max_retries=2, base_delay_ms=1, jitter=False))

    result, events = await _run(core.stream, config)

    # 初始调用 + 2 次重试
    assert core.call_count == 3
    assert result[-1]["stop_reason"] == "error"

    assert len(_events_of(events, "auto_retry_start")) == 2
    ends = [
        e for e in events if e["type"] == "message_end" and e["message"].get("role") == "assistant"
    ]
    assert len(ends) == 1
    assert ends[0]["message"]["stop_reason"] == "error"

    retry_ends = _events_of(events, "auto_retry_end")
    assert len(retry_ends) == 1
    assert retry_ends[0]["success"] is False
    assert retry_ends[0]["final_error"] == "503 Service Unavailable"


# ============================================================================
# 不重试的路径
# ============================================================================


@pytest.mark.asyncio
async def test_retry_disabled() -> None:
    core = _make_faux([_llm_error("500 Internal Server Error")])
    config = _make_config(RetryPolicy(enabled=False))

    result, events = await _run(core.stream, config)

    assert core.call_count == 1
    assert result[-1]["stop_reason"] == "error"
    assert not _events_of(events, "auto_retry_start")


@pytest.mark.asyncio
async def test_non_retryable_error_fast_fail() -> None:
    core = _make_faux([_llm_error("insufficient_quota")])
    config = _make_config()  # 默认启用，但 quota 不可重试

    result, events = await _run(core.stream, config)

    assert core.call_count == 1
    assert result[-1]["stop_reason"] == "error"
    assert not _events_of(events, "auto_retry_start")


@pytest.mark.asyncio
async def test_abort_during_stream_not_retried() -> None:
    core = _make_faux([_llm_error("500 Internal Server Error")])
    signal = asyncio.Event()

    async def _aborting_stream(model: Model, context: object, options: object):
        signal.set()  # 首次调用即中止
        return await core.stream(model, context, options)

    config = _make_config(RetryPolicy(max_retries=3, base_delay_ms=1, jitter=False))

    with pytest.raises(asyncio.CancelledError):
        await _run(_aborting_stream, config, signal=signal)

    # 中止不重试
    assert core.call_count == 1
