"""AgentHarness 配置、队列与 hook 错误路径补充测试。"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai.types import UserMessage

from pi_agent import _harness as harness_module
from pi_agent._harness_types import (
    AgentHarnessError,
    AgentHarnessStreamOptionsPatch,
    BeforeProviderPayloadResult,
    BeforeProviderRequestResult,
)

from test_harness import _make_harness, _make_model, _make_harness_tool


@pytest.mark.asyncio
async def test_configuration_getters_and_setters() -> None:
    harness = _make_harness()
    assert harness.get_model().id == "test-model"

    next_model = _make_model("next")
    await harness.set_model(next_model)
    assert harness.get_model() is next_model

    assert harness.get_thinking_level() == "off"
    await harness.set_thinking_level("high")
    assert harness.get_thinking_level() == "high"

    settings = harness.get_compaction_settings()
    await harness.set_compaction_settings(settings)
    assert harness.get_compaction_settings() is settings

    tool = _make_harness_tool("reader")
    await harness.set_tools([tool], active_tool_names=["reader"])
    assert [item.name for item in harness.get_tools()] == ["reader"]
    assert [item.name for item in harness.get_active_tools()] == ["reader"]

    await harness.set_active_tools([])
    assert harness.get_active_tools() == []

    resources = harness.get_resources()
    await harness.set_resources(resources)
    assert harness.get_resources().skills == resources.skills

    options = harness.get_stream_options()
    await harness.set_stream_options(
        options,
        AgentHarnessStreamOptionsPatch(headers={"X-Test": "1"}),
    )
    assert harness.get_stream_options().headers == {"X-Test": "1"}

    assert harness.get_steering_mode() == "one-at-a-time"
    await harness.set_steering_mode("all")
    assert harness.get_steering_mode() == "all"
    await harness.set_follow_up_mode("all")
    assert harness.get_follow_up_mode() == "all"

    assert await harness.get_leaf_id() is None


@pytest.mark.asyncio
async def test_follow_up_while_idle_raises() -> None:
    harness = _make_harness()
    with pytest.raises(AgentHarnessError) as excinfo:
        await harness.follow_up("hello")
    assert excinfo.value.code == "invalid_state"


@pytest.mark.asyncio
async def test_append_message_while_idle_writes_session() -> None:
    harness = _make_harness()
    await harness.append_message(UserMessage(role="user", content="persist"))
    entries = await harness._session.find_entries()
    assert entries[-1]["message"]["content"] == "persist"


@pytest.mark.asyncio
async def test_request_shutdown_is_idempotent() -> None:
    harness = _make_harness()
    harness.request_shutdown()
    harness.request_shutdown()
    await harness.wait_for_shutdown()

    fresh = _make_harness()
    with pytest.raises(AgentHarnessError) as excinfo:
        await fresh.wait_for_shutdown()
    assert excinfo.value.code == "invalid_state"


@pytest.mark.asyncio
async def test_drain_queue_modes_and_rollback(monkeypatch) -> None:
    harness = _make_harness()
    first = UserMessage(role="user", content="a")
    second = UserMessage(role="user", content="b")

    queue = [first, second]
    drained = await harness._drain_queue(queue, "all")
    assert drained == [first, second]
    assert queue == []

    queue = [first, second]
    drained = await harness._drain_queue(queue, "one-at-a-time")
    assert drained == [first]
    assert queue == [second]

    async def fail_queue_update() -> None:
        raise RuntimeError("queue update failed")

    monkeypatch.setattr(harness, "_emit_queue_update", fail_queue_update)
    queue = [first]
    with pytest.raises(AgentHarnessError):
        await harness._drain_queue(queue, "all")
    assert queue == [first]


@pytest.mark.asyncio
async def test_callable_system_prompt_sync_and_async() -> None:
    sync_harness = _make_harness(system_prompt=lambda _ctx: "SYNC")
    state = await sync_harness._create_turn_state()
    assert state.system_prompt == "SYNC"

    async def prompt_fn(_ctx) -> str:
        return "ASYNC"

    async_harness = _make_harness(system_prompt=prompt_fn)
    state = await async_harness._create_turn_state()
    assert state.system_prompt == "ASYNC"


@pytest.mark.asyncio
async def test_apply_request_options_merges_patch(monkeypatch) -> None:
    harness = _make_harness()
    patched_options = harness.get_stream_options()
    patched_options.max_retries = 3
    patched_options.max_retry_delay_ms = 10
    patched_options.headers = {"X-Test": "1"}
    patched_options.cache_retention = "long"
    patched_options.transport = "websocket"

    async def fake_request(*_args):
        return patched_options

    monkeypatch.setattr(harness, "_emit_before_provider_request", fake_request)
    merged = await harness._apply_request_options(_make_model(), "s1", {})
    assert merged["max_retries"] == 3
    assert merged["max_retry_delay_ms"] == 10
    assert merged["headers"] == {"X-Test": "1"}
    assert merged["cache_retention"] == "long"
    assert merged["transport"] == "websocket"


@pytest.mark.asyncio
async def test_before_provider_hooks_await_async_handlers() -> None:
    harness = _make_harness()
    request_events: list = []

    async def on_request(event) -> BeforeProviderRequestResult:
        request_events.append(event)
        return BeforeProviderRequestResult(
            stream_options=AgentHarnessStreamOptionsPatch(transport="websocket")
        )

    harness.on("before_provider_request", on_request)
    current = await harness._emit_before_provider_request(
        _make_model(),
        "s1",
        harness.get_stream_options(),
    )
    assert current.transport == "websocket"
    assert len(request_events) == 1

    async def on_payload(event) -> BeforeProviderPayloadResult:
        return BeforeProviderPayloadResult(payload={"replaced": True})

    harness.on("before_provider_payload", on_payload)
    payload = await harness._emit_before_provider_payload(_make_model(), {"x": 1})
    assert payload == {"replaced": True}


@pytest.mark.asyncio
async def test_abort_collects_hook_errors() -> None:
    harness = _make_harness()

    def fail_queue(event) -> None:
        raise RuntimeError("queue boom")

    def fail_abort(event) -> None:
        raise RuntimeError("abort boom")

    harness.on("queue_update", fail_queue)
    harness.subscribe(fail_abort)
    with pytest.raises(AgentHarnessError) as excinfo:
        await harness.abort()
    assert excinfo.value.code == "hook"


@pytest.mark.asyncio
async def test_execute_turn_without_assistant_message_raises(monkeypatch) -> None:
    async def empty_loop(**kwargs) -> list:
        return []

    monkeypatch.setattr(harness_module, "run_agent_loop", empty_loop)
    harness = _make_harness()
    turn_state = await harness._create_turn_state()
    with pytest.raises(AgentHarnessError) as excinfo:
        await harness._execute_turn(turn_state, "hi", asyncio.Event())
    assert excinfo.value.code == "invalid_state"
