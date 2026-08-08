"""AgentSession 统计（turn timings / cache stats）测试。"""

from __future__ import annotations

import pytest
from pi_agent import Agent, AgentOptions
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime


def _make_session(tmp_path, responses=None, **kwargs):
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    core = faux_provider()
    if responses:
        core.set_responses(responses)
    models.add_provider(core.provider)
    runtime = ModelRuntime(models, store)
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=runtime.stream,
        )
    )
    return AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=model,
        model_runtime=runtime,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_turn_timings_recorded(tmp_path):
    session = _make_session(
        tmp_path,
        responses=[faux_assistant_message("stats reply")],
    )
    initial = session.get_session_stats()
    assert initial["turnTimings"] == {
        "turnCount": 0,
        "totalMs": 0,
        "averageMs": 0,
        "lastMs": 0,
    }
    assert initial["cacheStats"] == {
        "missedTokens": 0,
        "missedCost": 0.0,
        "missCount": 0,
        "hitTokens": 0,
        "hitRate": None,
    }

    await session.prompt("hi")
    stats = session.get_session_stats()
    assert stats["turnTimings"]["turnCount"] == 1
    assert stats["turnTimings"]["lastMs"] >= 0
    assert stats["turnTimings"]["totalMs"] >= stats["turnTimings"]["lastMs"]


def test_cache_stats_from_messages(tmp_path):
    session = _make_session(tmp_path)
    session._agent.state.messages = [
        {
            "role": "assistant",
            "provider": "faux",
            "model": "faux-1",
            "content": [{"type": "text", "text": "first"}],
            "usage": {"input": 30000, "cache_read": 0, "cache_write": 0},
        },
        {
            "role": "assistant",
            "provider": "faux",
            "model": "faux-1",
            "content": [{"type": "text", "text": "second"}],
            "usage": {"input": 20000, "cache_read": 10000, "cache_write": 0},
        },
    ]
    stats = session.get_session_stats()
    assert stats["cacheStats"]["missCount"] == 1
    assert stats["cacheStats"]["missedTokens"] == 20000
    assert stats["hitRate"] == pytest.approx(10000 / 60000)
    assert stats["cacheStats"]["hitTokens"] == 10000
    assert stats["cacheStats"]["hitRate"] == pytest.approx(10000 / 60000)


def test_cost_aggregated_from_usage(tmp_path):
    """coding-agent 层应聚合 pi_ai 计算的逐请求 usage.cost。"""
    session = _make_session(tmp_path)
    session._agent.state.messages = [
        {
            "role": "assistant",
            "provider": "faux",
            "model": "faux-1",
            "content": [{"type": "text", "text": "first"}],
            "usage": {
                "input": 1_000_000,
                "output": 500_000,
                "cache_read": 0,
                "cache_write": 0,
                "total_tokens": 1_500_000,
                "cost": {
                    "input": 0.14,
                    "output": 0.14,
                    "cache_read": 0.0,
                    "cache_write": 0.0,
                    "total": 0.28,
                },
            },
        },
        {
            "role": "assistant",
            "provider": "faux",
            "model": "faux-1",
            "content": [{"type": "text", "text": "second"}],
            "usage": {
                "input": 10,
                "output": 10,
                "cache_read": 0,
                "cache_write": 0,
                "total_tokens": 20,
                "cost": {
                    "input": 0.0,
                    "output": 0.0,
                    "cache_read": 0.0,
                    "cache_write": 0.0,
                    "total": 0.01,
                },
            },
        },
    ]
    stats = session.get_session_stats()
    assert stats["cost"] == pytest.approx(0.29)
    assert stats["tokens"]["input"] == 1_000_010


def test_cache_miss_notice_emitted_when_enabled(tmp_path):
    session = _make_session(tmp_path, show_cache_miss_notices=lambda: True)
    events: list[dict] = []
    session.subscribe(events.append)
    session._agent.state.messages = [
        {
            "role": "assistant",
            "provider": "faux",
            "model": "faux-1",
            "content": [],
            "usage": {"input": 30000, "cache_read": 0, "cache_write": 0},
        },
        {
            "role": "assistant",
            "provider": "faux",
            "model": "faux-1",
            "content": [],
            "usage": {"input": 20000, "cache_read": 0, "cache_write": 0},
        },
    ]
    session._maybe_emit_cache_miss_notice()
    assert any(event.get("type") == "cache_miss_notice" for event in events)
    notice = next(event for event in events if event.get("type") == "cache_miss_notice")
    assert notice.get("reasons") == []


def test_cache_miss_notice_suppressed_when_disabled(tmp_path):
    session = _make_session(tmp_path)
    events: list[dict] = []
    session.subscribe(events.append)
    session._agent.state.messages = [
        {
            "role": "assistant",
            "provider": "faux",
            "model": "faux-1",
            "content": [],
            "usage": {"input": 30000, "cache_read": 0, "cache_write": 0},
        },
        {
            "role": "assistant",
            "provider": "faux",
            "model": "faux-1",
            "content": [],
            "usage": {"input": 20000, "cache_read": 0, "cache_write": 0},
        },
    ]
    session._maybe_emit_cache_miss_notice()
    assert not any(event.get("type") == "cache_miss_notice" for event in events)


def test_hit_rate_none_when_no_tokens(tmp_path):
    session = _make_session(tmp_path)
    assert session.get_session_stats()["hitRate"] is None
