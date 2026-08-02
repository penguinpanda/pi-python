"""AgentSession 自动压缩集成测试（对齐 TS _checkCompaction）。

覆盖：
    - 溢出（is_context_overflow）→ 压缩 + 自动重试
    - 溢出恢复单次尝试保护
    - 阈值（should_compact）→ 压缩（不重试）
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions, set_default_stream_fn
from pi_ai import Models
from pi_ai._types import Model
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_ai.utils.retry import RetryPolicy

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.compaction import CompactionSettings


@pytest.fixture
def faux_env():
    """注册 Faux Provider 的 Models + 全局默认流函数。"""
    core = faux_provider()
    models = Models()
    models.add_provider(core.provider)
    set_default_stream_fn(models.stream)
    yield models, core
    set_default_stream_fn(None)


def _llm_error(text: str):
    return faux_assistant_message([], stop_reason="error", error_message=text)


def _llm_ok(text: str = "ok"):
    return faux_assistant_message(text)


def _asst(text: str, usage: dict | None = None):
    msg = faux_assistant_message(text)
    if usage is not None:
        msg["usage"] = usage
    return msg


def _usage(input_: int, output: int = 100) -> dict:
    return {
        "input": input_,
        "output": output,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": input_ + output,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    }


def _realistic_model(context_window: int = 128_000) -> Model:
    return Model(
        id="faux-1", provider="faux", api="openai-completions", name="Faux",
        input=["text"], output=["text"], context_window=context_window, max_tokens=16384,
    )


def _make_session(
    models: Models,
    session_manager: SessionManager,
    cwd: str | Path,
    *,
    model: Model | None = None,
    compaction_settings: CompactionSettings | None = None,
    agent_retry_policy: RetryPolicy | None = RetryPolicy(enabled=False),
) -> AgentSession:
    """构建 Agent + AgentSession。

    agent_retry_policy 默认关闭 agent 内部重试（隔离测试 session 层压缩）。
    """
    model = model or models.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(AgentOptions(
        system_prompt="You are a helpful coding assistant.",
        model=model,
        retry_policy=agent_retry_policy,
    ))
    return AgentSession(
        agent=agent,
        session_manager=session_manager,
        cwd=str(cwd),
        model=model,
        compaction_settings=compaction_settings,
    )


def _last_assistant(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg
    return None


async def _preload_history(mgr: SessionManager, count: int = 6) -> None:
    """预置历史：count 轮 user + assistant（各 ~26 tokens），使压缩有内容可摘要。"""
    for i in range(count):
        await mgr.append_message({"role": "user", "content": f"h{i} " + "x" * 100, "timestamp": i})
        await mgr.append_message(_asst(f"a{i} " + "y" * 100))


# ============================================================================
# 溢出 → 压缩 + 自动重试
# ============================================================================


class TestOverflowCompaction:
    async def test_overflow_compacts_and_retries(self, faux_env, tmp_path):
        """溢出错误 → 移除错误消息 → 压缩 → continue_() 重试成功。"""
        models, core = faux_env
        core.set_responses([
            _llm_error("prompt is too long: 213462 tokens > 200000 maximum"),
            faux_assistant_message("## Goal\ncompacted summary"),
            _llm_ok("retried ok"),
        ])

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        await _preload_history(mgr)
        session = _make_session(
            models, mgr, tmp_path,
            model=_realistic_model(),
            compaction_settings=CompactionSettings(keep_recent_tokens=40),
        )
        events: list[dict] = []
        session.subscribe(lambda e: events.append(e))

        try:
            await session.prompt("continue")
            await session.wait_for_idle()
        finally:
            await session.dispose()

        # 3 次 LLM 调用：溢出错误 + 摘要 + 重试
        assert core.call_count == 3

        # 会话写入压缩条目
        entries = mgr.get_entries()
        assert any(e["type"] == "compaction" for e in entries)

        # 上下文以 compactionSummary 开头
        messages = session.get_messages()
        assert messages[0]["role"] == "compactionSummary"
        assert "compacted summary" in messages[0]["summary"]

        # 重试成功
        last = _last_assistant(messages)
        assert last is not None
        assert last["stop_reason"] == "stop"
        assert last["content"][0]["text"] == "retried ok"

        # 事件
        starts = [e for e in events if e["type"] == "compaction_start"]
        ends = [e for e in events if e["type"] == "compaction_end"]
        assert len(starts) == 1 and starts[0]["reason"] == "overflow"
        assert len(ends) == 1 and ends[0]["willRetry"] is True

    async def test_overflow_recovery_single_attempt(self, faux_env, tmp_path):
        """重试仍溢出 → 第二次压缩+重试被阻止（单次尝试保护）。"""
        models, core = faux_env
        core.set_responses([
            _llm_error("prompt is too long: 213462 tokens > 200000 maximum"),
            faux_assistant_message("## Goal\nfirst compact"),
            _llm_error("prompt is too long: 213462 tokens > 200000 maximum"),
        ])

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        await _preload_history(mgr)
        session = _make_session(
            models, mgr, tmp_path,
            model=_realistic_model(),
            compaction_settings=CompactionSettings(keep_recent_tokens=40),
        )
        events: list[dict] = []
        session.subscribe(lambda e: events.append(e))

        try:
            await session.prompt("continue")
            await session.wait_for_idle()
        finally:
            await session.dispose()

        # 溢出 + 摘要 + 重试仍溢出 = 3 次调用；第二次摘要未发生
        assert core.call_count == 3

        starts = [e for e in events if e["type"] == "compaction_start"]
        ends = [e for e in events if e["type"] == "compaction_end"]
        assert len(starts) == 1  # 只有一次压缩
        # 第二次溢出：compaction_end 带 errorMessage 且 willRetry False
        assert ends[-1]["willRetry"] is False
        assert "Context overflow recovery failed" in ends[-1].get("errorMessage", "")


# ============================================================================
# 阈值 → 压缩（不重试）
# ============================================================================


class TestThresholdCompaction:
    async def test_threshold_compacts_without_retry(self, faux_env, tmp_path):
        """usage 超阈值 → 压缩，不重试（状态保留在错误/溢出消息之外）。"""
        models, core = faux_env
        core.set_responses([
            faux_assistant_message("## Goal\nthreshold compacted"),
        ])

        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        # 历史 + 一条带大 usage 的 assistant（触发阈值）
        await _preload_history(mgr, count=5)
        await mgr.append_message(_asst("big usage", usage=_usage(1_950)))

        model = _realistic_model(context_window=2_000)
        session = _make_session(
            models, mgr, tmp_path,
            model=model,
            compaction_settings=CompactionSettings(reserve_tokens=100, keep_recent_tokens=40),
        )
        events: list[dict] = []
        session.subscribe(lambda e: events.append(e))

        try:
            # 直接触发压缩检查（等价于 prompt 后的检查）
            triggered = await session._check_compaction()
            await session.wait_for_idle()
        finally:
            await session.dispose()

        assert triggered is True
        # 仅摘要 LLM 调用
        assert core.call_count == 1

        entries = mgr.get_entries()
        assert any(e["type"] == "compaction" for e in entries)

        messages = session.get_messages()
        assert messages[0]["role"] == "compactionSummary"

        starts = [e for e in events if e["type"] == "compaction_start"]
        ends = [e for e in events if e["type"] == "compaction_end"]
        assert len(starts) == 1 and starts[0]["reason"] == "threshold"
        assert len(ends) == 1 and ends[0]["willRetry"] is False

    async def test_normal_response_does_not_compact(self, faux_env, tmp_path):
        """正常小上下文响应不触发压缩。"""
        models, core = faux_env
        core.set_responses([_llm_ok("fine")])

        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        await _preload_history(mgr, count=3)
        session = _make_session(
            models, mgr, tmp_path,
            model=_realistic_model(),
            compaction_settings=CompactionSettings(keep_recent_tokens=40),
        )
        events: list[dict] = []
        session.subscribe(lambda e: events.append(e))

        try:
            triggered = await session._check_compaction()
            await session.wait_for_idle()
        finally:
            await session.dispose()

        assert triggered is False
        assert core.call_count == 0  # 未消费摘要响应
        assert not any(e["type"] == "compaction" for e in mgr.get_entries())
        assert not any(e["type"] == "compaction_start" for e in events)

    async def test_compaction_disabled(self, faux_env, tmp_path):
        """compaction_settings.enabled=False → 溢出也不压缩。"""
        models, core = faux_env
        core.set_responses([
            _llm_error("prompt is too long: 213462 tokens > 200000 maximum"),
        ])

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        await _preload_history(mgr)
        session = _make_session(
            models, mgr, tmp_path,
            model=_realistic_model(),
            compaction_settings=CompactionSettings(enabled=False),
        )

        try:
            await session.prompt("continue")
            await session.wait_for_idle()
        finally:
            await session.dispose()

        # 只消耗了错误响应，无摘要、无压缩条目
        assert core.call_count == 1
        assert not any(e["type"] == "compaction" for e in mgr.get_entries())
