"""AgentSession turn 级重试测试（Phase 4）。

验证 session 层自动重试：
    prompt() 后检查末条消息为可重试错误 → 移除错误消息 → continue_() 恢复状态机

为隔离 session 层，Agent 内部重试默认关闭（retry_policy=RetryPolicy(enabled=False)），
session 层 turn_retry_policy 单独控制。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions, set_default_stream_fn
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_ai.utils.retry import RetryPolicy

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager


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


_DEFAULT_AGENT_RETRY_POLICY = RetryPolicy(enabled=False)


def _make_session(
    models: Models,
    session_manager: SessionManager,
    cwd: str | Path,
    *,
    turn_retry_policy: RetryPolicy | None = None,
    agent_retry_policy: RetryPolicy | None = _DEFAULT_AGENT_RETRY_POLICY,
) -> AgentSession:
    """构建 Agent + AgentSession。

    agent_retry_policy 默认关闭 agent 内部重试（隔离测试 session 层）。
    """
    model = models.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            retry_policy=agent_retry_policy,
        )
    )
    return AgentSession(
        agent=agent,
        session_manager=session_manager,
        cwd=str(cwd),
        model=model,
        turn_retry_policy=turn_retry_policy,
    )


def _last_assistant(messages):
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg
    return None


# ============================================================================
# turn 级重试：失败 → continue_() 成功
# ============================================================================


class TestSessionTurnRetry:
    async def test_turn_retry_succeeds(self, faux_env, tmp_path):
        """失败 → 移除错误消息 → continue_() → 成功。"""
        models, core = faux_env
        core.set_responses([_llm_error("500 Internal Server Error"), _llm_ok("retried ok")])

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        session = _make_session(
            models,
            mgr,
            tmp_path,
            turn_retry_policy=RetryPolicy(max_retries=3, base_delay_ms=1, jitter=False),
        )

        try:
            await session.prompt("Hi")
            await session.wait_for_idle()
        finally:
            await session.dispose()

        # 两次调用：第一次失败，第二次（continue_）成功
        assert core.call_count == 2

        messages = session.get_messages()
        last = _last_assistant(messages)
        assert last is not None
        assert last["stop_reason"] == "stop"

        # 状态机恢复：错误消息已被移除，transcript 只有一条 assistant
        assistants = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistants) == 1

    async def test_turn_retry_exhausted_ends_with_error(self, faux_env, tmp_path):
        """重试耗尽 → 仍以错误结束，且状态机保持最后一条错误消息。"""
        models, core = faux_env
        core.set_responses(
            [
                _llm_error("503 Service Unavailable"),
                _llm_error("503 Service Unavailable"),
                _llm_error("503 Service Unavailable"),
            ]
        )

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        session = _make_session(
            models,
            mgr,
            tmp_path,
            turn_retry_policy=RetryPolicy(max_retries=2, base_delay_ms=1, jitter=False),
        )

        try:
            await session.prompt("Hi")
            await session.wait_for_idle()
        finally:
            await session.dispose()

        # 初始调用 + 2 次 turn 重试
        assert core.call_count == 3

        last = _last_assistant(session.get_messages())
        assert last is not None
        assert last["stop_reason"] == "error"

    async def test_turn_retry_disabled(self, faux_env, tmp_path):
        """turn 级重试关闭 → 不重试。"""
        models, core = faux_env
        core.set_responses([_llm_error("500 Internal Server Error")])

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        session = _make_session(
            models,
            mgr,
            tmp_path,
            turn_retry_policy=RetryPolicy(enabled=False),
        )

        try:
            await session.prompt("Hi")
            await session.wait_for_idle()
        finally:
            await session.dispose()

        assert core.call_count == 1
        last = _last_assistant(session.get_messages())
        assert last["stop_reason"] == "error"

    async def test_non_retryable_error_no_turn_retry(self, faux_env, tmp_path):
        """不可重试错误 → 不触发 turn 重试。"""
        models, core = faux_env
        core.set_responses([_llm_error("insufficient_quota")])

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        session = _make_session(
            models,
            mgr,
            tmp_path,
            turn_retry_policy=RetryPolicy(max_retries=3, base_delay_ms=1, jitter=False),
        )

        try:
            await session.prompt("Hi")
            await session.wait_for_idle()
        finally:
            await session.dispose()

        assert core.call_count == 1
        last = _last_assistant(session.get_messages())
        assert last["stop_reason"] == "error"

    async def test_abort_during_turn_retry_backoff(self, faux_env, tmp_path):
        """退避等待期间 abort → 停止重试，不再 continue_()。"""
        models, core = faux_env
        # 长退避：保证 prompt() 停在退避等待中
        core.set_responses([_llm_error("500 Internal Server Error"), _llm_ok("ok")])

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        session = _make_session(
            models,
            mgr,
            tmp_path,
            turn_retry_policy=RetryPolicy(max_retries=3, base_delay_ms=60000, jitter=False),
        )

        try:
            task = asyncio.create_task(session.prompt("Hi"))

            # 等待第一次调用完成并进入退避等待
            while core.call_count < 1:
                await asyncio.sleep(0)

            await session.abort()
            await task  # 应正常返回（不抛异常）

            # 只调用了一次（失败），未继续重试
            assert core.call_count == 1
        finally:
            await session.dispose()
