"""AgentSession 自动压缩集成测试（对齐 TS _checkCompaction）。

覆盖：
    - 溢出（is_context_overflow）→ 压缩 + 自动重试
    - 溢出恢复单次尝试保护
    - 阈值（should_compact）→ 压缩（不重试）
"""

from __future__ import annotations

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
        id="faux-1",
        provider="faux",
        api="openai-completions",
        name="Faux",
        input=["text"],
        output=["text"],
        context_window=context_window,
        max_tokens=16384,
    )


_DEFAULT_AGENT_RETRY_POLICY = RetryPolicy(enabled=False)


def _make_session(
    models: Models,
    session_manager: SessionManager,
    cwd: str | Path,
    *,
    model: Model | None = None,
    compaction_settings: CompactionSettings | None = None,
    agent_retry_policy: RetryPolicy | None = _DEFAULT_AGENT_RETRY_POLICY,
) -> AgentSession:
    """构建 Agent + AgentSession。

    agent_retry_policy 默认关闭 agent 内部重试（隔离测试 session 层压缩）。
    """
    model = model or models.get_model("faux", "faux-1")
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


class TestSummarizationAuthOverride:
    @pytest.mark.asyncio
    async def test_resolver_returns_auth(self, faux_env, tmp_path):
        """_get_summarization_request_auth 返回 apiKey/headers/env/baseUrl 覆盖。"""
        from pi_ai.auth.types import AuthResult

        models, _core = faux_env
        model = _realistic_model()

        class _FakeRuntime:
            async def get_auth(self, m):
                return AuthResult(
                    auth={
                        "api_key": "sk-ovr",
                        "headers": {"X-A": "1", "X-Dropped": None},
                        "base_url": "https://override.example.com/v1",
                    },
                    env={"PI_TEST": "1"},
                )

        session = AgentSession(
            agent=Agent(
                AgentOptions(
                    system_prompt="x",
                    model=model,
                    retry_policy=RetryPolicy(enabled=False),
                )
            ),
            session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
            cwd=str(tmp_path),
            model=model,
            model_runtime=_FakeRuntime(),
        )
        try:
            auth = await session._get_summarization_request_auth(model)
        finally:
            await session.dispose()
        assert auth["apiKey"] == "sk-ovr"
        assert auth["headers"] == {"X-A": "1"}  # None 值被过滤
        assert auth["env"] == {"PI_TEST": "1"}
        assert auth["model"] is not model
        assert auth["model"].base_url == "https://override.example.com/v1"

    @pytest.mark.asyncio
    async def test_manual_compact_forwards_auth(self, faux_env, tmp_path):
        """手动 /compact 把摘要级认证覆盖传进 stream_fn 选项。"""
        from pi_ai.auth.types import AuthResult

        models, core = faux_env
        core.set_responses([faux_assistant_message("## Goal\ncompacted")])
        model = _realistic_model()

        class _FakeRuntime:
            async def get_auth(self, m):
                return AuthResult(
                    auth={
                        "api_key": "sk-ovr",
                        "headers": {"X-A": "1"},
                        "base_url": "https://override.example.com/v1",
                    },
                    env={"PI_TEST": "1"},
                )

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        await _preload_history(mgr)
        session = AgentSession(
            agent=Agent(
                AgentOptions(
                    system_prompt="x",
                    model=model,
                    retry_policy=RetryPolicy(enabled=False),
                )
            ),
            session_manager=mgr,
            cwd=str(tmp_path),
            model=model,
            model_runtime=_FakeRuntime(),
            compaction_settings=CompactionSettings(keep_recent_tokens=40),
        )
        captured: list[tuple] = []
        original = models.stream

        async def capturing_stream(model, context, options=None):
            captured.append((model, dict(options or {})))
            return await original(model, context, options)

        session._agent.stream_function = capturing_stream
        try:
            result = await session.compact()
        finally:
            await session.dispose()
        assert result is not None
        assert captured
        for request_model, opts in captured:
            # baseUrl 通过替换后的 requestModel 生效（对齐 TS requestModel）。
            assert request_model.base_url == "https://override.example.com/v1"
            assert opts.get("api_key") == "sk-ovr"
            assert opts.get("headers") == {"X-A": "1"}
            assert opts.get("env") == {"PI_TEST": "1"}


# ============================================================================
# 溢出 → 压缩 + 自动重试
# ============================================================================


class TestOverflowCompaction:
    async def test_overflow_compacts_and_retries(self, faux_env, tmp_path):
        """溢出错误 → 移除错误消息 → 压缩 → continue_() 重试成功。"""
        models, core = faux_env
        core.set_responses(
            [
                _llm_error("prompt is too long: 213462 tokens > 200000 maximum"),
                faux_assistant_message("## Goal\ncompacted summary"),
                _llm_ok("retried ok"),
            ]
        )

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        await _preload_history(mgr)
        session = _make_session(
            models,
            mgr,
            tmp_path,
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
        core.set_responses(
            [
                _llm_error("prompt is too long: 213462 tokens > 200000 maximum"),
                faux_assistant_message("## Goal\nfirst compact"),
                _llm_error("prompt is too long: 213462 tokens > 200000 maximum"),
            ]
        )

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        await _preload_history(mgr)
        session = _make_session(
            models,
            mgr,
            tmp_path,
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
        core.set_responses(
            [
                faux_assistant_message("## Goal\nthreshold compacted"),
            ]
        )

        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        # 历史 + 一条带大 usage 的 assistant（触发阈值）
        await _preload_history(mgr, count=5)
        await mgr.append_message(_asst("big usage", usage=_usage(1_950)))

        model = _realistic_model(context_window=2_000)
        session = _make_session(
            models,
            mgr,
            tmp_path,
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
            models,
            mgr,
            tmp_path,
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
        core.set_responses(
            [
                _llm_error("prompt is too long: 213462 tokens > 200000 maximum"),
            ]
        )

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        await _preload_history(mgr)
        session = _make_session(
            models,
            mgr,
            tmp_path,
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


@pytest.mark.asyncio
async def test_cache_first_build_context_truncates_large_tool_result(faux_env, tmp_path):
    """cache_first 开启时 _build_context_messages 截断大工具输出。"""
    models, _core = faux_env
    mgr = SessionManager.in_memory(cwd=str(tmp_path))
    await mgr.append_message({"role": "user", "content": "hi", "timestamp": 1})
    await mgr.append_message(
        {
            "role": "toolResult",
            "tool_call_id": "call-1",
            "tool_name": "bash",
            "content": [{"type": "text", "text": "x" * 5000}],
            "is_error": False,
            "timestamp": 2,
        }
    )
    session = _make_session(
        models,
        mgr,
        str(tmp_path),
        compaction_settings=CompactionSettings(cache_first=True),
    )
    try:
        messages = session._build_context_messages()
    finally:
        await session.dispose()
    tool_result = messages[-1]
    assert tool_result["role"] == "toolResult"
    assert tool_result["content"][0]["text"] == "[output truncated]"
    assert tool_result["tool_call_id"] == "call-1"


@pytest.mark.asyncio
async def test_cache_first_disabled_keeps_full_tool_result(faux_env, tmp_path):
    """cache_first 关闭时上下文保留完整工具输出。"""
    models, _core = faux_env
    mgr = SessionManager.in_memory(cwd=str(tmp_path))
    await mgr.append_message({"role": "user", "content": "hi", "timestamp": 1})
    await mgr.append_message(
        {
            "role": "toolResult",
            "tool_call_id": "call-1",
            "tool_name": "bash",
            "content": [{"type": "text", "text": "x" * 5000}],
            "is_error": False,
            "timestamp": 2,
        }
    )
    session = _make_session(models, mgr, str(tmp_path))
    try:
        messages = session._build_context_messages()
    finally:
        await session.dispose()
    assert messages[-1]["content"][0]["text"] == "x" * 5000


@pytest.mark.asyncio
async def test_cache_first_turn_prefix_stable_effect(faux_env, tmp_path):
    """effect 级：cache_first 下第二轮请求的前缀与第一轮字节一致。"""
    models, core = faux_env
    core.set_responses([_asst("r1"), _asst("r2")])
    mgr = SessionManager.in_memory(cwd=str(tmp_path))
    await mgr.append_message({"role": "user", "content": "init", "timestamp": 1})
    await mgr.append_message(
        {
            "role": "toolResult",
            "tool_call_id": "call-1",
            "tool_name": "bash",
            "content": [{"type": "text", "text": "x" * 5000}],
            "is_error": False,
            "timestamp": 2,
        }
    )
    model = _realistic_model(context_window=1000)
    session = _make_session(
        models,
        mgr,
        str(tmp_path),
        model=model,
        compaction_settings=CompactionSettings(cache_first=True, reserve_tokens=0),
    )
    seen: list[list[dict]] = []
    original = models.stream

    async def capturing(model, context, options=None):
        seen.append(list(context.messages))
        return await original(model, context, options)

    session._agent.stream_function = capturing
    try:
        await session.prompt("first")
        await session.prompt("second")
    finally:
        await session.dispose()
    assert len(seen) == 2
    assert seen[1][:-2] == seen[0]
    tool_result = seen[0][1]
    assert tool_result["role"] == "toolResult"
    assert tool_result["content"][0]["text"] == "[output truncated]"
