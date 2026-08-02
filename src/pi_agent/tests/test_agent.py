"""_agent.py 模块测试。"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai._types import (
    AssistantMessage,
    Model,
    TextContent,
    UserMessage,
)
from pi_ai.providers.faux import FauxCore, faux_assistant_message, faux_provider

from pi_agent._agent import Agent, AgentOptions
from pi_agent._stream_fn import set_default_stream_fn
from pi_agent._types import (
    AgentEvent,
    AgentTool,
    AgentToolResult,
    StreamFn,
)


# ============================================================================
# 辅助
# ============================================================================


def _make_model() -> Model:
    return Model(
        id="test-model",
        provider="test",
        api="openai-completions",
        name="Test",
        supportsToolCalling=True,
    )


def _make_faux(responses: list[AssistantMessage]) -> FauxCore:
    """创建脚本化 Faux 响应序列，返回 FauxCore。

    其 `.stream` 签名即 StreamFn，可直接注入 Agent 或注册为默认。
    """
    core = faux_provider()
    core.set_responses(responses)
    return core


def _make_faux_stream_fn(text: str = "Hello!") -> StreamFn:
    """创建返回固定文本响应的 Faux stream_fn。"""
    return _make_faux([faux_assistant_message(text)]).stream


def _make_modeled_stream_fn(responses: list[str]) -> StreamFn:
    """创建按调用次数返回不同文本的 Faux stream_fn（顺序消费响应序列）。"""
    return _make_faux([faux_assistant_message(r) for r in responses]).stream


def _make_tool(name: str, result_text: str = "tool result") -> AgentTool:
    """创建测试用工具。"""
    async def _execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(type="text", text=result_text)],
        )

    return AgentTool(
        name=name,
        description=f"Tool: {name}",
        input_schema={"type": "object", "properties": {}},
        label=name,
        execute=_execute,
    )


# ============================================================================
# 测试
# ============================================================================


class TestAgentPrompt:
    """prompt() 端到端测试。"""

    @pytest.mark.asyncio
    async def test_basic_prompt(self):
        """基本 prompt() 完成一轮对话。"""
        stream_fn = _make_faux_stream_fn("Hi there!")
        set_default_stream_fn(stream_fn)

        agent = Agent(AgentOptions(
            model=_make_model(),
            system_prompt="You are helpful.",
        ))
        await agent.prompt("Hello")

        # 验证状态：isStreaming 应为 False
        assert agent.state.is_streaming is False
        # 验证 messages 中有 user 和 assistant
        roles = [m.get("role") for m in agent.state.messages]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_prompt_with_explicit_stream_fn(self):
        """显式传 stream_fn 优于全局默认。"""
        stream_fn = _make_faux_stream_fn("explicit!")
        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=stream_fn,
        ))
        await agent.prompt("Hi")
        assert agent.state.is_streaming is False

    @pytest.mark.asyncio
    async def test_multiple_prompts(self):
        """多次 prompt() 调用。"""
        stream_fn = _make_modeled_stream_fn(["First!", "Second!"])
        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=stream_fn,
        ))
        await agent.prompt("Q1")
        await agent.prompt("Q2")
        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("user") == 2
        assert roles.count("assistant") == 2

    @pytest.mark.asyncio
    async def test_events_subscription(self):
        """subscribe() 接收生命周期事件。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=stream_fn,
        ))

        received: list[AgentEvent] = []
        agent.subscribe(lambda e: received.append(e))

        await agent.prompt("Hi")

        event_types = [e["type"] for e in received]
        assert "agent_start" in event_types
        assert "agent_end" in event_types

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """取消订阅后不再接收事件。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=stream_fn,
        ))

        received: list[AgentEvent] = []
        unsub = agent.subscribe(lambda e: received.append(e))
        unsub()

        await agent.prompt("Hi")
        assert len(received) == 0


class TestAgentContinue:
    """continue_() 测试。"""

    @pytest.mark.asyncio
    async def test_continue_from_user_message(self):
        """最后一条是 user 消息时可以 continue。"""
        stream_fn = _make_modeled_stream_fn(["First!", "Second!"])
        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=stream_fn,
        ))
        # 先发一条 user 消息但让 prompt 完成（assistant 回复后状态正常）
        await agent.prompt("Q1")

        # 手动追加一条 user 消息（模拟外部注入）
        agent.state._append_message(UserMessage(role="user", content="Q2"))

        await agent.continue_()

        roles = [m.get("role") for m in agent.state.messages]
        # Q1 user, A1 assistant, Q2 user, A2 assistant
        assert roles.count("user") >= 1
        assert roles.count("assistant") >= 2

    @pytest.mark.asyncio
    async def test_continue_after_assistant_raises(self):
        """最后一条是 assistant 时 continue 应抛异常。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=stream_fn,
        ))
        await agent.prompt("Hi")
        # 现在最后一条是 assistant
        with pytest.raises(RuntimeError, match="Cannot continue"):
            await agent.continue_()


class TestAgentAbort:
    """abort() 测试。"""

    @pytest.mark.asyncio
    async def test_abort_stops_run(self):
        """abort() 后 agent loop 停止，agent_end 事件正确发出。"""
        # 用 Faux 慢速流模拟长回复：abort 在流式输出过程中触发，
        # agent loop 的 signal 检查会中断循环并发出 agent_end。
        core = faux_provider(tokens_per_second=5)
        core.set_responses([faux_assistant_message("A" * 500)])

        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=core.stream,
        ))

        received: list[AgentEvent] = []
        agent.subscribe(lambda e: received.append(e))

        # 在后台启动 prompt，稍后 abort
        async def _run_and_abort():
            await asyncio.sleep(0.05)  # 给 prompt 时间启动
            agent.abort()

        # 不应超时：abort 应让 loop 干净停止
        await asyncio.wait_for(
            asyncio.gather(agent.prompt("Hi"), _run_and_abort()),
            timeout=2.0,
        )

        event_types = [e["type"] for e in received]
        assert "agent_end" in event_types


class TestAgentMutualExclusion:
    """互斥运行测试。"""

    @pytest.mark.asyncio
    async def test_concurrent_prompts_raise(self):
        """两个并发 prompt() 抛 RuntimeError。"""
        # Faux 慢速流：第一个 prompt 仍在流式输出时，第二个 prompt 抛 RuntimeError。
        core = faux_provider(tokens_per_second=4)
        core.set_responses([faux_assistant_message("A" * 20)])

        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=core.stream,
        ))

        async def _first():
            await agent.prompt("First")

        async def _second():
            await asyncio.sleep(0.02)
            with pytest.raises(RuntimeError, match="already running"):
                await agent.prompt("Second")

        await asyncio.gather(_first(), _second())


class TestAgentReset:
    """reset() 测试。"""

    @pytest.mark.asyncio
    async def test_reset_clears_messages(self):
        """reset() 后 messages 清空。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=stream_fn,
        ))
        await agent.prompt("Hi")
        assert len(agent.state.messages) > 0

        agent.reset()
        assert len(agent.state.messages) == 0
        assert agent.state.error_message is None

    @pytest.mark.asyncio
    async def test_reset_while_running_raises(self):
        """运行时 reset 抛异常。"""
        # Faux 慢速流：prompt 仍在流式输出时 reset 抛 RuntimeError。
        core = faux_provider(tokens_per_second=4)
        core.set_responses([faux_assistant_message("A" * 20)])

        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=core.stream,
        ))

        # 启动 prompt 但不等待
        task = asyncio.create_task(agent.prompt("Hi"))
        await asyncio.sleep(0.05)

        with pytest.raises(RuntimeError, match="Cannot reset"):
            agent.reset()

        agent.abort()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass


class TestAgentState:
    """AgentState 状态管理测试。"""

    @pytest.mark.asyncio
    async def test_streaming_flag(self):
        """isStreaming 在 prompt 完成后为 False。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=stream_fn,
        ))
        assert agent.state.is_streaming is False
        await agent.prompt("Hi")
        assert agent.state.is_streaming is False

    @pytest.mark.asyncio
    async def test_error_response_sets_error_message(self):
        """LLM 返回 error stopReason → agent_end 事件 + state.error_message。"""
        core = _make_faux([
            faux_assistant_message([], stop_reason="error", error_message="Boom!"),
        ])

        agent = Agent(AgentOptions(
            model=_make_model(),
            stream_fn=core.stream,
        ))

        called: list[str] = []
        agent.subscribe(lambda e: called.append(e["type"]))

        await agent.prompt("Hi")

        # agent_end 应发出
        assert "agent_end" in called
        # AgentState 记录错误消息
        assert agent.state.error_message == "Boom!"
