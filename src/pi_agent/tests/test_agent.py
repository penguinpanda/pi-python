"""_agent.py 模块测试。"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai.types import (
    AssistantMessage,
    Model,
    TextContent,
    UserMessage,
)
from pi_ai.providers.faux import (
    FauxCore,
    faux_assistant_message,
    faux_provider,
    faux_tool_call,
)

from pi_agent._agent import Agent, AgentOptions
from pi_agent._stream_fn import set_default_stream_fn
from pi_agent._types import (
    AgentEvent,
    AgentLoopTurnUpdate,
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

        agent = Agent(
            AgentOptions(
                model=_make_model(),
                system_prompt="You are helpful.",
            )
        )
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
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
        await agent.prompt("Hi")
        assert agent.state.is_streaming is False

    @pytest.mark.asyncio
    async def test_multiple_prompts(self):
        """多次 prompt() 调用。"""
        stream_fn = _make_modeled_stream_fn(["First!", "Second!"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
        await agent.prompt("Q1")
        await agent.prompt("Q2")
        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("user") == 2
        assert roles.count("assistant") == 2

    @pytest.mark.asyncio
    async def test_events_subscription(self):
        """subscribe() 接收生命周期事件。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )

        received: list[AgentEvent] = []
        agent.subscribe(lambda e, signal: received.append(e))

        await agent.prompt("Hi")

        event_types = [e["type"] for e in received]
        assert "agent_start" in event_types
        assert "agent_end" in event_types

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """取消订阅后不再接收事件。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )

        received: list[AgentEvent] = []
        unsub = agent.subscribe(lambda e, signal: received.append(e))
        unsub()

        await agent.prompt("Hi")
        assert len(received) == 0


class TestAgentContinue:
    """continue_() 测试。"""

    @pytest.mark.asyncio
    async def test_continue_from_user_message(self):
        """最后一条是 user 消息时可以 continue。"""
        stream_fn = _make_modeled_stream_fn(["First!", "Second!"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
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
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
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

        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=core.stream,
            )
        )

        received: list[AgentEvent] = []
        agent.subscribe(lambda e, signal: received.append(e))

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


class TestAgentSettled:
    """agent_settled 在正常完成 / abort / 崩溃三条路径都发出。"""

    @pytest.mark.asyncio
    async def test_settled_emitted_on_normal_completion(self):
        events: list[str] = []
        agent = Agent(AgentOptions(model=_make_model(), stream_fn=_make_faux_stream_fn("hi")))
        agent.subscribe(lambda e, signal: events.append(e["type"]))

        await agent.prompt("hello")

        assert events[-1] == "agent_settled"
        assert events.count("agent_settled") == 1

    @pytest.mark.asyncio
    async def test_settled_emitted_after_abort(self):
        events: list[str] = []
        core = faux_provider(tokens_per_second=5)
        core.set_responses([faux_assistant_message("A" * 500)])
        agent = Agent(AgentOptions(model=_make_model(), stream_fn=core.stream))
        agent.subscribe(lambda e, signal: events.append(e["type"]))

        async def _run_and_abort():
            await asyncio.sleep(0.05)
            agent.abort()

        await asyncio.wait_for(
            asyncio.gather(agent.prompt("Hi"), _run_and_abort()),
            timeout=2.0,
        )

        assert events[-1] == "agent_settled"

    @pytest.mark.asyncio
    async def test_settled_emitted_on_crash(self):
        events: list[str] = []

        def broken_stream_fn(model, context, options):
            raise RuntimeError("boom")

        agent = Agent(AgentOptions(model=_make_model(), stream_fn=broken_stream_fn))
        agent.subscribe(lambda e, signal: events.append(e["type"]))

        # 对齐 TS runWithLifecycle：crash 被捕获并正常结束（不向调用方抛异常），
        # settled 与合成 agent_end 照常发出，error_message 记录在 state，
        # 且 state 末条为 error assistant 消息（下游 retry/compaction 可识别）。
        await agent.prompt("Hi")

        assert "agent_end" in events
        assert events[-1] == "agent_settled"
        assert agent.state.error_message == "boom"
        tail = agent.state.messages[-1]
        assert tail.get("role") == "assistant"
        assert tail.get("stop_reason") == "error"
        assert tail.get("error_message") == "boom"


class TestAgentMutualExclusion:
    """互斥运行测试。"""

    @pytest.mark.asyncio
    async def test_concurrent_prompts_raise(self):
        """两个并发 prompt() 抛 RuntimeError。"""
        # Faux 慢速流：第一个 prompt 仍在流式输出时，第二个 prompt 抛 RuntimeError。
        core = faux_provider(tokens_per_second=4)
        core.set_responses([faux_assistant_message("A" * 20)])

        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=core.stream,
            )
        )

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
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
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

        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=core.stream,
            )
        )

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
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
        assert agent.state.is_streaming is False
        await agent.prompt("Hi")
        assert agent.state.is_streaming is False

    @pytest.mark.asyncio
    async def test_error_response_sets_error_message(self):
        """LLM 返回 error stop_reason → agent_end 事件 + state.error_message。"""
        core = _make_faux(
            [
                faux_assistant_message([], stop_reason="error", error_message="Boom!"),
            ]
        )

        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=core.stream,
            )
        )

        called: list[str] = []
        agent.subscribe(lambda e, signal: called.append(e["type"]))

        await agent.prompt("Hi")

        # agent_end 应发出
        assert "agent_end" in called
        # AgentState 记录错误消息
        assert agent.state.error_message == "Boom!"


class TestAgentMessageQueues:
    """1.1/1.2：双消息队列 + 双重嵌套循环（Agent 层集成）。"""

    def test_queue_api_and_modes(self):
        """steer/follow_up/clear/has_queued_messages + QueueMode setter。"""
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=_make_faux_stream_fn(),
            )
        )
        assert agent.has_queued_messages() is False

        agent.steer(UserMessage(role="user", content="s1"))
        agent.follow_up(UserMessage(role="user", content="f1"))
        assert agent.has_queued_messages() is True

        agent.clear_steering_queue()
        assert agent.has_queued_messages() is True

        agent.clear_all_queues()
        assert agent.has_queued_messages() is False

        agent.steering_mode = "all"
        agent.follow_up_mode = "one-at-a-time"
        assert agent.steering_mode == "all"
        assert agent.follow_up_mode == "one-at-a-time"

    @pytest.mark.asyncio
    async def test_steer_before_prompt_injected_first_turn(self):
        """prompt 前 steer() → 首轮注入。"""
        stream_fn = _make_modeled_stream_fn(["A1"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
        agent.steer(UserMessage(role="user", content="nudge"))
        await agent.prompt("Q")

        contents = [m.get("content") for m in agent.state.messages]
        assert "nudge" in contents
        assert agent.has_queued_messages() is False

    @pytest.mark.asyncio
    async def test_follow_up_processed_in_same_run(self):
        """prompt 前 follow_up() → agent 停止后自动追加一轮。"""
        stream_fn = _make_modeled_stream_fn(["A1", "A2"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
        agent.follow_up(UserMessage(role="user", content="follow up"))
        await agent.prompt("Q")

        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("assistant") == 2
        contents = [m.get("content") for m in agent.state.messages]
        assert "follow up" in contents

    @pytest.mark.asyncio
    async def test_one_at_a_time_follow_up_mode(self):
        """one-at-a-time：每个 follow-up 占一轮。"""
        stream_fn = _make_modeled_stream_fn(["A1", "A2", "A3"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
                follow_up_mode="one-at-a-time",
            )
        )
        agent.follow_up(UserMessage(role="user", content="F1"))
        agent.follow_up(UserMessage(role="user", content="F2"))
        await agent.prompt("Q0")

        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("assistant") == 3
        assert agent.has_queued_messages() is False

    @pytest.mark.asyncio
    async def test_all_follow_up_mode_drains_all(self):
        """all：一次注入全部 follow-up。"""
        stream_fn = _make_modeled_stream_fn(["A1", "A2"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
                follow_up_mode="all",
            )
        )
        agent.follow_up(UserMessage(role="user", content="F1"))
        agent.follow_up(UserMessage(role="user", content="F2"))
        await agent.prompt("Q0")

        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("assistant") == 2
        contents = [m.get("content") for m in agent.state.messages]
        assert "F1" in contents and "F2" in contents

    @pytest.mark.asyncio
    async def test_steer_during_run_injected_next_turn(self):
        """运行中 steer() → 下一个 turn 边界注入（CLI 交互场景）。"""
        core = faux_provider(tokens_per_second=200)
        core.set_responses(
            [
                faux_assistant_message("A" * 200),
                faux_assistant_message("B" * 20),
            ]
        )
        original_stream = core.stream
        steer_done = asyncio.Event()

        async def _delayed_stream(model, context, options=None):
            # 首轮流式输出前等待 steer 被调用，保证 steer 落在 initial poll 之后
            if not steer_done.is_set():
                await steer_done.wait()
            return await original_stream(model, context, options)

        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=_delayed_stream,
            )
        )

        async def _steer_while_running():
            await asyncio.sleep(0.01)  # 确保 initial steering poll 已完成
            agent.steer(UserMessage(role="user", content="nudge"))
            steer_done.set()

        await asyncio.gather(agent.prompt("Q"), _steer_while_running())

        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("assistant") == 2
        contents = [m.get("content") for m in agent.state.messages]
        assert "nudge" in contents

    @pytest.mark.asyncio
    async def test_reset_clears_queues(self):
        """reset() 同时清空双消息队列。"""
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=_make_faux_stream_fn(),
            )
        )
        agent.steer(UserMessage(role="user", content="s"))
        agent.follow_up(UserMessage(role="user", content="f"))
        agent.reset()
        assert agent.has_queued_messages() is False


class TestAgentContinueQueues:
    """1.2：continue() 队列排空路径（对齐 TS continue()）。"""

    @pytest.mark.asyncio
    async def test_continue_after_assistant_drains_steering(self):
        """最后一条是 assistant + 有 steering → 作为 prompt 续跑。"""
        stream_fn = _make_modeled_stream_fn(["A1", "A2"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
        await agent.prompt("Q")
        assert agent.state.messages[-1].get("role") == "assistant"

        agent.steer(UserMessage(role="user", content="nudge"))
        await agent.continue_()

        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("assistant") == 2
        contents = [m.get("content") for m in agent.state.messages]
        assert "nudge" in contents
        assert agent.has_queued_messages() is False

    @pytest.mark.asyncio
    async def test_continue_after_assistant_drains_follow_up(self):
        """最后一条是 assistant + 只有 follow-up → 作为 prompt 续跑。"""
        stream_fn = _make_modeled_stream_fn(["A1", "A2"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
        await agent.prompt("Q")

        agent.follow_up(UserMessage(role="user", content="follow"))
        await agent.continue_()

        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("assistant") == 2
        contents = [m.get("content") for m in agent.state.messages]
        assert "follow" in contents

    @pytest.mark.asyncio
    async def test_continue_steering_one_at_a_time(self):
        """one-at-a-time：continue 只消费一条 steering，其余下一轮注入。"""
        stream_fn = _make_modeled_stream_fn(["A1", "A2", "A3"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )
        await agent.prompt("Q")

        agent.steer(UserMessage(role="user", content="s1"))
        agent.steer(UserMessage(role="user", content="s2"))
        await agent.continue_()

        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("assistant") == 3
        contents = [m.get("content") for m in agent.state.messages]
        assert "s1" in contents and "s2" in contents
        assert agent.has_queued_messages() is False

    @pytest.mark.asyncio
    async def test_continue_steering_mode_all(self):
        """all：continue 一次消费全部 steering。"""
        stream_fn = _make_modeled_stream_fn(["A1", "A2"])
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
                steering_mode="all",
            )
        )
        await agent.prompt("Q")

        agent.steer(UserMessage(role="user", content="s1"))
        agent.steer(UserMessage(role="user", content="s2"))
        await agent.continue_()

        roles = [m.get("role") for m in agent.state.messages]
        assert roles.count("assistant") == 2
        contents = [m.get("content") for m in agent.state.messages]
        assert "s1" in contents and "s2" in contents

    @pytest.mark.asyncio
    async def test_continue_empty_transcript_raises(self):
        """无消息时 continue 抛异常（对齐 TS）。"""
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=_make_faux_stream_fn(),
            )
        )
        with pytest.raises(RuntimeError, match="No messages to continue from"):
            await agent.continue_()


class TestAsyncListeners:
    """1.3：async 监听器 + AbortSignal 传播。"""

    @pytest.mark.asyncio
    async def test_async_listener_awaited_in_subscription_order(self):
        """监听器按订阅顺序 await（async 监听器先完成后才调用下一个）。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )

        order: list[str] = []

        async def _first(event, signal):
            if event["type"] == "agent_end":
                await asyncio.sleep(0.01)
                order.append("first")

        def _second(event, signal):
            if event["type"] == "agent_end":
                order.append("second")

        agent.subscribe(_first)
        agent.subscribe(_second)

        await agent.prompt("Hi")

        assert order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_listener_receives_abort_signal(self):
        """监听器收到当前运行的 abort signal（运行中非 None）。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )

        seen: list[asyncio.Event | None] = []

        def _record(event, signal):
            if event["type"] == "agent_start":
                seen.append(signal)

        agent.subscribe(_record)
        await agent.prompt("Hi")

        assert len(seen) == 1
        assert seen[0] is not None
        assert isinstance(seen[0], asyncio.Event)

    @pytest.mark.asyncio
    async def test_signal_same_object_across_events(self):
        """同一运行内所有事件共享同一个 signal 对象。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )

        signals: list[asyncio.Event] = []

        def _record(event, signal):
            if event["type"] in ("agent_start", "agent_end") and signal is not None:
                signals.append(signal)

        agent.subscribe(_record)
        await agent.prompt("Hi")

        assert len(signals) == 2
        assert signals[0] is signals[1]

    @pytest.mark.asyncio
    async def test_signal_set_after_abort(self):
        """abort 后 agent_end 监听器看到的 signal 已置位。"""
        core = faux_provider(tokens_per_second=200)
        core.set_responses([faux_assistant_message("A" * 300)])

        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=core.stream,
            )
        )

        agent_end_signal_state: list[bool] = []

        def _record(event, signal):
            if event["type"] == "agent_end" and signal is not None:
                agent_end_signal_state.append(signal.is_set())

        agent.subscribe(_record)

        async def _run_and_abort():
            await asyncio.sleep(0.05)
            agent.abort()

        await asyncio.wait_for(
            asyncio.gather(agent.prompt("Hi"), _run_and_abort()),
            timeout=2.0,
        )

        assert agent_end_signal_state == [True]

    @pytest.mark.asyncio
    async def test_wait_for_idle_waits_for_listener_settlement(self):
        """wait_for_idle 等待 agent_end 监听器 settle（而非仅 _active）。"""
        stream_fn = _make_faux_stream_fn("Hello!")
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=stream_fn,
            )
        )

        listener_done = asyncio.Event()

        async def _slow_listener(event, signal):
            if event["type"] == "agent_end":
                await asyncio.sleep(0.05)
                listener_done.set()

        agent.subscribe(_slow_listener)

        task = asyncio.create_task(agent.prompt("Hi"))
        await agent.wait_for_idle()
        await task

        assert listener_done.is_set()


def test_default_convert_to_llm_filters_non_standard_roles():
    """默认转换器只透传 user/assistant/toolResult（对齐 TS defaultConvertToLlm）。"""
    from pi_agent._agent import _default_convert_to_llm

    messages = [
        {"role": "user", "content": "hi", "timestamp": 1},
        {
            "role": "assistant",
            "content": [TextContent(type="text", text="ok")],
            "timestamp": 2,
        },
        {
            "role": "toolResult",
            "tool_call_id": "t1",
            "tool_name": "x",
            "content": [],
            "is_error": False,
            "timestamp": 3,
        },
        {"role": "compactionSummary", "summary": "old history", "timestamp": 4},
        {
            "role": "bashExecution",
            "command": "ls",
            "output": "",
            "exitCode": 0,
            "cancelled": False,
            "truncated": False,
            "timestamp": 5,
        },
    ]
    llm_messages = _default_convert_to_llm(messages)
    assert [m["role"] for m in llm_messages] == ["user", "assistant", "toolResult"]


def test_convert_to_llm_wraps_custom_roles():
    """应用层转换器包装 bashExecution/compactionSummary/branchSummary/custom。"""
    from pi_agent._messages import convert_to_llm

    messages = [
        {
            "role": "bashExecution",
            "command": "npm run lint",
            "output": "All checks passed",
            "exitCode": 0,
            "cancelled": False,
            "truncated": False,
            "timestamp": 1,
        },
        {
            "role": "bashExecution",
            "command": "secret-command",
            "output": "hidden",
            "exitCode": 1,
            "cancelled": False,
            "truncated": False,
            "excludeFromContext": True,
            "timestamp": 2,
        },
        {"role": "compactionSummary", "summary": "compacted", "timestamp": 3},
        {"role": "branchSummary", "summary": "branched", "timestamp": 4},
        {"role": "custom", "customType": "note", "content": "hello custom", "timestamp": 5},
        {"role": "user", "content": "keep", "timestamp": 6},
    ]
    llm_messages = convert_to_llm(messages)
    assert [m["role"] for m in llm_messages] == ["user"] * 5
    assert "Ran `npm run lint`" in str(llm_messages[0]["content"])
    assert "secret-command" not in str(llm_messages[0]["content"])
    assert "compacted" in str(llm_messages[1]["content"])
    assert "branched" in str(llm_messages[2]["content"])
    assert llm_messages[3]["content"] == [{"type": "text", "text": "hello custom"}]
    assert llm_messages[4]["content"] == "keep"


class TestAgentStreamOptionForwarding:
    """AgentOptions 的 thinking_budgets / transport 透传给 StreamOptions。"""

    @pytest.mark.asyncio
    async def test_thinking_budgets_and_transport_forwarded(self):
        captured: list[dict] = []
        core = _make_faux([faux_assistant_message("Hello")])
        original = core.stream

        async def _capturing_stream(model, context, options=None):
            captured.append(dict(options or {}))
            return await original(model, context, options)

        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=_capturing_stream,
                thinking_budgets={"high": 4096},
                transport="sse",
            )
        )
        await agent.prompt("Hi")

        assert captured
        assert captured[0]["thinking_budgets"] == {"high": 4096}
        assert captured[0]["transport"] == "sse"


class TestAgentPrepareNextTurnWithContext:
    """prepare_next_turn_with_context 接收 PrepareNextTurnContext 并可替换状态。"""

    @pytest.mark.asyncio
    async def test_receives_turn_context(self):
        received: list = []

        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=_make_faux_stream_fn("Hello"),
                prepare_next_turn_with_context=lambda ctx: received.append(ctx) or None,
            )
        )
        await agent.prompt("Hi")

        assert len(received) == 1
        ctx = received[0]
        assert ctx.message.get("role") == "assistant"
        assert ctx.tool_results == []
        assert ctx.context.messages[-1].get("role") == "assistant"
        roles = [m.get("role") for m in ctx.new_messages]
        assert roles == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_with_context_update_replaces_model(self):
        seen_models: list[str] = []
        core = _make_faux(
            [
                faux_assistant_message(
                    [faux_tool_call("finish", {})],
                    stop_reason="tool_call",
                ),
                faux_assistant_message("done"),
            ]
        )
        original = core.stream

        async def _capturing_stream(model, context, options=None):
            seen_models.append(model.id)
            return await original(model, context, options)

        model_b = _make_model()
        model_b.id = "model-b"
        agent = Agent(
            AgentOptions(
                model=_make_model(),
                stream_fn=_capturing_stream,
                tools=[_make_tool("finish")],
                prepare_next_turn_with_context=lambda ctx: AgentLoopTurnUpdate(model=model_b),
            )
        )
        await agent.prompt("Run the tool")

        assert seen_models == ["test-model", "model-b"]
