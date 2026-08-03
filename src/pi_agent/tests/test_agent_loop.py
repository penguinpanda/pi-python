"""_agent_loop.py 模块测试。

使用 Faux Provider 注入脚本化的 LLM 事件流，验证 agent 循环的各个路径。
"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai._types import (
    AssistantMessage,
    Model,
    TextContent,
    UserMessage,
)
from pi_ai.providers.faux import FauxCore, faux_assistant_message, faux_provider, faux_tool_call

from pi_agent._agent_loop import (
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
)
from pi_agent._types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
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

    其 `.stream` 签名即 StreamFn，可直接传给 run_agent_loop。
    """
    core = faux_provider()
    core.set_responses(responses)
    return core


def _make_stream_fn(message: AssistantMessage) -> StreamFn:
    """创建返回单条脚本化响应的 Faux stream_fn。"""
    return _make_faux([message]).stream


def _make_counting_stream_fn(messages: list[AssistantMessage]) -> StreamFn:
    """创建按调用次数顺序消费脚本化响应的 Faux stream_fn。"""
    return _make_faux(list(messages)).stream


def _make_llm_text_response(text: str) -> AssistantMessage:
    """创建纯文本 LLM 响应的 AssistantMessage。"""
    return faux_assistant_message(text)


def _make_llm_tool_response(
    tool_name: str,
    args: dict,
    tool_call_id: str = "tc-1",
) -> AssistantMessage:
    """创建工具调用 LLM 响应的 AssistantMessage。"""
    return faux_assistant_message(
        [faux_tool_call(tool_name, args, tool_call_id=tool_call_id)],
        stop_reason="tool_call",
    )


async def _collect_events(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    stream_fn: StreamFn,
    *,
    signal: asyncio.Event | None = None,
) -> tuple[list[AgentMessage], list[AgentEvent]]:
    """运行 agent loop 并收集所有事件。"""
    events: list[AgentEvent] = []

    async def _emit(evt: AgentEvent) -> None:
        events.append(evt)

    result = await run_agent_loop(
        prompts=prompts,
        context=context,
        config=config,
        emit=_emit,
        signal=signal,
        stream_fn=stream_fn,
    )
    return result, events


def _find_events(events: list[AgentEvent], event_type: str) -> list[AgentEvent]:
    """从事件列表中筛选指定类型的事件。"""
    return [e for e in events if e["type"] == event_type]


def _make_tool(name: str, result_text: str = "ok") -> AgentTool:
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


class TestSingleTurnText:
    """正常单轮纯文本对话。"""

    @pytest.mark.asyncio
    async def test_basic(self):
        prompts = [UserMessage(role="user", content="Hello")]
        context = AgentContext(system_prompt="You are helpful.", messages=[])

        final = _make_llm_text_response("Hi there!")
        stream_fn = _make_stream_fn(final)

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # 事件序列检查
        event_types = [e["type"] for e in events]
        assert "agent_start" in event_types
        assert "turn_start" in event_types
        assert "message_start" in event_types
        assert "message_end" in event_types
        assert "turn_end" in event_types
        assert "agent_end" in event_types

        # agent_start 在 turn_start 之前
        agent_start_idx = event_types.index("agent_start")
        turn_start_idx = event_types.index("turn_start")
        assert agent_start_idx < turn_start_idx

        # 结果包含助手消息
        assert len(result) >= 2  # user + assistant at minimum
        roles = [m.get("role") for m in result]
        assert "assistant" in roles


class TestToolCallLoop:
    """工具调用循环测试。"""

    @pytest.mark.asyncio
    async def test_single_tool_call_then_text(self):
        """LLM 先调用工具，拿到结果后给文本回复。"""
        tool = _make_tool("search", "search results here")
        prompts = [UserMessage(role="user", content="Search for X")]
        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[],
            tools=[tool],
        )

        # Turn 1: tool call
        tc_final = _make_llm_tool_response("search", {"q": "X"})
        # Turn 2: text response
        text_final = _make_llm_text_response("Found it!")

        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # 应该有两次 turn
        turn_start_events = _find_events(events, "turn_start")
        assert len(turn_start_events) == 2

        turn_end_events = _find_events(events, "turn_end")
        assert len(turn_end_events) == 2

        # 工具执行事件
        tool_start = _find_events(events, "tool_execution_start")
        assert len(tool_start) == 1
        assert tool_start[0]["tool_name"] == "search"

        tool_end = _find_events(events, "tool_execution_end")
        assert len(tool_end) == 1
        assert tool_end[0]["is_error"] is False

        # 结果中包含 toolResult 消息
        roles = [m.get("role") for m in result]
        assert "toolResult" in roles


    @pytest.mark.asyncio
    async def test_tool_execution_error(self):
        """工具执行异常 → 转化为 is_error=true 的结果。"""
        async def _failing_execute(tool_call_id, params, signal=None, on_update=None):
            raise ValueError("tool failed")

        tool = AgentTool(
            name="bad_tool",
            description="Always fails",
            input_schema={"type": "object", "properties": {}},
            label="bad_tool",
            execute=_failing_execute,
        )

        prompts = [UserMessage(role="user", content="Use bad_tool")]
        context = AgentContext(
            system_prompt="test",
            messages=[],
            tools=[tool],
        )

        tc_final = _make_llm_tool_response("bad_tool", {})
        text_final = _make_llm_text_response("ok")
        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        tool_end = _find_events(events, "tool_execution_end")
        assert len(tool_end) == 1
        assert tool_end[0]["is_error"] is True


class TestToolArgumentValidation:
    """工具参数 schema 校验（_execute_tool_calls 阶段1）。"""

    @pytest.mark.asyncio
    async def test_invalid_arguments_produce_error_result(self):
        """参数不符合 schema → 校验失败 → is_error 结果（不执行工具）。"""
        executed: list[dict] = []

        async def _execute(tool_call_id, params, signal=None, on_update=None):
            executed.append(params)
            return AgentToolResult(content=[TextContent(type="text", text="ok")])

        tool = AgentTool(
            name="add",
            description="Add two integers",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            label="add",
            execute=_execute,
        )
        prompts = [UserMessage(role="user", content="Add 1 and 2")]
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])

        # Turn 1: 缺少 b，且 a 为字符串 "1"（integer 可转换，但 b 缺失 → 失败）
        tc_final = _make_llm_tool_response("add", {"a": "1"})
        text_final = _make_llm_text_response("done")
        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        tool_end = _find_events(events, "tool_execution_end")
        assert len(tool_end) == 1
        assert tool_end[0]["is_error"] is True
        assert tool_end[0]["tool_name"] == "add"
        # 工具不应被执行
        assert executed == []
        # 错误详情标记 invalid_arguments
        assert tool_end[0]["result"].details["error"] == "invalid_arguments"
        # 错误结果回给 LLM（后续文本回复正常完成）
        roles = [m.get("role") for m in result]
        assert "toolResult" in roles

    @pytest.mark.asyncio
    async def test_valid_arguments_coerced_before_execute(self):
        """参数通过校验时，转换后的参数传给 execute。"""
        received: list[dict] = []

        async def _execute(tool_call_id, params, signal=None, on_update=None):
            received.append(dict(params))
            return AgentToolResult(content=[TextContent(type="text", text="ok")])

        tool = AgentTool(
            name="add",
            description="Add two integers",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            label="add",
            execute=_execute,
        )
        prompts = [UserMessage(role="user", content="Add 1 and 2")]
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])

        # LLM 返回字符串 "1"/"2" → 校验后转换为 int
        tc_final = _make_llm_tool_response("add", {"a": "1", "b": "2"})
        text_final = _make_llm_text_response("done")
        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        tool_end = _find_events(events, "tool_execution_end")
        assert len(tool_end) == 1
        assert tool_end[0]["is_error"] is False
        assert received == [{"a": 1, "b": 2}]


class TestLengthTruncation:
    """stop_reason="length" 截断保护。"""

    @pytest.mark.asyncio
    async def test_truncated_tool_calls_marked_error(self):
        """LLM 返回 length stop_reason → 所有工具调用标记为错误（不执行）。"""
        tool = _make_tool("search")

        prompts = [UserMessage(role="user", content="Search")]
        context = AgentContext(
            system_prompt="test",
            messages=[],
            tools=[tool],
        )

        # 模拟 length 截断响应
        final = faux_assistant_message(
            [faux_tool_call("search", {"q": "incomplete"}, tool_call_id="tc-1")],
            stop_reason="length",
        )
        stream_fn = _make_stream_fn(final)

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # 工具不应被实际执行（无 tool_execution_start）
        tool_start = _find_events(events, "tool_execution_start")
        assert len(tool_start) == 0

        # 结果中包含 toolResult 消息（错误结果）
        tool_results = [m for m in result if m.get("role") == "toolResult"]
        assert len(tool_results) == 1


class TestHooks:
    """beforeToolCall / afterToolCall 钩子测试。"""

    @pytest.mark.asyncio
    async def test_before_tool_call_block(self):
        """beforeToolCall 返回 block=True → 阻止执行。"""
        tool = _make_tool("search")

        prompts = [UserMessage(role="user", content="Search")]
        context = AgentContext(
            system_prompt="test",
            messages=[],
            tools=[tool],
        )

        tc_final = _make_llm_tool_response("search", {"q": "X"})
        text_final = _make_llm_text_response("ok")
        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        def _before(ctx):
            from pi_agent._types import BeforeToolCallResult
            return BeforeToolCallResult(block=True, reason="not allowed")

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            before_tool_call=_before,
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # 工具不应被实际执行
        tool_start = _find_events(events, "tool_execution_start")
        assert len(tool_start) == 0

        # 工具结果应是 is_error=true
        tool_end = _find_events(events, "tool_execution_end")
        assert len(tool_end) == 1
        assert tool_end[0]["is_error"] is True


    @pytest.mark.asyncio
    async def test_after_tool_call_override(self):
        """afterToolCall 覆盖工具结果。"""
        tool = _make_tool("search", "original result")

        prompts = [UserMessage(role="user", content="Search")]
        context = AgentContext(
            system_prompt="test",
            messages=[],
            tools=[tool],
        )

        tc_final = _make_llm_tool_response("search", {"q": "X"})
        text_final = _make_llm_text_response("ok")
        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        def _after(ctx):
            from pi_agent._types import AfterToolCallResult
            return AfterToolCallResult(
                content=[TextContent(type="text", text="overridden!")],
                terminate=True,
            )

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            after_tool_call=_after,
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # 结果中 toolResult 的 content 应被覆盖
        tool_results = [m for m in result if m.get("role") == "toolResult"]
        assert len(tool_results) == 1
        content = tool_results[0].get("content", [])
        assert len(content) == 1
        assert content[0]["text"] == "overridden!"


class TestTerminate:
    """terminate=true 提前退出。"""

    @pytest.mark.asyncio
    async def test_terminate_stops_loop(self):
        """工具返回 terminate=true → 不再继续下一轮。"""
        async def _terminating_execute(tool_call_id, params, signal=None, on_update=None):
            result = AgentToolResult(
                content=[TextContent(type="text", text="done")],
                terminate=True,
            )
            return result

        tool = AgentTool(
            name="finish",
            description="Finishes the task",
            input_schema={"type": "object", "properties": {}},
            label="finish",
            execute=_terminating_execute,
        )

        prompts = [UserMessage(role="user", content="Finish")]
        context = AgentContext(
            system_prompt="test",
            messages=[],
            tools=[tool],
        )

        tc_final = _make_llm_tool_response("finish", {})
        stream_fn = _make_stream_fn(tc_final)

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # 只应有一次 turn（terminate 阻止了后续 turn）
        turn_start = _find_events(events, "turn_start")
        assert len(turn_start) == 1


class TestShouldStopAfterTurn:
    """shouldStopAfterTurn 钩子。"""

    @pytest.mark.asyncio
    async def test_stop_after_turn(self):
        """shouldStopAfterTurn 返回 True → 立即停止。"""
        prompts = [UserMessage(role="user", content="Hi")]
        context = AgentContext(system_prompt="test", messages=[])

        final = _make_llm_text_response("Hello")
        stream_fn = _make_stream_fn(final)

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            should_stop_after_turn=lambda ctx: True,
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # 应该成功完成（agent_end 存在）
        agent_end = _find_events(events, "agent_end")
        assert len(agent_end) == 1


class TestCancellation:
    """取消信号测试。"""

    @pytest.mark.asyncio
    async def test_cancellation_during_loop(self):
        """在循环中设置取消信号 → CancelledError。"""
        prompts = [UserMessage(role="user", content="Hi")]
        context = AgentContext(system_prompt="test", messages=[])

        final = _make_llm_text_response("Hello")
        stream_fn = _make_stream_fn(final)

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        signal = asyncio.Event()
        signal.set()  # 立即设置

        async def _noop(event: AgentEvent) -> None:
            return

        with pytest.raises(asyncio.CancelledError):
            await run_agent_loop(
                prompts=prompts,
                context=context,
                config=config,
                emit=_noop,
                signal=signal,
                stream_fn=stream_fn,
            )


class TestLLMError:
    """LLM 返回错误测试。"""

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self):
        """LLM 返回 error 事件 → 通过 agent_end 传播。"""
        prompts = [UserMessage(role="user", content="Hi")]
        context = AgentContext(system_prompt="test", messages=[])

        error_msg = faux_assistant_message(
            [], stop_reason="error", error_message="API call failed"
        )
        stream_fn = _make_stream_fn(error_msg)

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # 应该有 agent_end 事件
        agent_end = _find_events(events, "agent_end")
        assert len(agent_end) == 1


class TestRunAgentLoopContinue:
    """run_agent_loop_continue 测试。"""

    @pytest.mark.asyncio
    async def test_continue_from_existing_context(self):
        """从已有上下文继续执行。"""
        context = AgentContext(
            system_prompt="You are helpful.",
            messages=[
                UserMessage(role="user", content="Hello"),
                {
                    "role": "assistant",
                    "content": [TextContent(type="text", text="Hi!")],
                    "api": "test",
                    "provider": "test",
                    "model": "test",
                },
            ],
        )

        final = _make_llm_text_response("How can I help?")
        stream_fn = _make_stream_fn(final)

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        events: list[AgentEvent] = []

        async def _emit(e: AgentEvent) -> None:
            events.append(e)

        result = await run_agent_loop_continue(
            context=context,
            config=config,
            emit=_emit,
            signal=None,
            stream_fn=stream_fn,
        )

        # 验证事件序列
        assert len(_find_events(events, "agent_start")) == 1
        assert len(_find_events(events, "agent_end")) == 1
        assert len(result) >= 2


class TestToolLifecycle:
    """Tool 生命周期钩子（before_execute / after_execute）。"""

    @pytest.mark.asyncio
    async def test_before_execute_replaces_args(self):
        """before_execute 返回的 dict 应替换传给 execute 的参数。"""
        seen_args: list[dict] = []

        async def _execute(tool_call_id, params, signal=None, on_update=None):
            seen_args.append(params)
            return AgentToolResult(
                content=[TextContent(type="text", text="ok")],
            )

        async def _before(params, ctx):
            return {"q": "replaced"}

        tool = AgentTool(
            name="search",
            description="Tool: search",
            input_schema={"type": "object", "properties": {}},
            label="search",
            execute=_execute,
            before_execute=_before,
        )
        prompts = [UserMessage(role="user", content="Search")]
        context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[tool])

        tc_final = _make_llm_tool_response("search", {"q": "original"})
        text_final = _make_llm_text_response("done")
        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        assert seen_args == [{"q": "replaced"}]
        # tool_execution_start 事件也应携带替换后的参数
        tool_start = _find_events(events, "tool_execution_start")
        assert tool_start[0]["args"] == {"q": "replaced"}

    @pytest.mark.asyncio
    async def test_after_execute_replaces_result(self):
        """after_execute 返回的新值应替换最终工具结果。"""
        async def _execute(tool_call_id, params, signal=None, on_update=None):
            return AgentToolResult(
                content=[TextContent(type="text", text="original")],
            )

        async def _after(result):
            return AgentToolResult(
                content=[TextContent(type="text", text="post-processed")],
            )

        tool = AgentTool(
            name="search",
            description="Tool: search",
            input_schema={"type": "object", "properties": {}},
            label="search",
            execute=_execute,
            after_execute=_after,
        )
        prompts = [UserMessage(role="user", content="Search")]
        context = AgentContext(system_prompt="You are helpful.", messages=[], tools=[tool])

        tc_final = _make_llm_tool_response("search", {"q": "x"})
        text_final = _make_llm_text_response("done")
        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # toolResult 消息应包含 after_execute 替换后的文本
        tr_messages = [m for m in result if m.get("role") == "toolResult"]
        assert len(tr_messages) == 1
        text = "".join(
            b["text"] for b in tr_messages[0]["content"] if b["type"] == "text"
        )
        assert text == "post-processed"


class TestPromptCacheWiring:
    """AgentLoopConfig.session_id / cache_retention 透传到 stream options。"""

    @pytest.mark.asyncio
    async def test_session_id_and_retention_flow_to_options(self):
        captured: dict = {}

        core = _make_faux([_make_llm_text_response("ok")])
        original_stream = core.stream

        async def capturing_stream(model, context, options=None):
            captured["session_id"] = options.get("session_id") if options else None
            captured["cache_retention"] = options.get("cache_retention") if options else None
            return await original_stream(model, context, options)

        prompts = [UserMessage(role="user", content="Hello")]
        context = AgentContext(system_prompt="You are helpful.", messages=[])
        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            session_id="session-abc",
            cache_retention="long",
        )

        await _collect_events(prompts, context, config, capturing_stream)

        assert captured["session_id"] == "session-abc"
        assert captured["cache_retention"] == "long"

    @pytest.mark.asyncio
    async def test_no_session_id_leaves_options_unset(self):
        captured: dict = {}

        core = _make_faux([_make_llm_text_response("ok")])
        original_stream = core.stream

        async def capturing_stream(model, context, options=None):
            captured["session_id"] = options.get("session_id") if options else None
            captured["cache_retention"] = options.get("cache_retention") if options else None
            return await original_stream(model, context, options)

        prompts = [UserMessage(role="user", content="Hello")]
        context = AgentContext(system_prompt="You are helpful.", messages=[])
        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        await _collect_events(prompts, context, config, capturing_stream)

        assert captured["session_id"] is None
        assert captured["cache_retention"] is None


class TestFollowUpLoop:
    """1.1 双重嵌套循环：Follow-up 外层测试。"""

    @pytest.mark.asyncio
    async def test_follow_up_continues_after_text_stop(self):
        """纯文本回复后存在 follow-up → 追加一轮再停止。"""
        prompts = [UserMessage(role="user", content="Hi")]
        context = AgentContext(system_prompt="test", messages=[])

        stream_fn = _make_counting_stream_fn([
            _make_llm_text_response("First answer"),
            _make_llm_text_response("Second answer"),
        ])

        follow_up_msg = UserMessage(role="user", content="And then?")
        poll_count = 0

        async def _get_follow_up() -> list[AgentMessage]:
            nonlocal poll_count
            poll_count += 1
            return [follow_up_msg] if poll_count == 1 else []

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            get_follow_up_messages=_get_follow_up,
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # 两次 turn（首轮 + follow-up 追加轮）
        assert len(_find_events(events, "turn_start")) == 2
        # follow-up 消息被注入结果
        assert follow_up_msg in result
        # 只应有一个 agent_end
        assert len(_find_events(events, "agent_end")) == 1

    @pytest.mark.asyncio
    async def test_follow_up_drives_tool_turn(self):
        """follow-up 触发的下一轮可以再次使用工具。"""
        tool = _make_tool("search", "results")
        prompts = [UserMessage(role="user", content="First")]
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])

        stream_fn = _make_counting_stream_fn([
            _make_llm_text_response("First answer"),
            _make_llm_tool_response("search", {"q": "X"}),
            _make_llm_text_response("Final answer"),
        ])

        follow_up_msg = UserMessage(role="user", content="Now search")
        poll_count = 0

        async def _get_follow_up() -> list[AgentMessage]:
            nonlocal poll_count
            poll_count += 1
            return [follow_up_msg] if poll_count == 1 else []

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            get_follow_up_messages=_get_follow_up,
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        assert len(_find_events(events, "turn_start")) == 3
        assert len(_find_events(events, "tool_execution_start")) == 1
        assert follow_up_msg in result
        assert len(_find_events(events, "agent_end")) == 1

    @pytest.mark.asyncio
    async def test_follow_up_empty_returns_normally(self):
        """无 follow-up 时保持原有单轮行为。"""
        prompts = [UserMessage(role="user", content="Hi")]
        context = AgentContext(system_prompt="test", messages=[])

        final = _make_llm_text_response("Hello")
        stream_fn = _make_stream_fn(final)

        async def _get_follow_up() -> list[AgentMessage]:
            return []

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            get_follow_up_messages=_get_follow_up,
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        assert len(_find_events(events, "turn_start")) == 1
        assert len(_find_events(events, "agent_end")) == 1


class TestSteeringLoop:
    """1.1 双重嵌套循环：Steering 注入测试。"""

    @pytest.mark.asyncio
    async def test_initial_steering_injected_first_turn(self):
        """运行前已入队的 steering 消息在首轮注入。"""
        prompts = [UserMessage(role="user", content="Hi")]
        context = AgentContext(system_prompt="test", messages=[])

        stream_fn = _make_counting_stream_fn([
            _make_llm_text_response("A1"),
            _make_llm_text_response("A2"),
        ])

        steering_msg = UserMessage(role="user", content="nudge")
        poll_count = 0

        async def _get_steering() -> list[AgentMessage]:
            nonlocal poll_count
            poll_count += 1
            return [steering_msg] if poll_count == 1 else []

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            get_steering_messages=_get_steering,
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        assert steering_msg in result
        assert len(_find_events(events, "turn_start")) == 1

    @pytest.mark.asyncio
    async def test_steering_injected_between_tool_turns(self):
        """工具轮结束后轮询 steering → 注入下一轮。"""
        tool = _make_tool("search")
        prompts = [UserMessage(role="user", content="Search")]
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])

        stream_fn = _make_counting_stream_fn([
            _make_llm_tool_response("search", {"q": "X"}),
            _make_llm_text_response("done"),
        ])

        steering_msg = UserMessage(role="user", content="Wait, refine")
        poll_count = 0

        async def _get_steering() -> list[AgentMessage]:
            nonlocal poll_count
            poll_count += 1
            # 首次轮询（运行前）返回空；工具轮结束后返回引导消息
            return [steering_msg] if poll_count == 2 else []

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            get_steering_messages=_get_steering,
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        assert steering_msg in result
        assert len(_find_events(events, "turn_start")) == 2
        assert len(_find_events(events, "tool_execution_start")) == 1


class TestToolCallContexts:
    """1.5：beforeToolCall / afterToolCall 专用 context 对象。"""

    @pytest.mark.asyncio
    async def test_before_tool_call_receives_context(self):
        """beforeToolCall 收到 BeforeToolCallContext（含 assistant 消息与 toolCall）。"""
        tool = _make_tool("search", "results")
        prompts = [UserMessage(role="user", content="Search")]
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])

        tc_final = _make_llm_tool_response("search", {"q": "X"})
        text_final = _make_llm_text_response("done")
        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        received: list = []

        def _before(hook_ctx):
            received.append(hook_ctx)
            return None

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            before_tool_call=_before,
        )

        await _collect_events(prompts, context, config, stream_fn)

        assert len(received) == 1
        hook_ctx = received[0]
        assert hook_ctx.tool_call["id"] == "tc-1"
        assert hook_ctx.tool_call["name"] == "search"
        assert hook_ctx.args == {"q": "X"}
        # assistant_message 携带触发该调用的 toolCall 块
        blocks = hook_ctx.assistant_message.get("content", [])
        assert any(b.get("type") == "toolCall" for b in blocks)
        # context 携带本轮 messages（user prompt + assistant 消息）
        assert len(hook_ctx.context.messages) >= 2

    @pytest.mark.asyncio
    async def test_after_tool_call_receives_context(self):
        """afterToolCall 收到 AfterToolCallContext（含 result 与 is_error）。"""
        tool = _make_tool("search", "results")
        prompts = [UserMessage(role="user", content="Search")]
        context = AgentContext(system_prompt="test", messages=[], tools=[tool])

        tc_final = _make_llm_tool_response("search", {"q": "X"})
        text_final = _make_llm_text_response("done")
        stream_fn = _make_counting_stream_fn([tc_final, text_final])

        received: list = []

        def _after(hook_ctx):
            received.append(hook_ctx)
            return None

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            after_tool_call=_after,
        )

        await _collect_events(prompts, context, config, stream_fn)

        assert len(received) == 1
        hook_ctx = received[0]
        assert hook_ctx.is_error is False
        assert hook_ctx.result.content[0]["text"] == "results"
        assert hook_ctx.tool_call["name"] == "search"
        assert hook_ctx.assistant_message.get("role") == "assistant"
        assert len(hook_ctx.context.messages) >= 2


class TestParallelToolExecution:
    """1.4：Parallel 工具执行。"""

    @staticmethod
    def _two_call_message() -> AssistantMessage:
        """单条 assistant 消息携带两个 toolCall。"""
        return faux_assistant_message(
            [
                faux_tool_call("a", {}, tool_call_id="tc-a"),
                faux_tool_call("b", {}, tool_call_id="tc-b"),
            ],
            stop_reason="tool_call",
        )

    @pytest.mark.asyncio
    async def test_parallel_executes_concurrently(self):
        """parallel 模式：两个工具并发执行（屏障证明重叠）。"""
        barrier = asyncio.Event()
        started: list[int] = [0]

        async def _slow_execute(tool_call_id, params, signal=None, on_update=None):
            started[0] += 1
            if started[0] == 2:
                barrier.set()
            await barrier.wait()
            return AgentToolResult(
                content=[TextContent(type="text", text="ok")],
            )

        tools = [
            AgentTool(
                name="a",
                description="Tool: a",
                input_schema={"type": "object", "properties": {}},
                label="a",
                execute=_slow_execute,
            ),
            AgentTool(
                name="b",
                description="Tool: b",
                input_schema={"type": "object", "properties": {}},
                label="b",
                execute=_slow_execute,
            ),
        ]
        prompts = [UserMessage(role="user", content="Run both")]
        context = AgentContext(system_prompt="test", messages=[], tools=tools)
        text_final = _make_llm_text_response("done")
        stream_fn = _make_counting_stream_fn([self._two_call_message(), text_final])

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            tool_execution="parallel",
        )

        result, events = await asyncio.wait_for(
            _collect_events(prompts, context, config, stream_fn),
            timeout=2.0,
        )

        # 两个工具都已启动（并发 → 任一工具等待前 started 已达 2；顺序会死锁超时）
        assert started[0] == 2
        assert len(_find_events(events, "tool_execution_start")) == 2
        assert len(_find_events(events, "tool_execution_end")) == 2
        # ToolResultMessage 按 assistant 原始顺序
        trs = [m for m in result if m.get("role") == "toolResult"]
        assert [t.get("tool_name") for t in trs] == ["a", "b"]
        # 工具批次后文本轮正常完成
        assert len(_find_events(events, "turn_start")) == 2

    @pytest.mark.asyncio
    async def test_sequential_tool_forces_batch_sequential(self):
        """config=parallel 但批次内工具声明 sequential → 整批回退顺序执行。"""
        a_done = asyncio.Event()
        trace: list[str] = []

        def _make_tool(name: str, text: str, execution_mode) -> AgentTool:
            async def _execute(tool_call_id, params, signal=None, on_update=None):
                trace.append(name + ":start")
                if name == "a":
                    await asyncio.sleep(0.05)
                    a_done.set()
                    trace.append(name + ":end")
                else:
                    trace.append(f"b:saw_a_done={a_done.is_set()}")
                    trace.append(name + ":end")
                return AgentToolResult(
                    content=[TextContent(type="text", text=text)],
                )

            return AgentTool(
                name=name,
                description=f"Tool: {name}",
                input_schema={"type": "object", "properties": {}},
                label=name,
                execute=_execute,
                execution_mode=execution_mode,
            )

        tools = [
            _make_tool("a", "A-result", "parallel"),
            _make_tool("b", "B-result", "sequential"),
        ]
        prompts = [UserMessage(role="user", content="Run both")]
        context = AgentContext(system_prompt="test", messages=[], tools=tools)
        text_final = _make_llm_text_response("done")
        stream_fn = _make_counting_stream_fn([self._two_call_message(), text_final])

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            tool_execution="parallel",
        )

        result, events = await _collect_events(prompts, context, config, stream_fn)

        # b 在 a 结束后才启动（顺序回退，无并发）
        assert trace == [
            "a:start", "a:end", "b:start", "b:saw_a_done=True", "b:end",
        ]
        assert len(_find_events(events, "tool_execution_end")) == 2

    @pytest.mark.asyncio
    async def test_config_sequential_forces_sequential(self):
        """config="sequential" → 即使工具默认 parallel 也整批顺序执行。"""
        a_done = asyncio.Event()
        trace: list[str] = []

        def _make_tool(name: str, text: str) -> AgentTool:
            async def _execute(tool_call_id, params, signal=None, on_update=None):
                trace.append(name + ":start")
                if name == "a":
                    await asyncio.sleep(0.05)
                    a_done.set()
                    trace.append(name + ":end")
                else:
                    trace.append(f"b:saw_a_done={a_done.is_set()}")
                    trace.append(name + ":end")
                return AgentToolResult(
                    content=[TextContent(type="text", text=text)],
                )

            return AgentTool(
                name=name,
                description=f"Tool: {name}",
                input_schema={"type": "object", "properties": {}},
                label=name,
                execute=_execute,
            )

        tools = [_make_tool("a", "A-result"), _make_tool("b", "B-result")]
        prompts = [UserMessage(role="user", content="Run both")]
        context = AgentContext(system_prompt="test", messages=[], tools=tools)
        text_final = _make_llm_text_response("done")
        stream_fn = _make_counting_stream_fn([self._two_call_message(), text_final])

        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
            tool_execution="sequential",
        )

        await _collect_events(prompts, context, config, stream_fn)

        assert trace == [
            "a:start", "a:end", "b:start", "b:saw_a_done=True", "b:end",
        ]


class TestAgentLoopEventStream:
    """1.6：agentLoop() / agentLoopContinue() EventStream 包装。"""

    @pytest.mark.asyncio
    async def test_agent_loop_stream_yields_events_and_result(self):
        prompts = [UserMessage(role="user", content="Hi")]
        context = AgentContext(system_prompt="test", messages=[])
        final = _make_llm_text_response("Hello!")
        stream_fn = _make_stream_fn(final)
        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        stream = agent_loop(prompts, context, config, None, stream_fn)

        events: list[AgentEvent] = []
        async for event in stream:
            events.append(event)

        types = [e["type"] for e in events]
        assert "agent_start" in types
        assert types[-1] == "agent_end"

        messages = await stream.result()
        assert [m.get("role") for m in messages] == ["user", "assistant"]

    @pytest.mark.asyncio
    async def test_agent_loop_continue_stream(self):
        context = AgentContext(
            system_prompt="test",
            messages=[UserMessage(role="user", content="Hi")],
        )
        final = _make_llm_text_response("Hello!")
        stream_fn = _make_stream_fn(final)
        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )

        stream = agent_loop_continue(context, config, None, stream_fn)

        events: list[AgentEvent] = []
        async for event in stream:
            events.append(event)
        assert events[-1]["type"] == "agent_end"

        messages = await stream.result()
        assert messages[-1].get("role") == "assistant"

    @pytest.mark.asyncio
    async def test_agent_loop_continue_validates_context(self):
        """agentLoopContinue 对空 / assistant 结尾的 context 同步抛异常（对齐 TS）。"""
        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )
        stream_fn = _make_stream_fn(_make_llm_text_response("x"))

        empty_ctx = AgentContext(system_prompt="test", messages=[])
        with pytest.raises(ValueError, match="no messages"):
            agent_loop_continue(empty_ctx, config, None, stream_fn)

        assistant_last_ctx = AgentContext(
            system_prompt="test",
            messages=[
                UserMessage(role="user", content="Q"),
                {
                    "role": "assistant",
                    "content": [TextContent(type="text", text="A")],
                    "api": "test",
                    "provider": "test",
                    "model": "test",
                },
            ],
        )
        with pytest.raises(ValueError, match="assistant"):
            agent_loop_continue(assistant_last_ctx, config, None, stream_fn)
