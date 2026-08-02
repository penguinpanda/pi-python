"""_agent_loop.py 模块测试。

使用 Faux Provider 注入脚本化的 LLM 事件流，验证 agent 循环的各个路径。
"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai._types import (
    AssistantMessage,
    Model,
    ModelCapabilities,
    TextContent,
    UserMessage,
)
from pi_ai.providers.faux import FauxCore, faux_assistant_message, faux_provider, faux_tool_call

from pi_agent._agent_loop import run_agent_loop, run_agent_loop_continue
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
        capabilities=ModelCapabilities(tools=True),
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

        def _before(tc_id, tc_name, args, ctx):
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

        def _after(tc_id, tc_name, result, is_error, ctx):
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
