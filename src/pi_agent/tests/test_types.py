"""_types.py 模块测试。"""

from pi_ai.types import Model, TextContent
from pi_agent._types import (
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentState,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
    PrepareNextTurnContext,
)


def _make_model() -> Model:
    return Model(
        id="test-model",
        provider="test",
        api="openai-completions",
        name="Test Model",
    )


def _make_tool(name: str = "test_tool") -> AgentTool:
    async def _execute(tool_call_id, params, signal=None, on_update=None):
        return AgentToolResult(
            content=[TextContent(type="text", text=f"result from {name}")],
        )

    return AgentTool(
        name=name,
        description=f"Tool: {name}",
        input_schema={"type": "object", "properties": {}},
        label=name,
        execute=_execute,
    )


class TestAgentTool:
    def test_construct(self):
        tool = _make_tool()
        assert tool.name == "test_tool"
        assert tool.execution_mode == "parallel"

    def test_result_defaults(self):
        result = AgentToolResult(
            content=[TextContent(type="text", text="ok")],
        )
        assert result.content[0]["text"] == "ok"
        assert result.details is None
        assert result.usage is None
        assert result.terminate is False
        assert result.added_tool_names is None


class TestAgentState:
    def test_defensive_copy_tools(self):
        tool = _make_tool()
        state = AgentState(
            system_prompt="test",
            model=_make_model(),
        )
        state.tools = [tool]
        # 修改返回的 list 不影响内部
        tools_copy = state.tools
        tools_copy.append(_make_tool("other"))
        assert len(state.tools) == 1

    def test_defensive_copy_messages(self):
        state = AgentState(
            system_prompt="test",
            model=_make_model(),
        )
        msg = TextContent(type="text", text="hello")
        state.messages = [msg]  # type: ignore[assignment]
        msgs_copy = state.messages
        msgs_copy.append(TextContent(type="text", text="world"))
        assert len(state.messages) == 1

    def test_append_message(self):
        state = AgentState(
            system_prompt="test",
            model=_make_model(),
        )
        msg = TextContent(type="text", text="hello")
        state._append_message(msg)  # type: ignore[arg-type]
        assert len(state.messages) == 1

    def test_pending_tool_calls_default_empty(self):
        state = AgentState(
            system_prompt="test",
            model=_make_model(),
        )
        assert state.pending_tool_calls == set()


class TestAgentContext:
    def test_construct_minimal(self):
        ctx = AgentContext(system_prompt="test", messages=[])
        assert ctx.system_prompt == "test"
        assert ctx.messages == []
        assert ctx.tools is None

    def test_construct_with_tools(self):
        tool = _make_tool()
        ctx = AgentContext(system_prompt="test", messages=[], tools=[tool])
        assert ctx.tools is not None
        assert len(ctx.tools) == 1


class TestAgentEvent:
    def test_agent_start(self):
        evt: AgentEvent = {"type": "agent_start"}
        assert evt["type"] == "agent_start"

    def test_agent_end(self):
        evt: AgentEvent = {"type": "agent_end", "messages": []}
        assert evt["type"] == "agent_end"
        assert evt["messages"] == []

    def test_turn_start(self):
        evt: AgentEvent = {"type": "turn_start"}
        assert evt["type"] == "turn_start"

    def test_turn_end(self):
        msg: dict = {
            "role": "assistant",
            "content": [],
            "api": "test",
            "provider": "test",
            "model": "test",
        }
        evt: AgentEvent = {"type": "turn_end", "message": msg, "tool_results": []}
        assert evt["type"] == "turn_end"
        assert evt["tool_results"] == []

    def test_message_start(self):
        msg: dict = {"role": "user", "content": "hi"}
        evt: AgentEvent = {"type": "message_start", "message": msg}
        assert evt["type"] == "message_start"

    def test_tool_execution_start(self):
        evt: AgentEvent = {
            "type": "tool_execution_start",
            "tool_call_id": "tcid-1",
            "tool_name": "search",
            "args": {"q": "test"},
        }
        assert evt["tool_call_id"] == "tcid-1"
        assert evt["tool_name"] == "search"

    def test_tool_execution_end(self):
        result = AgentToolResult(
            content=[TextContent(type="text", text="ok")],
        )
        evt: AgentEvent = {
            "type": "tool_execution_end",
            "tool_call_id": "tcid-1",
            "tool_name": "search",
            "result": result,
            "is_error": False,
        }
        assert evt["is_error"] is False


class TestAgentLoopConfig:
    def test_construct_minimal(self):
        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )
        assert config.model.id == "test-model"
        assert config.tool_execution == "parallel"
        assert config.transform_context is None

    def test_all_hooks_default_to_none(self):
        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )
        assert config.get_api_key is None
        assert config.should_stop_after_turn is None
        assert config.prepare_next_turn is None
        assert config.before_tool_call is None
        assert config.after_tool_call is None

    def test_stream_option_fields_default_to_none(self):
        config = AgentLoopConfig(
            model=_make_model(),
            convert_to_llm=lambda msgs: list(msgs),  # type: ignore[arg-type,return-value]
        )
        assert config.thinking_budgets is None
        assert config.transport is None

    def test_prepare_next_turn_context_construct(self):
        ctx = PrepareNextTurnContext(
            message={
                "role": "assistant",
                "content": [TextContent(type="text", text="ok")],
                "api": "test",
                "provider": "test",
                "model": "test-model",
            },
            tool_results=[],
            context=AgentContext(system_prompt="test", messages=[]),
            new_messages=[],
        )
        assert ctx.message["role"] == "assistant"
        assert ctx.context.system_prompt == "test"
        assert ctx.new_messages == []


class TestBeforeAfterToolCall:
    def test_before_default(self):
        r = BeforeToolCallResult()
        assert r.block is False
        assert r.reason == ""

    def test_before_block(self):
        r = BeforeToolCallResult(block=True, reason="not allowed")
        assert r.block is True

    def test_after_default(self):
        r = AfterToolCallResult()
        assert r.content is None
        assert r.is_error is None
        assert r.terminate is None

    def test_after_override(self):
        r = AfterToolCallResult(
            content=[TextContent(type="text", text="overridden")],
            terminate=True,
        )
        assert r.content is not None
        assert r.terminate is True
