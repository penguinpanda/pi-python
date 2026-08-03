"""
pi-agent-core  最小核心 LLM Agent 循环

纯函数 Agent 循环 + 有状态 Agent 包装类。

用法:
    from pi_agent import Agent, set_default_stream_fn
    from pi_ai import create_default_models

    models = create_default_models()
    set_default_stream_fn(models.stream)

    agent = Agent()
    await agent.prompt("Hello!")
"""

from ._agent import Agent, AgentOptions
from ._agent_loop import (
    agent_loop,
    agent_loop_continue,
    run_agent_loop,
    run_agent_loop_continue,
)
from ._harness import AgentHarness, AgentHarnessSession
from ._harness_types import (
    AbortResult,
    AgentHarnessError,
    AgentHarnessOptions,
    AgentHarnessResources,
    AgentHarnessStreamOptions,
    AgentHarnessStreamOptionsPatch,
    BeforeAgentStartResult,
    CompactResult,
    ContextResult,
    NavigateTreeResult,
    PromptTemplate,
    SessionBeforeCompactResult,
    SessionBeforeTreeResult,
    Skill,
    ToolCallResult,
    ToolResultPatch,
)
from ._stream_fn import get_default_stream_fn, set_default_stream_fn
from ._types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentState,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    QueueMode,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
)

__all__ = [
    # Agent
    "Agent",
    "AgentOptions",
    # Loop
    "run_agent_loop",
    "run_agent_loop_continue",
    "agent_loop",
    "agent_loop_continue",
    # Harness
    "AgentHarness",
    "AgentHarnessSession",
    "AgentHarnessError",
    "AgentHarnessOptions",
    "AgentHarnessResources",
    "AgentHarnessStreamOptions",
    "AgentHarnessStreamOptionsPatch",
    "Skill",
    "PromptTemplate",
    "AbortResult",
    "BeforeAgentStartResult",
    "CompactResult",
    "ContextResult",
    "NavigateTreeResult",
    "SessionBeforeCompactResult",
    "SessionBeforeTreeResult",
    "ToolCallResult",
    "ToolResultPatch",
    # Stream
    "set_default_stream_fn",
    "get_default_stream_fn",
    # Types
    "AgentTool",
    "AgentToolResult",
    "AgentState",
    "AgentContext",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentMessage",
    "QueueMode",
    "StreamFn",
    "ThinkingLevel",
    "ToolExecutionMode",
    "BeforeToolCallResult",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "AfterToolCallContext",
]
