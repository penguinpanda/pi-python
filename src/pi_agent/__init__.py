"""
pi-agent-core  LLM Agent 循环

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
from ._harness import AgentHarness
from ._harness_types import (
    AbortResult,
    AgentHarnessError,
    AgentHarnessEvent,
    AgentHarnessEventResultMap,
    AgentHarnessOptions,
    AgentHarnessPhase,
    AgentHarnessResources,
    AgentHarnessStreamOptions,
    AgentHarnessStreamOptionsPatch,
    AgentHarnessTool,
    BeforeAgentStartResult,
    BeforeProviderPayloadResult,
    CompactResult,
    ContextResult,
    NavigateOptions,
    NavigateTreeResult,
    PromptTemplate,
    Result,
    SessionBeforeCompactResult,
    SessionBeforeTreeResult,
    Skill,
    ToolCallResult,
    ToolResultPatch,
    TreePreparation,
)
from ._stream_fn import get_default_stream_fn, set_default_stream_fn
from .branch_summarization import (
    BranchSummaryError,
    collect_entries_for_branch_summary,
    generate_branch_summary,
)
from .compaction import (
    CompactionError,
    CompactionPreparation,
    CompactionResult,
    CompactionSettings,
    DEFAULT_COMPACTION_SETTINGS,
    compact,
    prepare_compaction,
)
from .env import (
    ExecutionError,
    FileError,
    FileInfo,
    PythonExecutionEnv,
    ShellExecOptions,
    ShellResult,
    err,
    get_or_throw,
    ok,
)
from .prompt_templates import (
    format_prompt_template_invocation,
    load_prompt_templates,
    load_sourced_prompt_templates,
    substitute_args,
)
from .proxy import (
    ProxyMessageEventStream,
    build_proxy_request_options,
    process_proxy_event,
    stream_proxy,
)
from .session import (
    InMemorySessionStorage,
    InMemorySessionStore,
    JsonlSessionStorage,
    JsonlSessionStore,
    ScanningSessionSearch,
    Session,
    SessionError,
    SessionRepo,
    SessionStorage,
    SessionStore,
    build_session_context,
    create_in_memory_session_repo,
    create_in_memory_session_store,
    create_jsonl_session_repo,
    create_jsonl_session_store,
    rebuild_session_search_index,
)
from .skills import format_skill_invocation, load_sourced_skills, load_skills
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
    BashExecutionMessage,
    BeforeToolCallContext,
    BeforeToolCallResult,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    QueueMode,
    StreamFn,
    ThinkingLevel,
    ToolExecutionMode,
)
from .tools import (
    BashToolOptions,
    ExecutionToolContext,
    ReadToolOptions,
    create_bash_tool,
    create_edit_tool,
    create_read_tool,
    create_write_tool,
)
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    format_size,
    truncate_head,
    truncate_tail,
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
    "AgentHarnessError",
    "AgentHarnessEvent",
    "AgentHarnessEventResultMap",
    "AgentHarnessOptions",
    "AgentHarnessPhase",
    "AgentHarnessResources",
    "AgentHarnessStreamOptions",
    "AgentHarnessStreamOptionsPatch",
    "AgentHarnessTool",
    "Result",
    "NavigateOptions",
    "TreePreparation",
    "BeforeProviderPayloadResult",
    # Session
    "Session",
    "SessionError",
    "SessionStorage",
    "SessionStore",
    "SessionRepo",
    "InMemorySessionStorage",
    "InMemorySessionStore",
    "JsonlSessionStorage",
    "JsonlSessionStore",
    "ScanningSessionSearch",
    "rebuild_session_search_index",
    "build_session_context",
    "create_in_memory_session_store",
    "create_in_memory_session_repo",
    "create_jsonl_session_store",
    "create_jsonl_session_repo",
    # 环境抽象
    "PythonExecutionEnv",
    "FileError",
    "ExecutionError",
    "FileInfo",
    "ShellExecOptions",
    "ShellResult",
    "ok",
    "err",
    "get_or_throw",
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_BYTES",
    "format_size",
    "truncate_head",
    "truncate_tail",
    # 内置工具
    "create_read_tool",
    "create_write_tool",
    "create_edit_tool",
    "create_bash_tool",
    "ExecutionToolContext",
    "ReadToolOptions",
    "BashToolOptions",
    # Skills / Templates
    "load_skills",
    "load_sourced_skills",
    "format_skill_invocation",
    "load_prompt_templates",
    "load_sourced_prompt_templates",
    "substitute_args",
    "format_prompt_template_invocation",
    # Compaction / 分支摘要
    "CompactionSettings",
    "DEFAULT_COMPACTION_SETTINGS",
    "CompactionError",
    "CompactionPreparation",
    "CompactionResult",
    "prepare_compaction",
    "compact",
    "BranchSummaryError",
    "collect_entries_for_branch_summary",
    "generate_branch_summary",
    # Proxy
    "stream_proxy",
    "ProxyMessageEventStream",
    "build_proxy_request_options",
    "process_proxy_event",
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
    "BashExecutionMessage",
    "BranchSummaryMessage",
    "CompactionSummaryMessage",
    "CustomMessage",
    "QueueMode",
    "StreamFn",
    "ThinkingLevel",
    "ToolExecutionMode",
    "BeforeToolCallResult",
    "AfterToolCallResult",
    "BeforeToolCallContext",
    "AfterToolCallContext",
]
