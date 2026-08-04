"""
pi-coding-agent 类型定义（Coding-Agent Types）

本模块定义 coding-agent 层的专有类型，建立在 pi_ai 和 pi_agent 类型之上。

分层:
- TypedDict: 会话持久化条目（SessionHeader, SessionMessageEntry）
- dataclass: 配置对象（AgentSessionConfig, PrintModeOptions）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from pi_agent import Agent, AgentMessage
from pi_ai import Model

# ---------------------------------------------------------------------------
# 会话持久化类型
# ---------------------------------------------------------------------------

CURRENT_SESSION_VERSION = 3


class SessionHeader(TypedDict):
    """JSONL 文件首行 —— 会话元数据。"""

    type: Literal["session"]
    version: int
    id: str
    timestamp: str
    cwd: str


class SessionMessageEntry(TypedDict):
    """JSONL 消息条目 —— 唯一需要的条目类型（最小核心）。"""

    type: Literal["message"]
    id: str
    parentId: str | None
    timestamp: str
    message: AgentMessage


class CompactionEntry(TypedDict):
    """JSONL 压缩条目 —— 摘要替代旧历史，标记保留起点。"""

    type: Literal["compaction"]
    id: str
    parentId: str | None
    timestamp: str
    summary: str
    firstKeptEntryId: str
    tokensBefore: int


class ModelChangeEntry(TypedDict):
    """JSONL 模型切换条目 —— 记录 setModel/cycleModel 历史。"""

    type: Literal["model_change"]
    id: str
    parentId: str | None
    timestamp: str
    provider: str
    modelId: str


class ThinkingLevelChangeEntry(TypedDict):
    """JSONL 思考级别切换条目。"""

    type: Literal["thinking_level_change"]
    id: str
    parentId: str | None
    timestamp: str
    thinkingLevel: str


class BranchSummaryEntry(TypedDict, total=False):
    """JSONL 分支摘要条目 —— 记录跨分支导航的上下文摘要。"""

    type: Literal["branch_summary"]
    id: str
    parentId: str | None
    timestamp: str
    fromId: str
    summary: str
    details: object
    usage: object
    fromHook: bool


class LabelEntry(TypedDict):
    """JSONL 标签条目 —— 指向被标记的条目。"""

    type: Literal["label"]
    id: str
    parentId: str | None
    timestamp: str
    targetId: str
    label: str | None


class SessionInfoEntry(TypedDict, total=False):
    """JSONL 会话信息条目（名称等）。"""

    type: Literal["session_info"]
    id: str
    parentId: str | None
    timestamp: str
    name: str


class CustomEntry(TypedDict, total=False):
    """JSONL 自定义条目 —— 状态持久化，不进入 LLM 上下文。"""

    type: Literal["custom"]
    id: str
    parentId: str | None
    timestamp: str
    customType: str
    data: object


class CustomMessageEntry(TypedDict, total=False):
    """JSONL 自定义消息条目。"""

    type: Literal["custom_message"]
    id: str
    parentId: str | None
    timestamp: str
    customType: str
    content: object
    display: bool
    details: object


class LeafEntry(TypedDict):
    """JSONL leaf 指针条目 —— 持久化当前分支叶节点。"""

    type: Literal["leaf"]
    id: str
    parentId: str | None
    timestamp: str
    targetId: str | None


class ActiveToolsChangeEntry(TypedDict):
    """JSONL 工具集变更条目。"""

    type: Literal["active_tools_change"]
    id: str
    parentId: str | None
    timestamp: str
    activeToolNames: list[str]


# 会话条目联合（消息 + 压缩 + 模型/思考级别变更）。
SessionEntry = (
    SessionMessageEntry
    | CompactionEntry
    | ModelChangeEntry
    | ThinkingLevelChangeEntry
    | BranchSummaryEntry
    | LabelEntry
    | SessionInfoEntry
    | CustomEntry
    | CustomMessageEntry
    | LeafEntry
    | ActiveToolsChangeEntry
)


# ---------------------------------------------------------------------------
# 配置类型
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentSessionConfig:
    """AgentSession 构造参数。"""

    agent: Agent
    cwd: str
    model: Model
    session_id: str
    tools: list  # list[AgentTool] — 延迟导入以避免循环


@dataclass(slots=True)
class PrintModeOptions:
    """Print 模式执行选项。"""

    message: str
    system_prompt: str | None = None
