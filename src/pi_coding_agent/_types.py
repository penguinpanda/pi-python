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
