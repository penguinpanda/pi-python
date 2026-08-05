"""RPC 协议类型与响应构造（对齐 TS modes/rpc/rpc-types.ts）。"""

from __future__ import annotations

from typing import Any, TypedDict

from typing_extensions import NotRequired


class RpcSessionState(TypedDict, total=False):
    """get_state 的响应数据。"""

    model: dict[str, Any] | None
    thinkingLevel: str
    isStreaming: bool
    isCompacting: bool
    steeringMode: str
    followUpMode: str
    sessionFile: str | None
    sessionId: str
    sessionName: str | None
    autoCompactionEnabled: bool
    messageCount: int
    pendingMessageCount: int


class RpcSlashCommand(TypedDict):
    """get_commands 返回的命令条目。"""

    name: str
    source: str  # "extension" | "prompt" | "skill"
    sourceInfo: dict[str, Any]
    description: NotRequired[str]


def success_response(
    command_id: str | None,
    command: str,
    data: Any = None,
) -> dict[str, Any]:
    """构造成功响应（data=None 时不带 data 字段）。"""
    response: dict[str, Any] = {
        "id": command_id,
        "type": "response",
        "command": command,
        "success": True,
    }
    if data is not None:
        response["data"] = data
    return response


def error_response(
    command_id: str | None,
    command: str,
    message: str,
) -> dict[str, Any]:
    """构造错误响应。"""
    return {
        "id": command_id,
        "type": "response",
        "command": command,
        "success": False,
        "error": message,
    }


__all__ = [
    "RpcSessionState",
    "RpcSlashCommand",
    "success_response",
    "error_response",
]
