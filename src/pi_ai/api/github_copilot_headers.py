"""GitHub Copilot 动态请求头（对齐 TS api/github-copilot-headers.ts）。

Copilot 要求 X-Initiator 标识请求是用户发起还是 agent 发起，并在携带
图片时要求 Copilot-Vision-Request 头；缺少这些头请求可能被拒。
"""

from __future__ import annotations

from typing import Any

from ..types import Message


def infer_copilot_initiator(messages: list[Message]) -> str:
    """末条消息非 user → "agent"，否则 "user"（对齐 TS inferCopilotInitiator）。"""
    last = messages[-1] if messages else None
    return "agent" if last is not None and last["role"] != "user" else "user"


def has_copilot_vision_input(messages: list[Message]) -> bool:
    """user/toolResult 消息是否携带图片（对齐 TS hasCopilotVisionInput）。"""
    for message in messages:
        if message["role"] not in ("user", "toolResult"):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    return True
    return False


def build_copilot_dynamic_headers(
    *,
    messages: list[Message],
    has_images: bool,
) -> dict[str, Any]:
    """构建 Copilot 动态头（对齐 TS buildCopilotDynamicHeaders）。"""
    headers: dict[str, Any] = {
        "X-Initiator": infer_copilot_initiator(messages),
        "Openai-Intent": "conversation-edits",
    }
    if has_images:
        headers["Copilot-Vision-Request"] = "true"
    return headers


__all__ = [
    "build_copilot_dynamic_headers",
    "has_copilot_vision_input",
    "infer_copilot_initiator",
]
