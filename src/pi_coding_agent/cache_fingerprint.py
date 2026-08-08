"""上下文指纹：检测相邻请求间 context 变化来源。"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _serialize_tools(tools: list[Any]) -> Any:
    try:
        from pi_ai.api._shared import to_openai_tools

        return to_openai_tools(tools, supports_strict_mode=True)
    except Exception:
        return [{"name": getattr(tool, "name", None)} for tool in tools]


def compute_context_fingerprint(
    system_prompt: str | None,
    messages: list[Any],
    tools: list[Any],
) -> dict[str, str]:
    """返回 {system, messages, tools} 三部分 sha256 指纹。"""
    return {
        "system": _sha256(_stable_json(system_prompt)),
        "messages": _sha256(_stable_json(messages)),
        "tools": _sha256(_stable_json(_serialize_tools(tools))),
    }


def classify_context_change(
    previous: dict[str, str] | None,
    current: dict[str, str] | None,
    messages: list[Any],
) -> list[str]:
    """把相邻请求的指纹差异归类为可读原因。"""
    if previous is None or current is None:
        return []
    reasons: list[str] = []
    messages_changed = previous.get("messages") != current.get("messages")
    if messages_changed and any(
        isinstance(message.get("role"), str)
        and message.get("role") in ("compactionSummary", "branchSummary")
        for message in messages
    ):
        reasons.append("compaction")
    if previous.get("system") != current.get("system"):
        reasons.append("system")
    if previous.get("tools") != current.get("tools"):
        reasons.append("tools")
    if messages_changed and not reasons:
        reasons.append("append")
    return reasons


__all__ = ["classify_context_change", "compute_context_fingerprint"]
