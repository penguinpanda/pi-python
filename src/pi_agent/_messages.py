"""pi_agent._messages — AgentMessage → LLM Message 转换（应用层丰富版）。

对齐 TS `packages/agent/src/harness/messages.ts` 与
`packages/coding-agent/src/core/messages.ts` 的 convertToLlm：

- user / assistant / toolResult 直接透传
- bashExecution 包装为 user 消息（excludeFromContext 时跳过，!! 前缀）
- compactionSummary / branchSummary 包装为 user 消息
- custom 包装为 user 消息
- 其余 role（含 system）过滤

`pi_agent._agent._default_convert_to_llm` 只做最小过滤（对齐 TS agent 包默认），
需要完整转换的应用层（AgentHarness / pi_coding_agent）使用本模块。
"""

from __future__ import annotations

from typing import Any, cast

from pi_ai.types import Message

from ._types import AgentMessage

# 压缩/分支摘要消息包装（对齐 TS messages.ts）。
COMPACTION_SUMMARY_PREFIX = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"
BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
)
BRANCH_SUMMARY_SUFFIX = "</summary>"


def bash_execution_to_text(msg: dict[str, Any]) -> str:
    """把 bashExecution 消息转为 LLM user 消息文本（对齐 TS bashExecutionToText）。"""
    command = str(msg.get("command", ""))
    output = str(msg.get("output", ""))
    text = f"Ran `{command}`\n"
    if output:
        text += f"```\n{output}\n```"
    else:
        text += "(no output)"
    if msg.get("cancelled"):
        text += "\n\n(command cancelled)"
    elif msg.get("exitCode") not in (None, 0):
        text += f"\n\nCommand exited with code {msg.get('exitCode')}"
    if msg.get("truncated") and msg.get("fullOutputPath"):
        text += f"\n\n[Output truncated. Full output: {msg.get('fullOutputPath')}]"
    return text


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """AgentMessage → LLM Message 转换（应用层丰富版，对齐 TS convertToLlm）。"""
    result: list[Message] = []
    for m in messages:
        role = m.get("role", "")
        if role in ("user", "assistant", "toolResult"):
            result.append(m)
        elif role == "bashExecution":
            if cast(dict[str, Any], m).get("excludeFromContext"):
                continue
            result.append(
                {
                    "role": "user",
                    "content": bash_execution_to_text(cast(dict[str, Any], m)),
                    "timestamp": m.get("timestamp"),
                }
            )
        elif role in ("compactionSummary", "branchSummary"):
            summary = m.get("summary", "")
            prefix = (
                COMPACTION_SUMMARY_PREFIX if role == "compactionSummary" else BRANCH_SUMMARY_PREFIX
            )
            suffix = (
                COMPACTION_SUMMARY_SUFFIX if role == "compactionSummary" else BRANCH_SUMMARY_SUFFIX
            )
            result.append(
                {
                    "role": "user",
                    "content": prefix + summary + suffix,
                    "timestamp": m.get("timestamp"),
                }
            )
        elif role == "custom":
            result.append(
                {
                    "role": "user",
                    "content": m.get("content"),
                    "timestamp": m.get("timestamp"),
                }
            )
    return result


__all__ = [
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "bash_execution_to_text",
    "convert_to_llm",
]
