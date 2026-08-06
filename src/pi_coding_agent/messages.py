"""pi_coding_agent.messages — 编码代理消息转换（对齐 TS coding-agent messages.ts）。

复用 pi_agent._messages 的完整实现：bashExecution / compactionSummary /
branchSummary / custom 包装为 user 消息，user / assistant / toolResult 透传。
"""

from __future__ import annotations

from pi_agent._messages import (
    BRANCH_SUMMARY_PREFIX,
    BRANCH_SUMMARY_SUFFIX,
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
    bash_execution_to_text,
    convert_to_llm,
)

__all__ = [
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "bash_execution_to_text",
    "convert_to_llm",
]
