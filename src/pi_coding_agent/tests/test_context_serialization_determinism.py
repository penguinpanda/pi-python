"""上下文序列化确定性回归测试（P3）。"""

from __future__ import annotations

import asyncio
import json

from pi_ai import Model
from pi_ai.types import Tool
from pi_ai.api._shared import to_openai_messages, to_openai_tools
from pi_agent.compaction_utils import apply_cache_first_truncation
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.messages import convert_to_llm


def _model() -> Model:
    return Model(
        id="deepseek-v4-flash",
        provider="deepseek",
        api="openai-completions",
        name="DeepSeek V4 Flash",
        input=["text"],
        output=["text"],
        compat={
            "thinkingFormat": "deepseek",
            "requiresReasoningContentOnAssistantMessages": True,
        },
    )


def _tool() -> Tool:
    return Tool(
        name="read",
        description="Read a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "fix login bug", "timestamp": 1},
        {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "id": "c1", "name": "read", "arguments": {"path": "auth.py"}}
            ],
            "timestamp": 2,
        },
        {
            "role": "toolResult",
            "tool_call_id": "c1",
            "tool_name": "read",
            "content": [{"type": "text", "text": "x" * 5000}],
            "is_error": False,
            "timestamp": 3,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "auth.py updated"}],
            "timestamp": 4,
        },
    ]


def _serialize(messages: list[dict], model: Model, tools: list[Tool]) -> str:
    llm_messages = convert_to_llm(messages)
    openai_messages = to_openai_messages(llm_messages, model)
    openai_tools = to_openai_tools(tools, supports_strict_mode=True)
    return json.dumps(
        [openai_messages, openai_tools],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_serialization_deterministic_same_context():
    first = _serialize(_messages(), _model(), [_tool()])
    second = _serialize(_messages(), _model(), [_tool()])
    assert first == second


def test_cache_first_pruned_serialization_stable_across_resume(tmp_path):
    mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
    for message in _messages():
        asyncio.run(mgr.append_message(message))

    def build(manager: SessionManager) -> str:
        pruned = apply_cache_first_truncation(manager.build_context())
        return _serialize(pruned, _model(), [_tool()])

    first = build(mgr)
    assert build(mgr) == first

    reopened = SessionManager.open(mgr.session_path)
    assert build(reopened) == first
