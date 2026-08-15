"""Coding-agent JSON wire event normalization.

The Python port stores agent events with snake_case field names, while the
TypeScript coding-agent JSON/RPC protocols use camelCase field names
(toolCallId, toolName, assistantMessageEvent, etc.).  Normalize at the wire
boundary only; in-memory events stay snake_case.
"""

from __future__ import annotations

from typing import Any

# Wire keys where the coding-agent JSON protocol uses camelCase.
_CAMEL_CASE_KEYS = {
    "assistant_message_event": "assistantMessageEvent",
    "tool_call_id": "toolCallId",
    "tool_name": "toolName",
    "is_error": "isError",
    "partial_result": "partialResult",
    "tool_results": "toolResults",
    "stop_reason": "stopReason",
    "error_message": "errorMessage",
    "raw_arguments": "rawArguments",
    "mime_type": "mimeType",
    "base_url": "baseUrl",
    "cache_read": "cacheRead",
    "cache_write": "cacheWrite",
    "total_tokens": "totalTokens",
    "first_kept_entry_id": "firstKeptEntryId",
    "tokens_before": "tokensBefore",
    "parent_id": "parentId",
    "session_id": "sessionId",
    "model_id": "modelId",
    "custom_type": "customType",
    "full_output_path": "fullOutputPath",
    "exit_code": "exitCode",
    "turn_timings": "turnTimings",
    "started_at_ms": "startedAtMs",
    "duration_ms": "durationMs",
    "message_count": "messageCount",
    "thinking_level": "thinkingLevel",
    "previous_model": "previousModel",
    "from_hook": "fromHook",
    "target_id": "targetId",
    "old_leaf_id": "oldLeafId",
    "new_leaf_id": "newLeafId",
    "entry_id": "entryId",
    "from_entry_id": "fromEntryId",
    "source_leaf_id": "sourceLeafId",
    "result_entry_id": "resultEntryId",
    "original_prompt": "originalPrompt",
    "initial_messages": "initialMessages",
    "custom_instructions": "customInstructions",
    "auto_compaction_enabled": "autoCompactionEnabled",
    "pending_message_count": "pendingMessageCount",
}


def _convert_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _CAMEL_CASE_KEYS.get(str(key), str(key)): _convert_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_convert_value(item) for item in value]
    return value


def to_json_event(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize an agent event for JSON/RPC output.

    Also removes the cumulative ``partial`` snapshot from
    ``assistant_message_event``/``assistantMessageEvent`` (TS toJsonEvent).
    """
    result = dict(event)
    if result.get("type") == "message_update":
        assistant_event = result.get("assistant_message_event")
        if isinstance(assistant_event, dict):
            assistant_event = dict(assistant_event)
            assistant_event.pop("partial", None)
            result["assistant_message_event"] = assistant_event
    return _convert_value(result)


__all__ = ["to_json_event"]
