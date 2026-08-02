"""pi_ai.utils — 内部工具函数包。"""

from ._event_stream import AssistantMessageEventStream, EventStream
from .diagnostics import (
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    extract_diagnostic_error,
    format_thrown_value,
)
from .estimate import (
    CHARS_PER_TOKEN,
    ESTIMATED_IMAGE_CHARS,
    ContextUsageEstimate,
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_text_and_image_content_tokens,
    estimate_text_tokens,
    estimate_tools_tokens,
)
from .overflow import (
    NON_OVERFLOW_PATTERNS,
    OVERFLOW_PATTERNS,
    get_overflow_patterns,
    is_context_overflow,
)
from .retry import (
    RetryCallbacks,
    RetryPolicy,
    compute_backoff_delay,
    is_retryable_error,
    retry_assistant_call,
)
from .validation import (
    ValidationError,
    coerce_with_json_schema,
    validate_arguments,
    validate_tool_arguments,
    validate_tool_call,
)
from .json_parse import (
    parse_json_with_repair,
    parse_streaming_json,
    repair_json,
)
from .partial_json import partial_json
from .provider_env import get_provider_env_value
from .uuid import uuidv7

__all__ = [
    "AssistantMessageEventStream",
    "EventStream",
    "RetryCallbacks",
    "RetryPolicy",
    "compute_backoff_delay",
    "is_retryable_error",
    "retry_assistant_call",
    # Diagnostics
    "format_thrown_value",
    "extract_diagnostic_error",
    "create_assistant_message_diagnostic",
    "append_assistant_message_diagnostic",
    # Tool 参数校验
    "ValidationError",
    "coerce_with_json_schema",
    "validate_arguments",
    "validate_tool_arguments",
    "validate_tool_call",
    # JSON 修复 / 流式解析
    "repair_json",
    "parse_json_with_repair",
    "parse_streaming_json",
    "partial_json",
    # Provider 环境检测
    "get_provider_env_value",
    # UUID v7
    "uuidv7",
    # Token / Context
    "ContextUsageEstimate",
    "CHARS_PER_TOKEN",
    "ESTIMATED_IMAGE_CHARS",
    "calculate_context_tokens",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "estimate_text_and_image_content_tokens",
    "estimate_text_tokens",
    "estimate_tools_tokens",
    "is_context_overflow",
    "get_overflow_patterns",
    "OVERFLOW_PATTERNS",
    "NON_OVERFLOW_PATTERNS",
]
