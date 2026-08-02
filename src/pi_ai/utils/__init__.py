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
