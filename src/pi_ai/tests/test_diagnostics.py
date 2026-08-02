"""pi_ai.utils.diagnostics 单元测试。

用例移植自 TS `packages/ai/src/utils/diagnostics.ts` 语义，
并验证 diagnostics 已接入 build_error_message / retry 错误路径。
"""

from __future__ import annotations

import pytest

from pi_ai._types import AssistantMessage, Model, ModelCapabilities
from pi_ai.api._shared import build_error_message
from pi_ai.utils.diagnostics import (
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    extract_diagnostic_error,
    format_thrown_value,
)
from pi_ai.utils.retry import RetryPolicy, retry_assistant_call


def _make_model() -> Model:
    return Model(
        id="m",
        provider="openai",
        api="openai-completions",
        capabilities=ModelCapabilities(tools=True),
    )


# ============================================================================
# format_thrown_value / extract_diagnostic_error
# ============================================================================


class TestFormatThrownValue:
    def test_exception_message(self):
        assert format_thrown_value(ValueError("boom")) == "boom"

    def test_exception_without_message_uses_name(self):
        assert format_thrown_value(ValueError()) == "ValueError"

    def test_string_passthrough(self):
        assert format_thrown_value("hello") == "hello"

    def test_other_values(self):
        assert format_thrown_value(42) == "42"
        assert format_thrown_value(None) == "None"


class TestExtractDiagnosticError:
    def test_exception(self):
        info = extract_diagnostic_error(ValueError("boom"))
        assert info["name"] == "ValueError"
        assert info["message"] == "boom"
        assert "stack" not in info

    def test_exception_with_code(self):
        class _ApiError(RuntimeError):
            code = 429

        info = extract_diagnostic_error(_ApiError("rate limited"))
        assert info["code"] == 429

    def test_exception_with_stack_attribute(self):
        class _WithStack(RuntimeError):
            def __init__(self, message: str):
                super().__init__(message)
                self.stack = "traceback line"

        info = extract_diagnostic_error(_WithStack("boom"))
        assert info["stack"] == "traceback line"
        assert info["name"] == "_WithStack"

    def test_non_exception(self):
        info = extract_diagnostic_error("plain string")
        assert info == {"name": "ThrownValue", "message": "plain string"}


# ============================================================================
# create / append
# ============================================================================


class TestCreateAppend:
    def test_create_with_error(self):
        diag = create_assistant_message_diagnostic("provider_error", ValueError("boom"))
        assert diag["type"] == "provider_error"
        assert isinstance(diag["timestamp"], int)
        assert diag["error"] == {"name": "ValueError", "message": "boom"}

    def test_create_without_error_and_details(self):
        diag = create_assistant_message_diagnostic("plain")
        assert diag == {"type": "plain", "timestamp": diag["timestamp"]}
        assert "error" not in diag
        assert "details" not in diag

    def test_create_with_details(self):
        diag = create_assistant_message_diagnostic(
            "x", details={"attempts": 3, "max_retries": 3}
        )
        assert diag["details"] == {"attempts": 3, "max_retries": 3}

    def test_append_keeps_previous(self):
        msg: AssistantMessage = {"role": "assistant", "content": [], "api": "a", "provider": "p", "model": "m"}
        append_assistant_message_diagnostic(msg, {"type": "one", "timestamp": 1})
        append_assistant_message_diagnostic(msg, {"type": "two", "timestamp": 2})
        assert [d["type"] for d in msg["diagnostics"]] == ["one", "two"]


# ============================================================================
# 接入验证：build_error_message / retry_assistant_call
# ============================================================================


class TestWiring:
    def test_build_error_message_has_diagnostic(self):
        msg = build_error_message(_make_model(), ValueError("boom"))
        assert msg["stop_reason"] == "error"
        assert msg["error_message"] == "boom"
        diagnostics = msg["diagnostics"]
        assert len(diagnostics) == 1
        assert diagnostics[0]["type"] == "provider_error"
        assert diagnostics[0]["error"] == {"name": "ValueError", "message": "boom"}

    @pytest.mark.asyncio
    async def test_retry_exhausted_attaches_diagnostic(self):
        def _error_msg(text: str) -> AssistantMessage:
            return {
                "role": "assistant",
                "content": [],
                "api": "responses",
                "provider": "faux",
                "model": "faux",
                "stop_reason": "error",
                "error_message": text,
            }

        calls = 0

        async def _produce() -> AssistantMessage:
            nonlocal calls
            calls += 1
            return _error_msg("502 Bad Gateway")

        policy = RetryPolicy(enabled=True, max_retries=2, base_delay_ms=1, jitter=False)
        result = await retry_assistant_call(_produce, policy=policy)

        # 初始调用 + 2 次重试后放弃
        assert calls == 3
        assert result["stop_reason"] == "error"
        diagnostics = result.get("diagnostics", [])
        assert len(diagnostics) == 1
        assert diagnostics[0]["type"] == "retry_exhausted"
        assert diagnostics[0]["details"] == {"attempts": 2, "max_retries": 2}

    @pytest.mark.asyncio
    async def test_retry_non_retryable_attaches_diagnostic(self):
        async def _produce() -> AssistantMessage:
            return {
                "role": "assistant",
                "content": [],
                "api": "responses",
                "provider": "faux",
                "model": "faux",
                "stop_reason": "error",
                "error_message": "quota exceeded",
            }

        policy = RetryPolicy(enabled=True, max_retries=2, base_delay_ms=1, jitter=False)
        result = await retry_assistant_call(_produce, policy=policy)

        # 不可重试：不重试，直接附加诊断
        diagnostics = result.get("diagnostics", [])
        assert len(diagnostics) == 1
        assert diagnostics[0]["type"] == "retry_exhausted"
        assert diagnostics[0]["details"] == {"attempts": 0, "max_retries": 2}

    @pytest.mark.asyncio
    async def test_retry_success_no_diagnostic(self):
        async def _produce() -> AssistantMessage:
            return {
                "role": "assistant",
                "content": [],
                "api": "responses",
                "provider": "faux",
                "model": "faux",
                "stop_reason": "stop",
            }

        result = await retry_assistant_call(_produce, policy=RetryPolicy())
        assert "diagnostics" not in result
