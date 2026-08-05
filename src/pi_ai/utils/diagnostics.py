"""pi_ai.utils.diagnostics — 诊断信息工具。

对应 TypeScript `packages/ai/src/utils/diagnostics.ts` 的移植：

    DiagnosticErrorInfo                  → DiagnosticErrorInfo
    AssistantMessageDiagnostic           → AssistantMessageDiagnostic
    formatThrownValue                    → format_thrown_value
    extractDiagnosticError               → extract_diagnostic_error
    createAssistantMessageDiagnostic     → create_assistant_message_diagnostic
    appendAssistantMessageDiagnostic     → append_assistant_message_diagnostic

用途：给 AssistantMessage 附加结构化诊断（type / timestamp / error / details），
用于 provider 错误定位与流式调试。
"""

from __future__ import annotations

from typing import Any

from ..types.message import AssistantMessage, AssistantMessageDiagnostic, DiagnosticErrorInfo
from ..types.common import now_ms


def format_thrown_value(value: Any) -> str:
    """将任意抛出值规范化为字符串（对齐 TS formatThrownValue）。

    - BaseException → 其消息（空则用类名）
    - str → 原样
    - 其他 → str()
    """
    if isinstance(value, BaseException):
        message = str(value)
        return message or value.__class__.__name__
    if isinstance(value, str):
        return value
    return str(value)


def extract_diagnostic_error(error: Any) -> DiagnosticErrorInfo:
    """将任意抛出值（含非 Exception）规范化为 DiagnosticErrorInfo。

    对齐 TS extractDiagnosticError：
    - 非 BaseException → { name: "ThrownValue", message }
    - BaseException → name / message，附 code（str|int）与 stack（str，可用时）
    """
    if not isinstance(error, BaseException):
        return {"name": "ThrownValue", "message": format_thrown_value(error)}

    info: DiagnosticErrorInfo = {
        "name": error.__class__.__name__,
        "message": str(error) or error.__class__.__name__,
    }
    code = getattr(error, "code", None)
    if isinstance(code, (str, int)):
        info["code"] = code
    stack = getattr(error, "stack", None)
    if isinstance(stack, str):
        info["stack"] = stack
    return info


def create_assistant_message_diagnostic(
    type: str,
    error: Any = None,
    details: dict[str, Any] | None = None,
) -> AssistantMessageDiagnostic:
    """构造一条 AssistantMessageDiagnostic（对齐 TS createAssistantMessageDiagnostic）。

    参数:
        type: 诊断类型（如 provider_error / retry_exhausted）
        error: 抛出值；None 时不带 error 字段
        details: 附加信息；None 时不带 details 字段
    """
    diagnostic: AssistantMessageDiagnostic = {"type": type, "timestamp": now_ms()}
    if error is not None:
        diagnostic["error"] = extract_diagnostic_error(error)
    if details is not None:
        diagnostic["details"] = details
    return diagnostic


def append_assistant_message_diagnostic(
    message: AssistantMessage,
    diagnostic: AssistantMessageDiagnostic,
) -> None:
    """不可变追加一条诊断到 AssistantMessage（对齐 TS appendAssistantMessageDiagnostic）。

    原地更新 message["diagnostics"]（新列表），不修改原列表。
    """
    message["diagnostics"] = [*(message.get("diagnostics") or []), diagnostic]


__all__ = [
    "format_thrown_value",
    "extract_diagnostic_error",
    "create_assistant_message_diagnostic",
    "append_assistant_message_diagnostic",
]
