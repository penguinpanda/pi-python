"""
Print 模式 — 单次问答，输出 assistant 最终回复到 stdout。

用法:
    exit_code = await run_print_mode(session, "read README.md")
    sys.exit(exit_code)
"""

from __future__ import annotations

import json
import sys
from typing import Any, cast

from pi_agent import AgentEvent, AgentMessage

from ._session import AgentSession


def _emit_text(text: str) -> None:
    """写一行纯文本到 stdout（BrokenPipeError 由调用方处理）。"""
    print(text, flush=True)


def _emit_json(obj: dict) -> None:
    """写一行 JSON 到 stdout（BrokenPipeError 由调用方处理）。"""
    print(json.dumps(obj, ensure_ascii=False, default=str), flush=True)


async def run_print_mode(session: AgentSession, message: str, images: list | None = None) -> int:
    """运行 Print 模式：发送消息 → 等待完成 → 提取文本 → 输出到 stdout。

    Returns:
        0: 成功
        1: 错误/中止
    """
    final_text: str = ""
    has_error = False

    def on_event(event: AgentEvent) -> None:
        nonlocal final_text, has_error
        event_type = event.get("type")

        if event_type == "agent_end":
            messages = cast(list[AgentMessage], event.get("messages") or [])
            # 从后往前找最后一条 assistant 消息
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    # 提取纯文本
                    content = msg.get("content", [])
                    if not isinstance(content, list):
                        content = []
                    text_parts: list[str] = []
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(str(block.get("text", "")))
                    final_text = "".join(text_parts)

                    stop_reason = msg.get("stop_reason", "stop")
                    if stop_reason in ("error", "aborted"):
                        has_error = True
                    break

    unsub = session.subscribe(on_event)

    try:
        await session.prompt(message, images)
        await session.wait_for_idle()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        unsub()
        await session.dispose()

    if final_text:
        try:
            _emit_text(final_text)
        except BrokenPipeError:
            return 0
    if has_error:
        return 1
    return 0


async def run_print_mode_json(
    session: AgentSession, message: str, images: list | None = None
) -> int:
    """Print 模式 JSON Lines 输出：逐条 agent 事件 + 结束摘要。"""
    has_error = False
    pipe_closed = False

    def on_event(event: AgentEvent) -> None:
        nonlocal pipe_closed
        if pipe_closed:
            return
        try:
            _emit_json(cast(dict[Any, Any], event))
        except BrokenPipeError:
            # 下游提前关闭管道（如 `--json | grep -m1`）：停止输出，静默收尾。
            pipe_closed = True

    unsub = session.subscribe(on_event)
    try:
        await session.prompt(message, images)
        await session.wait_for_idle()
    except Exception as exc:
        try:
            _emit_json({"type": "error", "error": str(exc)})
        except BrokenPipeError:
            return 1
        return 1
    finally:
        unsub()
        await session.dispose()

    if pipe_closed:
        return 0

    messages = session.get_messages()
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            if msg.get("stop_reason") in ("error", "aborted"):
                has_error = True
            break
    try:
        _emit_json({"type": "done", "messages": messages})
    except BrokenPipeError:
        return 0
    return 1 if has_error else 0
