"""
Print 模式 — 单次问答，输出 assistant 最终回复到 stdout。

用法:
    exit_code = await run_print_mode(session, "read README.md")
    sys.exit(exit_code)
"""

from __future__ import annotations

import sys

from pi_agent import AgentEvent
from pi_ai import TextContent

from ._session import AgentSession


async def run_print_mode(session: AgentSession, message: str) -> int:
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
            messages = event.get("messages", [])
            # 从后往前找最后一条 assistant 消息
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    # 提取纯文本
                    content = msg.get("content", [])
                    text_parts: list[str] = []
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    final_text = "".join(text_parts)

                    stop_reason = msg.get("stopReason", "end")
                    if stop_reason in ("error", "aborted"):
                        has_error = True
                    break

    unsub = session.subscribe(on_event)

    try:
        await session.prompt(message)
        await session.wait_for_idle()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        unsub()
        await session.dispose()

    if final_text:
        print(final_text)
    if has_error:
        return 1
    return 0
