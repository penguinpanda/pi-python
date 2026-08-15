"""
Print 模式 — 单次问答，输出 assistant 最终回复到 stdout。

用法:
    exit_code = await run_print_mode(session, "read README.md")
    sys.exit(exit_code)
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
from typing import Any, Callable, cast

from pi_agent import AgentEvent

from ._session import AgentSession


def _emit_text(text: str) -> None:
    """写一行纯文本到 stdout（BrokenPipeError 由调用方处理）。"""
    print(text, flush=True)


def _emit_json(obj: dict) -> None:
    """写一行 JSON 到 stdout（BrokenPipeError 由调用方处理）。"""
    print(json.dumps(obj, ensure_ascii=False, default=str), flush=True)


def _to_json_event(event: AgentEvent) -> dict:
    """JSON 输出规范化（对齐 TS json-event.ts toJsonEvent）。

    message_update 携带的累计 partial 快照体积大且下游不需要，
    仅保留 delta 事件（assistant_message_event 去掉 partial 字段）。
    """
    if event.get("type") != "message_update":
        return cast(dict, event)
    assistant_event = dict(cast(dict, event.get("assistant_message_event") or {}))
    assistant_event.pop("partial", None)
    result = dict(cast(dict, event))
    result["assistant_message_event"] = assistant_event
    return result


def _install_signal_handlers(dispose: Callable[[], Any]) -> list[tuple[int, Any]]:
    """SIGTERM → 143 / SIGHUP（非 Windows）→ 129（对齐 TS print-mode.ts）。

    收到信号时先触发会话清理（中止运行、杀子进程），再以对应码退出。
    """
    handlers: list[tuple[int, Any]] = []
    loop = asyncio.get_running_loop()

    def _make_handler(code: int):
        def _handler() -> None:
            dispose()
            sys.exit(code)

        return _handler

    # Windows 无 SIGHUP；用 getattr 兼容（直接引用会在 Windows 抛 AttributeError）。
    sighup = getattr(signal, "SIGHUP", None)
    for sig, code in ((signal.SIGTERM, 143), (sighup, 129)):
        if sig is None:
            continue
        if sig == sighup and sys.platform == "win32":
            continue
        handler = _make_handler(code)
        try:
            loop.add_signal_handler(sig, handler)
        except (NotImplementedError, RuntimeError):
            continue
        handlers.append((sig, handler))
    return handlers


def _remove_signal_handlers(handlers: list[tuple[int, Any]]) -> None:
    loop = asyncio.get_running_loop()
    for sig, _handler in handlers:
        try:
            loop.remove_signal_handler(sig)
        except (NotImplementedError, RuntimeError):
            pass


async def run_print_mode(
    session: AgentSession,
    message: str | list[str],
    images: list | None = None,
) -> int:
    """运行 Print 模式：发送消息（可多条）→ 等待完成 → 输出最后一条 assistant 文本。

    Returns:
        0: 成功
        1: 错误/中止
    """
    messages = [message] if isinstance(message, str) else list(message)
    disposed = False

    def dispose() -> None:
        nonlocal disposed
        if disposed:
            return
        disposed = True
        asyncio.get_running_loop().create_task(session.dispose())

    handlers = _install_signal_handlers(dispose)

    unsub = session.subscribe(lambda _event: None)

    try:
        for index, prompt in enumerate(messages):
            await session.prompt(prompt, images if index == 0 else None)
        await session.wait_for_idle()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        unsub()
        _remove_signal_handlers(handlers)
        if not disposed:
            await session.dispose()

    # 对齐 TS：state 最后一条 assistant 消息；error/aborted → stderr + 1。
    state_messages = session.get_messages()
    last = state_messages[-1] if state_messages else None
    if last is not None and last.get("role") == "assistant":
        stop_reason = last.get("stop_reason", "stop")
        if stop_reason in ("error", "aborted"):
            print(
                last.get("error_message") or f"Request {stop_reason}",
                file=sys.stderr,
            )
            return 1
        text_parts: list[str] = []
        content = last.get("content", [])
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
        final_text = "".join(text_parts)
        if final_text:
            try:
                _emit_text(final_text)
            except BrokenPipeError:
                return 0
    return 0


async def run_print_mode_json(
    session: AgentSession,
    message: str | list[str],
    images: list | None = None,
) -> int:
    """Print 模式 JSON Lines 输出：session header + 逐条 agent 事件（对齐 TS print-mode）。"""
    messages = [message] if isinstance(message, str) else list(message)
    has_error = False
    pipe_closed = False
    disposed = False

    def dispose() -> None:
        nonlocal disposed
        if disposed:
            return
        disposed = True
        asyncio.get_running_loop().create_task(session.dispose())

    handlers = _install_signal_handlers(dispose)

    def on_event(event: AgentEvent) -> None:
        nonlocal pipe_closed
        if pipe_closed:
            return
        try:
            _emit_json(_to_json_event(event))
        except BrokenPipeError:
            # 下游提前关闭管道（如 `--json | grep -m1`）：停止输出，静默收尾。
            pipe_closed = True

    unsub = session.subscribe(on_event)
    try:
        # session header 作为首条记录（对齐 TS getHeader()）。
        manager = session.session_manager
        get_header = getattr(manager, "get_header", None)
        header = get_header() if get_header is not None else None
        if header is not None and not pipe_closed:
            try:
                _emit_json(header)
            except BrokenPipeError:
                pipe_closed = True
        for index, prompt in enumerate(messages):
            await session.prompt(prompt, images if index == 0 else None)
        await session.wait_for_idle()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        unsub()
        _remove_signal_handlers(handlers)
        if not disposed:
            await session.dispose()

    if pipe_closed:
        return 0

    messages_state = session.get_messages()
    for msg in reversed(messages_state):
        if msg.get("role") == "assistant":
            if msg.get("stop_reason") in ("error", "aborted"):
                has_error = True
            break
    return 1 if has_error else 0
