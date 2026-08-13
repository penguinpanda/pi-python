"""Shell 输出捕获与截断（Phase 4.3 bash 工具辅助）。

对齐 TS `harness/utils/shell-output.ts`：增量捕获、双倍缓冲尾部、
超限时写 full-output 临时文件。
"""

from __future__ import annotations

import asyncio
from typing import Any

from .env import ExecutionEnv, ShellExecOptions, to_execution_error
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    truncate_tail,
)


class ShellCaptureProgress:
    def __init__(
        self,
        output: str,
        truncation: TruncationResult,
        full_output_path: str | None,
        last_line_bytes: int,
    ) -> None:
        self.output = output
        self.truncation = truncation
        self.full_output_path = full_output_path
        self.last_line_bytes = last_line_bytes


class ShellCaptureResult:
    def __init__(
        self,
        output: str,
        truncation: TruncationResult,
        full_output_path: str | None,
        last_line_bytes: int,
        exit_code: int | None,
        cancelled: bool,
        execution_error: Any = None,
    ) -> None:
        self.output = output
        self.truncation = truncation
        self.full_output_path = full_output_path
        self.last_line_bytes = last_line_bytes
        self.exit_code = exit_code
        self.cancelled = cancelled
        self.execution_error = execution_error


def sanitize_binary_output(text: str) -> str:
    """过滤二进制控制字符。"""
    chars: list[str] = []
    for char in text:
        code = ord(char)
        if code in (0x09, 0x0A, 0x0D):
            chars.append(char)
            continue
        if code <= 0x1F:
            continue
        if 0xFFF9 <= code <= 0xFFFB:
            continue
        chars.append(char)
    return "".join(chars)


def _trim_to_last_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8", errors="replace")


async def execute_shell_with_capture(
    env: ExecutionEnv,
    command: str,
    options: dict[str, Any] | None = None,
):
    """执行 shell 命令并增量捕获输出（对齐 TS executeShellWithCapture）。"""
    options = options or {}
    max_output_bytes = DEFAULT_MAX_BYTES * 2
    on_chunk = options.get("onChunk")

    tail_output = ""
    total_bytes = 0
    completed_lines = 0
    has_open_line = False
    current_line_bytes = 0
    full_output_path: str | None = None
    full_output_requested = False
    accepting_output = True
    full_output_tasks: set[asyncio.Task] = set()
    full_output_lock = asyncio.Lock()
    capture_error: BaseException | None = None

    def _schedule_full_output(coro) -> None:
        task = asyncio.create_task(coro)
        full_output_tasks.add(task)
        task.add_done_callback(full_output_tasks.discard)

    async def _ensure_full_output_file(initial_content: str) -> None:
        nonlocal capture_error, full_output_path
        async with full_output_lock:
            if capture_error is not None:
                return
            temp_file = await env.create_temp_file({"prefix": "bash-", "suffix": ".log"})
            if not temp_file[0]:
                capture_error = to_execution_error(temp_file[1])
                return
            full_output_path = temp_file[1]
            append_result = await env.append_file(full_output_path, initial_content)
            if not append_result[0]:
                capture_error = to_execution_error(append_result[1])

    async def _append_full_output(text: str) -> None:
        nonlocal capture_error
        async with full_output_lock:
            if capture_error is not None or full_output_path is None:
                return
            append_result = await env.append_file(full_output_path, text)
            if not append_result[0]:
                capture_error = to_execution_error(append_result[1])

    def mark_full_output_requested(initial_content: str) -> None:
        nonlocal full_output_requested
        if full_output_requested or capture_error is not None:
            return
        full_output_requested = True
        _schedule_full_output(_ensure_full_output_file(initial_content))

    def on_chunk_internal(text: str) -> None:
        nonlocal tail_output, total_bytes, completed_lines, has_open_line, current_line_bytes
        if not accepting_output:
            return
        try:
            cleaned = sanitize_binary_output(text).replace("\r", "")
            cleaned_bytes = len(cleaned.encode("utf-8", errors="replace"))
            total_bytes += cleaned_bytes
            newline_count = cleaned.count("\n")
            completed_lines += newline_count
            last_newline = cleaned.rfind("\n")
            if last_newline >= 0:
                trailing = cleaned[last_newline + 1 :]
                current_line_bytes = len(trailing.encode("utf-8", errors="replace"))
                has_open_line = bool(trailing)
            elif cleaned:
                current_line_bytes += cleaned_bytes
                has_open_line = True

            tail_output += cleaned
            total_lines = completed_lines + (1 if has_open_line else 0)
            if (
                total_bytes > DEFAULT_MAX_BYTES or total_lines > DEFAULT_MAX_LINES
            ) and not full_output_requested:
                mark_full_output_requested(tail_output)
            elif full_output_requested:
                _schedule_full_output(_append_full_output(cleaned))
            tail_output = _trim_to_last_utf8_bytes(tail_output, max_output_bytes)
            if on_chunk is not None:
                on_chunk(cleaned, lambda: create_progress())
        except BaseException as error:
            nonlocal capture_error
            capture_error = error

    def create_progress() -> ShellCaptureProgress:
        tail_truncation = truncate_tail(tail_output)
        total_lines = completed_lines + (1 if has_open_line else 0)
        truncated = total_lines > DEFAULT_MAX_LINES or total_bytes > DEFAULT_MAX_BYTES
        truncation = TruncationResult(
            content=tail_truncation.content,
            truncated=truncated,
            truncated_by=(tail_truncation.truncated_by if truncated else None),
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=tail_truncation.output_lines,
            output_bytes=tail_truncation.output_bytes,
            last_line_partial=tail_truncation.last_line_partial,
            first_line_exceeds_limit=tail_truncation.first_line_exceeds_limit,
        )
        return ShellCaptureProgress(
            output=truncation.content if truncated else tail_output,
            truncation=truncation,
            full_output_path=full_output_path,
            last_line_bytes=current_line_bytes,
        )

    exec_options = ShellExecOptions(
        cwd=options.get("cwd"),
        env=options.get("env"),
        inherit_env=options.get("inheritEnv", True),
        unset_env=options.get("unsetEnv"),
        timeout=options.get("timeout"),
        abort_signal=options.get("abortSignal"),
        on_stdout=on_chunk_internal,
        on_stderr=on_chunk_internal,
    )
    result = await env.exec(command, exec_options)
    accepting_output = False
    progress = create_progress()
    if progress.truncation.truncated and not full_output_requested:
        mark_full_output_requested(tail_output)

    if full_output_tasks:
        await asyncio.gather(*full_output_tasks, return_exceptions=True)
    if capture_error is not None:
        return (False, to_execution_error(capture_error))
    progress = create_progress()

    if not result[0]:
        execution_error = result[1]
        aborted = options.get("abortSignal") is not None and options["abortSignal"].is_set()
        if execution_error.code == "aborted" or aborted:
            return (
                True,
                ShellCaptureResult(
                    output=progress.output,
                    truncation=progress.truncation,
                    full_output_path=full_output_path,
                    last_line_bytes=progress.last_line_bytes,
                    exit_code=None,
                    cancelled=True,
                ),
            )
        if options.get("returnExecutionErrors"):
            return (
                True,
                ShellCaptureResult(
                    output=progress.output,
                    truncation=progress.truncation,
                    full_output_path=full_output_path,
                    last_line_bytes=progress.last_line_bytes,
                    exit_code=None,
                    cancelled=False,
                    execution_error=execution_error,
                ),
            )
        return result

    shell_result = result[1]
    cancelled = options.get("abortSignal") is not None and options["abortSignal"].is_set()
    return (
        True,
        ShellCaptureResult(
            output=progress.output,
            truncation=progress.truncation,
            full_output_path=full_output_path,
            last_line_bytes=progress.last_line_bytes,
            exit_code=None if cancelled else shell_result.exit_code,
            cancelled=cancelled,
        ),
    )
