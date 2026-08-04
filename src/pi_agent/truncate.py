"""工具输出截断（Phase 4.3 辅助）。

对齐 TS `harness/utils/truncate.ts`：行数 + 字节数双限制，UTF-8 边界安全。
"""

from __future__ import annotations

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500


class TruncationResult:
    def __init__(
        self,
        content: str,
        truncated: bool,
        truncated_by: str | None,
        total_lines: int,
        total_bytes: int,
        output_lines: int,
        output_bytes: int,
        last_line_partial: bool = False,
        first_line_exceeds_limit: bool = False,
        max_lines: int = DEFAULT_MAX_LINES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.content = content
        self.truncated = truncated
        self.truncated_by = truncated_by
        self.total_lines = total_lines
        self.total_bytes = total_bytes
        self.output_lines = output_lines
        self.output_bytes = output_bytes
        self.last_line_partial = last_line_partial
        self.first_line_exceeds_limit = first_line_exceeds_limit
        self.max_lines = max_lines
        self.max_bytes = max_bytes


def utf8_byte_length(content: str) -> int:
    return len(content.encode("utf-8", errors="replace"))


def split_lines_for_counting(content: str) -> list[str]:
    if content == "":
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{(size / 1024):.1f}KB"
    return f"{(size / (1024 * 1024)):.1f}MB"


def truncate_head(
    content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> TruncationResult:
    """从头截断（保留开头），绝不返回半行。"""
    total_bytes = utf8_byte_length(content)
    lines = split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes
        )

    first_line_bytes = utf8_byte_length(lines[0]) if lines else 0
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            first_line_exceeds_limit=True,
        )

    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by: str = "lines"
    for index, line in enumerate(lines[:max_lines]):
        line_bytes = utf8_byte_length(line) + (1 if index > 0 else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines_arr.append(line)
        output_bytes_count += line_bytes
    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=utf8_byte_length(output_content),
    )


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    """从末尾向前的 UTF-8 安全截断。"""
    if max_bytes <= 0:
        return ""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    # 从后往前按字符收集，保持 UTF-8 边界
    result_chars: list[str] = []
    current_bytes = 0
    for char in reversed(text):
        char_bytes = len(char.encode("utf-8", errors="replace"))
        if current_bytes + char_bytes > max_bytes:
            break
        result_chars.append(char)
        current_bytes += char_bytes
    return "".join(reversed(result_chars))


def truncate_tail(
    content: str, max_lines: int = DEFAULT_MAX_LINES, max_bytes: int = DEFAULT_MAX_BYTES
) -> TruncationResult:
    """从尾截断（保留结尾，适合 bash 输出）。可能返回半行。"""
    total_bytes = utf8_byte_length(content)
    lines = split_lines_for_counting(content)
    total_lines = len(lines)

    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes
        )

    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by: str = "lines"
    last_line_partial = False
    for index in range(len(lines) - 1, -1, -1):
        if len(output_lines_arr) >= max_lines:
            break
        line = lines[index]
        line_bytes = utf8_byte_length(line) + (1 if output_lines_arr else 0)
        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not output_lines_arr:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines_arr.insert(0, truncated_line)
                output_bytes_count = utf8_byte_length(truncated_line)
                last_line_partial = True
            break
        output_lines_arr.insert(0, line)
        output_bytes_count += line_bytes
    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=utf8_byte_length(output_content),
        last_line_partial=last_line_partial,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True
