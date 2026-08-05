"""read 工具（对齐 TS harness/tools/read.ts）。"""

from __future__ import annotations

from typing import Any

from pi_ai.types import ImageContent, TextContent

from .._types import AgentTool, AgentToolResult
from ..env import FileError, get_or_throw
from ..truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)
from .image import detect_supported_image_mime_type, encode_base64
from .image_pipeline import process_image
from .path_utils import resolve_read_tool_path


class ReadToolOptions:
    def __init__(
        self,
        auto_resize_images: bool = True,
        image_processor=process_image,
    ) -> None:
        self.auto_resize_images = auto_resize_images
        self.image_processor = image_processor


def create_read_tool(options: ReadToolOptions | None = None) -> AgentTool:
    options = options or ReadToolOptions()

    async def execute(
        tool_call_id, params, signal=None, on_update=None, context=None
    ) -> AgentToolResult:
        env = context.env
        path = params["path"]
        absolute_path = await resolve_read_tool_path(env, path, signal)
        bytes_result = await env.read_binary_file(absolute_path, signal)
        try:
            data = get_or_throw(bytes_result)
        except FileError as exc:
            if exc.code == "not_found":
                raise ValueError(
                    f"{exc} The file is not in the working directory. Report this to "
                    "the user directly; do not search the whole disk "
                    "(e.g. find /, grep -r /, locate)."
                ) from exc
            raise

        mime_type = detect_supported_image_mime_type(data)
        if mime_type:
            if options.image_processor is not None:
                processed = await options.image_processor(
                    data,
                    mime_type,
                    {
                        "autoResizeImages": options.auto_resize_images,
                    },
                )
                if not processed["ok"]:
                    return AgentToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=f"Read image file [{mime_type}]\n{processed['message']}",
                            )
                        ],
                        details=None,
                    )
                hints = "\n".join(processed["hints"]) if processed["hints"] else ""
                text = f"Read image file [{processed['mimeType']}]" + (
                    f"\n{hints}" if hints else ""
                )
                return AgentToolResult(
                    content=[
                        TextContent(type="text", text=text),
                        ImageContent(
                            type="image",
                            url=None,
                            data=processed["data"],
                            mime_type=processed["mimeType"],
                        ),
                    ],
                    details=None,
                )
            if mime_type == "image/bmp":
                return AgentToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text="Read image file [image/bmp]\n[Image omitted: configure an imageProcessor to convert BMP images.]",
                        )
                    ],
                    details=None,
                )
            return AgentToolResult(
                content=[
                    TextContent(type="text", text=f"Read image file [{mime_type}]"),
                    ImageContent(
                        type="image",
                        url=None,
                        data=encode_base64(data),
                        mime_type=mime_type,
                    ),
                ],
                details=None,
            )

        text_content = data.decode("utf-8", errors="replace")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, (params.get("offset") or 1) - 1)
        start_line_display = start_line + 1
        if start_line >= total_file_lines:
            raise ValueError(
                f"Offset {params.get('offset')} is beyond end of file ({total_file_lines} lines total)"
            )

        if "limit" in params:
            end_line = min(start_line + params["limit"], total_file_lines)
            selected = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_line:])
            user_limited_lines = None

        truncation: TruncationResult = truncate_head(selected)
        details: Any = None
        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(
                len(all_lines[start_line].encode("utf-8", errors="replace"))
            )
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
            )
            details = {"truncation": truncation}
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                )
            details = {"truncation": truncation}
        elif user_limited_lines is not None and start_line + user_limited_lines < total_file_lines:
            remaining = total_file_lines - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
        else:
            output_text = truncation.content

        return AgentToolResult(
            content=[TextContent(type="text", text=output_text)],
            details=details,
        )

    return AgentTool(
        name="read",
        label="read",
        prompt_snippet="Read file contents",
        description=(
            f"Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
            f"Images are sent as attachments. For text files, output is truncated to {DEFAULT_MAX_LINES} lines "
            f"or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). Use offset/limit for large files. "
            "When you need the full file, continue with offset until complete. "
            "Only operate on files inside the current working directory. If a file is "
            "not found, report that to the user; do not search the whole disk "
            "(e.g. find /, grep -r /, locate)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-indexed)",
                },
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
        execute=execute,
    )
