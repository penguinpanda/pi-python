"""edit 工具（对齐 TS harness/tools/edit.ts）。"""

from __future__ import annotations

import json

from pi_ai.types import TextContent

from .._types import AgentTool, AgentToolResult
from .edit_diff import (
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path


def _prepare_edit_arguments(input_data: dict) -> dict:
    """兼容 legacy oldText/newText 参数与字符串化 edits。"""
    args = dict(input_data)
    if isinstance(args.get("edits"), str):
        try:
            parsed = json.loads(args["edits"])
            if isinstance(parsed, list):
                args["edits"] = parsed
        except json.JSONDecodeError:
            pass
    if isinstance(args.get("oldText"), str) and isinstance(args.get("newText"), str):
        legacy_edits = args.get("edits")
        edits = list(legacy_edits) if isinstance(legacy_edits, list) else []
        edits.append({"oldText": args["oldText"], "newText": args["newText"]})
        args.pop("oldText", None)
        args.pop("newText", None)
        args["edits"] = edits
    return args


def _edit_access_error(path: str, error) -> ValueError:
    return ValueError(f"Could not edit file: {path}. Error code: {error.code}.")


def create_edit_tool() -> AgentTool:
    async def execute(
        tool_call_id, params, signal=None, on_update=None, context=None
    ) -> AgentToolResult:
        env = context.env
        # prepare_arguments 已在 schema 校验前归一化（对齐 TS prepareEditArguments）；
        # 此处保留幂等归一化作为防御（无 loop 直接调用 execute 的场景）。
        input_data = _prepare_edit_arguments(params)
        path = input_data["path"]
        edits = input_data.get("edits")
        if not isinstance(edits, list) or len(edits) == 0:
            raise ValueError(
                "Edit tool input is invalid. edits must contain at least one replacement."
            )

        absolute_path = await resolve_tool_path(env, path, signal)

        async def _edit() -> AgentToolResult:
            if signal is not None and signal.is_set():
                raise ValueError("Operation aborted")
            info = await env.file_info(absolute_path, signal)
            if not info[0]:
                raise _edit_access_error(path, info[1])
            if info[1].kind not in ("file", "symlink"):
                raise ValueError(f"Could not edit file: {path}. Path is not a file.")

            read_result = await env.read_text_file(absolute_path, signal)
            if not read_result[0]:
                raise _edit_access_error(path, read_result[1])
            if signal is not None and signal.is_set():
                raise ValueError("Operation aborted")

            bom, content = strip_bom(read_result[1])
            original_ending = detect_line_ending(content)
            normalized = normalize_to_lf(content)
            applied = apply_edits_to_normalized_content(normalized, edits, path)
            if signal is not None and signal.is_set():
                raise ValueError("Operation aborted")

            final_content = bom + restore_line_endings(applied["newContent"], original_ending)
            write_result = await env.write_file(absolute_path, final_content, signal)
            if not write_result[0]:
                raise _edit_access_error(path, write_result[1])
            if signal is not None and signal.is_set():
                raise ValueError("Operation aborted")

            diff_result = generate_diff_string(applied["baseContent"], applied["newContent"])
            return AgentToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Successfully replaced {len(edits)} block(s) in {path}.",
                    )
                ],
                details={
                    "diff": diff_result["diff"],
                    "patch": generate_unified_patch(
                        path, applied["baseContent"], applied["newContent"]
                    ),
                    "firstChangedLine": diff_result["firstChangedLine"],
                },
            )

        return await with_file_mutation_queue(env, absolute_path, _edit)

    return AgentTool(
        name="edit",
        label="edit",
        prepare_arguments=_prepare_edit_arguments,
        prompt_snippet=(
            "Make precise file edits with exact text replacement, "
            "including multiple disjoint edits in one call"
        ),
        prompt_guidelines=[
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
        ],
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match a unique, "
            "non-overlapping region of the original file. If two changes affect the same block or nearby lines, "
            "merge them into one edit instead of emitting overlapping edits."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative or absolute)",
                },
                "edits": {
                    "type": "array",
                    "description": (
                        "One or more targeted replacements. Each edit is matched against "
                        "the original file, not incrementally. Do not include overlapping "
                        "or nested edits. If two changes touch the same block or nearby "
                        "lines, merge them into one edit instead."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": (
                                    "Exact text for one targeted replacement. It must be "
                                    "unique in the original file and must not overlap with "
                                    "any other edits[].oldText in the same call."
                                ),
                            },
                            "newText": {
                                "type": "string",
                                "description": "Replacement text for this targeted edit.",
                            },
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
        execute=execute,
    )
