"""编辑差异工具（对齐 TS harness/tools/edit-diff.ts）。

支持：LF 规范化、Unicode 模糊匹配、BOM 保留、原始行尾恢复、
统一 patch（difflib）与展示用 diff 字符串。
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any


def detect_line_ending(content: str) -> str:
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1:
        return "\n"
    if crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def strip_bom(content: str) -> tuple[str, str]:
    if content.startswith("\ufeff"):
        return "\ufeff", content[1:]
    return "", content


def normalize_for_fuzzy_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", normalized)
    normalized = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', normalized)
    normalized = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", normalized)
    normalized = re.sub(r"[\u00A0\u2002-\u200A\u202F\u205F\u3000]", " ", normalized)
    return normalized


def _split_lines_with_endings(content: str) -> list[str]:
    return re.findall(r"[^\n]*\n|[^\n]+", content)


def fuzzy_find_text(content: str, old_text: str) -> dict[str, Any]:
    exact_index = content.find(old_text)
    if exact_index != -1:
        return {
            "found": True,
            "index": exact_index,
            "matchLength": len(old_text),
            "usedFuzzyMatch": False,
            "contentForReplacement": content,
        }
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old)
    if fuzzy_index == -1:
        return {
            "found": False,
            "index": -1,
            "matchLength": 0,
            "usedFuzzyMatch": False,
            "contentForReplacement": content,
        }
    return {
        "found": True,
        "index": fuzzy_index,
        "matchLength": len(fuzzy_old),
        "usedFuzzyMatch": True,
        "contentForReplacement": fuzzy_content,
    }


def _count_occurrences(content: str, old_text: str) -> int:
    return normalize_for_fuzzy_match(content).count(normalize_for_fuzzy_match(old_text))


def _apply_replacements(content: str, replacements: list[dict[str, Any]], offset: int = 0) -> str:
    result = content
    for replacement in reversed(replacements):
        match_index = replacement["matchIndex"] - offset
        result = (
            result[:match_index]
            + replacement["newText"]
            + result[match_index + replacement["matchLength"] :]
        )
    return result


def _get_replacement_line_range(
    lines: list[dict[str, int]],
    replacement: dict[str, Any],
) -> tuple[int, int]:
    replacement_start = replacement["matchIndex"]
    replacement_end = replacement["matchIndex"] + replacement["matchLength"]
    start_line = -1
    for index, line in enumerate(lines):
        if line["start"] <= replacement_start < line["end"]:
            start_line = index
            break
    if start_line == -1:
        raise ValueError("Replacement range is outside the base content.")
    end_line = start_line
    while end_line < len(lines) and lines[end_line]["end"] < replacement_end:
        end_line += 1
    if end_line >= len(lines):
        raise ValueError("Replacement range is outside the base content.")
    return start_line, end_line + 1


def _apply_replacements_preserving_unchanged_lines(
    original_content: str,
    base_content: str,
    replacements: list[dict[str, Any]],
) -> str:
    original_lines = _split_lines_with_endings(original_content)
    base_lines_raw = _split_lines_with_endings(base_content)
    base_lines: list[dict[str, int]] = []
    offset = 0
    for line in base_lines_raw:
        base_lines.append({"start": offset, "end": offset + len(line)})
        offset += len(line)
    if len(original_lines) != len(base_lines):
        raise ValueError(
            "Cannot preserve unchanged lines because the base content has a different line count."
        )

    groups: list[dict[str, Any]] = []
    sorted_replacements = sorted(replacements, key=lambda r: r["matchIndex"])
    for replacement in sorted_replacements:
        start_line, end_line = _get_replacement_line_range(base_lines, replacement)
        if groups and start_line < groups[-1]["endLine"]:
            groups[-1]["endLine"] = max(groups[-1]["endLine"], end_line)
            groups[-1]["replacements"].append(replacement)
            continue
        groups.append({"startLine": start_line, "endLine": end_line, "replacements": [replacement]})

    original_line_index = 0
    result = ""
    for group in groups:
        result += "".join(original_lines[original_line_index : group["startLine"]])
        group_start_offset = base_lines[group["startLine"]]["start"]
        group_end_offset = base_lines[group["endLine"] - 1]["end"]
        result += _apply_replacements(
            base_content[group_start_offset:group_end_offset],
            group["replacements"],
            group_start_offset,
        )
        original_line_index = group["endLine"]
    result += "".join(original_lines[original_line_index:])
    return result


def _not_found_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"Could not find the exact text in {path}. "
            "The old text must match exactly including all whitespace and newlines."
        )
    return (
        f"Could not find edits[{edit_index}] in {path}. "
        "The oldText must match exactly including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> str:
    if total_edits == 1:
        return (
            f"Found {occurrences} occurrences of the text in {path}. "
            "The text must be unique. Please provide more context to make it unique."
        )
    return (
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each oldText must be unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return f"oldText must not be empty in {path}."
    return f"edits[{edit_index}].oldText must not be empty in {path}."


def _no_change_error(path: str, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing as expected."
        )
    return f"No changes made to {path}. The replacements produced identical content."


def apply_edits_to_normalized_content(
    normalized_content: str,
    edits: list[dict[str, str]],
    path: str,
) -> dict[str, str]:
    normalized_edits = [
        {"oldText": normalize_to_lf(edit["oldText"]), "newText": normalize_to_lf(edit["newText"])}
        for edit in edits
    ]
    for index, edit in enumerate(normalized_edits):
        if len(edit["oldText"]) == 0:
            raise ValueError(_empty_old_text_error(path, index, len(normalized_edits)))

    initial_matches = [
        fuzzy_find_text(normalized_content, edit["oldText"]) for edit in normalized_edits
    ]
    used_fuzzy = any(match["usedFuzzyMatch"] for match in initial_matches)
    replacement_base = (
        normalize_for_fuzzy_match(normalized_content) if used_fuzzy else normalized_content
    )

    matched_edits: list[dict[str, Any]] = []
    for index, edit in enumerate(normalized_edits):
        match = fuzzy_find_text(replacement_base, edit["oldText"])
        if not match["found"]:
            raise ValueError(_not_found_error(path, index, len(normalized_edits)))
        occurrences = _count_occurrences(replacement_base, edit["oldText"])
        if occurrences > 1:
            raise ValueError(_duplicate_error(path, index, len(normalized_edits), occurrences))
        matched_edits.append(
            {
                "editIndex": index,
                "matchIndex": match["index"],
                "matchLength": match["matchLength"],
                "newText": edit["newText"],
            }
        )

    matched_edits.sort(key=lambda m: m["matchIndex"])
    for index in range(1, len(matched_edits)):
        previous = matched_edits[index - 1]
        current = matched_edits[index]
        if previous["matchIndex"] + previous["matchLength"] > current["matchIndex"]:
            raise ValueError(
                f"edits[{previous['editIndex']}] and edits[{current['editIndex']}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    base_content = normalized_content
    new_content = (
        _apply_replacements_preserving_unchanged_lines(
            normalized_content, replacement_base, matched_edits
        )
        if used_fuzzy
        else _apply_replacements(replacement_base, matched_edits)
    )
    if base_content == new_content:
        raise ValueError(_no_change_error(path, len(normalized_edits)))
    return {"baseContent": base_content, "newContent": new_content}


def generate_unified_patch(
    path: str, old_content: str, new_content: str, context_lines: int = 4
) -> str:
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")
    old_has_final_newline = bool(old_lines) and old_lines[-1] == ""
    new_has_final_newline = bool(new_lines) and new_lines[-1] == ""
    if old_has_final_newline:
        old_lines.pop()
    if new_has_final_newline:
        new_lines.pop()

    if old_lines == new_lines and old_has_final_newline != new_has_final_newline:
        final_newline_output: list[str] = [
            f"--- {path}\n",
            f"+++ {path}\n",
            f"@@ -1,{len(old_lines)} +1,{len(new_lines)} @@\n",
        ]
        for line in old_lines[:-1]:
            final_newline_output.append(f" {line}\n")
        if old_lines:
            last_line = old_lines[-1]
            final_newline_output.append(f"-{last_line}\n")
            if not old_has_final_newline:
                final_newline_output.append("\\ No newline at end of file\n")
            final_newline_output.append(f"+{last_line}\n")
            if not new_has_final_newline:
                final_newline_output.append("\\ No newline at end of file\n")
        return "".join(final_newline_output)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=path,
        tofile=path,
        lineterm="\n",
        n=context_lines,
    )

    output: list[str] = []
    last_old_line = old_lines[-1] if old_lines else None
    last_new_line = new_lines[-1] if new_lines else None
    for line in diff:
        if line.endswith("\n"):
            output.append(line)
            continue
        prefix = line[:1]
        body = line[1:]
        output.append(line + "\n")
        if not old_has_final_newline and prefix in (" ", "-") and body == last_old_line:
            output.append("\\ No newline at end of file\n")
        if not new_has_final_newline and prefix in (" ", "+") and body == last_new_line:
            output.append("\\ No newline at end of file\n")
    return "".join(output)


def _split_diff_part(value: str) -> list[str]:
    """Split one diffLines part; drop the trailing synthetic empty element."""
    raw = value.split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    return raw


def generate_diff_string(
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> dict[str, Any]:
    """生成带行号的展示用 diff（逐分支对齐 TS generateDiffString）。"""
    old_all = old_content.split("\n")
    new_all = new_content.split("\n")
    old_has_final_newline = bool(old_all) and old_all[-1] == ""
    new_has_final_newline = bool(new_all) and new_all[-1] == ""
    old_lines = old_all[:-1] if old_has_final_newline else old_all
    new_lines = new_all[:-1] if new_has_final_newline else new_all

    if old_lines == new_lines and old_has_final_newline != new_has_final_newline:
        # Diff.diffLines 把末尾 newline 差异编码为最后一行的 replace。
        last = len(old_lines)
        parts = [
            ("equal", 0, last - 1, 0, last - 1),
            ("replace", last - 1, last, last - 1, last),
        ]
    else:
        differ = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
        parts = list(differ.get_opcodes())

    output: list[str] = []
    old_line_num = 1
    new_line_num = 1
    last_was_change = False
    first_changed_line: int | None = None
    width = max(len(str(len(old_all))), len(str(len(new_all))))
    for part_index, (tag, i1, i2, j1, j2) in enumerate(parts):
        old_block = old_lines[i1:i2]
        new_block = new_lines[j1:j2]
        is_change = tag in ("replace", "delete", "insert")

        if is_change:
            if first_changed_line is None:
                first_changed_line = new_line_num
            for line in old_block:
                output.append(f"-{str(old_line_num).rjust(width)} {line}")
                old_line_num += 1
            for line in new_block:
                output.append(f"+{str(new_line_num).rjust(width)} {line}")
                new_line_num += 1
            last_was_change = True
            continue

        raw = old_block
        next_part_is_change = part_index < len(parts) - 1 and parts[part_index + 1][0] in (
            "replace",
            "delete",
            "insert",
        )
        has_leading_change = last_was_change
        has_trailing_change = next_part_is_change

        if has_leading_change and has_trailing_change:
            if len(raw) <= context_lines * 2:
                for line in raw:
                    output.append(f" {str(old_line_num).rjust(width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
            else:
                leading = raw[:context_lines]
                trailing = raw[-context_lines:]
                skipped = len(raw) - len(leading) - len(trailing)
                for line in leading:
                    output.append(f" {str(old_line_num).rjust(width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
                output.append(f" {'':{width}} ...")
                old_line_num += skipped
                new_line_num += skipped
                for line in trailing:
                    output.append(f" {str(old_line_num).rjust(width)} {line}")
                    old_line_num += 1
                    new_line_num += 1
        elif has_leading_change:
            shown = raw[:context_lines]
            skipped = len(raw) - len(shown)
            for line in shown:
                output.append(f" {str(old_line_num).rjust(width)} {line}")
                old_line_num += 1
                new_line_num += 1
            if skipped > 0:
                output.append(f" {'':{width}} ...")
                old_line_num += skipped
                new_line_num += skipped
        elif has_trailing_change:
            skipped = max(0, len(raw) - context_lines)
            if skipped > 0:
                output.append(f" {'':{width}} ...")
                old_line_num += skipped
                new_line_num += skipped
            for line in raw[skipped:]:
                output.append(f" {str(old_line_num).rjust(width)} {line}")
                old_line_num += 1
                new_line_num += 1
        else:
            old_line_num += len(raw)
            new_line_num += len(raw)

        last_was_change = False

    return {"diff": "\n".join(output), "firstChangedLine": first_changed_line}
