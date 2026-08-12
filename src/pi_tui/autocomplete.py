"""自动补全 provider 栈（对齐 TS CombinedAutocompleteProvider）。

provider 签名保持 `(text: str) -> list[dict] | Awaitable[list[dict]]`；
新增统一 provider 时使用 `get_suggestions()` / `apply_completion()`。
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .engine.fuzzy import fuzzy_filter

AutocompleteProvider = Callable[[str], list[dict[str, Any]] | Awaitable[list[dict[str, Any]]]]

# fd 子进程超时（秒）：慢目录 / 卡住的 fd 不得冻结 TUI；
# 调用方可通过 _walk_directory_with_fd 的 timeout 参数覆盖。
DEFAULT_FD_TIMEOUT_SECONDS = 5.0

_PATH_DELIMITERS = frozenset((" ", "\t", '"', "'", "="))
_FD_ARGS_BASE = (
    "--max-results",
    "100",
    "--type",
    "f",
    "--type",
    "d",
    "--follow",
    "--hidden",
    "--exclude",
    ".git",
    "--exclude",
    ".git/*",
    "--exclude",
    ".git/**",
)


@dataclass(frozen=True, slots=True)
class AutocompleteItem:
    """统一候选（对齐 TS AutocompleteItem）。"""

    value: str
    label: str = ""
    description: str = ""
    kind: str = "text"
    source: str = ""


@dataclass(frozen=True, slots=True)
class AutocompleteSuggestions:
    """一次补全的候选与待替换前缀。"""

    items: list[AutocompleteItem]
    prefix: str = ""
    kind: str = "text"


class CombinedAutocompleteProvider:
    """统一 slash / argument / path 补全；兼容旧的 providers 合并 API。"""

    def __init__(
        self,
        providers: list[AutocompleteProvider] | None = None,
        *,
        commands: list[Any] | None = None,
        base_path: str | Path | None = None,
        fd_path: str | None = None,
    ) -> None:
        self._providers = list(providers or [])
        self._commands = list(commands or [])
        self._base_path = str(base_path or ".")
        self._fd_path = fd_path if fd_path is not None else shutil.which("fd")
        self.trigger_characters = ["/", "@"]

    def add(self, provider: AutocompleteProvider) -> None:
        self._providers.append(provider)

    @property
    def providers(self) -> list[AutocompleteProvider]:
        return list(self._providers)

    async def collect(self, text: str) -> list[dict[str, Any]]:
        """并发收集所有 provider 结果，按 value 去重后保持注册顺序。"""
        if not self._providers:
            return []

        async def _run(provider: AutocompleteProvider) -> list[dict[str, Any]]:
            try:
                result = provider(text)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception:
                return []
            if not isinstance(result, list):
                return []
            return [item for item in result if isinstance(item, dict)]

        results = await asyncio.gather(*(_run(provider) for provider in self._providers))
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for items in results:
            for item in items:
                value = str(item.get("value", item.get("label", "")))
                if not value or value in seen:
                    continue
                seen.add(value)
                merged.append(item)
        return merged

    # ------------------------------------------------------------------
    # 统一建议
    # ------------------------------------------------------------------

    async def get_suggestions(
        self,
        text: str,
        *,
        force: bool = False,
        cursor: int | None = None,
    ) -> AutocompleteSuggestions | None:
        """按 TS 顺序返回建议：@ 文件 > slash > 参数 > 路径。"""
        if cursor is None:
            cursor = len(text)
        cursor = max(0, min(cursor, len(text)))
        before = text[:cursor]

        at_prefix = _extract_at_prefix(before)
        if at_prefix is not None:
            raw, _is_at, is_quoted = _parse_path_prefix(at_prefix)
            items = await self._get_fuzzy_file_suggestions(
                raw,
                is_at_prefix=True,
                is_quoted_prefix=is_quoted,
            )
            if items:
                return AutocompleteSuggestions(items=items, prefix=at_prefix, kind="attachment")

        if before.startswith("/"):
            space_index = before.find(" ")
            if space_index == -1:
                items = self._command_suggestions(before[1:])
                if items:
                    return AutocompleteSuggestions(
                        items=items,
                        prefix=before,
                        kind="command",
                    )
            else:
                command_name = before[1:space_index]
                argument_prefix = before[space_index + 1 :]
                items = await self._argument_suggestions(command_name, argument_prefix)
                if items:
                    return AutocompleteSuggestions(
                        items=items,
                        prefix=argument_prefix,
                        kind="argument",
                    )

        path_prefix = self._extract_path_prefix(before, force=force)
        if path_prefix is not None:
            items = await asyncio.to_thread(self._get_file_suggestions, path_prefix)
            if items:
                return AutocompleteSuggestions(items=items, prefix=path_prefix, kind="path")
        if self._providers:
            extension_items = await self.collect(text)
            if extension_items:
                return AutocompleteSuggestions(
                    items=[
                        AutocompleteItem(
                            value=str(item.get("value", item.get("label", ""))),
                            label=str(item.get("label", item.get("value", ""))),
                            description=str(item.get("description", "") or ""),
                            kind="text",
                            source="extension",
                        )
                        for item in extension_items
                    ],
                    prefix=text,
                    kind="text",
                )
        return None

    def apply_completion(
        self,
        text: str,
        suggestion: AutocompleteItem,
        prefix: str,
        cursor: int | None = None,
    ) -> tuple[str, int]:
        """把候选替换进文本，返回 (new_text, new_cursor)。"""
        if cursor is None:
            cursor = len(text)
        cursor = max(0, min(cursor, len(text)))
        before = text[:cursor]
        after = text[cursor:]
        if before.endswith(prefix):
            before = before[: len(before) - len(prefix)]

        if suggestion.kind == "command":
            replacement = f"/{suggestion.value} "
        elif suggestion.kind == "attachment":
            replacement = suggestion.value
            if suggestion.value.endswith("/"):
                replacement = suggestion.value
        else:
            replacement = suggestion.value
        new_before = before + replacement
        return new_before + after, len(new_before)

    # ------------------------------------------------------------------
    # slash / argument
    # ------------------------------------------------------------------

    def _command_suggestions(self, prefix: str) -> list[AutocompleteItem]:
        commands: list[dict[str, str]] = []
        seen: set[str] = set()
        for command in self._commands:
            name = _command_name(command)
            if not name or name in seen:
                continue
            seen.add(name)
            description = _command_description(command)
            hint = _command_hint(command)
            full = f"{hint} — {description}" if hint else description
            commands.append({"name": name, "label": name, "description": full})
        filtered = fuzzy_filter(commands, prefix, lambda item: item["name"])
        return [
            AutocompleteItem(
                value=item["name"],
                label=item["label"],
                description=item["description"] or "",
                kind="command",
                source="command",
            )
            for item in filtered
        ]

    async def _argument_suggestions(
        self,
        command_name: str,
        argument_prefix: str,
    ) -> list[AutocompleteItem]:
        for command in self._commands:
            if _command_name(command) != command_name:
                continue
            callback = _command_argument_completions(command)
            if callback is None:
                return []
            try:
                result = callback(argument_prefix)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception:
                return []
            return _normalize_argument_items(result)
        return []

    # ------------------------------------------------------------------
    # 路径补全
    # ------------------------------------------------------------------

    def _extract_path_prefix(self, before: str, *, force: bool) -> str | None:
        quoted = _extract_quoted_prefix(before)
        if quoted is not None:
            return quoted
        last_delimiter = _find_last_delimiter(before)
        path_prefix = before if last_delimiter == -1 else before[last_delimiter + 1 :]
        if force:
            return path_prefix
        if "/" in path_prefix or path_prefix.startswith(".") or path_prefix.startswith("~/"):
            return path_prefix
        if path_prefix == "" and before.endswith(" "):
            return path_prefix
        return None

    async def _get_fuzzy_file_suggestions(
        self,
        raw_prefix: str,
        *,
        is_at_prefix: bool,
        is_quoted_prefix: bool,
    ) -> list[AutocompleteItem]:
        if self._fd_path is None or not raw_prefix:
            return []

        scoped = _resolve_scoped_query(raw_prefix, self._base_path)
        base_dir = scoped["base_dir"] if scoped is not None else self._base_path
        query = scoped["query"] if scoped is not None else raw_prefix
        display_base = scoped["display_base"] if scoped is not None else ""
        entries = await _walk_directory_with_fd(base_dir, self._fd_path, query)
        scored = [
            (self._score_entry(entry_path, query, is_directory), entry_path, is_directory)
            for entry_path, is_directory in entries
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        items: list[AutocompleteItem] = []
        for _score, entry_path, is_directory in scored[:20]:
            path_without_slash = (
                entry_path[:-1] if is_directory and entry_path.endswith("/") else entry_path
            )
            display_path = _scoped_display_path(display_base, path_without_slash)
            value = _build_completion_value(
                display_path,
                is_directory=is_directory,
                is_at_prefix=is_at_prefix,
                is_quoted_prefix=is_quoted_prefix,
            )
            items.append(
                AutocompleteItem(
                    value=value,
                    label=Path(path_without_slash).name + ("/" if is_directory else ""),
                    description=display_path,
                    kind="attachment" if is_at_prefix else "path",
                    source="path",
                )
            )
        return items

    def _get_file_suggestions(
        self,
        prefix: str,
        *,
        is_at_prefix: bool = False,
        is_quoted_prefix: bool = False,
    ) -> list[AutocompleteItem]:
        raw, raw_is_at, raw_is_quoted = _parse_path_prefix(prefix)
        if not is_at_prefix and raw_is_at:
            is_at_prefix = True
        if not is_quoted_prefix and raw_is_quoted:
            is_quoted_prefix = True
        expanded = _expand_home(raw)
        try:
            if _is_root_prefix(raw, expanded):
                search_dir = (
                    Path(expanded)
                    if raw.startswith(("~", "/"))
                    else Path(self._base_path) / expanded
                )
                search_prefix = ""
            elif raw.endswith("/"):
                search_dir = (
                    Path(expanded)
                    if raw.startswith(("~", "/"))
                    else Path(self._base_path) / expanded
                )
                search_prefix = ""
            else:
                parent = Path(expanded).parent
                search_dir = (
                    parent if raw.startswith(("~", "/")) else Path(self._base_path) / parent
                )
                search_prefix = Path(expanded).name

            if not search_dir.is_dir():
                return []
            suggestions: list[AutocompleteItem] = []
            for entry in sorted(search_dir.iterdir(), key=lambda item: item.name.lower()):
                name = entry.name
                if not name.lower().startswith(search_prefix.lower()):
                    continue
                is_directory = entry.is_dir()
                relative_path = _relative_completion_path(raw, name, is_at_prefix)
                path_value = f"{relative_path}/" if is_directory else relative_path
                value = _build_completion_value(
                    path_value,
                    is_directory=is_directory,
                    is_at_prefix=is_at_prefix,
                    is_quoted_prefix=is_quoted_prefix,
                )
                suggestions.append(
                    AutocompleteItem(
                        value=value,
                        label=name + ("/" if is_directory else ""),
                        description=relative_path,
                        kind="attachment" if is_at_prefix else "path",
                        source="path",
                    )
                )
            suggestions.sort(
                key=lambda item: (
                    not item.label.endswith("/"),
                    item.label.lower(),
                )
            )
            return suggestions
        except OSError:
            return []

    def _score_entry(self, file_path: str, query: str, is_directory: bool) -> float:
        name = Path(file_path).name.lower()
        lower_query = query.lower()
        if not lower_query:
            return 1.0
        if name == lower_query:
            score = 100.0
        elif name.startswith(lower_query):
            score = 80.0
        elif lower_query in name:
            score = 50.0
        elif lower_query in file_path.lower():
            score = 30.0
        else:
            return 0.0
        if is_directory and score > 0:
            score += 10.0
        return score


def _normalize_argument_items(result: Any) -> list[AutocompleteItem]:
    if not isinstance(result, list):
        return []
    items: list[AutocompleteItem] = []
    for raw in result:
        if isinstance(raw, AutocompleteItem):
            items.append(raw)
            continue
        if isinstance(raw, dict):
            items.append(
                AutocompleteItem(
                    value=str(raw.get("value", "")),
                    label=str(raw.get("label", raw.get("value", ""))),
                    description=str(raw.get("description", "") or ""),
                    kind="argument",
                    source=str(raw.get("source", "argument")),
                )
            )
    return items


def _command_name(command: Any) -> str:
    if isinstance(command, dict):
        return str(command.get("name", ""))
    return str(getattr(command, "name", ""))


def _command_description(command: Any) -> str:
    if isinstance(command, dict):
        return str(command.get("description", "") or "")
    return str(getattr(command, "description", "") or "")


def _command_hint(command: Any) -> str:
    if isinstance(command, dict):
        return str(command.get("argument_hint", command.get("argumentHint", "")) or "")
    return str(getattr(command, "argument_hint", "") or "")


def _command_argument_completions(command: Any) -> Callable[[str], Any] | None:
    if isinstance(command, dict):
        return command.get("get_argument_completions") or command.get("argument_completions")
    return getattr(command, "get_argument_completions", None) or getattr(
        command, "argument_completions", None
    )


def _find_last_delimiter(text: str) -> int:
    for index in range(len(text) - 1, -1, -1):
        if text[index] in _PATH_DELIMITERS:
            return index
    return -1


def _extract_quoted_prefix(before: str) -> str | None:
    in_quotes = False
    quote_start = -1
    for index, char in enumerate(before):
        if char == '"':
            in_quotes = not in_quotes
            if in_quotes:
                quote_start = index
    if not in_quotes:
        return None
    if quote_start > 0 and before[quote_start - 1] == "@":
        if quote_start - 1 > 0 and before[quote_start - 2] not in _PATH_DELIMITERS:
            return None
        return before[quote_start - 1 :]
    if quote_start > 0 and before[quote_start - 1] not in _PATH_DELIMITERS:
        return None
    return before[quote_start:]


def _extract_at_prefix(before: str) -> str | None:
    quoted = _extract_quoted_prefix(before)
    if quoted is not None and quoted.startswith('@"'):
        return quoted
    last_delimiter = _find_last_delimiter(before)
    token_start = 0 if last_delimiter == -1 else last_delimiter + 1
    if token_start < len(before) and before[token_start] == "@":
        return before[token_start:]
    return None


def _parse_path_prefix(prefix: str) -> tuple[str, bool, bool]:
    if prefix.startswith('@"'):
        return prefix[2:], True, True
    if prefix.startswith('"'):
        return prefix[1:], False, True
    if prefix.startswith("@"):
        return prefix[1:], True, False
    return prefix, False, False


def _expand_home(path: str) -> str:
    if path == "~":
        return str(Path.home())
    if path.startswith("~/"):
        return str(Path.home() / path[2:])
    return path


def _is_root_prefix(raw: str, expanded: str) -> bool:
    return raw in ("", "./", "../", "~", "~/", "/") or (raw.startswith("@") and raw[1:] == "")


def _relative_completion_path(raw_prefix: str, name: str, is_at_prefix: bool) -> str:
    if raw_prefix.endswith("/"):
        return raw_prefix + name
    if "/" in raw_prefix or "\\" in raw_prefix:
        parent = str(Path(raw_prefix).parent)
        if parent == ".":
            return name
        return f"{parent}/{name}"
    if raw_prefix.startswith("~"):
        parent = str(Path(raw_prefix).parent)
        if parent == "~":
            return f"~/{name}"
        return f"{parent}/{name}"
    return name


def _build_completion_value(
    path: str,
    *,
    is_directory: bool,
    is_at_prefix: bool,
    is_quoted_prefix: bool,
) -> str:
    needs_quotes = is_quoted_prefix or " " in path
    prefix = "@" if is_at_prefix else ""
    if not needs_quotes:
        return f"{prefix}{path}"
    return f'{prefix}"{path}"'


def _resolve_scoped_query(
    raw_query: str,
    base_path: str,
) -> dict[str, str] | None:
    normalized = raw_query.replace("\\", "/")
    slash_index = normalized.rfind("/")
    if slash_index == -1:
        return None
    display_base = normalized[: slash_index + 1]
    query = normalized[slash_index + 1 :]
    if display_base.startswith("~/"):
        base_dir = _expand_home(display_base)
    elif display_base.startswith("/"):
        base_dir = display_base
    else:
        base_dir = str(Path(base_path) / display_base)
    if not Path(base_dir).is_dir():
        return None
    return {"base_dir": base_dir, "query": query, "display_base": display_base}


def _scoped_display_path(display_base: str, relative_path: str) -> str:
    if not display_base:
        return relative_path
    if display_base == "/":
        return f"/{relative_path}"
    return f"{display_base}{relative_path}"


def _kill_fd_process(process: asyncio.subprocess.Process | None) -> None:
    """终止 fd 子进程（POSIX 上杀整个进程组，避免残留孙进程持管道）。"""
    if process is None or process.returncode is not None or process.pid is None:
        return
    if os.name == "nt":
        try:
            process.kill()
        except OSError:
            pass
        return
    killpg = getattr(os, "killpg", None)
    if killpg is not None:
        try:
            killpg(process.pid, 9)
        except OSError:
            try:
                process.kill()
            except OSError:
                pass
    else:
        try:
            process.kill()
        except OSError:
            pass


async def _reap_fd_process(process: asyncio.subprocess.Process | None) -> None:
    """等待已终止的 fd 子进程退出（shield：调用方可能已被取消）。"""
    if process is None or process.returncode is not None:
        return
    try:
        await asyncio.shield(process.wait())
    except Exception:
        pass


async def _walk_directory_with_fd(
    base_dir: str,
    fd_path: str,
    query: str,
    *,
    timeout: float | None = None,
) -> list[tuple[str, bool]]:
    args = ["--base-directory", base_dir, *_FD_ARGS_BASE]
    if "/" in query:
        args.append("--full-path")
    if query:
        args.append(_build_fd_query(query))
    timeout = DEFAULT_FD_TIMEOUT_SECONDS if timeout is None else timeout
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            fd_path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name != "nt",
        )
        stdout, _stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        # 慢目录 / 卡住的 fd：终止子进程并返回空结果，避免冻结 TUI。
        _kill_fd_process(process)
        await _reap_fd_process(process)
        return []
    except asyncio.CancelledError:
        # 上层取消（abort 旧请求）时同样终止 fd 子进程，避免叠加残留进程。
        _kill_fd_process(process)
        await _reap_fd_process(process)
        raise
    except OSError:
        return []
    if process is None or process.returncode != 0 or not stdout:
        return []
    entries: list[tuple[str, bool]] = []
    for line in stdout.decode(errors="replace").splitlines():
        display = line.replace("\\", "/")
        is_directory = display.endswith("/")
        path = display[:-1] if is_directory else display
        if path == ".git" or path.startswith(".git/") or "/.git/" in path:
            continue
        entries.append((display, is_directory))
    return entries


def _build_fd_query(query: str) -> str:
    normalized = query.replace("\\", "/")
    if "/" not in normalized:
        return normalized
    has_trailing = normalized.endswith("/")
    trimmed = normalized.strip("/")
    if not trimmed:
        return normalized
    segments = [re.escape(segment) for segment in trimmed.split("/") if segment]
    pattern = r"[\\/]".join(segments)
    if has_trailing:
        pattern += r"[\\/]"
    return pattern


__all__ = [
    "AutocompleteItem",
    "AutocompleteProvider",
    "AutocompleteSuggestions",
    "CombinedAutocompleteProvider",
]
