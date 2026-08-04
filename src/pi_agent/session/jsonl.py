"""JSONL Session 持久化（Phase 3.2）。

对齐 TS `harness/session/jsonl-storage.ts` 与 `jsonl-repo.ts`：

- 每行一个 JSON 对象；首行为 SessionHeader（version 3）
- 追加写入（append-only），原子逐行 append
- 目录布局：`{sessions_root}/{encoded_cwd}/{timestamp}_{sessionId}.jsonl`
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from .repo import (
    SessionRepo,
    create_session_id,
    create_timestamp,
    get_entries_to_fork,
    get_path_to_root_or_compaction,
)
from .search import ScanningSessionSearch
from .session import _get_session_name, _get_session_stats
from .types import (
    JsonlSessionMetadata,
    LeafEntry,
    SessionEntryCursorOptions,
    SessionError,
    SessionForkOptions,
    SessionSnapshot,
    SessionTreeEntry,
)


def _encode_cwd(cwd: str) -> str:
    return "--" + re.sub(r"[/\\:]", "-", re.sub(r"^[/\\]", "", cwd)) + "--"


def _update_label_cache(
    labels_by_id: dict[str, str],
    entry: SessionTreeEntry,
) -> None:
    if entry["type"] != "label":
        return
    label = (entry.get("label") or "").strip()
    target_id = cast(Any, entry)["targetId"]
    if label:
        labels_by_id[target_id] = label
    else:
        labels_by_id.pop(target_id, None)


def _build_labels_by_id(entries: list[SessionTreeEntry]) -> dict[str, str]:
    labels_by_id: dict[str, str] = {}
    for entry in entries:
        _update_label_cache(labels_by_id, entry)
    return labels_by_id


def _generate_entry_id(by_id: dict[str, SessionTreeEntry]) -> str:
    from pi_ai.utils.uuid import uuidv7

    for _ in range(100):
        entry_id = uuidv7()[-8:]
        if entry_id not in by_id:
            return entry_id
    return uuidv7()


def _leaf_id_after_entry(entry: SessionTreeEntry) -> str | None:
    return entry["targetId"] if entry["type"] == "leaf" else entry["id"]


def _invalid_session(
    file_path: str, message: str, cause: BaseException | None = None
) -> SessionError:
    return SessionError(
        "invalid_session",
        f"Invalid JSONL session file {file_path}: {message}",
        cause,
    )


def _invalid_entry(
    file_path: str, line_number: int, message: str, cause: BaseException | None = None
) -> SessionError:
    return SessionError(
        "invalid_entry",
        f"Invalid JSONL session file {file_path}: line {line_number} {message}",
        cause,
    )


def _parse_header_line(line: str, file_path: str) -> dict[str, Any]:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise _invalid_session(
            file_path, "first line is not a valid session header", error
        ) from error
    if not isinstance(parsed, dict):
        raise _invalid_session(file_path, "first line is not a valid session header")
    if parsed.get("type") != "session":
        raise _invalid_session(file_path, "first line is not a valid session header")
    if parsed.get("version") != 3:
        raise _invalid_session(file_path, "unsupported session version")
    if not isinstance(parsed.get("id"), str) or not parsed["id"]:
        raise _invalid_session(file_path, "session header is missing id")
    if not isinstance(parsed.get("timestamp"), str) or not parsed["timestamp"]:
        raise _invalid_session(file_path, "session header is missing timestamp")
    if not isinstance(parsed.get("cwd"), str) or not parsed["cwd"]:
        raise _invalid_session(file_path, "session header is missing cwd")
    if parsed.get("parentSession") is not None and not isinstance(parsed.get("parentSession"), str):
        raise _invalid_session(file_path, "session header parentSession must be a string")
    return parsed


def _parse_entry_line(line: str, file_path: str, line_number: int) -> SessionTreeEntry:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise _invalid_entry(file_path, line_number, "is not valid JSON", error) from error
    if not isinstance(parsed, dict):
        raise _invalid_entry(file_path, line_number, "is not a valid session entry")
    if not isinstance(parsed.get("type"), str):
        raise _invalid_entry(file_path, line_number, "is missing entry type")
    if not isinstance(parsed.get("id"), str) or not parsed["id"]:
        raise _invalid_entry(file_path, line_number, "is missing entry id")
    if parsed.get("parentId") is not None and not isinstance(parsed.get("parentId"), str):
        raise _invalid_entry(file_path, line_number, "has invalid parentId")
    if not isinstance(parsed.get("timestamp"), str) or not parsed["timestamp"]:
        raise _invalid_entry(file_path, line_number, "is missing timestamp")
    return cast(SessionTreeEntry, parsed)


def _header_to_metadata(header: dict[str, Any], path: str) -> JsonlSessionMetadata:
    metadata: JsonlSessionMetadata = {
        "id": header["id"],
        "createdAt": header["timestamp"],
        "cwd": header["cwd"],
        "path": path,
    }
    if header.get("parentSession") is not None:
        metadata["parentSessionPath"] = header["parentSession"]
    if header.get("metadata") is not None:
        metadata["metadata"] = header["metadata"]
    return metadata


def _load_storage(
    file_path: Path,
) -> tuple[dict[str, Any], list[SessionTreeEntry], str | None]:
    content = file_path.read_text(encoding="utf-8")
    lines = [line for line in content.split("\n") if line.strip()]
    if not lines:
        raise _invalid_session(str(file_path), "missing session header")
    header = _parse_header_line(lines[0], str(file_path))
    entries: list[SessionTreeEntry] = []
    leaf_id: str | None = None
    for index, line in enumerate(lines[1:], start=2):
        entry = _parse_entry_line(line, str(file_path), index)
        entries.append(entry)
        leaf_id = _leaf_id_after_entry(entry)
    return header, entries, leaf_id


class JsonlSessionStorage:
    """JSONL 文件会话存储。每行一个 JSON 条目，追加写入。"""

    def __init__(
        self,
        file_path: str,
        header: dict[str, Any],
        entries: list[SessionTreeEntry],
        leaf_id: str | None,
    ) -> None:
        self._file_path = Path(file_path)
        self._metadata = _header_to_metadata(header, str(self._file_path))
        self._entries = list(entries)
        self._by_id = {entry["id"]: entry for entry in self._entries}
        self._labels_by_id = _build_labels_by_id(self._entries)
        self._leaf_id = leaf_id

    @staticmethod
    async def open(file_path: str) -> "JsonlSessionStorage":
        header, entries, leaf_id = _load_storage(Path(file_path))
        return JsonlSessionStorage(file_path, header, entries, leaf_id)

    @staticmethod
    async def create(
        file_path: str,
        *,
        cwd: str,
        session_id: str,
        parent_session_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "JsonlSessionStorage":
        header: dict[str, Any] = {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": create_timestamp(),
            "cwd": cwd,
        }
        if parent_session_path is not None:
            header["parentSession"] = parent_session_path
        if metadata is not None:
            header["metadata"] = metadata
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")
        return JsonlSessionStorage(file_path, header, [], None)

    async def get_metadata(self) -> JsonlSessionMetadata:
        return dict(self._metadata)

    async def get_leaf_id(self) -> str | None:
        if self._leaf_id is not None and self._leaf_id not in self._by_id:
            raise SessionError("invalid_session", f"Entry {self._leaf_id} not found")
        return self._leaf_id

    async def set_leaf_id(self, leaf_id: str | None) -> LeafEntry:
        if leaf_id is not None and leaf_id not in self._by_id:
            raise SessionError("not_found", f"Entry {leaf_id} not found")
        entry: LeafEntry = {
            "type": "leaf",
            "id": _generate_entry_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": create_timestamp(),
            "targetId": leaf_id,
        }
        self._append_line(entry)
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry
        self._leaf_id = leaf_id
        return entry

    def _append_line(self, entry: SessionTreeEntry) -> None:
        with self._file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def create_entry_id(self) -> str:
        return _generate_entry_id(self._by_id)

    async def append_entry(self, entry: SessionTreeEntry) -> None:
        self._append_line(entry)
        self._entries.append(entry)
        self._by_id[entry["id"]] = entry
        _update_label_cache(self._labels_by_id, entry)
        self._leaf_id = _leaf_id_after_entry(entry)

    async def get_entry(self, entry_id: str) -> SessionTreeEntry | None:
        return self._by_id.get(entry_id)

    async def find_entries(self, entry_type: str) -> list[SessionTreeEntry]:
        return [entry for entry in self._entries if entry["type"] == entry_type]

    async def get_label(self, entry_id: str) -> str | None:
        return self._labels_by_id.get(entry_id)

    async def get_session_name(self) -> str | None:
        return _get_session_name(self._entries)

    async def get_session_stats(self):
        return _get_session_stats(self._entries)

    async def get_path_to_root_or_compaction(self, leaf_id: str | None) -> list[SessionTreeEntry]:
        return get_path_to_root_or_compaction(self._entries, leaf_id)

    async def get_entries(
        self, options: SessionEntryCursorOptions | None = None
    ) -> list[SessionTreeEntry]:
        start = (options or {}).get("afterEntrySeq", 0)
        limit = (options or {}).get("limit")
        end = None if limit is None else start + limit
        return self._entries[start:end]


def _load_metadata(file_path: Path) -> JsonlSessionMetadata:
    lines = [line for line in file_path.read_text(encoding="utf-8").split("\n") if line.strip()]
    if not lines:
        raise _invalid_session(str(file_path), "missing session header")
    header = _parse_header_line(lines[0], str(file_path))
    return _header_to_metadata(header, str(file_path))


class JsonlSessionStore:
    """JSONL 会话存储仓库（对齐 TS JsonlSessionStore）。"""

    def __init__(self, sessions_root: str) -> None:
        self._sessions_root = Path(sessions_root)

    def _session_dir(self, cwd: str) -> Path:
        return self._sessions_root / _encode_cwd(cwd)

    def _session_file_path(self, cwd: str, session_id: str, timestamp: str) -> Path:
        safe_timestamp = timestamp.replace(":", "-").replace(".", "-")
        return self._session_dir(cwd) / f"{safe_timestamp}_{session_id}.jsonl"

    async def create(self, options: dict[str, Any] | None = None) -> JsonlSessionMetadata:
        options = options or {}
        cwd = options["cwd"]
        session_id = options.get("id") or create_session_id()
        file_path = self._session_file_path(cwd, session_id, create_timestamp())
        storage = await JsonlSessionStorage.create(
            str(file_path),
            cwd=cwd,
            session_id=session_id,
            parent_session_path=options.get("parentSessionPath"),
            metadata=options.get("metadata"),
        )
        return await storage.get_metadata()

    async def open(self, metadata: JsonlSessionMetadata) -> JsonlSessionStorage:
        path = Path(metadata["path"])
        if not path.exists():
            raise SessionError("not_found", f"Session not found: {metadata['path']}")
        return await JsonlSessionStorage.open(str(path))

    async def load(self, metadata: JsonlSessionMetadata) -> SessionSnapshot:
        storage = await self.open(metadata)
        return {
            "metadata": await storage.get_metadata(),
            "leafId": await storage.get_leaf_id(),
            "entries": await storage.get_entries(),
        }

    async def list(self, options: dict[str, Any] | None = None) -> list[JsonlSessionMetadata]:
        options = options or {}
        if options.get("cwd") is not None:
            dirs = [self._session_dir(options["cwd"])]
        else:
            if not self._sessions_root.exists():
                return []
            dirs = [entry for entry in self._sessions_root.iterdir() if entry.is_dir()]
        sessions: list[JsonlSessionMetadata] = []
        for directory in dirs:
            if not directory.exists():
                continue
            for file_path in directory.iterdir():
                if file_path.is_dir() or not file_path.name.endswith(".jsonl"):
                    continue
                try:
                    sessions.append(_load_metadata(file_path))
                except SessionError as error:
                    if error.code != "invalid_session":
                        raise
        sessions.sort(key=lambda s: s["createdAt"], reverse=True)
        return sessions

    async def get_entries(
        self, metadata: JsonlSessionMetadata, options: SessionEntryCursorOptions | None = None
    ) -> list[SessionTreeEntry]:
        return await (await self.open(metadata)).get_entries(options)

    async def create_entry_id(self, metadata: JsonlSessionMetadata) -> str:
        return await (await self.open(metadata)).create_entry_id()

    async def append_entry(self, metadata: JsonlSessionMetadata, entry: SessionTreeEntry) -> None:
        await (await self.open(metadata)).append_entry(entry)

    async def set_leaf_id(self, metadata: JsonlSessionMetadata, leaf_id: str | None) -> LeafEntry:
        return await (await self.open(metadata)).set_leaf_id(leaf_id)

    async def delete(self, metadata: JsonlSessionMetadata) -> None:
        Path(metadata["path"]).unlink(missing_ok=True)

    async def fork(
        self, source: JsonlSessionMetadata, options: SessionForkOptions | None = None
    ) -> JsonlSessionMetadata:
        options = options or {}
        source_storage = await self.open(source)
        forked_entries = await get_entries_to_fork(source_storage, options)
        cwd = options.get("cwd", source["cwd"])
        session_id = options.get("id") or create_session_id()
        file_path = self._session_file_path(cwd, session_id, create_timestamp())
        storage = await JsonlSessionStorage.create(
            str(file_path),
            cwd=cwd,
            session_id=session_id,
            parent_session_path=options.get("parentSessionPath") or source["path"],
            metadata=options.get("metadata") or source.get("metadata"),
        )
        for entry in forked_entries:
            await storage.append_entry(entry)
        return await storage.get_metadata()


def create_jsonl_session_store(sessions_root: str) -> JsonlSessionStore:
    return JsonlSessionStore(sessions_root)


def create_jsonl_session_repo(sessions_root: str) -> SessionRepo:
    store = create_jsonl_session_store(sessions_root)
    return SessionRepo(store=store, search=ScanningSessionSearch(store))
