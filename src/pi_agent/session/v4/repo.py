"""JSONL v4 会话仓库（对齐 TS `harness/session/jsonl/repo.ts`）。"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from pi_ai.utils.uuid import uuidv7

from .codec import invalid_file, metadata_from_header, parse_header
from .converter import convert_v3_file_to_v4, v3_header_metadata
from .json_validation import assert_json_serializable
from .jsonl_types import (
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlV4Header,
)
from .session import Session
from .storage import JsonlSessionStorage
from .types import ForkOptions, SessionError, SessionMetadata


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def _validate_session_id(session_id: str) -> None:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise SessionError(
            "invalid_payload",
            "Session id must be non-empty, contain only alphanumeric characters, "
            "'-', '_', and '.', and start and end with an alphanumeric character",
        )


def _session_directory_name(cwd: str) -> str:
    return "--" + re.sub(r"[/\\:]", "-", re.sub(r"^[/\\]", "", cwd)) + "--"


def _session_file_name(created_at: int, session_id: str) -> str:
    iso = (
        datetime.fromtimestamp(created_at / 1000, tz=timezone.utc)
        .isoformat()
        .replace(":", "-")
        .replace(".", "-")
    )
    return f"{iso}_{session_id}.jsonl"


def _now_ms() -> int:
    import time

    return time.time_ns() // 1_000_000


def _mtime_ms(path: Path) -> int:
    return path.stat().st_mtime_ns // 1_000_000


def _migration_enabled() -> bool:
    """PI_SESSION_FORMAT=v3 时禁用惰性迁移（过渡期调试）。"""
    return os.environ.get("PI_SESSION_FORMAT", "auto") != "v3"


def _metadata_from_first_line(first_line: str, path: str, modified_at: int) -> JsonlSessionMetadata:
    """按首行 version 分发到 v4 / v3 元数据解析。"""
    if '"kind":"header"' in first_line or first_line.startswith('{"kind": "header"'):
        return metadata_from_header(parse_header(first_line, path), path, modified_at)
    return v3_header_metadata(first_line, path, modified_at)


class JsonlSessionRepo:
    """JSONL v4 会话仓库：create / open / list / delete / fork。"""

    def __init__(self, sessions_root: str | Path) -> None:
        self._sessions_root = Path(sessions_root).resolve()

    async def create(self, options: JsonlSessionCreateOptions | None = None) -> Session:
        options = options or {}
        session_id = options.get("id") or uuidv7()
        _validate_session_id(session_id)
        cwd = str(Path(options.get("cwd", ".")).resolve())
        if await self._session_id_exists(session_id, cwd):
            raise SessionError("already_exists", f"Session already exists: {session_id}")

        created_at = _now_ms()
        session_dir = self._session_dir(cwd)
        session_dir.mkdir(parents=True, exist_ok=True)
        path = str(session_dir / _session_file_name(created_at, session_id))
        if options.get("metadata") is not None:
            assert_json_serializable(options["metadata"])
        header: JsonlV4Header = {
            "kind": "header",
            "version": 4,
            "id": session_id,
            "createdAt": created_at,
            "cwd": cwd,
        }
        if options.get("parentSessionId") is not None:
            header["parentSessionId"] = options["parentSessionId"]
        if options.get("metadata") is not None:
            header["metadata"] = options["metadata"]
        storage = await JsonlSessionStorage.create(path, header)
        return Session(storage)

    async def open(self, metadata: SessionMetadata) -> Session:
        jsonl_metadata = cast(JsonlSessionMetadata, metadata)
        storage = await self._load_storage(jsonl_metadata)
        loaded = await storage.get_metadata()
        if loaded["id"] != metadata["id"]:
            raise SessionError(
                "invalid_entry", f"Session id does not match header: {metadata['id']}"
            )
        return Session(storage)

    def metadata_for_path(
        self, path: str | Path, cwd_override: str | None = None
    ) -> JsonlSessionMetadata:
        """按文件路径读取首行并构造元数据（v3/v4 均可）。"""
        file_path = Path(path)
        first_line = file_path.read_text(encoding="utf-8").split("\n", 1)[0]
        if not first_line:
            raise invalid_file(str(file_path), 1, "is missing a header")
        metadata = _metadata_from_first_line(first_line, str(file_path), _mtime_ms(file_path))
        if cwd_override is not None:
            metadata["cwd"] = cwd_override
        return metadata

    async def open_path(self, path: str | Path, cwd_override: str | None = None) -> Session:
        """打开任意路径的会话文件（v3 文件自动惰性转换）。"""
        return await self.open(self.metadata_for_path(path, cwd_override))

    async def _load_storage(self, jsonl_metadata: JsonlSessionMetadata) -> JsonlSessionStorage:
        """加载存储；v3 文件按 PI_SESSION_FORMAT 惰性转换为 v4。"""
        metadata = jsonl_metadata
        path = Path(metadata["path"])
        if not path.exists():
            raise SessionError("not_found", f"Session not found: {metadata['path']}")
        first_line = path.read_text(encoding="utf-8").split("\n", 1)[0]
        if '"kind":"header"' in first_line or first_line.startswith('{"kind": "header"'):
            return await JsonlSessionStorage.load(str(path))
        if not _migration_enabled():
            raise SessionError(
                "storage",
                "PI_SESSION_FORMAT=v3 禁用在 v4 仓库中惰性转换旧会话",
            )
        return await convert_v3_file_to_v4(str(path))

    async def list(
        self, options: JsonlSessionListOptions | None = None
    ) -> list[JsonlSessionMetadata]:
        options = options or {}
        directories: list[Path]
        if options.get("cwd") is not None:
            directory = self._session_dir(str(Path(options["cwd"]).resolve()))
            directories = [directory] if directory.exists() else []
        else:
            if not self._sessions_root.exists():
                directories = []
            else:
                directories = [entry for entry in self._sessions_root.iterdir() if entry.is_dir()]
        metadata: list[JsonlSessionMetadata] = []
        for directory in directories:
            for file_path in directory.iterdir():
                if file_path.is_dir() or not file_path.name.endswith(".jsonl"):
                    continue
                content = file_path.read_text(encoding="utf-8")
                first_line = content.split("\n", 1)[0]
                if not first_line:
                    raise invalid_file(str(file_path), 1, "is missing a header")
                metadata.append(
                    _metadata_from_first_line(first_line, str(file_path), _mtime_ms(file_path))
                )
        metadata.sort(key=lambda item: item["modifiedAt"], reverse=True)
        return metadata

    async def delete(self, metadata: SessionMetadata) -> None:
        path = Path(cast(JsonlSessionMetadata, metadata)["path"])
        path.unlink(missing_ok=True)
        Path(f"{path}.bak").unlink(missing_ok=True)

    async def fork(
        self,
        source: SessionMetadata,
        options: ForkOptions | None = None,
    ) -> Session:
        options = options or {}
        source_metadata = cast(JsonlSessionMetadata, source)
        source_storage = await self._load_storage(source_metadata)
        fork_options = cast(dict[str, Any], options)
        session_id = options.get("id") or uuidv7()
        _validate_session_id(session_id)
        cwd = str(Path(fork_options.get("cwd") or source_metadata.get("cwd") or ".").resolve())
        if await self._session_id_exists(session_id, cwd):
            raise SessionError("already_exists", f"Session already exists: {session_id}")
        created_at = _now_ms()
        session_dir = self._session_dir(cwd)
        session_dir.mkdir(parents=True, exist_ok=True)
        path = str(session_dir / _session_file_name(created_at, session_id))
        header: JsonlV4Header = {
            "kind": "header",
            "version": 4,
            "id": session_id,
            "createdAt": created_at,
            "cwd": cwd,
            "parentSessionId": options.get("parentSessionId") or source_metadata["id"],
        }
        if fork_options.get("metadata") is not None:
            assert_json_serializable(fork_options["metadata"])
            header["metadata"] = fork_options["metadata"]
        storage = await source_storage.fork(path, header, options)
        return Session(storage)

    def _session_dir(self, cwd: str) -> Path:
        return self._sessions_root / _session_directory_name(cwd)

    async def _session_id_exists(self, session_id: str, cwd: str) -> bool:
        suffix = f"_{session_id}.jsonl"
        directory = self._session_dir(cwd)
        if not directory.exists():
            return False
        return any(entry.is_file() and entry.name.endswith(suffix) for entry in directory.iterdir())


__all__ = ["JsonlSessionRepo", "_validate_session_id"]
