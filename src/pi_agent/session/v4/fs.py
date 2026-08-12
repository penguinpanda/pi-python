"""JSONL v4 会话文件系统抽象（对齐 TS `JsonlSessionRepoFileSystem`）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class FileInfo:
    path: str
    name: str
    mtime_ms: int
    kind: Literal["file", "directory", "symlink"]


class JsonlSessionRepoFileSystem(Protocol):
    """TS `JsonlSessionRepoFileSystem` 的 Python 协议。"""

    def absolute_path(self, path: str | Path) -> str: ...

    def join_path(self, parts: list[str]) -> str: ...

    def read_text_file(self, path: str | Path) -> str: ...

    def write_file(self, path: str | Path, content: str) -> None: ...

    def append_file(self, path: str | Path, content: str) -> None: ...

    def rename_file(self, source: str | Path, destination: str | Path) -> None: ...

    def file_info(self, path: str | Path) -> FileInfo: ...

    def list_dir(self, path: str | Path) -> list[FileInfo]: ...

    def exists(self, path: str | Path) -> bool: ...

    def create_dir(self, path: str | Path, *, recursive: bool = True) -> None: ...

    def remove(self, path: str | Path, *, force: bool = True) -> None: ...


def _kind(path: Path) -> Literal["file", "directory", "symlink"]:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    return "file"


class LocalFileSystem:
    """基于 `pathlib` 的默认文件系统实现。"""

    def absolute_path(self, path: str | Path) -> str:
        return str(Path(path).resolve())

    def join_path(self, parts: list[str]) -> str:
        return str(Path(*parts))

    def read_text_file(self, path: str | Path) -> str:
        return Path(path).read_text(encoding="utf-8")

    def write_file(self, path: str | Path, content: str) -> None:
        Path(path).write_text(content, encoding="utf-8")

    def append_file(self, path: str | Path, content: str) -> None:
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(content)

    def rename_file(self, source: str | Path, destination: str | Path) -> None:
        os.replace(source, destination)

    def file_info(self, path: str | Path) -> FileInfo:
        file_path = Path(path)
        stat = file_path.stat()
        return FileInfo(
            path=str(file_path),
            name=file_path.name,
            mtime_ms=stat.st_mtime_ns // 1_000_000,
            kind=_kind(file_path),
        )

    def list_dir(self, path: str | Path) -> list[FileInfo]:
        directory = Path(path)
        result: list[FileInfo] = []
        for entry in directory.iterdir():
            stat = entry.stat()
            result.append(
                FileInfo(
                    path=str(entry),
                    name=entry.name,
                    mtime_ms=stat.st_mtime_ns // 1_000_000,
                    kind=_kind(entry),
                )
            )
        return result

    def exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    def create_dir(self, path: str | Path, *, recursive: bool = True) -> None:
        Path(path).mkdir(parents=recursive, exist_ok=True)

    def remove(self, path: str | Path, *, force: bool = True) -> None:
        if force:
            Path(path).unlink(missing_ok=True)
        else:
            Path(path).unlink()


__all__ = ["FileInfo", "JsonlSessionRepoFileSystem", "LocalFileSystem"]
