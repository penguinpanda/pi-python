"""v4 版会话管理器（M3：应用层消费 v4 Session）。

与 `_session_manager.SessionManager` 保持同名 API，但底层使用
`pi_agent.session.v4` 的 `JsonlSessionRepo` / `Session`：
- 新会话写入 JSONL v4；
- 打开 v3 文件时由 `JsonlSessionRepo.open_path` 惰性转换；
- 同步访问器（get_entries / build_context / get_tree 等）读内存缓存，
  每次异步写入后刷新缓存。
"""

from __future__ import annotations

import asyncio
import inspect
import time
from pathlib import Path
from typing import Any, Protocol, cast

from filelock import FileLock
from pi_ai.utils.uuid import uuidv7

from pi_agent._types import AgentMessage
from pi_agent.session.v4.context import build_session_context
from pi_agent.session.v4.memory import InMemorySessionRepo
from pi_agent.session.v4.repo import JsonlSessionRepo
from pi_agent.session.v4.session import Session
from pi_agent.session.v4.jsonl_types import JsonlSessionCreateOptions, JsonlSessionListOptions
from pi_agent.session.v4.types import Entry, SessionMetadata

from ._session_manager import SessionInfo, SessionTreeNode


def _default_sessions_dir() -> Path:
    from ._config import get_sessions_dir

    return get_sessions_dir()


class SessionManagerLike(Protocol):
    """AgentSession 需要的会话管理器接口（同步门面 / V4SessionManager 通用）。"""

    @property
    def session_id(self) -> str: ...

    @property
    def session_path(self) -> Path | None: ...

    @property
    def session_name(self) -> str | None: ...

    @property
    def cwd(self) -> str: ...

    def is_persisted(self) -> bool: ...

    def set_session_name(self, name: str) -> None: ...

    def build_context(self) -> list[AgentMessage]: ...

    def get_entries(self) -> list[Any]: ...

    def get_entry(self, entry_id: str) -> Any | None: ...

    def get_leaf_id(self) -> str | None: ...

    async def fork(
        self,
        from_entry_id: str,
        *,
        position: str = "at",
        session_id: str | None = None,
        sessions_dir: str | Path | None = None,
    ) -> "SessionManagerLike": ...

    def get_branch(self, from_id: str | None = None) -> list[Any]: ...

    def get_tree(self) -> list[Any]: ...

    def get_last_model_change(self) -> tuple[str, str] | None: ...

    async def append_message(self, message: AgentMessage) -> str: ...

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict | None = None,
    ) -> str: ...

    async def append_model_change(self, provider: str, model_id: str) -> str: ...

    async def append_thinking_level_change(self, thinking_level: str) -> str: ...

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str: ...

    async def append_custom_message_entry(
        self,
        custom_type: str,
        content: Any,
        *,
        display: bool = True,
        details: Any = None,
    ) -> str: ...

    async def append_session_info(self, name: str | None) -> str: ...

    async def append_label(self, target_id: str, label: str | None) -> str: ...

    def set_label(self, target_id: str, label: str | None) -> None: ...

    async def move_to(
        self, entry_id: str | None, summary: dict[str, Any] | None = None
    ) -> str | None: ...


async def create_session_manager(
    cwd: str,
    sessions_dir: str | Path | None = None,
    session_id: str | None = None,
    parent_session_id: str | None = None,
    repo: Any = None,
) -> SessionManagerLike:
    """创建 v4 会话管理器。"""
    if repo is not None:
        return await V4SessionManager.from_repo(repo, cwd, session_id)
    return await V4SessionManager.create(
        cwd, sessions_dir, session_id, parent_session_id=parent_session_id
    )


async def open_session_manager(
    filepath: str | Path,
    cwd_override: str | None = None,
    repo: Any = None,
    metadata: SessionMetadata | None = None,
) -> SessionManagerLike:
    """打开会话；v3 文件由 v4 仓库惰性转换。"""
    if repo is not None:
        if metadata is None:
            raise ValueError("open_session_manager with repo requires metadata")
        return await V4SessionManager.open_with_repo(repo, metadata)
    return await V4SessionManager.open(filepath, cwd_override)


async def in_memory_session_manager(cwd: str, session_id: str | None = None) -> SessionManagerLike:
    return await V4SessionManager.in_memory(cwd, session_id)


async def list_sessions(
    directory: str | Path,
    cwd: str | None = None,
    *,
    detailed: bool = True,
) -> list[SessionInfo]:
    """列出 v4 会话。"""
    return await V4SessionManager.list_sessions(
        directory,
        cwd,
        detailed=detailed,
    )


async def fork_session_manager(
    manager: Any,
    from_entry_id: str,
    *,
    position: str = "at",
    session_id: str | None = None,
    sessions_dir: str | Path | None = None,
) -> Any:
    """fork 会话：同步门面 / 原生异步统一入口。"""
    result = manager.fork(
        from_entry_id,
        position=position,
        session_id=session_id,
        sessions_dir=sessions_dir,
    )
    if inspect.isawaitable(result):
        return await result
    return result


async def edit_session_message(
    manager: Any,
    entry_id: str,
    new_text: str,
    *,
    mode: str = "merge",
) -> str:
    """编辑历史 user 消息：同步门面 / 原生异步统一入口。"""
    result = manager.edit_message(entry_id, new_text, mode=mode)
    if inspect.isawaitable(result):
        return await result
    return result


class V4SessionManager:
    """v4 持久化会话管理器（API 对齐 SessionManager）。"""

    def __init__(
        self,
        *,
        cwd: str,
        session: Session,
        repo: JsonlSessionRepo | InMemorySessionRepo,
        session_path: Path | None = None,
        sessions_root: Path | None = None,
    ) -> None:
        self._cwd = cwd
        self._session = session
        self._repo = repo
        self._session_path = session_path
        self._sessions_root = sessions_root
        self._entries: list[Entry] = []
        self._by_id: dict[str, Entry] = {}
        self._leaf_id: str | None = None
        self._name: str | None = None
        self._labels: dict[str, str | None] = {}
        self._session_id = ""
        self._created_at: int | None = None
        self._lock = asyncio.Lock()
        self._scheduled_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # 工厂
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        cwd: str,
        sessions_dir: str | Path | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> "V4SessionManager":
        """创建新 v4 会话（目录布局与 SessionManager 一致）。"""
        root = Path(sessions_dir) if sessions_dir else _default_sessions_dir()
        repo = JsonlSessionRepo(root)
        options: JsonlSessionCreateOptions = {"cwd": cwd, "id": session_id or uuidv7()}
        if parent_session_id:
            options["parentSessionId"] = parent_session_id
        session = await repo.create(options)
        return await cls._from_session(cwd, session, repo, root)

    @classmethod
    async def from_repo(
        cls,
        repo: Any,
        cwd: str,
        session_id: str | None = None,
    ) -> "V4SessionManager":
        """在指定 v4 SessionRepo（如 PostgresV4SessionRepo）上创建会话。"""
        session = await repo.create({"cwd": cwd, "id": session_id or uuidv7()})
        return await cls._from_session(cwd, session, repo, None)

    @classmethod
    async def open_with_repo(
        cls,
        repo: Any,
        metadata: SessionMetadata,
    ) -> "V4SessionManager":
        """按元数据在指定 v4 SessionRepo 上打开会话。"""
        session = await repo.open(metadata)
        cwd = str(cast(dict[str, Any], metadata).get("cwd") or "")
        return await cls._from_session(cwd, session, repo, None)

    @classmethod
    async def open(
        cls,
        filepath: str | Path,
        cwd_override: str | None = None,
    ) -> "V4SessionManager":
        """打开已有会话文件（v3 自动惰性转换，v4 直接读取）。"""
        repo = JsonlSessionRepo(_default_sessions_dir())
        session = await repo.open_path(filepath, cwd_override)
        metadata = await session.get_metadata()
        cwd = cwd_override or cast(dict[str, Any], metadata)["cwd"]
        return await cls._from_session(
            cwd,
            session,
            repo,
            Path(filepath).parent.parent,
        )

    @classmethod
    async def in_memory(cls, cwd: str, session_id: str | None = None) -> "V4SessionManager":
        """非持久化 v4 会话（内存模式）。"""
        repo = InMemorySessionRepo()
        session = await repo.create({"id": session_id or uuidv7()})
        return await cls._from_session(cwd, session, repo, None)

    @classmethod
    async def _from_session(
        cls,
        cwd: str,
        session: Session,
        repo: JsonlSessionRepo | InMemorySessionRepo,
        sessions_root: Path | None,
    ) -> "V4SessionManager":
        metadata = await session.get_metadata()
        manager = cls(
            cwd=cwd,
            session=session,
            repo=repo,
            session_path=Path(cast(dict[str, Any], metadata)["path"])
            if isinstance(repo, JsonlSessionRepo)
            else None,
            sessions_root=sessions_root,
        )
        await manager._refresh()
        return manager

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_path(self) -> Path | None:
        return self._session_path

    @property
    def session_name(self) -> str | None:
        return self._name

    @property
    def cwd(self) -> str:
        return self._cwd

    def is_persisted(self) -> bool:
        return self._session_path is not None

    # ------------------------------------------------------------------
    # 同步访问器（读内存缓存）
    # ------------------------------------------------------------------

    def get_header(self) -> dict[str, Any] | None:
        """合成 session header（对齐 TS SessionManager.getHeader）。

        v4 JSONL 无 header 行，从缓存的 metadata 派生 {type, id, timestamp, cwd}。
        """
        if not self._session_id:
            return None
        header: dict[str, Any] = {
            "type": "session",
            "id": self._session_id,
            "timestamp": str(self._created_at or ""),
            "cwd": self._cwd,
        }
        return header

    def get_entries(self) -> list[Entry]:
        return list(self._entries)

    def get_entry(self, entry_id: str) -> Entry | None:
        return self._by_id.get(entry_id)

    def get_leaf_id(self) -> str | None:
        return self._leaf_id

    def get_label(self, target_id: str) -> str | None:
        return self._labels.get(target_id)

    def get_branch(self, from_id: str | None = None) -> list[Entry]:
        start = from_id if from_id is not None else self._leaf_id
        if start is None:
            return []
        path: list[Entry] = []
        current: Entry | None = self._by_id.get(start)
        seen: set[str] = set()
        while current is not None:
            if current["id"] in seen:
                raise ValueError(f"Session branch contains a cycle at {current['id']}")
            seen.add(current["id"])
            path.append(current)
            parent_id = current.get("parentId")
            current = self._by_id.get(parent_id) if parent_id is not None else None
        path.reverse()
        return path

    def build_context(self) -> list[AgentMessage]:
        return list(build_session_context(self.get_branch())["messages"])

    def get_last_model_change(self) -> tuple[str, str] | None:
        for entry in reversed(self._entries):
            if entry["type"] == "model_change":
                return (
                    cast(dict[str, Any], entry)["provider"],
                    cast(dict[str, Any], entry)["modelId"],
                )
        return None

    def get_tree(self) -> list[SessionTreeNode]:
        """构建会话树（v4 无 leaf/label 条目：标签来自 labels 缓存）。"""
        nodes: dict[str, SessionTreeNode] = {}
        for entry in self._entries:
            entry_id = entry["id"]
            nodes[entry_id] = SessionTreeNode(
                id=entry_id,
                parent_id=entry.get("parentId"),
                entry=cast(Any, entry),
            )
        for target_id, label in self._labels.items():
            node = nodes.get(target_id)
            if node is not None and label is not None:
                node.label = label
        roots: list[SessionTreeNode] = []
        for entry in self._entries:
            node = nodes[entry["id"]]
            parent_id = entry.get("parentId")
            if parent_id is None or parent_id == entry["id"]:
                roots.append(node)
                continue
            parent = nodes.get(parent_id)
            if parent is not None:
                parent.children.append(node)
            else:
                roots.append(node)

        def _sort(nodes_to_sort: list[SessionTreeNode]) -> None:
            for node in nodes_to_sort:
                node.children.sort(
                    key=lambda child: int(cast(Any, child.entry or {}).get("timestamp", 0) or 0)
                )
                _sort(node.children)

        _sort(roots)
        return roots

    # ------------------------------------------------------------------
    # 写入（异步，写后刷新缓存）
    # ------------------------------------------------------------------

    async def append_message(self, message: AgentMessage) -> str:
        entry_id = await self._session.append_message(message)
        await self._cache_entry(entry_id)
        return entry_id

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: dict | None = None,
    ) -> str:
        entry_id = await self._session.append_compaction(
            summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
        )
        await self._cache_entry(entry_id)
        return entry_id

    async def append_model_change(self, provider: str, model_id: str) -> str:
        entry_id = await self._session.append_model_change(provider, model_id)
        await self._cache_entry(entry_id)
        return entry_id

    async def append_thinking_level_change(self, thinking_level: str) -> str:
        entry_id = await self._session.append_thinking_level_change(thinking_level)
        await self._cache_entry(entry_id)
        return entry_id

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        entry_id = await self._session.append_custom_entry(custom_type, data)
        await self._cache_entry(entry_id)
        return entry_id

    async def append_custom_message_entry(
        self,
        custom_type: str,
        content: Any,
        *,
        display: bool = True,
        details: Any = None,
    ) -> str:
        entry_id = await self._session.append_custom_message_entry(
            custom_type, content, display=display, details=details
        )
        await self._cache_entry(entry_id)
        return entry_id

    async def append_session_info(self, name: str | None) -> str:
        if name is not None:
            await self._session.set_name(name)
        else:
            self._name = None
        self._name = await self._session.get_name()
        return ""

    def set_session_name(self, name: str) -> None:
        """同步设置会话名（异步持久化，任务纳入跟踪，close 时等待）。"""
        self._name = name
        self._schedule_tracked(self.append_session_info(name))

    async def append_label(self, target_id: str, label: str | None) -> str:
        await self._session.set_label(target_id, label)
        self._labels[target_id] = label
        return target_id

    def set_label(self, target_id: str, label: str | None) -> None:
        """同步设置标签（异步持久化，任务纳入跟踪，close 时等待）。"""
        self._labels[target_id] = label
        self._schedule_tracked(self.append_label(target_id, label))

    def _schedule_tracked(self, coroutine: Any) -> None:
        """在运行中的事件循环里调度协程并持有句柄（close 时等待完成）。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        task = asyncio.create_task(coroutine)
        self._scheduled_tasks.add(task)
        task.add_done_callback(self._scheduled_tasks.discard)

    async def move_to(
        self, entry_id: str | None, summary: dict[str, Any] | None = None
    ) -> str | None:
        result = await self._session.move_to(entry_id, summary)
        if result is not None:
            await self._cache_entry(result)
        else:
            self._leaf_id = entry_id
        return result

    async def fork(
        self,
        from_entry_id: str,
        *,
        position: str = "at",
        session_id: str | None = None,
        sessions_dir: str | Path | None = None,
    ) -> "V4SessionManager":
        """从指定条目 fork 新分支（v4 branch scope）。"""
        path = self.get_branch(from_entry_id)
        if not path:
            raise ValueError(f"Entry not found: {from_entry_id}")
        repo: JsonlSessionRepo | InMemorySessionRepo
        if sessions_dir is not None:
            root = Path(sessions_dir)
            repo = JsonlSessionRepo(root)
        elif self._sessions_root is not None:
            root = self._sessions_root
            repo = JsonlSessionRepo(root)
        else:
            repo = self._repo
            root = None
        session = await repo.fork(
            cast(SessionMetadata, await self._session.get_metadata()),
            cast(
                Any,
                {
                    "scope": "branch",
                    "entryId": from_entry_id,
                    "position": position,
                    "id": session_id or uuidv7(),
                    "cwd": self._cwd,
                },
            ),
        )
        return await V4SessionManager._from_session(self._cwd, session, repo, root)

    # ------------------------------------------------------------------
    # Operation records（M4）
    # ------------------------------------------------------------------

    async def start_operation(
        self,
        kind: str,
        *,
        run_id: str | None = None,
        source_leaf_id: str | None = None,
        original_prompt: list[AgentMessage] | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
        custom_instructions: str | None = None,
        result_entry_id: str | None = None,
        target_id: str | None = None,
        summarize: bool = False,
    ) -> str:
        """写入 operation_started 记录（run / compaction / navigation）。"""
        record_id = run_id or uuidv7()
        if kind == "run":
            intent: dict[str, Any] = {
                "kind": "run",
                "originalPrompt": list(original_prompt or []),
                "initialMessages": list(initial_messages or []),
            }
        elif kind == "compaction":
            intent = {"kind": "compaction", "resultEntryId": result_entry_id or ""}
        elif kind == "navigation":
            intent = {"kind": "navigation", "targetId": target_id, "summarize": summarize}
            if custom_instructions is not None:
                intent["customInstructions"] = custom_instructions
        else:
            raise ValueError(f"Unknown operation kind: {kind}")
        await self._session.append_record(
            {
                "type": "operation_started",
                "id": record_id,
                "lane": "main",
                "sourceLeafId": source_leaf_id or self._leaf_id,
                "intent": intent,
            }
        )
        return record_id

    async def finish_operation(
        self,
        run_id: str,
        outcome: str = "completed",
        error: dict[str, str] | None = None,
    ) -> None:
        """写入 operation_finished 记录。"""
        record: dict[str, Any] = {
            "type": "operation_finished",
            "id": uuidv7(),
            "lane": "main",
            "runId": run_id,
            "outcome": outcome,
        }
        if error is not None:
            record["error"] = error
        await self._session.append_record(record)

    async def record_usage(
        self,
        *,
        cause: str,
        usage: dict[str, Any],
        run_id: str | None = None,
        entry_id: str | None = None,
        attempt: int | None = None,
        stop_reason: str | None = None,
        details: Any = None,
    ) -> None:
        """写入 usage 记录（影响会话统计）。"""
        record: dict[str, Any] = {
            "type": "usage",
            "id": uuidv7(),
            "lane": "main",
            "cause": cause,
            "usage": usage,
        }
        if run_id is not None:
            record["runId"] = run_id
        if entry_id is not None:
            record["entryId"] = entry_id
        if attempt is not None:
            record["attempt"] = attempt
        if stop_reason is not None:
            record["stopReason"] = stop_reason
        if details is not None:
            record["details"] = details
        await self._session.append_record(record)

    async def record_write_deferred(self, target: dict[str, Any]) -> None:
        """写入 write_deferred 记录（延迟写入审计）。"""
        await self._session.append_record(
            {
                "type": "write_deferred",
                "id": uuidv7(),
                "lane": "main",
                "runId": "",
                "target": target,
            }
        )

    async def find_records(self, query: dict[str, Any] | None = None) -> list[dict]:
        return [
            cast(dict[str, Any], record)
            for record in await self._session.find_records(cast(Any, query))
        ]

    async def open_operations(self, lane: str = "main", limit: int = 2) -> list[dict]:
        return [
            cast(dict[str, Any], operation)
            for operation in await self._session.find_open_operations(lane, {"limit": limit})
        ]

    async def recovery_state(self, lane: str = "main") -> str:
        """0=idle、1=suspended、>=2=corrupt（对齐 TS findOpenOperations）。"""
        operations = await self.open_operations(lane, limit=2)
        if len(operations) == 0:
            return "idle"
        if len(operations) == 1:
            return "suspended"
        return "corrupt"

    async def get_session_stats(self) -> dict[str, Any]:
        return cast(dict[str, Any], await self._session.get_stats())

    # ------------------------------------------------------------------
    # 其他
    # ------------------------------------------------------------------

    async def edit_message(
        self,
        entry_id: str,
        new_text: str,
        *,
        mode: str = "merge",
    ) -> str:
        """把新文本合并/替换进一条历史 user 消息（v4 原生语义）。

        v4 仅追加：追加一条合并后的 user 消息并把 main lane 移到它，
        旧条目保留在文件中（旧分支仍可被树查看）。
        """
        if mode not in ("merge", "replace"):
            raise ValueError(f"Unknown edit mode: {mode}")
        target = self.get_entry(entry_id)
        if target is None:
            raise ValueError(f"Entry not found: {entry_id}")
        if target["type"] != "message":
            raise ValueError(f"Entry {entry_id} is not a message")
        message = cast(dict[str, Any], target).get("message")
        if not isinstance(message, dict) or message.get("role") != "user":
            raise ValueError(f"Entry {entry_id} is not a user message")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"Entry {entry_id} is not a plain-text user message")
        original = cast(str, message.get("content", ""))
        if mode == "replace":
            merged = new_text
        elif original:
            merged = f"{original}\n\n{new_text}".strip()
        else:
            merged = new_text
        new_message = cast(
            AgentMessage,
            {
                "role": "user",
                "content": merged,
                "timestamp": time.time_ns() // 1_000_000,
            },
        )
        new_id = await self._session.append_message(new_message)
        await self._session.move_lane("main", new_id)
        await self._cache_entry(new_id)
        return merged

    @staticmethod
    async def list_sessions(
        directory: str | Path,
        cwd: str | None = None,
        *,
        detailed: bool = True,
    ) -> list[SessionInfo]:
        """列出会话；始终扫描文件（v3 文件同样可列出）。"""
        repo = JsonlSessionRepo(directory)
        options: JsonlSessionListOptions | None = {"cwd": cwd} if cwd is not None else None
        metadata = await repo.list(options)
        if not detailed:
            return [
                SessionInfo(
                    path=item["path"],
                    session_id=item["id"],
                    cwd=item.get("cwd", ""),
                    modified=item.get("modifiedAt", 0) / 1000,
                    parent_session_id=item.get("parentSessionId"),
                )
                for item in metadata
            ]

        infos: list[SessionInfo] = []
        for item in metadata:
            info = SessionInfo(
                path=item["path"],
                session_id=item["id"],
                cwd=item.get("cwd", ""),
                modified=item.get("modifiedAt", 0) / 1000,
                parent_session_id=item.get("parentSessionId"),
            )
            try:
                session = await repo.open(cast(SessionMetadata, item))
                entries = await session.find_entries({"order": "oldestFirst"})
                info.name = await session.get_name()
                info.message_count = sum(
                    1 for entry in entries if cast(dict[str, Any], entry).get("type") == "message"
                )
                user_messages = [
                    _entry_text(entry) for entry in entries if _entry_is_user_message(entry)
                ]
                info.first_message = user_messages[0] if user_messages else ""
                text_parts = [info.session_id, info.cwd]
                if info.name:
                    text_parts.append(info.name)
                text_parts.extend(_entry_text(entry) for entry in entries)
                info.search_text = " ".join(text_parts)
            except Exception:
                # 单个会话读取失败不应让整个列表失败；保留基础元数据。
                info.search_text = f"{info.session_id} {info.cwd}"
            infos.append(info)
        return infos

    async def with_lock(self, fn):
        """文件级锁保护（对齐 SessionManager）。"""

        async def _run() -> Any:
            result = fn(self)
            if inspect.isawaitable(result):
                return await result
            return result

        if self._session_path is None:
            return await _run()
        lock = FileLock(str(self._session_path) + ".lock", timeout=30)
        lock.acquire()
        try:
            return await _run()
        finally:
            lock.release()

    async def close(self) -> None:
        """等待未完成的异步持久化任务，然后释放底层 repo。"""
        if self._scheduled_tasks:
            await asyncio.gather(*self._scheduled_tasks, return_exceptions=True)
        close = getattr(self._repo, "close", None)
        if close is not None:
            await close()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    async def _refresh(self) -> None:
        self._entries = await self._session.find_entries({"order": "oldestFirst"})
        self._by_id = {entry["id"]: entry for entry in self._entries}
        self._leaf_id = await self._session.get_leaf_id()
        self._name = await self._session.get_name()
        self._labels = {}
        for entry in self._entries:
            self._labels[entry["id"]] = await self._session.get_label(entry["id"])
        metadata = await self._session.get_metadata()
        self._session_id = metadata["id"]
        self._created_at = metadata.get("createdAt")

    async def _cache_entry(self, entry_id: str) -> None:
        """写入后增量更新缓存（避免大会话每次全量刷新）。"""
        entry = await self._session.get_entry(entry_id)
        if entry is not None:
            self._entries.append(entry)
            self._by_id[entry["id"]] = entry
            self._labels[entry["id"]] = await self._session.get_label(entry["id"])
        self._leaf_id = await self._session.get_leaf_id()
        self._name = await self._session.get_name()


def _entry_is_user_message(entry: Entry) -> bool:
    if entry.get("type") != "message":
        return False
    message = cast(dict[str, Any], entry).get("message")
    return isinstance(message, dict) and message.get("role") == "user"


def _entry_text(entry: Entry) -> str:
    """提取 message 条目的可搜索文本（对齐 TS SessionInfo.searchText 的轻量版本）。"""
    if entry.get("type") != "message":
        return ""
    message = cast(dict[str, Any], entry).get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


__all__ = [
    "SessionManagerLike",
    "V4SessionManager",
    "create_session_manager",
    "edit_session_message",
    "fork_session_manager",
    "open_session_manager",
    "in_memory_session_manager",
    "list_sessions",
]
