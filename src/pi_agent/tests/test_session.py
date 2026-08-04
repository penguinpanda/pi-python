"""Session 系统测试（Phase 3）。"""

from __future__ import annotations

import json

import pytest
from pi_ai._types import TextContent, UserMessage

from pi_agent.session import (
    InMemorySessionStorage,
    JsonlSessionStorage,
    ScanningSessionSearch,
    Session,
    SessionError,
    create_in_memory_session_repo,
    create_in_memory_session_store,
    create_jsonl_session_repo,
    create_jsonl_session_store,
    default_context_entry_transform,
    rebuild_session_search_index,
)


def _user_message(text: str):
    return UserMessage(role="user", content=text)


def _assistant_message(text: str):
    return {
        "role": "assistant",
        "content": [TextContent(type="text", text=text)],
        "api": "test",
        "provider": "test",
        "model": "test",
    }


async def _session_with_messages(*texts: str) -> Session:
    storage = InMemorySessionStorage()
    session = Session(storage)
    for text in texts:
        await session.append_message(_user_message(text))
    return session


# ============================================================================
# 3.1 DAG 会话树
# ============================================================================


class TestSessionDag:
    @pytest.mark.asyncio
    async def test_append_and_branch(self):
        session = await _session_with_messages("a", "b")

        branch = await session.get_branch()
        assert [e["type"] for e in branch] == ["message", "message"]
        assert branch[0]["message"]["content"] == "a"
        assert branch[1]["message"]["content"] == "b"
        assert await session.get_leaf_id() == branch[-1]["id"]

    @pytest.mark.asyncio
    async def test_build_context_messages(self):
        session = await _session_with_messages("hello", "world")
        context = await session.build_context()

        assert context["thinkingLevel"] == "off"
        assert context["model"] is None
        assert [m["content"] for m in context["messages"]] == ["hello", "world"]

    @pytest.mark.asyncio
    async def test_derive_state_from_entries(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        await session.append_message(_user_message("q"))
        await session.append_thinking_level_change("high")
        await session.append_message(_assistant_message("a"))
        await session.append_model_change("test", "model-x")
        await session.append_active_tools_change(["read", "write"])

        context = await session.build_context()
        assert context["thinkingLevel"] == "high"
        assert context["model"] == {"provider": "test", "modelId": "model-x"}
        assert context["activeToolNames"] == ["read", "write"]

    @pytest.mark.asyncio
    async def test_move_to_switches_branch(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        first_id = await session.append_message(_user_message("a"))
        await session.append_message(_user_message("b"))

        # 切回 a，再追加 c → 分支
        await session.move_to(first_id)
        await session.append_message(_user_message("c"))

        contents = [e["message"]["content"] for e in await session.get_branch()]
        assert contents == ["a", "c"]

    @pytest.mark.asyncio
    async def test_move_to_with_summary(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        first_id = await session.append_message(_user_message("a"))
        await session.append_message(_user_message("b"))

        summary_id = await session.move_to(
            first_id,
            summary={"summary": "back to a"},
        )
        assert summary_id is not None

        branch_types = [e["type"] for e in await session.get_branch()]
        assert branch_types == ["message", "branch_summary"]
        context = await session.build_context()
        summary_messages = [m for m in context["messages"] if m.get("role") == "branchSummary"]
        assert summary_messages[0]["summary"] == "back to a"

    @pytest.mark.asyncio
    async def test_label_and_session_name(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        entry_id = await session.append_message(_user_message("a"))

        await session.append_label(entry_id, "first")
        assert await session.get_label(entry_id) == "first"

        # 空 label 清除
        await session.append_label(entry_id, "")
        assert await session.get_label(entry_id) is None

        await session.append_session_name("  my session  ")
        assert await session.get_session_name() == "my session"

    @pytest.mark.asyncio
    async def test_label_missing_target_raises(self):
        session = await _session_with_messages("a")
        with pytest.raises(SessionError, match="not found"):
            await session.append_label("nope", "x")

    @pytest.mark.asyncio
    async def test_get_entries_cursor(self):
        session = await _session_with_messages("a", "b", "c")
        entries = await session.get_entries({"afterEntrySeq": 1, "limit": 1})
        assert [e["message"]["content"] for e in entries] == ["b"]

    @pytest.mark.asyncio
    async def test_session_stats(self):
        session = await _session_with_messages("a")
        stats = await session.get_session_stats()
        assert stats["messageCount"] == 1

    @pytest.mark.asyncio
    async def test_get_branch_from_id(self):
        session = await _session_with_messages("a", "b", "c")
        branch = await session.get_branch()
        target = branch[1]
        partial = await session.get_branch(target["id"])
        assert [e["message"]["content"] for e in partial] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_build_context_from_message_role_derives_model(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        await session.append_message(_assistant_message("answer"))
        context = await session.build_context()
        assert context["model"] == {"provider": "test", "modelId": "test"}


class TestCompactionContext:
    @pytest.mark.asyncio
    async def test_compaction_skips_summarized_entries(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        await session.append_message(_user_message("a"))
        b_id = await session.append_message(_user_message("b"))
        await session.append_message(_user_message("c"))

        await session.append_compaction(
            summary="a+b summarized",
            first_kept_entry_id=b_id,
            tokens_before=100,
        )

        context = await session.build_context()
        roles = [m.get("role") for m in context["messages"]]
        contents = [m.get("content") for m in context["messages"]]
        assert "compactionSummary" in roles
        # firstKeptEntryId=b（跳过 a），保留 b、c
        assert "a" not in contents
        assert "b" in contents and "c" in contents

    @pytest.mark.asyncio
    async def test_compaction_retained_tail(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        await session.append_message(_user_message("a"))
        await session.append_message(_user_message("b"))

        tail = [_user_message("tail-message")]
        await session.append_compaction(
            summary="summary",
            first_kept_entry_id=None,
            tokens_before=10,
            retained_tail=tail,
        )

        context = await session.build_context()
        contents = [m.get("content") for m in context["messages"]]
        assert "tail-message" in contents
        assert "a" not in contents and "b" not in contents

    @pytest.mark.asyncio
    async def test_default_transform_no_compaction_keeps_all(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        await session.append_message(_user_message("a"))
        entries = await session.get_branch()
        transformed = default_context_entry_transform(entries)
        assert len(transformed) == len(entries)


# ============================================================================
# 3.2 存储 / 仓库
# ============================================================================


class TestInMemoryStoreRepo:
    @pytest.mark.asyncio
    async def test_repo_create_open_delete(self):
        repo = create_in_memory_session_repo()
        session = await repo.create()
        await session.append_message(_user_message("hi"))

        metadata = await session.get_metadata()
        reopened = await repo.open(metadata)
        context = await reopened.build_context()
        assert [m["content"] for m in context["messages"]] == ["hi"]

        assert len(await repo.list()) == 1
        await repo.delete(metadata)
        assert len(await repo.list()) == 0

    @pytest.mark.asyncio
    async def test_fork_before_user_message(self):
        repo = create_in_memory_session_repo()
        session = await repo.create()
        await session.append_message(_user_message("a"))
        user_id = await session.append_message(_user_message("b"))
        await session.append_message(_user_message("c"))

        source = await session.get_metadata()
        forked = await repo.fork(source, {"entryId": user_id, "position": "before"})
        contents = [e["message"]["content"] for e in await forked.get_branch()]
        # fork 到 user 消息之前：只复制 a
        assert contents == ["a"]

    @pytest.mark.asyncio
    async def test_fork_at_entry(self):
        repo = create_in_memory_session_repo()
        session = await repo.create()
        await session.append_message(_user_message("a"))
        user_id = await session.append_message(_user_message("b"))
        await session.append_message(_user_message("c"))

        source = await session.get_metadata()
        forked = await repo.fork(source, {"entryId": user_id, "position": "at"})
        contents = [e["message"]["content"] for e in await forked.get_branch()]
        assert contents == ["a", "b"]

    @pytest.mark.asyncio
    async def test_fork_invalid_target(self):
        repo = create_in_memory_session_repo()
        session = await repo.create()
        await session.append_message(_user_message("a"))
        source = await session.get_metadata()
        with pytest.raises(SessionError, match="not found"):
            await repo.fork(source, {"entryId": "missing"})


class TestJsonlStorage:
    @pytest.mark.asyncio
    async def test_create_append_reopen(self, tmp_path):
        store = create_jsonl_session_store(str(tmp_path))
        metadata = await store.create({"cwd": "/tmp/project"})
        session = Session(await store.open(metadata))
        await session.append_message(_user_message("persisted"))

        reopened = Session(await store.open(metadata))
        context = await reopened.build_context()
        assert [m["content"] for m in context["messages"]] == ["persisted"]

        sessions = await store.list({"cwd": "/tmp/project"})
        assert len(sessions) == 1

    @pytest.mark.asyncio
    async def test_storage_open_roundtrip_via_file(self, tmp_path):
        file_path = tmp_path / "s.jsonl"
        storage = await JsonlSessionStorage.create(
            str(file_path),
            cwd="/tmp/x",
            session_id="s1",
        )
        session = Session(storage)
        await session.append_message(_user_message("roundtrip"))

        reopened = await JsonlSessionStorage.open(str(file_path))
        context = await Session(reopened).build_context()
        assert [m["content"] for m in context["messages"]] == ["roundtrip"]

    @pytest.mark.asyncio
    async def test_invalid_session_file_raises(self, tmp_path):
        file_path = tmp_path / "bad.jsonl"
        file_path.write_text("not-json\n", encoding="utf-8")
        with pytest.raises(SessionError, match="invalid"):
            await JsonlSessionStorage.open(str(file_path))

    @pytest.mark.asyncio
    async def test_invalid_entry_line_raises(self, tmp_path):
        file_path = tmp_path / "bad-entry.jsonl"
        file_path.write_text(
            json.dumps({"type": "session", "version": 3, "id": "x", "timestamp": "t", "cwd": "/"})
            + "\n"
            + '{"type":"message"}\n',
            encoding="utf-8",
        )
        with pytest.raises(SessionError, match="invalid"):
            await JsonlSessionStorage.open(str(file_path))

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        repo = create_jsonl_session_repo(str(tmp_path))
        session = await repo.create({"cwd": "/tmp/p"})
        metadata = await session.get_metadata()
        assert len(await repo.list()) == 1
        await repo.delete(metadata)
        assert len(await repo.list()) == 0

    @pytest.mark.asyncio
    async def test_jsonl_fork(self, tmp_path):
        store = create_jsonl_session_store(str(tmp_path))
        metadata = await store.create({"cwd": "/tmp/p"})
        session = Session(await store.open(metadata))
        await session.append_message(_user_message("a"))
        user_id = await session.append_message(_user_message("b"))

        forked_metadata = await store.fork(
            metadata,
            {"entryId": user_id, "position": "before", "cwd": "/tmp/p"},
        )
        forked = Session(await store.open(forked_metadata))
        contents = [e["message"]["content"] for e in await forked.get_branch()]
        assert contents == ["a"]


# ============================================================================
# 3.3 搜索
# ============================================================================


class TestSessionSearch:
    @pytest.mark.asyncio
    async def test_scanning_search_finds_entries(self):
        store = create_in_memory_session_store()
        metadata = await store.create()
        session = Session(await store.open(metadata))
        await session.append_message(_user_message("needle-in-haystack"))
        await session.append_message(_user_message("other"))

        search = ScanningSessionSearch(store)
        hits = await search.search({"text": "needle"})
        assert len(hits) == 1
        assert hits[0]["metadata"]["id"] == metadata["id"]
        assert "needle-in-haystack" in hits[0]["snippet"]

    @pytest.mark.asyncio
    async def test_search_cwd_filter(self, tmp_path):
        store = create_jsonl_session_store(str(tmp_path))
        await store.create({"cwd": "/project-a"})
        await store.create({"cwd": "/project-b"})

        search = ScanningSessionSearch(store)
        hits = await search.search({"text": "anything", "cwd": "/project-a"})
        assert len(hits) == 0  # 无匹配文本，但 cwd 过滤已生效（无抛错）

    @pytest.mark.asyncio
    async def test_rebuild_index(self):
        store = create_in_memory_session_store()
        metadata = await store.create()
        session = Session(await store.open(metadata))
        await session.append_message(_user_message("content"))

        replaced: list[tuple[str, int]] = []

        class FakeIndex:
            async def replace_session(self, meta, entries):
                replaced.append((meta["id"], len(entries)))

            async def upsert_entry(self, meta, entry):
                pass

            async def delete_session(self, meta):
                pass

        await rebuild_session_search_index(store, FakeIndex())
        assert replaced == [(metadata["id"], 1)]
