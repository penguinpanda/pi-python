"""JSONL v4 存储层测试（M1 验收）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pi_agent.session.v4.fs import LocalFileSystem
from pi_agent.session.v4.repo import JsonlSessionRepo
from pi_agent.session.v4.storage import JsonlSessionStorage, _publish_file_atomically
from pi_agent.session.v4.types import SessionError

from pi_agent.session.v4.testing.conformance import (
    create_assistant_message,
    create_user_message,
    entry_ids,
    operation_started,
    rejects_with_code,
)


@pytest.fixture
def repo(tmp_path: Path) -> JsonlSessionRepo:
    return JsonlSessionRepo(str(tmp_path / "sessions"))


class TestRoundtrip:
    @pytest.mark.asyncio
    async def test_create_append_reload_keeps_state(self, repo, tmp_path):
        session = await repo.create({"id": "one", "cwd": str(tmp_path)})
        root = await session.append_message(create_user_message("root"))
        await session.create_lane("thread", root)
        child = await session.view("thread").append_message(create_user_message("thread"))
        await session.set_name("Example")
        await session.set_label(root, "checkpoint")
        await session.append_record(operation_started("run", "main", "run"))
        metadata = await session.get_metadata()

        reloaded = await repo.open(metadata)
        assert await reloaded.get_log() == await session.get_log()
        assert await reloaded.get_lanes() == await session.get_lanes()
        assert await reloaded.get_name() == "Example"
        assert await reloaded.get_label(root) == "checkpoint"
        assert await reloaded.get_stats() == await session.get_stats()
        assert entry_ids(await reloaded.find_entries({"order": "oldestFirst"})) == [
            root,
            child,
        ]

    @pytest.mark.asyncio
    async def test_append_after_reload_continues_sequence(self, repo, tmp_path):
        session = await repo.create({"id": "one", "cwd": str(tmp_path)})
        await session.append_message(create_user_message("root"))
        metadata = await session.get_metadata()
        reloaded = await repo.open(metadata)

        entry_id = await reloaded.append_message(create_user_message("second"))
        assert (await reloaded.get_entry(entry_id))["seq"] == 2
        again = await repo.open(metadata)
        assert entry_ids(await again.find_entries({"order": "oldestFirst"})) == [
            (await reloaded.get_log())[0]["entry"]["id"],
            entry_id,
        ]


class TestRepo:
    @pytest.mark.asyncio
    async def test_lists_by_cwd_and_global(self, repo, tmp_path):
        await repo.create({"id": "a", "cwd": str(tmp_path / "p1")})
        await repo.create({"id": "b", "cwd": str(tmp_path / "p1")})
        await repo.create({"id": "c", "cwd": str(tmp_path / "p2")})

        all_ids = {item["id"] for item in await repo.list()}
        assert all_ids == {"a", "b", "c"}
        p1_ids = {item["id"] for item in await repo.list({"cwd": str(tmp_path / "p1")})}
        assert p1_ids == {"a", "b"}

    @pytest.mark.asyncio
    async def test_duplicate_create_rejected(self, repo, tmp_path):
        await repo.create({"id": "dup", "cwd": str(tmp_path)})
        await rejects_with_code(repo.create({"id": "dup", "cwd": str(tmp_path)}), "already_exists")

    @pytest.mark.asyncio
    async def test_invalid_session_id_rejected(self, repo, tmp_path):
        await rejects_with_code(
            repo.create({"id": "bad id!", "cwd": str(tmp_path)}), "invalid_payload"
        )

    @pytest.mark.asyncio
    async def test_open_missing_and_delete_idempotent(self, repo, tmp_path):
        session = await repo.create({"id": "one", "cwd": str(tmp_path)})
        metadata = await session.get_metadata()

        await repo.delete(metadata)
        await rejects_with_code(repo.open(metadata), "not_found")
        await repo.delete(metadata)

    @pytest.mark.asyncio
    async def test_open_id_mismatch_rejected(self, repo, tmp_path):
        session = await repo.create({"id": "one", "cwd": str(tmp_path)})
        metadata = dict(await session.get_metadata())
        metadata["id"] = "other"
        await rejects_with_code(repo.open(metadata), "invalid_entry")

    @pytest.mark.asyncio
    async def test_one_open_operation_per_lane(self, repo, tmp_path):
        session = await repo.create({"id": "one", "cwd": str(tmp_path)})
        await session.append_record(operation_started("run", "main", "run"))
        await rejects_with_code(
            session.append_record(operation_started("second", "main", "run")),
            "storage",
        )


class TestFork:
    @pytest.mark.asyncio
    async def test_forks_branch_and_reloads(self, repo, tmp_path):
        source = await repo.create({"id": "source", "cwd": str(tmp_path)})
        root = await source.append_message(create_user_message("root"))
        shared = await source.append_message(create_assistant_message("shared"))
        await source.create_lane("thread", shared)
        thread_child = await source.view("thread").append_message(create_user_message("thread"))
        main_child = await source.append_message(create_user_message("main"))
        await source.set_name("Source")
        await source.set_label(shared, "copied")
        await source.set_label(thread_child, "excluded")
        await source.append_record(operation_started("run", "main", "run"))

        fork = await repo.fork(
            await source.get_metadata(),
            {
                "scope": "branch",
                "entryId": main_child,
                "position": "at",
                "id": "branch-fork",
            },
        )
        assert entry_ids(await fork.find_entries({"order": "oldestFirst"})) == [
            root,
            shared,
            main_child,
        ]
        assert await fork.get_name() == "Source"
        assert await fork.get_label(shared) == "copied"
        assert await fork.get_label(thread_child) is None
        assert await fork.find_records() == []

        reloaded = await repo.open(await fork.get_metadata())
        assert entry_ids(await reloaded.find_entries({"order": "oldestFirst"})) == [
            root,
            shared,
            main_child,
        ]

    @pytest.mark.asyncio
    async def test_forks_tree_with_lanes_and_facts(self, repo, tmp_path):
        source = await repo.create({"id": "source", "cwd": str(tmp_path)})
        root = await source.append_message(create_user_message("root"))
        await source.create_lane("thread", root)
        main_child = await source.append_message(create_user_message("main"))
        thread_child = await source.view("thread").append_message(create_user_message("thread"))
        await source.set_label(thread_child, "thread-tip")

        fork = await repo.fork(await source.get_metadata(), {"scope": "tree", "id": "tree-fork"})
        assert entry_ids(await fork.find_entries({"order": "oldestFirst"})) == [
            root,
            main_child,
            thread_child,
        ]
        assert await fork.get_lanes() == [
            {"lane": "main", "leafId": main_child},
            {"lane": "thread", "leafId": thread_child},
        ]
        assert await fork.get_label(thread_child) == "thread-tip"
        assert (await fork.get_stats())["messageCount"] == 3


class TestRepair:
    @pytest.mark.asyncio
    async def test_torn_tail_is_repaired(self, repo, tmp_path):
        session = await repo.create({"id": "t1", "cwd": str(tmp_path)})
        root = await session.append_message(create_user_message("root"))
        path = Path((await session.get_metadata())["path"])
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                '{"kind":"entry","id":"torn-entry","type":"custom","customType":"note",'
                '"parentId":null,"seq":2,"timestamp":1'
            )

        storage = await JsonlSessionStorage.load(str(path))
        assert entry_ids(await storage.find_entries()) == [root]
        content = path.read_text(encoding="utf-8")
        assert content.endswith("\n")
        assert '"torn-entry"' not in content

        reopened = await repo.open(await session.get_metadata())
        entry_id = await reopened.append_message(create_user_message("after"))
        assert (await reopened.get_entry(entry_id))["seq"] == 2

    @pytest.mark.asyncio
    async def test_unterminated_tail_is_repaired(self, repo, tmp_path):
        session = await repo.create({"id": "tail", "cwd": str(tmp_path)})
        await session.append_message(create_user_message("root"))
        path = Path((await session.get_metadata())["path"])
        content = path.read_text(encoding="utf-8")
        path.write_text(content.rstrip("\n"), encoding="utf-8")

        storage = await JsonlSessionStorage.load(str(path))
        assert (await storage.get_stats())["messageCount"] == 1
        assert path.read_text(encoding="utf-8").endswith("\n")

    @pytest.mark.asyncio
    async def test_invalid_header_version_rejected(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(
            '{"kind":"header","version":3,"id":"x","createdAt":1,"cwd":"."}\n',
            encoding="utf-8",
        )
        with pytest.raises(SessionError) as excinfo:
            await JsonlSessionStorage.load(str(path))
        assert excinfo.value.code == "invalid_entry"

    @pytest.mark.asyncio
    async def test_unknown_mutation_kind_in_middle_rejected(self, repo, tmp_path):
        session = await repo.create({"id": "bad-mutation", "cwd": str(tmp_path)})
        await session.append_message(create_user_message("root"))
        path = Path((await session.get_metadata())["path"])
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"kind":"bogus","seq":2}\n')

        with pytest.raises(SessionError) as excinfo:
            await JsonlSessionStorage.load(str(path))
        assert excinfo.value.code == "invalid_entry"

    def test_publish_cleans_temp_and_keeps_original(self, tmp_path):
        destination = tmp_path / "dest.jsonl"
        destination.write_text("original\n", encoding="utf-8")

        def _boom(_temp: str) -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _publish_file_atomically(str(destination), _boom)
        assert destination.read_text(encoding="utf-8") == "original\n"
        assert not Path(f"{destination}.tmp").exists()


class _RecordingFileSystem:
    def __init__(self, inner: LocalFileSystem) -> None:
        self._inner = inner
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(name)
            return getattr(self._inner, name)(*args, **kwargs)

        return wrapper


class TestFileSystemAbstraction:
    @pytest.mark.asyncio
    async def test_repo_accepts_options_dict_and_injected_fs(self, tmp_path):
        recording = _RecordingFileSystem(LocalFileSystem())
        repo = JsonlSessionRepo({"sessionsRoot": str(tmp_path / "sessions"), "fs": recording})

        session = await repo.create({"id": "one", "cwd": str(tmp_path)})
        await session.append_message(create_user_message("root"))

        listed = await repo.list()
        assert [item["id"] for item in listed] == ["one"]
        assert "create_dir" in recording.calls
        assert "write_file" in recording.calls

    @pytest.mark.asyncio
    async def test_storage_drain_is_available(self, repo, tmp_path):
        session = await repo.create({"id": "drain", "cwd": str(tmp_path)})
        await session.append_message(create_user_message("root"))
        storage = session._storage
        await storage.drain()
        assert (await storage.get_stats())["messageCount"] == 1
