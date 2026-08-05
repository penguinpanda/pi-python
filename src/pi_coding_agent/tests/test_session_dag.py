"""SessionManager DAG 升级测试（fork / tree / 导航 / 迁移 / 锁）。"""

from __future__ import annotations

import asyncio
import json

from pi_ai._types import AssistantMessage, TextContent, UserMessage

from pi_coding_agent._session_manager import SessionManager


def _user(text: str) -> UserMessage:
    return UserMessage(role="user", content=text)


def _assistant(text: str) -> AssistantMessage:
    return {
        "role": "assistant",
        "content": [TextContent(type="text", text=text)],
        "api": "openai-completions",
        "provider": "faux",
        "model": "faux-1",
    }


class TestFork:
    def test_fork_creates_new_session(self, tmp_path):
        mgr = SessionManager.create(cwd="/tmp/proj", sessions_dir=str(tmp_path))
        e1 = asyncio.run(mgr.append_message(_user("one")))
        e2 = asyncio.run(mgr.append_message(_user("two")))
        asyncio.run(mgr.append_message(_user("three")))

        forked = mgr.fork(e2)
        assert forked.session_id != mgr.session_id
        assert forked.is_persisted()
        assert forked.get_leaf_id() == e2
        branch = forked.get_branch()
        assert [entry["id"] for entry in branch] == [e1, e2]

        # fork 会话可独立打开。
        reopened = SessionManager.open(forked.session_path)
        assert reopened.get_leaf_id() == e2
        assert [m.get("content") for m in reopened.build_context()] == ["one", "two"]

    def test_fork_missing_entry_raises(self, tmp_path):
        mgr = SessionManager.in_memory(cwd="/tmp")
        import pytest

        with pytest.raises(ValueError, match="Entry not found"):
            mgr.fork("nope")


class TestTree:
    def test_get_tree_after_fork_and_navigation(self, tmp_path):
        mgr = SessionManager.in_memory(cwd="/tmp")
        e1 = asyncio.run(mgr.append_message(_user("one")))
        e2 = asyncio.run(mgr.append_message(_user("two")))

        # 从 e1 分叉并导航回 e1，再追加新分支。
        asyncio.run(mgr.move_to(e1))
        e3 = asyncio.run(mgr.append_message(_user("three")))

        roots = mgr.get_tree()
        assert len(roots) == 1
        root = roots[0]
        assert root.id == e1
        assert {child.id for child in root.children} == {e2, e3}
        # 子节点按时间戳排序。
        assert [child.id for child in root.children] == [e2, e3]

    def test_get_branch(self, tmp_path):
        mgr = SessionManager.in_memory(cwd="/tmp")
        e1 = asyncio.run(mgr.append_message(_user("one")))
        e2 = asyncio.run(mgr.append_message(_user("two")))
        branch = mgr.get_branch(e2)
        assert [entry["id"] for entry in branch] == [e1, e2]
        assert mgr.get_entry(e1)["message"]["content"] == "one"


class TestMoveToAndBranchSummary:
    def test_move_to_sets_leaf_and_builds_context(self, tmp_path):
        mgr = SessionManager.in_memory(cwd="/tmp")
        e1 = asyncio.run(mgr.append_message(_user("one")))
        asyncio.run(mgr.append_message(_user("two")))
        asyncio.run(mgr.append_message(_user("three")))

        asyncio.run(mgr.move_to(e1))
        assert mgr.get_leaf_id() == e1
        # 上下文从 e1 重建，忽略 e2/e3。
        assert [m.get("content") for m in mgr.build_context()] == ["one"]

        # 追加新分支。
        e4 = asyncio.run(mgr.append_message(_user("four")))
        assert mgr.get_leaf_id() == e4
        assert [m.get("content") for m in mgr.build_context()] == ["one", "four"]

    def test_move_to_with_branch_summary(self, tmp_path):
        mgr = SessionManager.in_memory(cwd="/tmp")
        e1 = asyncio.run(mgr.append_message(_user("one")))
        asyncio.run(mgr.append_message(_user("two")))

        result = asyncio.run(
            mgr.move_to(
                e1,
                {"summary": "branch summary text", "details": {"readFiles": []}},
            )
        )
        assert result is not None
        entries = mgr.get_entries()
        assert entries[-1]["type"] == "branch_summary"
        assert entries[-1]["summary"] == "branch summary text"
        # leaf 指向 branch_summary 条目。
        assert mgr.get_leaf_id() == entries[-1]["id"]
        # 上下文仍为 e1 的消息（branch_summary 不进入上下文）。
        assert [m.get("content") for m in mgr.build_context()] == ["one"]


class TestExtendedEntries:
    def test_label_session_info_custom(self, tmp_path):
        mgr = SessionManager.in_memory(cwd="/tmp")
        e1 = asyncio.run(mgr.append_message(_user("one")))
        asyncio.run(mgr.append_label(e1, "important"))
        asyncio.run(mgr.append_session_info("My Session"))
        asyncio.run(mgr.append_custom_entry("state", {"key": "value"}))

        assert mgr.session_name == "My Session"
        entries = mgr.get_entries()
        assert entries[-3]["type"] == "label"
        assert entries[-3]["targetId"] == e1
        assert entries[-3]["label"] == "important"
        assert entries[-2]["type"] == "session_info"
        assert entries[-1]["type"] == "custom"
        assert entries[-1]["data"] == {"key": "value"}
        tree = mgr.get_tree()
        labeled = next(node for node in tree if node.id == e1)
        assert labeled.label == "important"
        assert labeled.label_timestamp is not None

    def test_persisted_leaf_restored(self, tmp_path):
        mgr = SessionManager.create(cwd="/tmp", sessions_dir=str(tmp_path))
        e1 = asyncio.run(mgr.append_message(_user("one")))
        asyncio.run(mgr.append_message(_user("two")))
        asyncio.run(mgr.move_to(e1))

        reopened = SessionManager.open(mgr.session_path)
        assert reopened.get_leaf_id() == e1
        assert [m.get("content") for m in reopened.build_context()] == ["one"]

    def test_persisted_session_name_restored(self, tmp_path):
        mgr = SessionManager.create(cwd="/tmp", sessions_dir=str(tmp_path))
        asyncio.run(mgr.append_message(_user("one")))
        asyncio.run(mgr.append_session_info("named"))
        reopened = SessionManager.open(mgr.session_path)
        assert reopened.session_name == "named"


class TestListAndLock:
    def test_list_sessions_sorted(self, tmp_path):
        SessionManager.create(cwd="/tmp/a", sessions_dir=str(tmp_path), session_id="older")
        SessionManager.create(cwd="/tmp/b", sessions_dir=str(tmp_path), session_id="newer")
        infos = SessionManager.list_sessions(tmp_path)
        assert [info.session_id for info in infos] == ["newer", "older"]
        assert infos[0].cwd == "/tmp/b"

    def test_with_lock(self, tmp_path):
        mgr = SessionManager.create(cwd="/tmp", sessions_dir=str(tmp_path))
        result = asyncio.run(mgr.with_lock(lambda m: m.session_id))
        assert result == mgr.session_id


class TestMigration:
    def test_v1_to_v3(self, tmp_path):
        path = tmp_path / "old.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "session",
                            "version": 1,
                            "id": "s1",
                            "timestamp": "t",
                            "cwd": "/tmp",
                        }
                    ),
                    json.dumps({"type": "message", "message": {"role": "user", "content": "hi"}}),
                    json.dumps(
                        {
                            "type": "compaction",
                            "summary": "sum",
                            "firstKeptEntryIndex": 1,
                            "tokensBefore": 10,
                        }
                    ),
                    json.dumps(
                        {"type": "message", "message": {"role": "hookMessage", "content": "x"}}
                    ),
                ]
            ),
            encoding="utf-8",
        )
        mgr = SessionManager.open(path)
        entries = mgr.get_entries()
        assert all(entry.get("id") for entry in entries)
        assert entries[0]["parentId"] is None
        assert entries[1]["parentId"] == entries[0]["id"]
        # firstKeptEntryIndex → firstKeptEntryId。
        compaction = entries[1]
        assert compaction["type"] == "compaction"
        assert compaction["firstKeptEntryId"] == entries[0]["id"]
        assert "firstKeptEntryIndex" not in compaction
        # hookMessage → custom。
        assert entries[2]["message"]["role"] == "custom"
        # 文件已重写为 v3。
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert header["version"] == 3

    def test_v2_to_v3(self, tmp_path):
        path = tmp_path / "v2.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "session",
                            "version": 2,
                            "id": "s2",
                            "timestamp": "t",
                            "cwd": "/tmp",
                        }
                    ),
                    json.dumps(
                        {
                            "type": "message",
                            "id": "a",
                            "parentId": None,
                            "timestamp": "t",
                            "message": {"role": "hookMessage", "content": "y"},
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )
        mgr = SessionManager.open(path)
        assert mgr.get_entries()[0]["message"]["role"] == "custom"
        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert header["version"] == 3
