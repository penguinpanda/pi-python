"""SessionManager 单元测试。"""

import json
import tempfile
from pathlib import Path

import pytest

from pi_coding_agent._session_manager import SessionManager
from pi_ai._types import UserMessage, TextContent


def test_open_expands_tilde(tmp_path, monkeypatch):
    """回归：SessionManager.open 应展开 ~（TUI /resume、/import 传字面 ~）。"""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))

    sessions_dir = home / ".pi" / "agent" / "sessions"
    sessions_dir.mkdir(parents=True)
    session_file = sessions_dir / "abc123.jsonl"
    header = {
        "type": "session",
        "version": 3,
        "id": "abc123",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "cwd": "/workspace",
    }
    session_file.write_text(json.dumps(header) + "\n", encoding="utf-8")

    manager = SessionManager.open("~/.pi/agent/sessions/abc123.jsonl")
    assert manager.session_id == "abc123"


class TestSessionManagerCreate:
    """测试创建新会话。"""

    def test_create_new_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager.create(cwd="/tmp/test", sessions_dir=tmpdir)
            assert mgr.session_id is not None
            assert mgr.cwd == "/tmp/test"
            assert mgr.is_persisted()
            assert len(mgr.get_entries()) == 0

    def test_create_with_custom_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = SessionManager.create(
                cwd="/tmp/test", sessions_dir=tmpdir, session_id="my-session"
            )
            assert mgr.session_id == "my-session"


class TestSessionManagerInMemory:
    """测试内存模式。"""

    def test_in_memory_session(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        assert mgr.session_id is not None
        assert not mgr.is_persisted()
        assert len(mgr.get_entries()) == 0


class TestSessionManagerOpen:
    """测试打开已有会话。"""

    def test_open_existing_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 先创建一个会话
            mgr1 = SessionManager.create(cwd="/tmp/test", sessions_dir=tmpdir)
            msg = UserMessage(role="user", content="hello")
            import asyncio

            asyncio.run(mgr1.append_message(msg))

            # 再打开
            session_path = mgr1.session_path
            mgr2 = SessionManager.open(session_path)
            assert mgr2.session_id == mgr1.session_id
            messages = mgr2.build_context()
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert messages[0]["content"] == "hello"

    def test_open_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            SessionManager.open("/nonexistent/path.jsonl")


class TestSessionManagerAppendMessage:
    """测试消息追加。"""

    def test_append_single_message(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        msg = UserMessage(role="user", content="hello")
        import asyncio

        entry_id = asyncio.run(mgr.append_message(msg))
        assert entry_id is not None
        assert len(mgr.get_entries()) == 1

    def test_append_multiple_messages(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        import asyncio

        msg1 = UserMessage(role="user", content="hello")
        asyncio.run(mgr.append_message(msg1))

        from pi_ai._types import AssistantMessage

        msg2: AssistantMessage = {
            "role": "assistant",
            "content": [TextContent(type="text", text="hi there")],
            "api": "openai-completions",
            "provider": "deepseek",
            "model": "deepseek-chat",
        }
        asyncio.run(mgr.append_message(msg2))

        assert len(mgr.get_entries()) == 2


class TestSessionManagerBuildContext:
    """测试上下文重建。"""

    def test_build_context_empty(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        messages = mgr.build_context()
        assert messages == []

    def test_build_context_single_message(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        msg = UserMessage(role="user", content="hello")
        import asyncio

        asyncio.run(mgr.append_message(msg))

        messages = mgr.build_context()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"

    def test_build_context_multiple_messages(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        import asyncio

        msg1 = UserMessage(role="user", content="hello")
        asyncio.run(mgr.append_message(msg1))

        from pi_ai._types import AssistantMessage

        msg2: AssistantMessage = {
            "role": "assistant",
            "content": [TextContent(type="text", text="hi")],
            "api": "openai-completions",
            "provider": "deepseek",
            "model": "deepseek-chat",
        }
        asyncio.run(mgr.append_message(msg2))

        messages = mgr.build_context()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_build_context_is_ordered(self):
        """验证消息按追加顺序返回。"""
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        import asyncio

        for i in range(5):
            msg = UserMessage(role="user", content=f"msg{i}")
            asyncio.run(mgr.append_message(msg))

        messages = mgr.build_context()
        assert len(messages) == 5
        for i, m in enumerate(messages):
            assert m["content"] == f"msg{i}"


class TestSessionManagerCompaction:
    """压缩条目：append_compaction + build_context。"""

    def test_append_compaction_replaces_old_history(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        import asyncio

        asyncio.run(mgr.append_message(UserMessage(role="user", content="old1")))
        e2 = asyncio.run(
            mgr.append_message(
                {
                    "role": "assistant",
                    "content": [TextContent(type="text", text="old2")],
                    "api": "openai-completions",
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                }
            )
        )
        asyncio.run(mgr.append_compaction("compacted summary", e2, 100))
        asyncio.run(mgr.append_message(UserMessage(role="user", content="new")))

        messages = mgr.build_context()
        assert len(messages) == 3
        assert messages[0]["role"] == "compactionSummary"
        assert messages[0]["summary"] == "compacted summary"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "new"

    def test_append_compaction_returns_entry_id(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        import asyncio

        e1 = asyncio.run(mgr.append_message(UserMessage(role="user", content="old")))
        entry_id = asyncio.run(mgr.append_compaction("summary", e1, 10))
        assert entry_id is not None
        entries = mgr.get_entries()
        assert entries[-1]["type"] == "compaction"
        assert entries[-1]["summary"] == "summary"
        assert entries[-1]["retainedTail"][0]["role"] == "user"

    def test_persisted_compaction_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import asyncio

            mgr1 = SessionManager.create(cwd="/tmp/test", sessions_dir=tmpdir)
            e1 = asyncio.run(mgr1.append_message(UserMessage(role="user", content="old")))
            asyncio.run(mgr1.append_compaction("summary", e1, 10))
            asyncio.run(mgr1.append_message(UserMessage(role="user", content="new")))

            session_path = mgr1.session_path
            mgr2 = SessionManager.open(session_path)
            messages = mgr2.build_context()
            assert len(messages) == 3
            assert messages[0]["role"] == "compactionSummary"
            assert messages[-1]["content"] == "new"

    def test_build_context_without_compaction_unchanged(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        import asyncio

        asyncio.run(mgr.append_message(UserMessage(role="user", content="a")))
        asyncio.run(mgr.append_message(UserMessage(role="user", content="b")))
        messages = mgr.build_context()
        assert [m["content"] for m in messages] == ["a", "b"]


class TestSessionLayout:
    """per-cwd 目录布局与旧平铺文件迁移。"""

    def test_create_uses_per_cwd_layout(self, tmp_path):
        mgr = SessionManager.create(
            cwd="/tmp/proj",
            sessions_dir=tmp_path,
            session_id="sid1",
        )
        assert mgr.session_path is not None
        relative = Path(mgr.session_path).relative_to(tmp_path)
        assert relative.parts[0] == "--tmp-proj--"
        assert relative.name.endswith("_sid1.jsonl")
        assert relative.name.count("_") >= 1

    def test_fork_uses_per_cwd_layout(self, tmp_path):
        mgr = SessionManager.create(cwd="/tmp/proj", sessions_dir=tmp_path)
        import asyncio

        asyncio.run(mgr.append_message(UserMessage(role="user", content="hi")))
        # fork 不传 sessions_dir 时沿用 create 的会话根目录。
        forked = mgr.fork(mgr.get_leaf_id())
        assert forked.session_path is not None
        relative = Path(forked.session_path).relative_to(tmp_path)
        assert relative.parts[0] == "--tmp-proj--"

    def test_list_sessions_scans_per_cwd_subdirs(self, tmp_path):
        import os

        older = SessionManager.create(cwd="/tmp/a", sessions_dir=tmp_path, session_id="older")
        newer = SessionManager.create(cwd="/tmp/b", sessions_dir=tmp_path, session_id="newer")
        os.utime(older.session_path, (1, 1))
        os.utime(newer.session_path, (2, 2))
        infos = SessionManager.list_sessions(tmp_path)
        assert [info.session_id for info in infos] == ["newer", "older"]
        assert infos[0].cwd == "/tmp/b"


class TestEditMessage:
    """/input：v4 仅追加语义下合并/替换历史 user 消息。"""

    def test_merge_appends_and_keeps_old_branch(self, tmp_path):
        import asyncio

        mgr = SessionManager.create(cwd="/tmp/proj", sessions_dir=tmp_path)
        e1 = asyncio.run(mgr.append_message(UserMessage(role="user", content="old")))
        asyncio.run(mgr.append_message(UserMessage(role="user", content="second")))
        asyncio.run(mgr.append_message(UserMessage(role="user", content="third")))

        merged = mgr.edit_message(e1, "new detail")

        assert merged == "old\n\nnew detail"
        leaf = mgr.get_leaf_id()
        assert leaf is not None and leaf != e1
        assert [m["content"] for m in mgr.build_context()] == [
            "old",
            "second",
            "third",
            "old\n\nnew detail",
        ]
        # 旧条目保留在文件中（树仍可见）。
        entries = mgr.get_entries()
        assert len(entries) == 4
        # 重开后 leaf 指向追加的合并消息。
        reopened = SessionManager.open(mgr.session_path)
        assert reopened.get_leaf_id() == leaf
        assert [m["content"] for m in reopened.build_context()][-1] == "old\n\nnew detail"

    def test_replace_mode(self, tmp_path):
        import asyncio

        mgr = SessionManager.create(cwd="/tmp/proj", sessions_dir=tmp_path)
        e1 = asyncio.run(mgr.append_message(UserMessage(role="user", content="old")))
        assert mgr.edit_message(e1, "replacement", mode="replace") == "replacement"
        assert [m["content"] for m in mgr.build_context()] == ["old", "replacement"]

    def test_edit_errors(self, tmp_path):
        import asyncio

        mgr = SessionManager.create(cwd="/tmp/proj", sessions_dir=tmp_path)
        e1 = asyncio.run(mgr.append_message(UserMessage(role="user", content="old")))
        e2 = asyncio.run(
            mgr.append_message(
                {
                    "role": "assistant",
                    "content": [TextContent(type="text", text="hi")],
                    "api": "openai-completions",
                    "provider": "deepseek",
                    "model": "deepseek-chat",
                }
            )
        )
        with pytest.raises(ValueError, match="Entry not found"):
            mgr.edit_message("nope", "x")
        with pytest.raises(ValueError, match="not a user message"):
            mgr.edit_message(e2, "x")
        with pytest.raises(ValueError, match="Unknown edit mode"):
            mgr.edit_message(e1, "x", mode="nope")
