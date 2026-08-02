"""SessionManager 单元测试。"""

import tempfile
from pathlib import Path

import pytest

from pi_coding_agent._session_manager import SessionManager
from pi_ai._types import UserMessage, TextContent


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
            session_path = Path(tmpdir) / f"{mgr1.session_id}.jsonl"
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

        e1 = asyncio.run(mgr.append_message(UserMessage(role="user", content="old1")))
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
        assert len(messages) == 2
        assert messages[0]["role"] == "compactionSummary"
        assert messages[0]["summary"] == "compacted summary"
        assert messages[0]["tokens_before"] == 100
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "new"

    def test_append_compaction_returns_entry_id(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        import asyncio

        e1 = asyncio.run(mgr.append_message(UserMessage(role="user", content="old")))
        entry_id = asyncio.run(mgr.append_compaction("summary", e1, 10))
        assert entry_id is not None
        entries = mgr.get_entries()
        assert entries[-1]["type"] == "compaction"
        assert entries[-1]["summary"] == "summary"
        assert entries[-1]["firstKeptEntryId"] == e1

    def test_persisted_compaction_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import asyncio

            mgr1 = SessionManager.create(cwd="/tmp/test", sessions_dir=tmpdir)
            e1 = asyncio.run(mgr1.append_message(UserMessage(role="user", content="old")))
            asyncio.run(mgr1.append_compaction("summary", e1, 10))
            asyncio.run(mgr1.append_message(UserMessage(role="user", content="new")))

            session_path = Path(tmpdir) / f"{mgr1.session_id}.jsonl"
            mgr2 = SessionManager.open(session_path)
            messages = mgr2.build_context()
            assert len(messages) == 2
            assert messages[0]["role"] == "compactionSummary"
            assert messages[1]["content"] == "new"

    def test_build_context_without_compaction_unchanged(self):
        mgr = SessionManager.in_memory(cwd="/tmp/test")
        import asyncio

        asyncio.run(mgr.append_message(UserMessage(role="user", content="a")))
        asyncio.run(mgr.append_message(UserMessage(role="user", content="b")))
        messages = mgr.build_context()
        assert [m["content"] for m in messages] == ["a", "b"]

