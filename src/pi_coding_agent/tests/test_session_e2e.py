"""AgentSession 端到端测试（Faux Provider，零网络依赖）。

覆盖 pi_coding_agent 全链路：
    Agent + 工具注入 + 会话持久化 + print 模式输出

通过 Faux Provider 脚本化响应，验证：
    - 纯文本一轮对话
    - 工具调用循环（read 工具真实执行）
    - print 模式 stdout / 退出码
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions, set_default_stream_fn
from pi_ai import Models
from pi_ai.providers.faux import (
    faux_assistant_message,
    faux_provider,
    faux_tool_call,
)

from pi_coding_agent._print_mode import run_print_mode
from pi_coding_agent._print_mode import run_print_mode_json
from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.tools import create_read_tool


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def faux_env():
    """注册 Faux Provider 的 Models + 全局默认流函数。

    返回 (models, core)：
        - core.set_responses(...) 脚本化响应
        - Agent 不显式传 stream_fn，走全局默认（与 CLI 路径一致）
    """
    core = faux_provider()
    models = Models()
    models.add_provider(core.provider)
    set_default_stream_fn(models.stream)
    yield models, core
    set_default_stream_fn(None)


def _make_session(
    models: Models,
    session_manager: SessionManager,
    cwd: str | Path,
    *,
    tools_override=None,
) -> AgentSession:
    """构建 Agent + AgentSession（走全局默认流函数）。"""
    model = models.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
        )
    )
    return AgentSession(
        agent=agent,
        session_manager=session_manager,
        cwd=str(cwd),
        model=model,
        tools_override=tools_override,
    )


def _final_assistant_text(messages) -> str:
    """提取最后一条 assistant 消息的纯文本。"""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            parts = [b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text"]
            return "".join(parts)
    return ""


# ============================================================================
# 场景 A：纯文本一轮对话 + 会话持久化
# ============================================================================


class TestPlainTextSession:
    async def test_single_prompt_roundtrip(self, faux_env, tmp_path):
        models, core = faux_env
        core.set_responses([faux_assistant_message("Hello from faux!")])

        mgr = SessionManager.create(cwd=str(tmp_path), sessions_dir=str(tmp_path / "sessions"))
        session = _make_session(models, mgr, tmp_path)

        try:
            await session.prompt("Hi there")
            await session.wait_for_idle()

            messages = session.get_messages()
            roles = [m.get("role") for m in messages]
            assert "user" in roles
            assert "assistant" in roles
            assert _final_assistant_text(messages) == "Hello from faux!"
        finally:
            await session.dispose()

        # 会话已持久化到 JSONL（header + user + assistant = 3 行）
        entries = mgr.get_entries()
        assert len(entries) == 2
        assert entries[0]["message"]["role"] == "user"
        assert entries[1]["message"]["role"] == "assistant"

        session_file = Path(tmp_path) / "sessions" / f"{mgr.session_id}.jsonl"
        assert session_file.exists()
        lines = session_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        assert lines[0].startswith('{"type": "session"')

    async def test_multiple_prompts_append(self, faux_env, tmp_path):
        models, core = faux_env
        core.set_responses(
            [
                faux_assistant_message("First reply"),
                faux_assistant_message("Second reply"),
            ]
        )

        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        session = _make_session(models, mgr, tmp_path)

        try:
            await session.prompt("Q1")
            await session.wait_for_idle()
            await session.prompt("Q2")
            await session.wait_for_idle()

            messages = session.get_messages()
            roles = [m.get("role") for m in messages]
            assert roles.count("user") == 2
            assert roles.count("assistant") == 2
        finally:
            await session.dispose()


# ============================================================================
# 场景 B：工具调用循环（read 工具真实执行）
# ============================================================================


class TestToolCallLoop:
    async def test_read_tool_executed(self, faux_env, tmp_path):
        models, core = faux_env
        (tmp_path / "notes.txt").write_text("hello world\n", encoding="utf-8")

        # Turn 1: 调用 read 工具；Turn 2: 文本回复
        core.set_responses(
            [
                faux_assistant_message(
                    [faux_tool_call("read", {"path": "notes.txt"}, tool_call_id="tc-1")],
                    stop_reason="tool_call",
                ),
                faux_assistant_message("Read complete. The file contains: hello world"),
            ]
        )

        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        session = _make_session(
            models,
            mgr,
            tmp_path,
            tools_override=[create_read_tool(str(tmp_path))],
        )

        try:
            await session.prompt("Read notes.txt")
            await session.wait_for_idle()

            messages = session.get_messages()
            roles = [m.get("role") for m in messages]
            assert "toolResult" in roles

            # 工具真实执行：toolResult 包含文件内容
            tool_result = next(m for m in messages if m.get("role") == "toolResult")
            content = tool_result.get("content", [])
            assert len(content) == 1
            assert "hello world" in content[0].get("text", "")

            # 最终 assistant 回复
            assert "Read complete" in _final_assistant_text(messages)

            # 会话历史完整（user + assistant(tool) + toolResult + assistant）
            assert len(messages) == 4
        finally:
            await session.dispose()


# ============================================================================
# 场景 C：print 模式（stdout / 退出码）
# ============================================================================


class TestPrintMode:
    async def test_print_mode_output(self, faux_env, tmp_path, capsys):
        models, core = faux_env
        core.set_responses([faux_assistant_message("Hello from print mode!")])

        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        session = _make_session(models, mgr, tmp_path)

        code = await run_print_mode(session, "hi")

        out = capsys.readouterr().out
        assert "Hello from print mode!" in out
        assert code == 0

    async def test_print_mode_error_exit_code(self, faux_env, tmp_path, capsys):
        """LLM 返回 error stop_reason → 退出码 1。"""
        models, core = faux_env
        core.set_responses(
            [
                faux_assistant_message(
                    [], stop_reason="error", error_message="No more faux responses queued"
                ),
            ]
        )

        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        session = _make_session(models, mgr, tmp_path)

        code = await run_print_mode(session, "hi")

        capsys.readouterr()
        assert code == 1

    async def test_json_print_mode(self, faux_env, tmp_path, capsys):
        models, core = faux_env
        core.set_responses([faux_assistant_message("json reply")])

        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        session = _make_session(models, mgr, tmp_path)

        code = await run_print_mode_json(session, "hi")

        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert code == 0
        assert len(lines) >= 2
        last = json.loads(lines[-1])
        assert last["type"] == "done"
        assert any(message.get("role") == "assistant" for message in last["messages"])

    async def test_json_print_mode_broken_pipe_quiet(self, faux_env, tmp_path, monkeypatch, capsys):
        """下游提前关闭管道（--json | grep -m1）→ 静默退出，不抛 traceback。"""
        from pi_coding_agent import _print_mode as print_mode_module

        models, core = faux_env
        core.set_responses([faux_assistant_message("json reply")])
        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        session = _make_session(models, mgr, tmp_path)

        def broken_pipe(*_args, **_kwargs):
            raise BrokenPipeError(32, "Broken pipe")

        monkeypatch.setattr(print_mode_module, "_emit_json", broken_pipe)

        code = await run_print_mode_json(session, "hi")

        capsys.readouterr()
        assert code == 0

    async def test_print_mode_broken_pipe_quiet(self, faux_env, tmp_path, monkeypatch, capsys):
        """纯文本 print 模式管道被关闭 → 静默退出。"""
        from pi_coding_agent import _print_mode as print_mode_module

        models, core = faux_env
        core.set_responses([faux_assistant_message("Hello from print mode!")])
        mgr = SessionManager.in_memory(cwd=str(tmp_path))
        session = _make_session(models, mgr, tmp_path)

        def broken_pipe(*_args, **_kwargs):
            raise BrokenPipeError(32, "Broken pipe")

        monkeypatch.setattr(print_mode_module, "_emit_text", broken_pipe)

        code = await run_print_mode(session, "hi")

        capsys.readouterr()
        assert code == 0
