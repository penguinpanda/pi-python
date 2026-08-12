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


@pytest.mark.asyncio
async def test_execute_bash_defers_while_streaming(faux_env, tmp_path):
    """Agent 运行中执行 `!cmd`：bashExecution 延迟到 agent_end 后才进入上下文。"""
    models, _core = faux_env
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    session = _make_session(models, manager, tmp_path)
    try:
        session._agent.state.is_streaming = True
        result = await session.execute_bash("echo deferred-bash")
        assert result.exit_code == 0
        assert session.get_messages() == []
        session._flush_pending_bash_messages()
        assert session.get_messages()[-1]["role"] == "bashExecution"
        assert session.get_messages()[-1]["command"] == "echo deferred-bash"
    finally:
        session._agent.state.is_streaming = False
        await session.dispose()


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

        entries = mgr.get_entries()
        messages = [entry for entry in entries if entry["type"] == "message"]
        assert len(messages) == 2
        assert messages[0]["message"]["role"] == "user"
        assert messages[1]["message"]["role"] == "assistant"

        assert mgr.session_path is not None
        session_file = Path(mgr.session_path)
        assert session_file.exists()
        lines = session_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 3
        assert lines[0].startswith('{"kind": "header", "version": 4')

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
        # 首条为 session header（对齐 TS getHeader），无非协议的 done 汇总事件
        first = json.loads(lines[0])
        assert first["type"] == "session"
        assert first["id"]
        last = json.loads(lines[-1])
        assert last["type"] == "agent_settled"
        event_types = [json.loads(line)["type"] for line in lines]
        assert "done" not in event_types
        assert "message_end" in event_types

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


@pytest.mark.asyncio
async def test_restrict_untrusted_tools_blocks_high_risk(tmp_path):
    """restrictUntrustedTools：未信任项目拦截 bash/write/edit，只读工具放行。"""
    from types import SimpleNamespace

    from pi_agent import Agent, AgentOptions
    from pi_ai import Model

    def _make(restrict: bool) -> AgentSession:
        agent = Agent(AgentOptions(system_prompt="x", model=None))
        return AgentSession(
            agent=agent,
            session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
            cwd=str(tmp_path),
            model=Model(id="m", provider="faux", api="openai-completions"),
            restrict_untrusted_tools=restrict,
        )

    def _ctx(name: str):
        return SimpleNamespace(
            tool_call={"id": "1", "name": name, "arguments": {}},
            args={},
        )

    session = _make(restrict=True)
    session.project_trusted = False
    for name in ("bash", "write", "edit"):
        result = await session._agent.before_tool_call(_ctx(name))
        assert result is not None and result.block is True
    for name in ("read", "grep", "find", "ls"):
        assert await session._agent.before_tool_call(_ctx(name)) is None

    session.project_trusted = True
    assert await session._agent.before_tool_call(_ctx("bash")) is None

    session2 = _make(restrict=False)
    session2.project_trusted = False
    assert await session2._agent.before_tool_call(_ctx("bash")) is None
