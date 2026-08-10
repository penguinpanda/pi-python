"""RpcClient 端到端测试：spawn `pi-python --mode rpc` 子进程，走真实 JSONL 管道。"""

from __future__ import annotations

import os
import sys

import pytest

from pi_coding_agent.rpc.rpc_client import RpcClient


def _rpc_env(tmp_path):
    """隔离 ~/.pi 并让子进程可导入本仓库包。"""
    deps = r"C:\Users\pengu\.codex\visualizations\2026\08\03\019fc699-764e-7f70-9ea5-7ffda55746b8\py-deps"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([part for part in [deps, os.path.abspath("src")] if part])
    env["USERPROFILE"] = str(tmp_path)
    env["HOME"] = str(tmp_path)
    return env


def _make_client(tmp_path) -> RpcClient:
    return RpcClient(
        {
            "command": [
                sys.executable,
                "-m",
                "pi_coding_agent",
                "--mode",
                "rpc",
                "--provider",
                "faux",
                "--model",
                "faux-1",
            ],
            "env": _rpc_env(tmp_path),
        }
    )


@pytest.mark.asyncio
async def test_rpc_roundtrip(tmp_path):
    client = _make_client(tmp_path)
    await client.start()
    try:
        # 状态
        state = await client.get_state()
        assert state["model"]["id"] == "faux-1"
        assert state["isStreaming"] is False
        assert state["messageCount"] == 0
        session_id = state["sessionId"]

        # 可用模型
        models = await client.get_available_models()
        assert any(model["id"] == "faux-1" for model in models)

        # 设置类命令
        await client.set_auto_compaction(False)
        await client.set_steering_mode("all")
        await client.set_follow_up_mode("one-at-a-time")
        state = await client.get_state()
        assert state["steeringMode"] == "all"
        assert state["autoCompactionEnabled"] is False

        # bash
        result = await client.bash("echo client-e2e-ok")
        assert result["exit_code"] == 0
        assert "client-e2e-ok" in result["output"]

        # prompt + 事件
        events = await client.prompt_and_wait("hello from client", timeout=30)
        assert any(event["type"] == "agent_settled" for event in events)

        # 消息与统计
        messages = await client.get_messages()
        assert any(message["role"] == "user" for message in messages)
        stats = await client.get_session_stats()
        assert stats["sessionId"] == session_id
        assert stats["totalMessages"] >= 2

        # 会话条目
        entries = await client.get_entries()
        assert entries["leafId"] is not None
        assert len(entries["entries"]) >= 1

        # new_session 替换会话
        result = await client.new_session()
        assert result["cancelled"] is False
        state = await client.get_state()
        assert state["sessionId"] != session_id
        assert state["messageCount"] == 0

        # export_html 导出独立 HTML 文件。
        export_path = str(tmp_path / "exported.html")
        result = await client.export_html(export_path)
        assert result["path"] == export_path
        assert (tmp_path / "exported.html").exists()
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_rpc_unknown_command_raises(tmp_path):
    client = _make_client(tmp_path)
    await client.start()
    try:
        response = await client.send("does_not_exist")
        assert response["success"] is False
        assert "Unknown command" in response["error"]
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_rpc_cycle_model_and_thinking(tmp_path):
    client = _make_client(tmp_path)
    await client.start()
    try:
        levels = await client.get_available_thinking_levels()
        assert levels == ["off"]

        result = await client.cycle_model()
        assert result is not None
        available = await client.get_available_models()
        available_ids = [model["id"] for model in available]
        assert result["model"]["id"] in available_ids
        assert result["model"]["id"] != "faux-1"

        await client.set_session_name("e2e-session")
        state = await client.get_state()
        assert state["sessionName"] == "e2e-session"
    finally:
        await client.stop()
