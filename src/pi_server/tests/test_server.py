"""pi server 测试：hello/命令分发/快照推送 + 子进程 spawn 验收。"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pi_agent import Agent, AgentOptions
from pi_ai import Model, Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime
from pi_protocol.framing import decode_frame, encode_frame, parse_server_message
from pi_protocol.schemas import (
    PROTOCOL_VERSION,
    ClientHello,
    RequestEnvelope,
    ResponseEnvelope,
    ServerHello,
    ServerHelloError,
)

from pi_server.handler import PiServer


def _runtime():
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    core = faux_provider()
    core.set_responses([faux_assistant_message("server reply")])
    models.add_provider(core.provider)
    return ModelRuntime(models, store)


def _session_factory(runtime):
    def factory(cwd):
        model = runtime.get_model("faux", "faux-1")
        assert model is not None
        agent = Agent(AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=runtime.stream,
        ))
        return AgentSession(
            agent=agent,
            session_manager=SessionManager.in_memory(cwd=cwd),
            cwd=cwd,
            model=model,
            model_runtime=runtime,
        )

    return factory


def _server(runtime=None, token="t"):
    runtime = runtime or _runtime()
    return PiServer(
        model_runtime=runtime,
        session_factory=_session_factory(runtime),
        token=token,
    )


async def _request(server, command: dict, request_id="q1") -> list[dict]:
    envelope = RequestEnvelope(
        type="request", id=request_id, request=command
    )
    return await server.handle_message(envelope)


class TestHello:
    async def test_hello_returns_server_hello(self):
        server = _server()
        messages = await server.handle_message(
            ClientHello(type="hello", version=2, token="t")
        )
        parsed = parse_server_message(messages[0])
        assert isinstance(parsed, ServerHello)
        assert parsed.version == PROTOCOL_VERSION
        assert parsed.connectionId
        assert parsed.snapshot.serverId == server.server_id

    async def test_hello_wrong_version(self):
        server = _server()
        messages = await server.handle_message(
            ClientHello(type="hello", version=1, token="t")
        )
        parsed = parse_server_message(messages[0])
        assert isinstance(parsed, ServerHelloError)
        assert parsed.error.code == "version"

    async def test_hello_wrong_token(self):
        server = _server()
        messages = await server.handle_message(
            ClientHello(type="hello", version=2, token="bad")
        )
        parsed = parse_server_message(messages[0])
        assert isinstance(parsed, ServerHelloError)
        assert parsed.error.code == "auth"


class TestCommands:
    async def test_create_returns_snapshot_event(self, tmp_path):
        server = _server()
        messages = await _request(server, {"command": "create", "cwd": str(tmp_path)})
        response = parse_server_message(messages[0])
        assert isinstance(response, ResponseEnvelope)
        assert response.ok is True
        assert response.result.command == "create"
        session_id = response.result.session.id
        event_types = [decode_frame(encode_frame(m))["event"]["type"] for m in messages[1:]]
        assert "session_snapshot" in event_types

        listed = parse_server_message(messages[0]).result.session
        assert listed.cwd == str(tmp_path)

    async def test_list(self):
        server = _server()
        await _request(server, {"command": "create", "cwd": "/tmp/a"})
        messages = await _request(server, {"command": "list"})
        response = parse_server_message(messages[0])
        assert response.result.command == "list"
        assert len(response.result.sessions) == 1

    async def test_prompt_returns_assistant_transcript(self, tmp_path):
        server = _server()
        created = parse_server_message(
            (await _request(server, {"command": "create", "cwd": str(tmp_path)}))[0]
        )
        session_id = created.result.session.id
        messages = await _request(
            server, {"command": "prompt", "sessionId": session_id, "text": "hi"}
        )
        response = parse_server_message(messages[0])
        assert response.ok is True
        assert response.result.command == "prompt"
        roles = [item.role for item in response.result.session.transcript]
        assert "user" in roles
        assert "assistant" in roles
        texts = [
            block.text
            for item in response.result.session.transcript
            if item.role == "assistant"
            for block in item.content
            if block.type == "text"
        ]
        assert any("server reply" in text for text in texts)

    async def test_attach_detach(self, tmp_path):
        server = _server()
        created = parse_server_message(
            (await _request(server, {"command": "create", "cwd": str(tmp_path)}))[0]
        )
        session_id = created.result.session.id
        detached = parse_server_message(
            (await _request(server, {"command": "detach", "sessionId": session_id}))[0]
        )
        assert detached.result.sessionId == session_id
        attached = parse_server_message(
            (await _request(server, {"command": "attach", "sessionId": session_id}))[0]
        )
        assert attached.result.session.attached is True

    async def test_set_thinking(self, tmp_path):
        store = AuthStorage.in_memory()
        models = Models(credentials=store)
        core = faux_provider(models=[
            Model(
                id="faux-1",
                provider="faux",
                api="openai-completions",
                reasoning=True,
            )
        ])
        core.set_responses([faux_assistant_message("ok")])
        models.add_provider(core.provider)
        runtime = ModelRuntime(models, store)
        server = _server(runtime)
        created = parse_server_message(
            (await _request(server, {"command": "create", "cwd": str(tmp_path)}))[0]
        )
        session_id = created.result.session.id
        messages = await _request(
            server,
            {"command": "set_thinking", "sessionId": session_id, "thinkingLevel": "low"},
        )
        response = parse_server_message(messages[0])
        assert response.ok is True
        assert response.result.session.thinkingLevel == "low"

    async def test_unknown_session_not_found(self):
        server = _server()
        messages = await _request(
            server, {"command": "prompt", "sessionId": "nope", "text": "hi"}
        )
        response = parse_server_message(messages[0])
        assert response.ok is False
        assert response.error.code == "not_found"

    async def test_invalid_line(self):
        server = _server()
        messages = await server.handle_line('{"type":"bogus"}\n')
        parsed = parse_server_message(messages[0])
        assert isinstance(parsed, ServerHelloError)
        assert parsed.error.code == "invalid_request"


@pytest.mark.asyncio
async def test_stdio_server_subprocess(tmp_path):
    """验收：spawn server → create/attach/prompt → 收到 session_snapshot 事件。"""
    script = (
        "import asyncio, sys\n"
        "sys.path.insert(0, r'SRC')\n"
        "from pi_ai import Model, Models\n"
        "from pi_ai.providers.faux import faux_provider, faux_assistant_message\n"
        "from pi_coding_agent.auth_storage import AuthStorage\n"
        "from pi_coding_agent.model_runtime import ModelRuntime\n"
        "from pi_agent import Agent, AgentOptions\n"
        "from pi_coding_agent._session import AgentSession\n"
        "from pi_coding_agent._session_manager import SessionManager\n"
        "from pi_server.handler import PiServer\n"
        "from pi_server.serve import run_stdio_server\n"
        "def factory(cwd):\n"
        "    store = AuthStorage.in_memory()\n"
        "    models = Models(credentials=store)\n"
        "    core = faux_provider()\n"
        "    core.set_responses([faux_assistant_message('subprocess reply')])\n"
        "    models.add_provider(core.provider)\n"
        "    runtime = ModelRuntime(models, store)\n"
        "    model = runtime.get_model('faux', 'faux-1')\n"
        "    agent = Agent(AgentOptions(system_prompt='x', model=model, stream_fn=runtime.stream))\n"
        "    return AgentSession(agent=agent, session_manager=SessionManager.in_memory(cwd=cwd), cwd=cwd, model=model, model_runtime=runtime)\n"
        "async def main():\n"
        "    server = PiServer(session_factory=factory)\n"
        "    return await run_stdio_server(server)\n"
        "sys.exit(asyncio.run(main()))\n"
    ).replace("SRC", str(Path(__file__).resolve().parents[2]))

    env = dict(os.environ)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    assert proc.stdin is not None and proc.stdout is not None

    async def _send(message: dict) -> dict:
        proc.stdin.write(encode_frame(message).encode("utf-8"))
        await proc.stdin.drain()
        request_id = message.get("id")
        while True:
            line = await proc.stdout.readline()
            data = json.loads(line)
            if data.get("type") == "hello":
                return data
            if data.get("type") == "response" and (
                request_id is None or data.get("id") == request_id
            ):
                return data
            # 事件行跳过（快照推送）。

    hello = await _send({"type": "hello", "version": 2, "token": "t"})
    assert hello["type"] == "hello"
    assert hello["version"] == 2

    create = await _send({"type": "request", "id": "1", "request": {"command": "create", "cwd": str(tmp_path)}})
    assert create["ok"] is True
    session_id = create["result"]["session"]["id"]

    attach = await _send({"type": "request", "id": "2", "request": {"command": "attach", "sessionId": session_id}})
    assert attach["ok"] is True

    prompt = await _send({"type": "request", "id": "3", "request": {"command": "prompt", "sessionId": session_id, "text": "hi"}})
    assert prompt["ok"] is True
    roles = [item["role"] for item in prompt["result"]["session"]["transcript"]]
    assert "assistant" in roles

    # 额外读事件流确认 session_snapshot 推送。
    seen_session_snapshot = False
    for _ in range(8):
        line = await proc.stdout.readline()
        if not line:
            break
        event = json.loads(line)
        if event.get("type") == "event" and event.get("event", {}).get("type") == "session_snapshot":
            seen_session_snapshot = True
            break
    assert seen_session_snapshot

    proc.stdin.close()
    await proc.wait()
