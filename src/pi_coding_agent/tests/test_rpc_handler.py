"""RpcMessageHandler / RpcUiContext 单元测试（Faux Provider）。"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from pi_agent import Agent, AgentOptions
from pi_ai import Model, Models
from pi_ai.providers.faux import faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime
from pi_coding_agent.rpc.rpc_mode import RpcMessageHandler, RpcUiContext
from pi_coding_agent.rpc.rpc_mode import run_rpc_mode


def _make_runtime(model_count: int = 3) -> ModelRuntime:
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    models_list = [
        Model(
            id=f"faux-{index}",
            provider="faux",
            api="openai-completions",
            name=f"Faux {index}",
            reasoning=(index % 2 == 0),
        )
        for index in range(1, model_count + 1)
    ]
    core = faux_provider(models=models_list)
    models.add_provider(core.provider)
    runtime = ModelRuntime(models, store)
    return runtime


def _make_session(runtime: ModelRuntime, tmp_path: Path) -> AgentSession:
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(AgentOptions(
        system_prompt="You are a helpful coding assistant.",
        model=model,
        stream_fn=runtime.stream,
    ))
    return AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=model,
        model_runtime=runtime,
    )


def _make_handler(runtime: ModelRuntime, tmp_path: Path, **kwargs) -> RpcMessageHandler:
    session = _make_session(runtime, tmp_path)
    return RpcMessageHandler(session, runtime, **kwargs)


def _rebuild_session(runtime: ModelRuntime, manager, tmp_path: Path) -> AgentSession:
    model = runtime.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(AgentOptions(
        system_prompt="You are a helpful coding assistant.",
        model=model,
        stream_fn=runtime.stream,
    ))
    return AgentSession(
        agent=agent,
        session_manager=manager,
        cwd=str(tmp_path),
        model=model,
        model_runtime=runtime,
    )


class TestBasicCommands:
    async def test_get_state(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1", "type": "get_state"})
        assert response["success"] is True
        assert response["command"] == "get_state"
        state = response["data"]
        assert state["model"]["id"] == "faux-1"
        assert state["thinkingLevel"] == "off"
        assert state["isStreaming"] is False
        assert state["messageCount"] == 0
        assert state["sessionId"]

    async def test_unknown_command(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1", "type": "nope"})
        assert response["success"] is False
        assert "Unknown command" in response["error"]

    async def test_parse_error_missing_type(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1"})
        assert response["success"] is False


class TestModelCommands:
    async def test_set_model(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({
            "id": "1",
            "type": "set_model",
            "provider": "faux",
            "modelId": "faux-2",
        })
        assert response["success"] is True
        assert response["data"]["id"] == "faux-2"
        assert handler.session.model.id == "faux-2"

    async def test_set_model_not_found(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({
            "id": "1",
            "type": "set_model",
            "provider": "faux",
            "modelId": "missing",
        })
        assert response["success"] is False
        assert "Model not found" in response["error"]

    async def test_cycle_model(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1", "type": "cycle_model"})
        assert response["success"] is True
        assert response["data"]["model"]["id"] == "faux-2"
        assert response["data"]["isScoped"] is False

    async def test_get_available_models(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1", "type": "get_available_models"})
        assert response["success"] is True
        ids = [model["id"] for model in response["data"]["models"]]
        assert "faux-1" in ids
        assert "faux-3" in ids


class TestThinkingCommands:
    async def test_set_thinking_level(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        # faux-1 非 reasoning → clamp 到 off，不报错。
        response = await handler.handle_command({
            "id": "1",
            "type": "set_thinking_level",
            "level": "medium",
        })
        assert response["success"] is True

    async def test_get_available_thinking_levels(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({
            "id": "1",
            "type": "get_available_thinking_levels",
        })
        assert response["success"] is True
        assert response["data"]["levels"] == ["off"]

    async def test_cycle_thinking_level_on_reasoning_model(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        await handler.handle_command({
            "id": "1", "type": "set_model", "provider": "faux", "modelId": "faux-2"
        })
        response = await handler.handle_command({"id": "2", "type": "cycle_thinking_level"})
        assert response["success"] is True
        # set_model 将级别设为 medium → 一次 cycle 到 high。
        assert response["data"]["level"] == "high"


class TestQueueAndSettingsCommands:
    async def test_set_steering_mode(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({
            "id": "1", "type": "set_steering_mode", "mode": "all"
        })
        assert response["success"] is True
        assert handler.session.steering_mode == "all"

        response = await handler.handle_command({
            "id": "2", "type": "set_steering_mode", "mode": "bogus"
        })
        assert response["success"] is False

    async def test_set_follow_up_mode(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        await handler.handle_command({"id": "1", "type": "set_follow_up_mode", "mode": "all"})
        assert handler.session.follow_up_mode == "all"

    async def test_auto_compaction_and_retry(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        await handler.handle_command({"id": "1", "type": "set_auto_compaction", "enabled": False})
        assert handler.session.auto_compaction_enabled is False
        await handler.handle_command({"id": "2", "type": "set_auto_retry", "enabled": False})
        await handler.handle_command({"id": "3", "type": "abort_retry"})
        state = await handler.handle_command({"id": "4", "type": "get_state"})
        assert state["data"]["autoCompactionEnabled"] is False


class TestPromptAndQueue:
    async def test_steer_and_follow_up_enqueue(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({
            "id": "1", "type": "steer", "message": "interrupt"
        })
        assert response["success"] is True
        assert handler.session.pending_message_count == 1

        response = await handler.handle_command({
            "id": "2", "type": "follow_up", "message": "then this"
        })
        assert response["success"] is True
        assert handler.session.pending_message_count == 2

    async def test_prompt_background(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({
            "id": "1", "type": "prompt", "message": "hi"
        })
        assert response["success"] is True
        # 等待后台 prompt 完成（faux 无脚本响应 → 立即返回 error）。
        for _ in range(100):
            if not handler._prompt_tasks:
                break
            await asyncio.sleep(0.01)
        messages = handler.session.get_messages()
        roles = [message.get("role") for message in messages]
        assert "user" in roles

    async def test_prompt_requires_message(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1", "type": "prompt"})
        assert response["success"] is False


class TestBashAndStats:
    async def test_bash(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({
            "id": "1", "type": "bash", "command": "echo rpc-ok"
        })
        assert response["success"] is True
        assert "rpc-ok" in response["data"]["output"]
        assert response["data"]["exit_code"] == 0

    async def test_bash_requires_command(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1", "type": "bash"})
        assert response["success"] is False

    async def test_get_session_stats(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        await handler.handle_command({"id": "1", "type": "prompt", "message": "hi"})
        for _ in range(100):
            if not handler._prompt_tasks:
                break
            await asyncio.sleep(0.01)
        response = await handler.handle_command({"id": "2", "type": "get_session_stats"})
        assert response["success"] is True
        stats = response["data"]
        assert stats["totalMessages"] >= 2
        assert stats["userMessages"] >= 1


class TestSessionCommands:
    async def test_get_messages(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1", "type": "get_messages"})
        assert response["success"] is True
        assert response["data"]["messages"] == []

    async def test_get_entries(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        await handler.handle_command({"id": "1", "type": "prompt", "message": "hi"})
        for _ in range(100):
            if not handler._prompt_tasks:
                break
            await asyncio.sleep(0.01)
        response = await handler.handle_command({"id": "2", "type": "get_entries"})
        assert response["success"] is True
        assert response["data"]["leafId"] is not None
        assert len(response["data"]["entries"]) >= 1

    async def test_get_entries_since(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        await handler.handle_command({"id": "1", "type": "prompt", "message": "hi"})
        for _ in range(100):
            if not handler._prompt_tasks:
                break
            await asyncio.sleep(0.01)
        entries = handler.session.session_manager.get_entries()
        first_id = entries[0]["id"]
        response = await handler.handle_command({
            "id": "2", "type": "get_entries", "since": first_id
        })
        assert response["success"] is True
        assert response["data"]["entries"] == entries[1:]

    async def test_get_entries_since_missing(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({
            "id": "1", "type": "get_entries", "since": "nope"
        })
        assert response["success"] is False

    async def test_set_session_name(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        await handler.handle_command({"id": "1", "type": "set_session_name", "name": "  My Task  "})
        assert handler.session.session_name == "My Task"
        response = await handler.handle_command({"id": "2", "type": "set_session_name", "name": " "})
        assert response["success"] is False

    async def test_get_last_assistant_text(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        await handler.handle_command({"id": "1", "type": "prompt", "message": "hi"})
        for _ in range(100):
            if not handler._prompt_tasks:
                break
            await asyncio.sleep(0.01)
        response = await handler.handle_command({"id": "2", "type": "get_last_assistant_text"})
        assert response["success"] is True
        assert isinstance(response["data"]["text"], (str, type(None)))

    async def test_get_commands_empty(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1", "type": "get_commands"})
        assert response["success"] is True
        assert response["data"]["commands"] == []

    async def test_get_commands_lists_extension_prompt_skill(self, tmp_path):
        """回归（E-03）：get_commands 聚合扩展命令 / 提示模板 / 技能三类条目。"""
        from pi_coding_agent.extensions import (
            Extension,
            ExtensionRunner,
            RegisteredCommand,
        )
        from pi_coding_agent.prompt_templates import PromptTemplateLoader
        from pi_coding_agent.skills import SkillLoader

        runtime = _make_runtime()

        extension_path = tmp_path / "ext.py"
        extension = Extension(
            path=str(extension_path),
            resolved_path=str(extension_path),
            commands={
                "hello": RegisteredCommand(
                    name="hello",
                    description="Say hello from extension",
                    handler=lambda ctx, args: "hi",
                    source_info={"source": "local", "path": str(extension_path)},
                )
            },
        )
        runner = ExtensionRunner(
            [extension], cwd=str(tmp_path), model_runtime=runtime
        )

        skills_dir = tmp_path / "skills"
        (skills_dir / "alpha").mkdir(parents=True)
        (skills_dir / "alpha" / "SKILL.md").write_text(
            "---\ndescription: Alpha skill\n---\nBody",
            encoding="utf-8",
        )
        skill_loader = SkillLoader(global_dir=skills_dir)
        skill_loader.load()

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "shorten.md").write_text(
            "---\ndescription: Shorten the following text to one sentence:\n---\nBody",
            encoding="utf-8",
        )
        template_loader = PromptTemplateLoader(global_dir=prompts_dir)
        template_loader.load()

        model = runtime.get_model("faux", "faux-1")
        assert model is not None
        agent = Agent(AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=runtime.stream,
        ))
        session = AgentSession(
            agent=agent,
            session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
            cwd=str(tmp_path),
            model=model,
            model_runtime=runtime,
            skill_loader=skill_loader,
            template_loader=template_loader,
            extension_runner=runner,
        )
        handler = RpcMessageHandler(session, runtime)

        response = await handler.handle_command({"id": "1", "type": "get_commands"})

        assert response["success"] is True
        commands = response["data"]["commands"]
        sources = {command["source"] for command in commands}
        assert sources == {"extension", "prompt", "skill"}
        hello = next(command for command in commands if command["name"] == "hello")
        assert hello["sourceInfo"]["path"] == str(extension_path)
        shorten = next(
            command for command in commands if command["name"] == "shorten"
        )
        assert shorten["sourceInfo"]["path"].endswith("shorten.md")
        alpha = next(
            command for command in commands if command["name"] == "skill:alpha"
        )
        assert alpha["sourceInfo"]["path"].endswith("SKILL.md")

    async def test_export_html(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({
            "id": "1",
            "type": "export_html",
            "outputPath": str(tmp_path / "session.html"),
        })
        assert response["success"] is True
        assert (tmp_path / "session.html").exists()


class TestDagCommands:
    async def _make_dag_handler(self, tmp_path):
        runtime = _make_runtime()
        session = _make_session(runtime, tmp_path)
        handler = RpcMessageHandler(
            session,
            runtime,
            session_rebuilder=lambda manager: _rebuild_session(runtime, manager, tmp_path),
        )
        return handler, session

    async def test_fork(self, tmp_path):
        handler, session = await self._make_dag_handler(tmp_path)
        await handler.handle_command({"id": "1", "type": "prompt", "message": "first"})
        for _ in range(100):
            if not handler._prompt_tasks:
                break
            await asyncio.sleep(0.01)
        entries = session.session_manager.get_entries()
        first_entry = entries[0]

        response = await handler.handle_command({
            "id": "2", "type": "fork", "entryId": first_entry["id"]
        })
        assert response["success"] is True
        assert response["data"]["text"] == "first"
        assert handler.session is not session
        state = await handler.handle_command({"id": "3", "type": "get_state"})
        assert state["data"]["sessionId"] != session.session_id

    async def test_clone(self, tmp_path):
        handler, session = await self._make_dag_handler(tmp_path)
        await handler.handle_command({"id": "1", "type": "prompt", "message": "clone-me"})
        for _ in range(100):
            if not handler._prompt_tasks:
                break
            await asyncio.sleep(0.01)
        response = await handler.handle_command({"id": "2", "type": "clone"})
        assert response["success"] is True
        assert response["data"]["cancelled"] is False

    async def test_get_tree(self, tmp_path):
        handler, _session = await self._make_dag_handler(tmp_path)
        await handler.handle_command({"id": "1", "type": "prompt", "message": "tree me"})
        for _ in range(100):
            if not handler._prompt_tasks:
                break
            await asyncio.sleep(0.01)
        response = await handler.handle_command({"id": "2", "type": "get_tree"})
        assert response["success"] is True
        assert response["data"]["leafId"] is not None
        assert len(response["data"]["tree"]) >= 1

    async def test_get_fork_messages(self, tmp_path):
        handler, session = await self._make_dag_handler(tmp_path)
        await handler.handle_command({"id": "1", "type": "prompt", "message": "fork target"})
        for _ in range(100):
            if not handler._prompt_tasks:
                break
            await asyncio.sleep(0.01)
        response = await handler.handle_command({"id": "2", "type": "get_fork_messages"})
        assert response["success"] is True
        messages = response["data"]["messages"]
        assert any(message["text"] == "fork target" for message in messages)


class TestNewSession:
    async def test_new_session_with_factory(self, tmp_path):
        runtime = _make_runtime()

        def factory():
            return _make_session(runtime, tmp_path)

        handler = _make_handler(runtime, tmp_path, session_factory=factory)
        old_session = handler.session
        response = await handler.handle_command({"id": "1", "type": "new_session"})
        assert response["success"] is True
        assert handler.session is not old_session
        assert len(handler.created_sessions) == 1

    async def test_new_session_without_factory(self, tmp_path):
        runtime = _make_runtime()
        handler = _make_handler(runtime, tmp_path)
        response = await handler.handle_command({"id": "1", "type": "new_session"})
        assert response["success"] is False


class TestRpcUiContext:
    async def test_select(self):
        emitted = []
        ui = RpcUiContext(emit=emitted.append)
        task = asyncio.create_task(ui.select("Pick", ["a", "b"]))
        await asyncio.sleep(0)
        request = emitted[0]
        assert request["method"] == "select"
        assert request["options"] == ["a", "b"]
        ui.resolve_response({"id": request["id"], "value": "b"})
        assert await task == "b"

    async def test_confirm_cancelled(self):
        emitted = []
        ui = RpcUiContext(emit=emitted.append)
        task = asyncio.create_task(ui.confirm("Sure?", "Really?"))
        await asyncio.sleep(0)
        ui.resolve_response({"id": emitted[0]["id"], "cancelled": True})
        assert await task is False

    async def test_input_timeout(self):
        emitted = []
        ui = RpcUiContext(emit=emitted.append)
        task = asyncio.create_task(ui.input("Name", timeout=0.05))
        await asyncio.sleep(0.1)
        assert await task is None

    async def test_notify_fire_and_forget(self):
        emitted = []
        ui = RpcUiContext(emit=emitted.append)
        ui.notify("hello", notify_type="info")
        assert emitted[0]["method"] == "notify"
        assert emitted[0]["message"] == "hello"
        assert not ui.has_pending()

    async def test_editor(self):
        emitted = []
        ui = RpcUiContext(emit=emitted.append)
        task = asyncio.create_task(ui.editor("Edit", prefill="x"))
        await asyncio.sleep(0)
        assert emitted[0]["method"] == "editor"
        ui.resolve_response({"id": emitted[0]["id"], "value": "new"})
        assert await task == "new"


class TestRunRpcMode:
    async def test_in_process_command_loop(self, tmp_path):
        runtime = _make_runtime()
        session = _make_session(runtime, tmp_path)
        stdin = io.BytesIO(
            b'{"id":"1","type":"get_state"}\n'
            b'{"id":"2","type":"get_commands"}\n'
            b'{"id":"3","type":"nope"}\n'
        )
        stdout = io.BytesIO()

        code = await run_rpc_mode(session, runtime, stdin=stdin, stdout=stdout)
        assert code == 0

        lines = [
            json.loads(line)
            for line in stdout.getvalue().decode("utf-8").splitlines()
            if line.strip()
        ]
        responses = [line for line in lines if line["type"] == "response"]
        assert [response["command"] for response in responses] == [
            "get_state",
            "get_commands",
            "nope",
        ]
        assert responses[0]["success"] is True
        assert responses[0]["data"]["model"]["id"] == "faux-1"
        assert responses[2]["success"] is False

    async def test_parse_error_response(self, tmp_path):
        runtime = _make_runtime()
        session = _make_session(runtime, tmp_path)
        stdin = io.BytesIO(b"{not json}\n")
        stdout = io.BytesIO()

        code = await run_rpc_mode(session, runtime, stdin=stdin, stdout=stdout)
        assert code == 0
        line = json.loads(stdout.getvalue().decode("utf-8").strip())
        assert line["command"] == "parse"
        assert line["success"] is False
