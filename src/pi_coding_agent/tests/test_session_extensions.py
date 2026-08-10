"""AgentSession × 扩展集成测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pi_agent import (
    AfterToolCallContext,
    Agent,
    AgentOptions,
    AgentToolResult,
    BeforeToolCallContext,
)
from pi_agent._agent_loop import _execute_tool_call
from pi_agent._types import AgentTool
from pi_agent.compaction import CompactionPreparation, CompactionSettings
from pi_ai import Context, Model, Models, TextContent
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_ai.utils._event_stream import EventStream

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.extensions.runner import ExtensionRunner
from pi_coding_agent.extensions.types import Extension, ExtensionAPI
from pi_coding_agent.prompt_templates import PromptTemplate
from pi_coding_agent.skills import Skill


def _make_session(
    tmp_path: Path,
    runner: ExtensionRunner | None,
    store_holder: dict,
    *,
    extension_state: dict | None = None,
    system_prompt_builder=None,
    skill_loader=None,
) -> AgentSession:
    models = Models(credentials=AuthStorage.in_memory())
    core = faux_provider()

    async def factory(context, _options, _state, _model):
        store_holder["messages"] = list(context.messages)
        return faux_assistant_message("ok")

    core.set_responses([factory])
    models.add_provider(core.provider)
    model = models.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=models.stream,
        )
    )
    return AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=model,
        extension_runner=runner,
        extension_state=extension_state,
        system_prompt_builder=system_prompt_builder,
        skill_loader=skill_loader,
    )


def _first_user_text(messages) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            parts = [
                block.get("text", "")
                for block in content or []
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(parts)
    return ""


async def _drain_background(runner: ExtensionRunner) -> None:
    """等待 runner 的后台任务（send_message / append_entry 等）完成。"""
    while runner._background_tasks:
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_input_event_transforms_prompt(tmp_path):
    def transform(event, ctx):
        return {"action": "transform", "text": f"EXT:{event['text']}"}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["input"] = [transform]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    await session.prompt("hello")
    await session.wait_for_idle()
    assert _first_user_text(holder["messages"]) == "EXT:hello"
    await session.dispose()


@pytest.mark.asyncio
async def test_agent_events_forwarded_to_extensions(tmp_path):
    seen: list[str] = []

    def on_message_end(event, ctx):
        seen.append(event.get("type"))
        return None

    def on_agent_settled(event, ctx):
        seen.append(event.get("type"))
        return None

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["message_end"] = [on_message_end]
    extension.handlers["agent_settled"] = [on_agent_settled]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    await session.prompt("hi")
    await session.wait_for_idle()
    assert "message_end" in seen
    assert "agent_settled" in seen
    await session.dispose()


@pytest.mark.asyncio
async def test_bind_session_provider_registration(tmp_path):
    from pi_ai import Models

    models = Models(credentials=AuthStorage.in_memory())
    models.add_provider(faux_provider().provider)
    from pi_coding_agent.model_runtime import ModelRuntime

    runtime = ModelRuntime(models, AuthStorage.in_memory())
    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.providers.append(
        (
            "acme",
            {
                "api_key": "sk-acme",
                "base_url": "https://acme.api/v1",
                "models": [{"id": "acme-1", "api": "openai-completions", "reasoning": False}],
            },
        )
    )
    runner = ExtensionRunner([extension], cwd=str(tmp_path), model_runtime=runtime)

    core = faux_provider()
    core.set_responses([faux_assistant_message("ok")])
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=runtime.get_model("faux", "faux-1"),
            stream_fn=models.stream,
        )
    )
    session = AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=runtime.get_model("faux", "faux-1"),
        model_runtime=runtime,
        extension_runner=runner,
    )
    try:
        model = runtime.get_model("acme", "acme-1")
        assert model is not None
        assert model.base_url == "https://acme.api/v1"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_session_start_event_emitted(tmp_path):
    seen: list[str] = []

    def on_session_start(event, ctx):
        seen.append(event.get("cwd"))

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["session_start"] = [on_session_start]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)
    await _drain_background(runner)
    assert str(tmp_path) in seen
    await session.dispose()


@pytest.mark.asyncio
async def test_before_agent_start_can_override_system_prompt(tmp_path):
    seen: dict = {}

    def on_before_agent_start(event, ctx):
        seen["prompt"] = event.get("prompt")
        seen["system_prompt"] = event.get("system_prompt")
        return {"system_prompt": "EXTENSION OVERRIDE"}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["before_agent_start"] = [on_before_agent_start]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    await session.prompt("hi")
    await session.wait_for_idle()
    assert seen["prompt"] == "hi"
    assert "helpful coding assistant" in seen["system_prompt"]
    # 覆盖只在当轮生效，结束后恢复原系统提示。
    assert "EXTENSION OVERRIDE" not in session._agent.state.system_prompt
    await session.dispose()


@pytest.mark.asyncio
async def test_model_and_thinking_events(tmp_path):
    seen: list[str] = []

    def on_model_select(event, ctx):
        seen.append(f"model:{event.get('modelId')}")

    def on_thinking_select(event, ctx):
        seen.append(f"thinking:{event.get('level')}")

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["model_select"] = [on_model_select]
    extension.handlers["thinking_level_select"] = [on_thinking_select]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    reasoning = Model(
        id="faux-2",
        provider="faux",
        api="faux",
        name="Faux Reasoning",
        max_tokens=4096,
        context_window=128000,
        reasoning=True,
    )
    await session.set_model(reasoning)
    session.set_thinking_level("low")
    await _drain_background(runner)

    assert "model:faux-2" in seen
    assert "thinking:low" in seen
    await session.dispose()


@pytest.mark.asyncio
async def test_tool_call_event_can_block_and_rewrite(tmp_path):
    def on_tool_call(event, ctx):
        command = str((event.get("input") or {}).get("command", ""))
        if "rm -rf" in command:
            return {"block": True, "reason": "Blocked by extension"}
        return {"input": {**event["input"], "command": f"echo {command}"}}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["tool_call"] = [on_tool_call]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    ctx = BeforeToolCallContext(
        assistant_message={"role": "assistant"},
        tool_call={"id": "call_1", "name": "bash", "arguments": {"command": "rm -rf /"}},
        args={"command": "rm -rf /"},
        context=Context(system_prompt="", messages=[], tools=[]),
    )
    blocked = await session._agent.before_tool_call(ctx)
    assert blocked is not None and blocked.block
    assert "Blocked by extension" in blocked.reason

    ctx.args = {"command": "ls"}
    rewritten = await session._agent.before_tool_call(ctx)
    assert rewritten is None
    assert ctx.args["command"] == "echo ls"
    await session.dispose()


@pytest.mark.asyncio
async def test_tool_result_event_can_modify(tmp_path):
    def on_tool_result(event, ctx):
        return {
            "content": [{"type": "text", "text": "patched"}],
            "details": {"from": "extension"},
        }

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["tool_result"] = [on_tool_result]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    ctx = AfterToolCallContext(
        assistant_message={"role": "assistant"},
        tool_call={"id": "call_1", "name": "read", "arguments": {}},
        args={},
        result=AgentToolResult(
            content=[TextContent(type="text", text="original")],
            details={},
        ),
        is_error=False,
        context=Context(system_prompt="", messages=[], tools=[]),
    )
    result = await session._agent.after_tool_call(ctx)
    assert result is not None
    assert result.content[0]["text"] == "patched"
    assert result.details == {"from": "extension"}
    await session.dispose()


@pytest.mark.asyncio
async def test_extension_api_send_message_append_entry_set_label(tmp_path):
    extension = Extension(path="<inline>", resolved_path="<inline>")
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)
    api = ExtensionAPI(extension, runner.runtime, cwd=str(tmp_path))

    api.send_message(
        [{"type": "text", "text": "custom payload"}],
        {"customType": "note", "details": {"level": "warn"}},
    )
    api.append_entry("bookmark", {"note": "here"})
    await _drain_background(runner)

    entries = session._session_manager.get_entries()
    custom_messages = [e for e in entries if e.get("type") == "custom_message"]
    custom_entries = [e for e in entries if e.get("type") == "custom"]
    assert custom_messages and custom_messages[-1]["customType"] == "note"
    assert custom_messages[-1].get("details") == {"level": "warn"}
    assert custom_entries and custom_entries[-1]["customType"] == "bookmark"

    first_entry_id = entries[0]["id"]
    api.set_label(first_entry_id, "mylabel")
    # SessionManager.set_label 使用模块级 _schedule_task（不在 runner 任务集里）。
    await asyncio.sleep(0)
    labels = [e for e in session._session_manager.get_entries() if e.get("type") == "label"]
    assert labels and labels[-1]["targetId"] == first_entry_id
    assert labels[-1]["label"] == "mylabel"
    await session.dispose()


@pytest.mark.asyncio
async def test_context_usage_and_has_ui(tmp_path):
    extension = Extension(path="<inline>", resolved_path="<inline>")
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    ctx = runner.create_context()
    assert ctx.has_ui is False
    runner.mode = "tui"
    assert ctx.has_ui is True

    await session.prompt("hi")
    await session.wait_for_idle()
    usage = runner.create_context().get_context_usage()
    assert usage is not None and "tokens" in usage
    await session.dispose()


def _fake_preparation() -> CompactionPreparation:
    return CompactionPreparation(
        first_kept_entry_id="entry-1",
        messages_to_summarize=[],
        turn_prefix_messages=[],
        retained_tail=[],
        is_split_turn=False,
        tokens_before=100,
        previous_summary=None,
        file_ops={"read": set(), "written": set(), "edited": set()},
        settings=CompactionSettings(),
    )


@pytest.mark.asyncio
async def test_manual_compact_cancelled_by_extension(tmp_path, monkeypatch):
    def on_before_compact(event, ctx):
        return {"cancel": True}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["session_before_compact"] = [on_before_compact]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)
    monkeypatch.setattr(
        "pi_coding_agent._session.prepare_compaction",
        lambda *args, **kwargs: _fake_preparation(),
    )

    result = await session.compact()
    assert result is None
    await session.dispose()


@pytest.mark.asyncio
async def test_manual_compact_extension_provided_result(tmp_path, monkeypatch):
    seen: list[dict] = []

    def on_before_compact(event, ctx):
        return {
            "compaction": {
                "summary": "EXT SUMMARY",
                "firstKeptEntryId": "entry-1",
                "tokensBefore": 99,
            }
        }

    def on_compact(event, ctx):
        seen.append(event)

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["session_before_compact"] = [on_before_compact]
    extension.handlers["session_compact"] = [on_compact]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)
    monkeypatch.setattr(
        "pi_coding_agent._session.prepare_compaction",
        lambda *args, **kwargs: _fake_preparation(),
    )

    result = await session.compact()
    assert result is not None
    assert result.summary == "EXT SUMMARY"
    assert result.tokens_before == 99
    await _drain_background(runner)
    assert seen and seen[-1]["fromExtension"] is True
    assert seen[-1]["reason"] == "manual"
    entries = session._session_manager.get_entries()
    assert any(entry.get("type") == "compaction" for entry in entries)
    await session.dispose()


@pytest.mark.asyncio
async def test_user_bash_result_replacement(tmp_path):
    def on_user_bash(event, ctx):
        return {
            "result": {
                "output": "fake output",
                "exitCode": 7,
                "cancelled": False,
                "truncated": False,
            }
        }

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["user_bash"] = [on_user_bash]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    result = await session.execute_bash("echo hi", exclude_from_context=True)
    assert result.output == "fake output"
    assert result.exit_code == 7
    bash_messages = [
        message for message in session.get_messages() if message.get("role") == "bashExecution"
    ]
    assert bash_messages and bash_messages[-1]["output"] == "fake output"
    assert bash_messages[-1]["excludeFromContext"] is True
    await session.dispose()


@pytest.mark.asyncio
async def test_user_bash_operations_override(tmp_path):
    def on_user_bash(event, ctx):
        def exec_op(command, cwd, options):
            return {
                "output": f"op:{command}",
                "exitCode": 0,
                "cancelled": False,
                "truncated": False,
            }

        return {"operations": {"exec": exec_op}}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["user_bash"] = [on_user_bash]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    result = await session.execute_bash("echo hi")
    assert result.output == "op:echo hi"
    await session.dispose()


def _make_session_with_manager(
    tmp_path: Path,
    runner: ExtensionRunner,
    holder: dict,
    manager: SessionManager,
) -> AgentSession:
    models = Models(credentials=AuthStorage.in_memory())
    core = faux_provider()

    async def factory(context, _options, _state, _model):
        holder["messages"] = list(context.messages)
        return faux_assistant_message("ok")

    core.set_responses([factory])
    models.add_provider(core.provider)
    model = models.get_model("faux", "faux-1")
    assert model is not None
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
            stream_fn=models.stream,
        )
    )
    return AgentSession(
        agent=agent,
        session_manager=manager,
        cwd=str(tmp_path),
        model=model,
        extension_runner=runner,
    )


@pytest.mark.asyncio
async def test_session_before_tree_cancel(tmp_path):
    def on_before_tree(event, ctx):
        return {"cancel": True}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["session_before_tree"] = [on_before_tree]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    first_id = await manager.append_message({"role": "user", "content": "hi"})
    await manager.append_message({"role": "assistant", "content": "ok"})
    session = _make_session_with_manager(tmp_path, runner, holder, manager)

    assert await session.navigate_to(first_id, summarize=False) is False
    assert manager.get_leaf_id() != first_id
    await session.dispose()


@pytest.mark.asyncio
async def test_session_before_tree_custom_summary(tmp_path):
    seen: list[dict] = []

    def on_before_tree(event, ctx):
        return {"summary": {"summary": "EXT TREE SUMMARY", "details": {}}}

    def on_tree(event, ctx):
        seen.append(event)

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["session_before_tree"] = [on_before_tree]
    extension.handlers["session_tree"] = [on_tree]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    manager = SessionManager.in_memory(cwd=str(tmp_path))
    first_id = await manager.append_message({"role": "user", "content": "hi"})
    await manager.append_message({"role": "assistant", "content": "ok"})
    session = _make_session_with_manager(tmp_path, runner, holder, manager)

    assert await session.navigate_to(first_id, summarize=False) is True
    await _drain_background(runner)
    assert seen and seen[-1]["fromExtension"] is True
    assert seen[-1]["newLeafId"] == first_id
    summaries = [entry for entry in manager.get_entries() if entry.get("type") == "branch_summary"]
    assert summaries and summaries[-1]["summary"] == "EXT TREE SUMMARY"
    await session.dispose()


@pytest.mark.asyncio
async def test_resources_discover_skills_and_templates(tmp_path):
    skill_md = tmp_path / "ext-skill" / "SKILL.md"
    skill_md.parent.mkdir()
    skill_md.write_text(
        "---\nname: ext-skill\ndescription: ext skill\n---\nSkill body", encoding="utf-8"
    )
    skill = Skill(
        name="ext-skill",
        description="ext skill",
        file_path=str(skill_md),
        base_dir=str(skill_md.parent),
        source="extension",
    )
    template = PromptTemplate(
        name="ext-tmpl",
        description="ext template",
        argument_hint=None,
        content="Hello $1",
        file_path=str(tmp_path / "ext.md"),
        source="extension",
    )

    def on_discover(event, ctx):
        return {"skills": [skill], "prompts": [template]}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["resources_discover"] = [on_discover]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    await runner.discover_resources()
    assert runner.get_discovered_skills() == [skill]
    assert runner.get_discovered_prompts() == [template]

    expanded = session.expand_prompt("/skill:ext-skill")
    assert '<skill name="ext-skill"' in expanded
    assert "Skill body" in expanded
    assert session.expand_prompt("/ext-tmpl world") == "Hello world"
    await session.dispose()


@pytest.mark.asyncio
async def test_resources_discover_paths_form(tmp_path):
    skill_md = tmp_path / "ext" / "SKILL.md"
    skill_md.parent.mkdir()
    skill_md.write_text(
        "---\nname: path-skill\ndescription: path skill\n---\nPath body",
        encoding="utf-8",
    )
    prompt_md = tmp_path / "ext" / "path.md"
    prompt_md.write_text("---\ndescription: path template\n---\nPath template $1", encoding="utf-8")

    def on_discover(event, ctx):
        return {
            "skillPaths": [str(skill_md)],
            "promptPaths": [str(prompt_md)],
        }

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["resources_discover"] = [on_discover]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    await runner.discover_resources()
    assert [skill.name for skill in runner.get_discovered_skills()] == ["path-skill"]
    assert [template.name for template in runner.get_discovered_prompts()] == ["path"]
    assert session.expand_prompt("/path tmpl") == "Path template tmpl"
    await session.dispose()


@pytest.mark.asyncio
async def test_before_provider_request_stream_options(tmp_path):
    seen: list[dict] = []

    async def base(model, context, options=None):
        seen.append(dict(options or {}))
        return "stream"

    def on_before(event, ctx):
        modified = dict(event["stream_options"])
        modified["custom"] = True
        return {"stream_options": modified}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["before_provider_request"] = [on_before]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    wrapped = session._wrap_stream_fn(base)
    result = await wrapped(
        None,
        Context(system_prompt="", messages=[], tools=[]),
        {"max_tokens": 10},
    )
    assert result == "stream"
    assert seen[-1]["max_tokens"] == 10
    assert seen[-1]["custom"] is True
    await session.dispose()


@pytest.mark.asyncio
async def test_after_provider_response_event(tmp_path):
    seen: list[str] = []

    def on_after(event, ctx):
        seen.append(event.get("type"))

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["after_provider_response"] = [on_after]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    stream = EventStream(
        is_complete=lambda event: event["type"] == "done",
        extract_result=lambda event: event["message"],
    )
    stream.push({"type": "done", "message": {"role": "assistant", "content": []}})

    async def base(model, context, options=None):
        return stream

    wrapped = session._wrap_stream_fn(base)

    async def consume():
        async for _event in await wrapped(
            None,
            Context(system_prompt="", messages=[], tools=[]),
            {},
        ):
            pass

    await consume()
    while session._after_response_tasks:
        await asyncio.sleep(0)
    assert seen == ["after_provider_response"]
    await session.dispose()


@pytest.mark.asyncio
async def test_send_user_message_deliver_as(tmp_path):
    extension = Extension(path="<inline>", resolved_path="<inline>")
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    runner._action_send_user_message(session, "steer me", {"deliverAs": "steer"})
    await _drain_background(runner)
    assert session.pending_message_count > 0
    before = session.pending_message_count
    runner._action_send_user_message(session, "then this", {"deliverAs": "followUp"})
    await _drain_background(runner)
    assert session.pending_message_count > before
    await session.dispose()


@pytest.mark.asyncio
async def test_session_info_changed_event(tmp_path):
    seen: list[dict] = []

    def on_info_changed(event, ctx):
        seen.append(event)

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["session_info_changed"] = [on_info_changed]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    session.set_session_name("newname")
    await _drain_background(runner)
    assert seen and seen[-1]["name"] == "newname"
    assert seen[-1]["previousName"] is None
    await session.dispose()


@pytest.mark.asyncio
async def test_before_provider_headers_merged(tmp_path):
    seen: list[dict] = []

    async def base(model, context, options=None):
        seen.append(dict(options or {}))
        return "stream"

    def on_headers(event, ctx):
        return {"headers": {"X-Ext": "1"}}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["before_provider_headers"] = [on_headers]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    wrapped = session._wrap_stream_fn(base)
    await wrapped(
        None,
        Context(system_prompt="", messages=[], tools=[]),
        {"headers": {"X-Base": "0"}},
    )
    assert seen[-1]["headers"] == {"X-Base": "0", "X-Ext": "1"}
    await session.dispose()


@pytest.mark.asyncio
async def test_bash_tool_session_env_and_spawn_hook(tmp_path):
    from pi_coding_agent.tools import create_all_tools

    tools = create_all_tools(
        str(tmp_path),
        bash_session_env_provider=lambda: {
            "PI_SESSION_ID": "abc",
            "PI_PROVIDER": "faux",
            "PI_MODEL": "m1",
        },
        bash_spawn_hook=lambda ctx: {"CI": "1"},
    )
    bash = next(tool for tool in tools if tool.name == "bash")
    result = await bash.execute(
        "call_1",
        {
            "command": (
                "python -c \"import os;print(os.environ.get('PI_SESSION_ID','')"
                "+'|'+os.environ.get('CI',''))\""
            )
        },
        signal=None,
        on_update=None,
        context=None,
    )
    text = "".join(block.get("text", "") for block in result.content if isinstance(block, dict))
    assert "abc|1" in text


@pytest.mark.asyncio
async def test_skill_expansion_emits_error_on_invalid_yaml(tmp_path):
    from pi_coding_agent.skills import SkillLoader

    skills_dir = tmp_path / "skills"
    (skills_dir / "bad").mkdir(parents=True)
    skill_file = skills_dir / "bad" / "SKILL.md"
    skill_file.write_text("---\ndescription: Bad\n---\n\nBody", encoding="utf-8")
    loader = SkillLoader(global_dir=skills_dir)
    loader.load()

    errors: list = []
    extension = Extension(path="<inline>", resolved_path="<inline>")
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    runner.on_error(errors.append)
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder, skill_loader=loader)
    try:
        skill_file.write_text("---\ndescription: [unclosed\n---\n\nBody", encoding="utf-8")
        text = session._expand_skill_command("/skill:bad arg")
        assert text == "/skill:bad arg"
        assert errors and errors[-1].event == "skill_expansion"
        assert errors[-1].extension_path == str(skill_file)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_tool_execution_update_event(tmp_path):
    async def tool_execute(tool_call_id, params, signal=None, on_update=None):
        if on_update is not None:
            on_update(
                AgentToolResult(
                    content=[TextContent(type="text", text="partial")],
                    details={},
                )
            )
        return AgentToolResult(
            content=[TextContent(type="text", text="done")],
            details={},
        )

    tool = AgentTool(
        name="demo",
        label="demo",
        description="demo tool",
        input_schema={"type": "object", "properties": {}},
        execute=tool_execute,
    )
    from pi_agent._agent_loop import _PreparedToolCall

    prepared = _PreparedToolCall(
        tc={"id": "c1", "name": "demo", "arguments": {}},
        tool=tool,
        args={},
        assistant_message={"role": "assistant"},
        context=Context(system_prompt="", messages=[], tools=[]),
    )
    events: list[dict] = []

    async def emit(event):
        events.append(event)

    await _execute_tool_call(prepared, emit, None)
    types = [event.get("type") for event in events]
    assert "tool_execution_start" in types
    assert "tool_execution_update" in types
    assert types.index("tool_execution_update") > types.index("tool_execution_start")


@pytest.mark.asyncio
async def test_context_event_can_filter_messages(tmp_path):
    seen: dict = {}

    async def base(model, context, options=None):
        seen["messages"] = list(context.messages or [])
        return "stream"

    def on_context(event, ctx):
        return {"messages": [{"role": "user", "content": "filtered"}]}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["context"] = [on_context]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    wrapped = session._wrap_stream_fn(base)
    ctx = Context(
        system_prompt="",
        messages=[{"role": "user", "content": "original"}],
        tools=[],
    )
    await wrapped(None, ctx, {})
    assert seen["messages"] == [{"role": "user", "content": "filtered"}]
    await session.dispose()


@pytest.mark.asyncio
async def test_session_before_switch_and_fork_cancel(tmp_path):
    called = {"new": 0, "fork": 0, "switch": 0}

    def on_before_switch(event, ctx):
        if event.get("position") == "before":
            return {"cancel": True}
        return None

    def on_before_fork(event, ctx):
        return {"cancel": True}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["session_before_switch"] = [on_before_switch]
    extension.handlers["session_before_fork"] = [on_before_fork]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))

    def _bump(key):
        def handler(*_args):
            called[key] += 1
            return "ok"

        return handler

    runner.bind(
        command_handlers={
            "new_session": _bump("new"),
            "fork": _bump("fork"),
            "switch_session": _bump("switch"),
        }
    )
    command_ctx = runner.create_command_context()

    assert await command_ctx.new_session() is None
    assert await command_ctx.fork("e1") is None
    # switch_session 的 position 是 "at"：不触发 cancel。
    assert await command_ctx.switch_session("path") == "ok"
    assert called == {"new": 0, "fork": 0, "switch": 1}


@pytest.mark.asyncio
async def test_model_registry_find_and_complete(tmp_path):
    from pi_ai import Models

    from pi_coding_agent.model_runtime import ModelRuntime

    models = Models(credentials=AuthStorage.in_memory())
    core = faux_provider()
    core.set_responses([faux_assistant_message("summarized")])
    models.add_provider(core.provider)
    runtime = ModelRuntime(models, AuthStorage.in_memory())

    extension = Extension(path="<inline>", resolved_path="<inline>")
    runner = ExtensionRunner([extension], cwd=str(tmp_path), model_runtime=runtime)
    agent = Agent(
        AgentOptions(
            system_prompt="x",
            model=runtime.get_model("faux", "faux-1"),
            stream_fn=models.stream,
        )
    )
    session = AgentSession(
        agent=agent,
        session_manager=SessionManager.in_memory(cwd=str(tmp_path)),
        cwd=str(tmp_path),
        model=runtime.get_model("faux", "faux-1"),
        model_runtime=runtime,
        extension_runner=runner,
    )
    try:
        ctx = runner.create_context()
        model = ctx.model_registry.find("faux", "faux-1")
        assert model is not None
        response = await ctx.model_registry.complete(
            model,
            Context(
                system_prompt="summarizer",
                messages=[{"role": "user", "content": "text"}],
                tools=[],
            ),
            {},
        )
        blocks = response.get("content") or []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        assert "summarized" in text
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_emit_input_streaming_behavior(tmp_path):
    seen: list = []

    def on_input(event, ctx):
        seen.append(event.get("streamingBehavior"))
        return {"action": "continue"}

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.handlers["input"] = [on_input]
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)

    await runner.emit_input("idle text")
    await runner.emit_input("steer text", streaming_behavior="steer")
    assert seen == [None, "steer"]
    await session.dispose()


@pytest.mark.asyncio
async def test_custom_message_enters_context(tmp_path):
    from pi_coding_agent.messages import convert_to_llm

    manager = SessionManager.in_memory(cwd=str(tmp_path))
    await manager.append_custom_message_entry(
        "note",
        [{"type": "text", "text": "hello custom"}],
        display=True,
    )
    messages = manager.build_context()
    custom = [message for message in messages if message.get("role") == "custom"]
    assert custom and custom[-1]["customType"] == "note"
    llm_messages = convert_to_llm(messages)
    assert any(message.get("role") == "user" for message in llm_messages)


@pytest.mark.asyncio
async def test_extension_tools_merged_and_normalized(tmp_path):
    from pi_agent import AgentToolResult
    from pi_coding_agent.extensions.types import ToolDefinition

    def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        return {
            "content": [{"type": "text", "text": f"hi {params['name']}"}],
            "details": {"ok": True},
            "terminate": True,
        }

    extension = Extension(path="<inline>", resolved_path="<inline>")
    extension.tools["greet"] = ToolDefinition(
        name="greet",
        label="Greet",
        description="Greet someone",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        execute=execute,
    )
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)
    try:
        tools = {tool.name: tool for tool in session._agent.state.tools}
        assert "greet" in tools
        result = await tools["greet"].execute("c1", {"name": "x"})
        assert isinstance(result, AgentToolResult)
        assert result.content[0]["text"] == "hi x"
        assert result.details == {"ok": True}
        assert result.terminate is True
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_extension_tool_prompt_metadata_preserved(tmp_path):
    from pi_coding_agent.extensions.types import ToolDefinition

    extension = Extension(path="<inline>", resolved_path="<inline>")
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    api = ExtensionAPI(extension, runner.runtime, cwd=str(tmp_path))
    api.register_tool(
        ToolDefinition(
            name="custom",
            label="Custom",
            description="Custom tool",
            prompt_snippet="Custom snippet",
            prompt_guidelines=["Use custom for custom things."],
            parameters={"type": "object", "properties": {}},
            execute=lambda *a, **k: {"content": [{"type": "text", "text": "ok"}]},
        )
    )
    holder: dict = {}
    extension_state: dict = {"runner": runner, "active_tools": []}
    session = _make_session(tmp_path, runner, holder, extension_state=extension_state)
    try:
        tool = next(tool for tool in session._agent.state.tools if tool.name == "custom")
        assert tool.prompt_snippet == "Custom snippet"
        assert tool.prompt_guidelines == ["Use custom for custom things."]
        assert session.extension_state["active_tools"] == session._agent.state.tools
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_set_active_tools_rebuilds_system_prompt(tmp_path):
    from pi_coding_agent.system_prompt import (
        BuildSystemPromptOptions,
        build_system_prompt,
        tool_prompt_guidelines_for,
        tool_snippets_for,
    )

    extension = Extension(path="<inline>", resolved_path="<inline>")
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    api = ExtensionAPI(extension, runner.runtime, cwd=str(tmp_path))
    api.register_tool(
        {
            "name": "custom",
            "label": "Custom",
            "description": "Custom tool",
            "prompt_snippet": "Custom snippet",
            "prompt_guidelines": ["Use custom for custom things."],
            "parameters": {"type": "object", "properties": {}},
            "execute": lambda *a, **k: {"content": [{"type": "text", "text": "ok"}]},
        }
    )
    holder: dict = {}
    extension_state: dict = {"runner": runner, "active_tools": []}

    def builder() -> str:
        active_tools = extension_state.get("active_tools") or []
        return build_system_prompt(
            BuildSystemPromptOptions(
                cwd=str(tmp_path),
                selected_tools=[tool.name for tool in active_tools],
                tool_snippets=tool_snippets_for(active_tools),
                prompt_guidelines=tool_prompt_guidelines_for(active_tools),
            )
        )

    session = _make_session(
        tmp_path,
        runner,
        holder,
        extension_state=extension_state,
        system_prompt_builder=builder,
    )
    try:
        assert "Use custom for custom things." in session._agent.state.system_prompt
        runner.runtime.get_action("set_active_tools")(["read"])
        assert (
            "Use read to examine files instead of cat or sed." in session._agent.state.system_prompt
        )
        assert "Use custom for custom things." not in session._agent.state.system_prompt
        assert [tool.name for tool in session.extension_state["active_tools"]] == ["read"]
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_get_all_tools_returns_prompt_guidelines_and_source_info(tmp_path):
    extension = Extension(path="<inline>", resolved_path="<inline>")
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    api = ExtensionAPI(extension, runner.runtime, cwd=str(tmp_path))
    api.register_tool(
        {
            "name": "custom",
            "label": "Custom",
            "description": "Custom tool",
            "prompt_guidelines": ["Use custom for custom things."],
            "parameters": {"type": "object", "properties": {}},
            "execute": lambda *a, **k: {"content": [{"type": "text", "text": "ok"}]},
        }
    )
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)
    try:
        tools = {tool["name"]: tool for tool in runner.runtime.get_action("get_all_tools")()}
        assert tools["custom"]["prompt_guidelines"] == ["Use custom for custom things."]
        assert tools["custom"]["source_info"] == {
            "source": extension.source,
            "path": extension.path,
        }
        assert "prompt_guidelines" in tools["read"]
        assert tools["read"]["source_info"] == {}
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_extension_context_session_manager_and_options(tmp_path):
    extension = Extension(path="<inline>", resolved_path="<inline>")
    runner = ExtensionRunner([extension], cwd=str(tmp_path))
    holder: dict = {}
    session = _make_session(tmp_path, runner, holder)
    try:
        ctx = runner.create_context()
        assert ctx.session_manager is session.session_manager
        assert ctx.signal is None
        assert ctx.scoped_models == []
        assert ctx.is_project_trusted() is False
        session.project_trusted = True
        assert ctx.is_project_trusted() is True
        options = ctx.get_system_prompt_options()
        assert options["cwd"] == str(tmp_path)
        assert "coding assistant" in options["systemPrompt"]
        assert options["thinkingLevel"] == "off"
    finally:
        await session.dispose()
