"""Phase 5.2 Harness 集成 / 压缩切割点 / 存储持久化测试。"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai._types import Model, TextContent
from pi_ai.providers.faux import faux_assistant_message, faux_provider, faux_tool_call

from pi_agent import (
    AgentHarness,
    AgentHarnessOptions,
    AgentHarnessResources,
    PythonExecutionEnv,
    Session,
    Skill,
    AgentTool,
    AgentToolResult,
    create_read_tool,
)
from pi_agent.compaction import DEFAULT_COMPACTION_SETTINGS, prepare_compaction
from pi_agent.session import InMemorySessionStorage, create_jsonl_session_store


def _make_model() -> Model:
    return Model(
        id="test-model",
        provider="test",
        api="openai-completions",
        name="Test",
        max_tokens=4096,
        context_window=128000,
    )


def _tool_response(tool_name: str, args: dict, tool_call_id: str = "tc-1"):
    return faux_assistant_message(
        [faux_tool_call(tool_name, args, tool_call_id=tool_call_id)],
        stop_reason="tool_call",
    )


def _assistant_with_usage(text: str):
    return {
        "role": "assistant",
        "content": [TextContent(type="text", text=text)],
        "api": "test",
        "provider": "test",
        "model": "test",
        "usage": {
            "input": 100,
            "output": 20,
            "cache_read": 0,
            "cache_write": 0,
            "total_tokens": 120,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        },
    }


class TestHarnessToolsIntegration:
    @pytest.mark.asyncio
    async def test_harness_runs_read_tool(self, tmp_path):
        (tmp_path / "data.txt").write_bytes(b"harness tool content")
        env = PythonExecutionEnv(str(tmp_path))
        tool = create_read_tool()
        core = faux_provider()
        core.set_responses(
            [
                _tool_response("read", {"path": "data.txt"}),
                faux_assistant_message("done reading"),
            ]
        )
        harness = AgentHarness(
            AgentHarnessOptions(
                model=_make_model(),
                tools=[tool],
                tool_context=type("ToolContext", (), {"env": env})(),
                stream_fn=core.stream,
            )
        )

        result = await harness.prompt("read the file")

        assert result["stop_reason"] == "stop"
        session_text = "\n".join(
            "".join(
                block.get("text", "")
                for block in m["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            )
            for m in harness._session.messages
            if isinstance(m.get("content"), list)
        )
        assert "harness tool content" in session_text
        roles = [m.get("role") for m in harness._session.messages]
        assert "toolResult" in roles

    @pytest.mark.asyncio
    async def test_thinking_level_flows_to_stream_reasoning(self):
        captured: list[dict] = []
        core = faux_provider([faux_assistant_message("ok")])
        original = core.stream

        async def _capturing_stream(model, context, options=None):
            captured.append(dict(options or {}))
            return await original(model, context, options)

        harness = AgentHarness(
            AgentHarnessOptions(
                model=_make_model(),
                stream_fn=_capturing_stream,
            )
        )
        await harness.set_thinking_level("low")
        await harness.prompt("Hi")

        assert captured[0].get("reasoning") == "low"

    @pytest.mark.asyncio
    async def test_steer_during_tool_run(self):
        tool_ready = asyncio.Event()

        async def _tool_execute(tool_call_id, params, signal=None, on_update=None, context=None):
            await tool_ready.wait()
            return AgentToolResult(content=[TextContent(type="text", text="tool done")])

        tool = AgentTool(
            name="slow",
            description="slow tool",
            input_schema={"type": "object", "properties": {}},
            label="slow",
            execute=_tool_execute,
        )
        core = faux_provider()
        core.set_responses(
            [
                _tool_response("slow", {}),
                faux_assistant_message("after slow"),
                faux_assistant_message("after steer"),
            ]
        )
        harness = AgentHarness(
            AgentHarnessOptions(
                model=_make_model(),
                tools=[tool],
                tool_context=None,
                stream_fn=core.stream,
            )
        )

        run_task = asyncio.create_task(harness.prompt("run slow"))
        await asyncio.sleep(0.05)
        await harness.steer("please continue")
        tool_ready.set()
        await asyncio.wait_for(run_task, timeout=5.0)

        text = "\n".join(
            "".join(
                block.get("text", "")
                for block in m["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            )
            for m in harness._session.messages
            if isinstance(m.get("content"), list)
        )
        assert "please continue" in text


class TestCompactionCutPoint:
    @pytest.mark.asyncio
    async def test_split_turn_cut_point(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        # 大文本使估算 token 超过 keep_recent_tokens → 切割点落在历史中间
        for index in range(30):
            await session.append_message({"role": "user", "content": f"q{index}"})
            await session.append_message(_assistant_with_usage("A" * 4000))
        entries = await session.get_branch()

        ok_flag, preparation = prepare_compaction(entries, DEFAULT_COMPACTION_SETTINGS)
        assert ok_flag is True
        assert preparation is not None
        # 历史足够大 → messages_to_summarize 非空
        assert len(preparation.messages_to_summarize) > 0
        assert preparation.tokens_before > 0
        # 切割点不在最旧位置
        assert preparation.first_kept_entry_id != entries[0]["id"]


class TestJsonlPersistence:
    @pytest.mark.asyncio
    async def test_label_and_name_persist_across_reopen(self, tmp_path):
        store = create_jsonl_session_store(str(tmp_path))
        metadata = await store.create({"cwd": "/project"})
        session = Session(await store.open(metadata))
        entry_id = await session.append_message({"role": "user", "content": "hello"})
        await session.append_label(entry_id, "important")
        await session.append_session_name("my project")

        reopened = Session(await store.open(metadata))
        assert await reopened.get_label(entry_id) == "important"
        assert await reopened.get_session_name() == "my project"


class TestHarnessSkillInstructions:
    @pytest.mark.asyncio
    async def test_skill_with_additional_instructions(self):
        skill = Skill(
            name="docs",
            description="Docs skill",
            content="Read the docs carefully.",
            file_path="/tmp/skills/docs/SKILL.md",
        )
        core = faux_provider()
        core.set_responses([faux_assistant_message("ok")])
        harness = AgentHarness(
            AgentHarnessOptions(
                model=_make_model(),
                resources=AgentHarnessResources(skills=[skill]),
                stream_fn=core.stream,
            )
        )

        await harness.skill("docs", "Focus on API reference")

        text = "\n".join(
            "".join(
                block.get("text", "")
                for block in m["content"]
                if isinstance(block, dict) and block.get("type") == "text"
            )
            for m in harness._session.messages
            if isinstance(m.get("content"), list)
        )
        assert "Read the docs carefully." in text
        assert "Focus on API reference" in text
