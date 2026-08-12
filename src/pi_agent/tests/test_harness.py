"""_harness.py 模块测试（Phase 2：AgentHarness 骨架）。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pi_ai import Models, RetryPolicy
from pi_ai.provider import create_provider
from pi_ai.types import Model, TextContent, UserMessage
from pi_ai.providers.faux import FauxCore, faux_assistant_message, faux_provider, faux_tool_call
from pi_telemetry import InMemoryTelemetryContext

from pi_agent import Session
from pi_agent._harness_types import (
    AgentHarnessError,
    AgentHarnessOptions,
    AgentHarnessResources,
    AgentHarnessStreamOptionsPatch,
    BeforeAgentStartResult,
    BeforeProviderPayloadResult,
    BeforeProviderRequestResult,
    CompactResult,
    ContextResult,
    NavigateOptions,
    PromptTemplate,
    SessionBeforeCompactResult,
    SessionBeforeTreeResult,
    Skill,
    ToolCallResult,
    ToolResultPatch,
)
from pi_agent._harness import AgentHarness
from pi_agent._types import AgentTool, AgentToolResult, StreamFn
from pi_agent.session.v4.memory import InMemorySessionRepo


@pytest.mark.asyncio
async def test_prompt_records_telemetry_span() -> None:
    telemetry = InMemoryTelemetryContext()
    harness = AgentHarness(
        _make_options(
            stream_fn=_make_stream_fn([_text_response("ok")]),
            telemetry_context=telemetry,
        )
    )
    await harness.prompt("hi")
    assert any(span.name == "pi.harness.prompt" for span in telemetry.spans)


# ============================================================================
# 辅助
# ============================================================================


def _make_model(model_id: str = "test-model") -> Model:
    return Model(
        id=model_id,
        provider="test",
        api="openai-completions",
        name=f"Test {model_id}",
    )


def _make_faux(responses: list) -> FauxCore:
    core = faux_provider()
    core.set_responses(responses)
    return core


def _make_stream_fn(responses: list) -> StreamFn:
    return _make_faux(responses).stream


def _make_models(*, responses: list | None = None, stream_fn=None) -> Models:
    if stream_fn is None:
        core = _make_faux(responses if responses is not None else [_text_response("Hello!")])
        stream_fn = core.stream
    provider = create_provider(
        "test",
        "Test",
        None,
        [_make_model()],
        stream_fn=stream_fn,
    )
    models = Models()
    models.add_provider(provider)
    return models


def _make_session() -> Session:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    repo = InMemorySessionRepo()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, repo.create({})).result()


def _make_options(
    *,
    stream_fn=None,
    responses: list | None = None,
    tools=None,
    model: Model | None = None,
    **kwargs,
) -> AgentHarnessOptions:
    return AgentHarnessOptions(
        model=model or _make_model(),
        session=_make_session(),
        models=_make_models(responses=responses, stream_fn=stream_fn),
        tools=tools,
        **kwargs,
    )


def _text_response(text: str):
    return faux_assistant_message(text)


def _tool_response(tool_name: str, args: dict | None = None, tool_call_id: str = "tc-1"):
    return faux_assistant_message(
        [faux_tool_call(tool_name, args or {}, tool_call_id=tool_call_id)],
        stop_reason="tool_call",
    )


def _make_harness_tool(
    name: str,
    result_text: str = "ok",
    *,
    execute_override=None,
) -> AgentTool:
    """harness 工具：execute 接收 context 作第 5 参（对齐 TS AgentHarnessTool）。"""

    async def _execute(tool_call_id, params, signal=None, on_update=None, context=None):
        if execute_override is not None:
            return await execute_override(tool_call_id, params, signal, on_update, context)
        return AgentToolResult(content=[TextContent(type="text", text=result_text)])

    return AgentTool(
        name=name,
        description=f"Tool: {name}",
        input_schema={"type": "object", "properties": {}},
        label=name,
        execute=_execute,
    )


async def _context_messages(harness: AgentHarness) -> list[dict]:
    """DAG Session 投影后的 LLM 上下文消息。"""
    context = await harness._session.build_context()
    return list(context["messages"])


async def _session_text(harness: AgentHarness) -> str:
    """把会话上下文消息内容拼接为纯文本（content 可能是 str 或块列表）。"""
    parts: list[str] = []
    for message in await _context_messages(harness):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.append(
                "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
            )
    return "\n".join(parts)


def _make_harness(
    *,
    responses: list | None = None,
    tools: list[AgentTool] | None = None,
    **kwargs,
) -> AgentHarness:
    options = _make_options(responses=responses, tools=tools, **kwargs)
    return AgentHarness(options)


# ============================================================================
# 2.1 prompt / skill / template
# ============================================================================


class TestHarnessPrompt:
    @pytest.mark.asyncio
    async def test_basic_prompt(self):
        harness = _make_harness()
        result = await harness.prompt("Hi")

        assert result.get("role") == "assistant"
        roles = [m.get("role") for m in await _context_messages(harness)]
        assert roles.count("user") == 1
        assert roles.count("assistant") == 1

    @pytest.mark.asyncio
    async def test_subscriber_receives_events_and_settled(self):
        harness = _make_harness()
        received: list[str] = []
        harness.subscribe(lambda e, signal: received.append(e["type"]))

        await harness.prompt("Hi")

        assert "agent_start" in received
        assert "agent_end" in received
        assert "settled" in received
        assert "save_point" in received

    @pytest.mark.asyncio
    async def test_prompt_while_busy_raises(self):
        core = faux_provider(tokens_per_second=200)
        core.set_responses([faux_assistant_message("A" * 200)])
        harness = AgentHarness(_make_options(stream_fn=core.stream))

        first = asyncio.create_task(harness.prompt("Q1"))
        await asyncio.sleep(0.05)

        with pytest.raises(AgentHarnessError, match="busy"):
            await harness.prompt("Q2")

        await harness.abort()
        try:
            await asyncio.wait_for(first, timeout=2.0)
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_wait_for_idle(self):
        core = faux_provider(tokens_per_second=200)
        core.set_responses([faux_assistant_message("A" * 100)])
        harness = AgentHarness(_make_options(stream_fn=core.stream))

        task = asyncio.create_task(harness.prompt("Hi"))
        await harness.wait_for_idle()
        await task

        assert harness._phase == "idle"


class TestHarnessSkillTemplate:
    @pytest.mark.asyncio
    async def test_skill_invocation(self):
        skill = Skill(
            name="docs",
            description="Docs skill",
            content="Read docs",
            file_path="/tmp/docs/SKILL.md",
        )
        harness = _make_harness(resources=AgentHarnessResources(skills=[skill]))

        await harness.skill("docs")

        text = await _session_text(harness)
        assert "docs" in text and "Read docs" in text

    @pytest.mark.asyncio
    async def test_unknown_skill(self):
        harness = _make_harness()
        with pytest.raises(AgentHarnessError, match="Unknown skill"):
            await harness.skill("nope")

    @pytest.mark.asyncio
    async def test_prompt_from_template(self):
        template = PromptTemplate(name="greet", content="Hi $1! $ARGUMENTS")
        harness = _make_harness(resources=AgentHarnessResources(prompt_templates=[template]))

        await harness.prompt_from_template("greet", ["World"])

        assert "Hi World! World" in await _session_text(harness)

    @pytest.mark.asyncio
    async def test_unknown_template(self):
        harness = _make_harness()
        with pytest.raises(AgentHarnessError, match="Unknown prompt template"):
            await harness.prompt_from_template("nope", [])


# ============================================================================
# 2.3 双事件系统
# ============================================================================


class TestHarnessDualEvents:
    @pytest.mark.asyncio
    async def test_on_context_transforms_messages(self):
        llm_inputs: list[list] = []
        core = _make_faux([_text_response("Hello!")])
        original = core.stream

        async def _capturing_stream(model, context, options=None):
            llm_inputs.append(list(context.messages))
            return await original(model, context, options)

        harness = AgentHarness(_make_options(stream_fn=_capturing_stream))

        async def _context_hook(event):
            return ContextResult(
                messages=list(event["messages"])
                + [{"role": "user", "content": "injected-by-context"}]
            )

        harness.on("context", _context_hook)
        await harness.prompt("Hi")

        # transform_context 只影响 LLM 输入，不写入会话
        assert len(llm_inputs) == 1
        llm_text = "\n".join(str(m.get("content")) for m in llm_inputs[0])
        assert "injected-by-context" in llm_text

    @pytest.mark.asyncio
    async def test_on_tool_call_blocks(self):
        executed: list[str] = []

        async def _execute(tool_call_id, params, signal=None, on_update=None, context=None):
            executed.append(tool_call_id)
            return AgentToolResult(content=[TextContent(type="text", text="ran")])

        tool = _make_harness_tool("search", execute_override=_execute)
        harness = _make_harness(
            tools=[tool],
            responses=[_tool_response("search"), _text_response("done")],
        )

        def _tool_call_hook(event):
            return ToolCallResult(block=True, reason="not allowed")

        harness.on("tool_call", _tool_call_hook)
        await harness.prompt("Search")

        assert executed == []
        # 被 block 的工具结果以错误形式回给 LLM，随后文本轮正常完成
        tool_results = [
            m for m in await _context_messages(harness) if m.get("role") == "toolResult"
        ]
        assert len(tool_results) == 1
        assert tool_results[0]["is_error"] is True

    @pytest.mark.asyncio
    async def test_on_tool_result_patches(self):
        tool = _make_harness_tool("search", "original")
        harness = _make_harness(
            tools=[tool],
            responses=[_tool_response("search"), _text_response("done")],
        )

        def _tool_result_hook(event):
            return ToolResultPatch(content=[TextContent(type="text", text="patched")])

        harness.on("tool_result", _tool_result_hook)
        await harness.prompt("Search")

        tool_results = [
            m for m in await _context_messages(harness) if m.get("role") == "toolResult"
        ]
        assert tool_results[0]["content"][0]["text"] == "patched"

    @pytest.mark.asyncio
    async def test_on_before_agent_start_injects_messages(self):
        harness = _make_harness()

        def _before_start(event):
            return BeforeAgentStartResult(
                messages=[{"role": "user", "content": "extra-context"}],
                system_prompt="Custom system prompt",
            )

        harness.on("before_agent_start", _before_start)
        await harness.prompt("Hi")

        assert "extra-context" in await _session_text(harness)

    @pytest.mark.asyncio
    async def test_on_before_provider_request_patches_options(self):
        captured: list[dict] = []
        core = _make_faux([_text_response("ok")])
        original = core.stream

        async def _capturing_stream(model, context, options=None):
            captured.append(dict(options or {}))
            return await original(model, context, options)

        harness = AgentHarness(_make_options(stream_fn=_capturing_stream))

        def _provider_hook(event):
            return BeforeProviderRequestResult(
                stream_options=AgentHarnessStreamOptionsPatch(max_retries=5)
            )

        harness.on("before_provider_request", _provider_hook)
        await harness.prompt("Hi")

        assert captured[0]["max_retries"] == 5

    @pytest.mark.asyncio
    async def test_next_turn_emits_queue_update(self):
        harness = _make_harness()
        received: list[dict] = []
        harness.subscribe(
            lambda e, signal: received.append(e) if e["type"] == "queue_update" else None
        )

        await harness.next_turn("later")

        assert len(received) == 1
        assert len(received[0]["next_turn"]) == 1


# ============================================================================
# 2.4 Save-point
# ============================================================================


class TestHarnessSavePoint:
    @pytest.mark.asyncio
    async def test_set_model_during_run_applies_next_turn(self):
        tool_ready = asyncio.Event()
        seen_models: list[str] = []

        async def _tool_execute(tool_call_id, params, signal=None, on_update=None, context=None):
            await tool_ready.wait()
            return AgentToolResult(content=[TextContent(type="text", text="done")])

        tool = _make_harness_tool("finish", execute_override=_tool_execute)
        core = _make_faux([_tool_response("finish"), _text_response("final answer")])
        original = core.stream

        async def _capturing_stream(model, context, options=None):
            seen_models.append(model.id)
            return await original(model, context, options)

        harness = AgentHarness(
            _make_options(model=_make_model("model-a"), stream_fn=_capturing_stream, tools=[tool])
        )

        # 第一轮流式输出后工具执行；工具等待测试任务切换模型
        async def _run():
            await harness.prompt("Go")

        run_task = asyncio.create_task(_run())
        await asyncio.sleep(0.05)
        await harness.set_model(_make_model("model-b"))
        tool_ready.set()
        await asyncio.wait_for(run_task, timeout=5.0)

        # 第一轮 LLM 调用用 model-a，prepare_next_turn 后第二轮用 model-b
        assert seen_models[0] == "model-a"
        assert seen_models[-1] == "model-b"

    @pytest.mark.asyncio
    async def test_save_point_event_reports_pending_mutations(self):
        tool_ready = asyncio.Event()

        async def _tool_execute(tool_call_id, params, signal=None, on_update=None, context=None):
            await tool_ready.wait()
            return AgentToolResult(content=[TextContent(type="text", text="done")])

        tool = _make_harness_tool("finish", execute_override=_tool_execute)
        harness = _make_harness(
            tools=[tool],
            responses=[_tool_response("finish"), _text_response("final")],
        )

        save_points: list[dict] = []
        harness.subscribe(
            lambda e, signal: save_points.append(e) if e["type"] == "save_point" else None
        )

        async def _run():
            await harness.prompt("Go")

        run_task = asyncio.create_task(_run())
        await asyncio.sleep(0.05)
        await harness.set_thinking_level("low")  # 运行中变更 → pending
        tool_ready.set()
        await asyncio.wait_for(run_task, timeout=5.0)

        assert any(sp["had_pending_mutations"] for sp in save_points)

    @pytest.mark.asyncio
    async def test_append_message_during_run_flushed_at_end(self):
        tool_ready = asyncio.Event()

        async def _tool_execute(tool_call_id, params, signal=None, on_update=None, context=None):
            await tool_ready.wait()
            return AgentToolResult(content=[TextContent(type="text", text="done")])

        tool = _make_harness_tool("finish", execute_override=_tool_execute)
        harness = _make_harness(
            tools=[tool],
            responses=[_tool_response("finish"), _text_response("final")],
        )

        run_task = asyncio.create_task(harness.prompt("Go"))
        await asyncio.sleep(0.05)
        await harness.append_message(UserMessage(role="user", content="queued-append"))
        tool_ready.set()
        await asyncio.wait_for(run_task, timeout=5.0)

        assert "queued-append" in await _session_text(harness)


# ============================================================================
# steer / follow_up / next_turn
# ============================================================================


class TestHarnessQueues:
    @pytest.mark.asyncio
    async def test_steer_during_run_injected(self):
        core = faux_provider(tokens_per_second=200)
        core.set_responses(
            [
                faux_assistant_message("A" * 100),
                faux_assistant_message("B" * 20),
            ]
        )
        harness = AgentHarness(_make_options(stream_fn=core.stream))

        async def _steer_mid_run():
            await asyncio.sleep(0.05)
            await harness.steer("nudge")

        await asyncio.gather(harness.prompt("Q"), _steer_mid_run())

        text = await _session_text(harness)
        assert "nudge" in text
        assert text.count("A" * 100) == 1

    @pytest.mark.asyncio
    async def test_follow_up_after_stop_injected(self):
        core = faux_provider(tokens_per_second=200)
        core.set_responses(
            [
                faux_assistant_message("A" * 50),
                faux_assistant_message("B" * 20),
            ]
        )
        harness = AgentHarness(_make_options(stream_fn=core.stream))

        async def _follow_up_mid_run():
            await asyncio.sleep(0.05)
            await harness.follow_up("one more")

        await asyncio.gather(harness.prompt("Q"), _follow_up_mid_run())

        assert "one more" in await _session_text(harness)
        # follow-up 追加一轮 → 两条 assistant
        assert sum(1 for m in await _context_messages(harness) if m.get("role") == "assistant") == 2

    @pytest.mark.asyncio
    async def test_next_turn_prepended(self):
        harness = _make_harness()
        await harness.next_turn("prepended")
        await harness.prompt("main")

        text = await _session_text(harness)
        assert text.index("prepended") < text.index("main")

    @pytest.mark.asyncio
    async def test_steer_while_idle_raises(self):
        harness = _make_harness()
        with pytest.raises(AgentHarnessError, match="Cannot steer while idle"):
            await harness.steer("nope")


# ============================================================================
# compact / navigateTree 骨架
# ============================================================================


class TestHarnessCompactNavigate:
    @pytest.mark.asyncio
    async def test_compact_nothing_to_compact(self):
        harness = _make_harness()
        with pytest.raises(AgentHarnessError, match="Nothing to compact"):
            await harness.compact()

    @pytest.mark.asyncio
    async def test_compact_hook_cancel(self):
        harness = _make_harness()
        harness.on("session_before_compact", lambda e: SessionBeforeCompactResult(cancel=True))

        with pytest.raises(AgentHarnessError, match="cancelled"):
            await harness.compact()

    @pytest.mark.asyncio
    async def test_compact_hook_provides_result(self):
        harness = _make_harness()
        provided = CompactResult(summary="from hook")
        harness.on(
            "session_before_compact",
            lambda e: SessionBeforeCompactResult(compaction=provided),
        )

        result = await harness.compact()
        assert result.summary == "from hook"

    @pytest.mark.asyncio
    async def test_navigate_tree_leaf_noop(self):
        harness = _make_harness()
        await harness.prompt("Hi")
        leaf_id = await harness.get_leaf_id()
        assert leaf_id is not None
        result = await harness.navigate_tree(leaf_id)
        assert result.cancelled is False

    @pytest.mark.asyncio
    async def test_navigate_tree_unknown_entry(self):
        harness = _make_harness()
        with pytest.raises(AgentHarnessError, match="not found"):
            await harness.navigate_tree("missing-entry")

    @pytest.mark.asyncio
    async def test_compact_generates_summary_and_entry(self):
        core = _make_faux(
            [
                _text_response("first answer"),
                faux_assistant_message("## Goal\ncompacted"),
            ]
        )
        harness = AgentHarness(_make_options(stream_fn=core.stream))
        compact_events: list[dict] = []
        harness.subscribe(
            lambda e, signal: compact_events.append(e) if e["type"] == "session_compact" else None
        )

        await harness.prompt("question")
        result = await harness.compact()

        assert "## Goal" in result.summary
        entries = await harness._session.get_branch()
        assert entries[-1]["type"] == "compaction"
        assert "fromHook" not in entries[-1]
        context = await harness._session.build_context()
        assert context["messages"][0]["role"] == "compactionSummary"
        assert compact_events and compact_events[-1]["from_hook"] is False

    @pytest.mark.asyncio
    async def test_compact_hook_provided_writes_entry(self):
        harness = _make_harness()
        await harness.prompt("Hi")
        provided = CompactResult(summary="from hook", first_kept_entry_id="nope", tokens_before=3)
        compact_events: list[dict] = []
        harness.subscribe(
            lambda e, signal: compact_events.append(e) if e["type"] == "session_compact" else None
        )
        harness.on(
            "session_before_compact",
            lambda e: SessionBeforeCompactResult(compaction=provided),
        )

        result = await harness.compact()

        assert result.summary == "from hook"
        entries = await harness._session.get_branch()
        assert entries[-1]["type"] == "compaction"
        assert entries[-1]["summary"] == "from hook"
        assert "fromHook" not in entries[-1]
        assert compact_events and compact_events[-1]["from_hook"] is True

    @pytest.mark.asyncio
    async def test_navigate_tree_moves_leaf(self):
        core = _make_faux([_text_response("a1"), _text_response("a2")])
        harness = AgentHarness(_make_options(stream_fn=core.stream))
        await harness.prompt("q1")
        first_leaf = await harness.get_leaf_id()
        assert first_leaf is not None
        first_entry = (await harness._session.get_branch())[0]["id"]
        tree_events: list[dict] = []
        harness.subscribe(
            lambda e, signal: tree_events.append(e) if e["type"] == "session_tree" else None
        )

        result = await harness.navigate_tree(first_entry)
        assert result.cancelled is False
        assert await harness.get_leaf_id() == first_entry

        await harness.prompt("q2")
        second_leaf = await harness.get_leaf_id()
        assert second_leaf is not None and second_leaf != first_leaf

        await harness.navigate_tree(first_leaf)
        assert await harness.get_leaf_id() == first_leaf
        assert any(e["new_leaf_id"] == first_leaf for e in tree_events)

    @pytest.mark.asyncio
    async def test_navigate_tree_summarize_generates_branch_summary(self):
        core = _make_faux(
            [
                _text_response("a1"),
                _text_response("a2"),
                faux_assistant_message("## Goal\nbranch summary"),
            ]
        )
        harness = AgentHarness(_make_options(stream_fn=core.stream))
        await harness.prompt("q1")
        first_leaf = await harness.get_leaf_id()
        assert first_leaf is not None
        first_entry = (await harness._session.get_branch())[0]["id"]
        await harness.navigate_tree(first_entry)
        await harness.prompt("q2")

        result = await harness.navigate_tree(first_leaf, NavigateOptions(summarize=True))

        assert result.cancelled is False
        assert result.summary_entry is not None
        assert result.summary_entry["type"] == "branch_summary"
        assert "fromHook" not in result.summary_entry
        context = await harness._session.build_context()
        roles = [m.get("role") for m in context["messages"]]
        assert "branchSummary" in roles

    @pytest.mark.asyncio
    async def test_navigate_tree_hook_cancel(self):
        harness = _make_harness()
        await harness.prompt("Hi")
        target = (await harness._session.get_branch())[0]["id"]
        harness.on("session_before_tree", lambda e: SessionBeforeTreeResult(cancel=True))

        result = await harness.navigate_tree(target)

        assert result.cancelled is True
        assert await harness.get_leaf_id() != target

    @pytest.mark.asyncio
    async def test_navigate_tree_hook_provides_summary(self):
        harness = _make_harness()
        await harness.prompt("Hi")
        target = (await harness._session.get_branch())[0]["id"]
        harness.on(
            "session_before_tree",
            lambda e: SessionBeforeTreeResult(summary="hook summary"),
        )

        result = await harness.navigate_tree(target, NavigateOptions(summarize=True))

        assert result.cancelled is False
        assert result.summary_entry is not None
        assert result.summary_entry["summary"] == "hook summary"
        assert "fromHook" not in result.summary_entry
        context = await harness._session.build_context()
        assert any(m.get("role") == "branchSummary" for m in context["messages"])


class TestHarnessLegacyFeatures:
    """对齐 TS legacy：payload/response hooks、retry 事件、生命周期拆分。"""

    @pytest.mark.asyncio
    async def test_before_provider_payload_replaces_payload(self):
        recorded: dict[str, Any] = {}
        original = _make_faux([_text_response("ok")]).stream

        async def _stream(model, context, options=None):
            on_payload = (options or {}).get("on_payload")
            if on_payload is not None:
                result = on_payload({"marker": 1}, model)
                if asyncio.iscoroutine(result):
                    result = await result
                recorded["payload"] = result
            return await original(model, context, options)

        harness = AgentHarness(_make_options(stream_fn=_stream))
        harness.on(
            "before_provider_payload",
            lambda event: BeforeProviderPayloadResult(payload={"replaced": True}),
        )

        await harness.prompt("Hi")

        assert recorded["payload"] == {"replaced": True}

    @pytest.mark.asyncio
    async def test_after_provider_response_emits_event(self):
        received: list[dict] = []
        original = _make_faux([_text_response("ok")]).stream

        async def _stream(model, context, options=None):
            on_response = (options or {}).get("on_response")
            if on_response is not None:
                result = on_response({"status": 201, "headers": {"x": "y"}}, model)
                if asyncio.iscoroutine(result):
                    await result
            return await original(model, context, options)

        harness = AgentHarness(_make_options(stream_fn=_stream))
        harness.subscribe(
            lambda e, signal: received.append(e) if e["type"] == "after_provider_response" else None
        )

        await harness.prompt("Hi")

        assert received and received[-1]["status"] == 201
        assert received[-1]["headers"] == {"x": "y"}

    @pytest.mark.asyncio
    async def test_compaction_retry_emits_events(self):
        calls = 0
        retry_events: list[dict] = []

        async def _stream(model, context, options=None):
            nonlocal calls
            calls += 1
            if calls == 1:
                response = _text_response("first answer")
            elif calls == 2:
                response = faux_assistant_message(
                    "",
                    stop_reason="error",
                    error_message="503 Service Unavailable",
                )
            else:
                response = faux_assistant_message("## Goal\ncompacted")
            core = _make_faux([response])
            return await core.stream(model, context, options)

        harness = AgentHarness(
            _make_options(
                stream_fn=_stream,
                retry=RetryPolicy(enabled=True, max_retries=1, base_delay_ms=1, jitter=False),
            )
        )
        harness.subscribe(
            lambda e, signal: (
                retry_events.append(e)
                if e["type"] in ("retry_scheduled", "retry_attempt_start", "retry_finished")
                else None
            )
        )

        await harness.prompt("question")
        result = await harness.compact()

        assert "## Goal" in result.summary
        assert calls == 3
        assert any(e["type"] == "retry_scheduled" for e in retry_events)
        assert any(e["type"] == "retry_attempt_start" for e in retry_events)
        assert any(e["type"] == "retry_finished" for e in retry_events)

    @pytest.mark.asyncio
    async def test_request_shutdown_and_wait_for_shutdown(self):
        harness = _make_harness()

        with pytest.raises(AgentHarnessError, match="Shutdown has not been requested"):
            await harness.wait_for_shutdown()

        harness.request_shutdown()
        await harness.wait_for_shutdown()

        with pytest.raises(AgentHarnessError, match="shut down"):
            await harness.prompt("Hi")


# ============================================================================
# abort / shutdown / 配置校验
# ============================================================================


class TestHarnessAbortShutdown:
    @pytest.mark.asyncio
    async def test_abort_clears_queues_and_stops_run(self):
        core = faux_provider(tokens_per_second=100)
        core.set_responses([faux_assistant_message("A" * 300)])
        harness = AgentHarness(_make_options(stream_fn=core.stream))

        run_task = asyncio.create_task(harness.prompt("Q"))
        await asyncio.sleep(0.05)
        await harness.steer("stale")
        await harness.follow_up("stale-fu")
        result = await harness.abort()
        await asyncio.wait_for(run_task, timeout=2.0)

        assert len(result.cleared_steer) == 1
        assert len(result.cleared_follow_up) == 1
        assert harness._phase == "idle"

    @pytest.mark.asyncio
    async def test_shutdown_prevents_operations(self):
        harness = _make_harness()
        await harness.shutdown()

        with pytest.raises(AgentHarnessError, match="shut down"):
            await harness.prompt("Hi")
        with pytest.raises(AgentHarnessError, match="shut down"):
            harness.subscribe(lambda e, signal: None)

    @pytest.mark.asyncio
    async def test_set_tools_validates_unknown_names(self):
        harness = _make_harness(tools=[_make_harness_tool("a")])
        with pytest.raises(AgentHarnessError, match="Unknown tool"):
            await harness.set_active_tools(["nope"])

    @pytest.mark.asyncio
    async def test_set_tools_duplicate_names(self):
        tool = _make_harness_tool("dup")
        harness = _make_harness()
        with pytest.raises(AgentHarnessError, match="Duplicate"):
            await harness.set_tools([tool, _make_harness_tool("dup")])
