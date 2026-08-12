"""评测 harness（完整对齐 TS packages/evals/src/pi-harness.ts）。

默认构造与 CLI 一致的 ModelRuntime（真实 providers + ~/.pi/agent 的
auth.json / models.json / models-store.json），模型由 PI_PROVIDER /
PI_MODEL 或显式 model 选项选择；测试通过 runtime 注入 faux provider
（零网络、可脚本化响应）。每次 run 都在隔离的临时 workspace / agent /
sessions 目录中创建 AgentSession，并在结束后把 session JSONL 快照写入
artifacts。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast, get_args

from pi_agent import Agent, AgentOptions, ThinkingLevel
from pi_ai import create_default_models
from pi_ai.models.models_store import FileModelsStore
from pi_ai.types.model import Model
from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager_v4 import SessionManagerLike, create_session_manager
from pi_coding_agent._config import get_agent_dir
from pi_coding_agent.extensions import ExtensionLoader, ExtensionRunner
from pi_coding_agent.model_runtime import ModelRuntime
from pi_coding_agent.prompt_templates import PromptTemplateLoader
from pi_coding_agent.skills import SkillLoader
from pi_coding_agent.system_prompt import (
    BuildSystemPromptOptions,
    build_system_prompt,
    load_project_context_files,
    tool_snippets_for,
)
from pi_coding_agent.tools import create_all_tools

from .vitest_evals.artifacts import PI_SESSION_SNAPSHOT_ARTIFACT
from .vitest_evals.harness import Harness, HarnessContext, HarnessRun, JsonValue, create_harness

PiCodingAgentInput: TypeAlias = str | list[dict[str, Any]]
PiCodingAgentOutputFn: TypeAlias = Callable[[dict[str, Any]], JsonValue]


@dataclass(slots=True)
class PiCodingAgentHarnessOptions:
    """createPiCodingAgentHarness 选项（对齐 TS PiCodingAgentHarnessOptions）。"""

    name: str = "pi-coding-agent"
    model: dict[str, str] | None = None
    # 推理强度（对齐 TS createPiCodingAgentHarness 的 thinkingLevel）。
    # None = 回退 PI_REASONING_LEVEL 环境变量，再默认 off；
    # 实际生效级别会被模型支持范围 clamp（见 AgentSession.set_thinking_level）。
    thinking_level: ThinkingLevel | None = None
    no_tools: bool | str = (
        False  # False / True / "all"（对齐 TS CreateAgentSessionOptions["noTools"]）
    )
    transform_system_prompt: Callable[[str], str] | None = None
    output: PiCodingAgentOutputFn | None = None
    # 测试注入：默认 None 时创建真实 ModelRuntime。
    runtime: ModelRuntime | None = None
    # 测试注入：在隔离 workspace 创建后、资源加载前调用（写扩展/上下文文件用）。
    workspace_setup: Callable[[Path], None] | None = None
    # 测试注入：每次 reload 前调用（模拟模型写入扩展后再加载）。
    reload_setup: Callable[[Path], None] | None = None


def resolve_model_selection(
    explicit: dict[str, str] | None = None,
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """解析模型选择：显式 > PI_PROVIDER/PI_MODEL 环境变量（对齐 TS resolveModelSelection）。"""
    env = environment if environment is not None else os.environ
    provider = ((explicit or {}).get("provider") or env.get("PI_PROVIDER") or "").strip()
    model_id = ((explicit or {}).get("id") or env.get("PI_MODEL") or "").strip()
    if not provider or not model_id:
        raise ValueError("Select a harness model explicitly or set both PI_PROVIDER and PI_MODEL.")
    return {"provider": provider, "id": model_id}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _transcript_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把会话消息归一化为 message / tool_call / tool_result 事件（对齐 TS toTranscriptEvents）。"""
    events: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            events.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": _content_text(message.get("content")),
                }
            )
        elif role == "assistant":
            text = _content_text(message.get("content"))
            if text:
                events.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": text,
                    }
                )
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    events.append(
                        {
                            "type": "tool_call",
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "arguments": block.get("arguments"),
                        }
                    )
        elif role == "toolResult":
            events.append(
                {
                    "type": "tool_result",
                    "toolCallId": message.get("tool_call_id"),
                    "name": message.get("tool_name"),
                    "content": _content_text(message.get("content")),
                    "isError": bool(message.get("is_error")),
                }
            )
    return events


def _has_pricing(model: Model) -> bool:
    """模型是否有任何非零单价（对齐 TS hasPricing 检查）。"""
    costs = [model.cost, *model.cost.tiers]
    return any(
        cost.input > 0 or cost.output > 0 or cost.cache_read > 0 or cost.cache_write > 0
        for cost in costs
    )


def _build_usage(model: Model, stats: dict[str, Any]) -> dict[str, Any]:
    """从会话统计构建 usage（对齐 TS SessionStats 映射）。"""
    tokens = stats["tokens"]
    usage: dict[str, Any] = {
        "provider": model.provider,
        "model": model.id,
        "inputTokens": tokens["input"],
        "outputTokens": tokens["output"],
        "totalTokens": tokens["total"],
        "toolCalls": stats["toolCalls"],
        "metadata": {
            "cacheReadTokens": tokens["cache_read"],
            "cacheWriteTokens": tokens["cache_write"],
        },
    }
    if _has_pricing(model):
        # pi_ai 已按模型单价逐请求计算 usage.cost，这里直接聚合（对齐 TS）。
        usage["metadata"]["estimatedCostUsd"] = stats["cost"]
    return usage


async def _abort_watcher(signal: asyncio.Event, session: AgentSession) -> None:
    await signal.wait()
    try:
        await session.abort()
    except Exception:
        pass


async def _prompt_agent(
    session: AgentSession,
    text: str,
    signal: asyncio.Event | None,
) -> str | None:
    """运行一次 prompt 并返回最后一条 assistant 文本（对齐 TS promptAgent）。"""
    if signal is not None and signal.is_set():
        raise RuntimeError("Eval run aborted.")
    watcher: asyncio.Task[None] | None = None
    if signal is not None:
        watcher = asyncio.create_task(_abort_watcher(signal, session))
    try:
        previous_count = len(session.get_messages())
        await session.prompt(text)
        new_messages = session.get_messages()[previous_count:]
        assistant = next(
            (message for message in reversed(new_messages) if message.get("role") == "assistant"),
            None,
        )
        if assistant is None:
            raise RuntimeError("Agent run completed without an assistant message.")
        stop_reason = assistant.get("stop_reason")
        if stop_reason not in (None, "stop"):
            raise RuntimeError(
                str(
                    assistant.get("error_message")
                    or f"Agent run ended with unexpected stop reason: {stop_reason}."
                )
            )
        output = session.get_last_assistant_text()
        has_tool_calls = any(
            isinstance(block, dict) and block.get("type") == "toolCall"
            for block in (cast(dict[str, Any], assistant).get("content") or [])
        )
        if not output and not has_tool_calls:
            raise RuntimeError("Agent run produced no assistant text or tool calls.")
        return output
    finally:
        if watcher is not None:
            watcher.cancel()


async def _reload_session(
    session: AgentSession,
    extension_loader: ExtensionLoader,
    runtime: ModelRuntime,
    reload_setup: Callable[[Path], None] | None = None,
) -> None:
    """重新加载技能/模板/扩展并重建系统提示（对齐 TS evalSession.reload()）。"""
    if reload_setup is not None:
        reload_setup(Path(session.cwd))
    old_runner = session.extension_runner
    if old_runner is not None:
        try:
            await old_runner.shutdown_all()
        except Exception:
            pass
    if session.skill_loader is not None:
        session.skill_loader.reload()
    if session.template_loader is not None:
        session.template_loader.reload()
    result = await extension_loader.load()
    new_runner = ExtensionRunner(
        result.extensions,
        runtime=result.runtime,
        cwd=session.cwd,
        model_runtime=runtime,
    )
    session.set_extension_runner(new_runner)
    session._pi_eval_extension_errors = [  # type: ignore[attr-defined]
        {"path": error.extension_path, "error": error.error} for error in result.errors
    ]
    await new_runner.discover_resources()
    session.rebuild_system_prompt()


async def _make_runtime(options: PiCodingAgentHarnessOptions) -> ModelRuntime:
    """测试注入优先；否则构造与 CLI 一致的默认运行时。

    默认运行时使用真实 providers + ~/.pi/agent 下的 auth.json / models.json /
    models-store.json（对齐 TS ModelRuntime.create()）；测试通过 options.runtime
    注入 faux provider。
    """
    if options.runtime is not None:
        return options.runtime
    return await ModelRuntime.create(
        providers=create_default_models().get_providers(),
        auth_path=str(get_agent_dir() / "auth.json"),
        models_path=str(get_agent_dir() / "models.json"),
        models_store=FileModelsStore(get_agent_dir() / "models-store.json"),
        allow_model_network=False,
        model_refresh_timeout_ms=15000,
    )


async def run_pi_coding_agent(
    input: PiCodingAgentInput,
    context: HarnessContext,
    options: PiCodingAgentHarnessOptions,
    signal: asyncio.Event | None = None,
) -> HarnessRun:
    """运行 pi-coding-agent eval（对齐 TS runPiCodingAgent）。"""
    started = time.perf_counter()
    selection = resolve_model_selection(options.model)
    thinking_level = options.thinking_level or os.environ.get("PI_REASONING_LEVEL") or "off"
    if thinking_level not in get_args(ThinkingLevel):
        raise ValueError(
            f"Invalid thinking level {thinking_level!r}; "
            f"expected one of {', '.join(get_args(ThinkingLevel))}."
        )
    runtime = await _make_runtime(options)
    model = runtime.get_model(selection["provider"], selection["id"])
    if model is None:
        raise ValueError(f"Eval model not found: {selection['provider']}/{selection['id']}")

    outcome_error: BaseException | None = None
    run: HarnessRun | None = None
    session: AgentSession | None = None
    session_manager: SessionManagerLike | None = None
    root = Path(tempfile.mkdtemp(prefix="pi-eval-"))
    try:
        cwd = root / "workspace"
        agent_dir = root / "agent"
        sessions_dir = root / "sessions"
        cwd.mkdir()
        agent_dir.mkdir()
        sessions_dir.mkdir()
        if options.workspace_setup is not None:
            options.workspace_setup(cwd)
        session_manager = await create_session_manager(str(cwd), sessions_dir=sessions_dir)
        context.set_artifact("runId", session_manager.session_id)

        skill_loader = SkillLoader(
            global_dir=agent_dir / "skills",
            project_dir=cwd / ".pi" / "skills",
        )
        skill_loader.load()
        template_loader = PromptTemplateLoader(
            global_dir=agent_dir / "prompts",
            project_dir=cwd / ".pi" / "prompts",
        )
        template_loader.load()
        extension_loader = ExtensionLoader(
            global_dir=agent_dir / "extensions",
            project_dir=cwd / ".pi" / "extensions",
            cwd=str(cwd),
        )
        extension_result = await extension_loader.load()
        extension_runner = ExtensionRunner(
            extension_result.extensions,
            runtime=extension_result.runtime,
            cwd=str(cwd),
            model_runtime=runtime,
        )
        await extension_runner.discover_resources()

        selected_tools: list[str] = []
        tool_snippets: dict[str, str] = {}
        if options.no_tools != "all" and options.no_tools is not True:
            default_tools = create_all_tools(str(cwd))
            selected_tools = [tool.name for tool in default_tools]
            tool_snippets = tool_snippets_for(default_tools)

        override_prompt: str | None = None

        def system_prompt_builder() -> str:
            if override_prompt is not None:
                return override_prompt
            skills = [*skill_loader.all(), *extension_runner.get_discovered_skills()]
            return build_system_prompt(
                BuildSystemPromptOptions(
                    cwd=str(cwd),
                    selected_tools=selected_tools,
                    tool_snippets=tool_snippets,
                    context_files=load_project_context_files(cwd, agent_dir),
                    skills=skills,
                )
            )

        default_prompt = system_prompt_builder()
        if options.transform_system_prompt is not None:
            transformed = options.transform_system_prompt(default_prompt)
            if not transformed.strip():
                raise ValueError("Transformed eval system prompt must not be empty.")
            override_prompt = transformed

        agent = Agent(
            AgentOptions(
                system_prompt=override_prompt or default_prompt,
                model=model,
                thinking_level=cast(ThinkingLevel, thinking_level),
                stream_fn=runtime.stream,
                session_id=session_manager.session_id,
            )
        )
        session = AgentSession(
            agent=agent,
            session_manager=session_manager,
            cwd=str(cwd),
            model=model,
            model_runtime=runtime,
            skill_loader=skill_loader,
            template_loader=template_loader,
            extension_runner=extension_runner,
            tools_override=[] if (options.no_tools == "all" or options.no_tools is True) else None,
            system_prompt_builder=system_prompt_builder,
        )
        session.project_trusted = True
        session._pi_eval_extension_errors = [  # type: ignore[attr-defined]
            {"path": error.extension_path, "error": error.error}
            for error in extension_result.errors
        ]
        if extension_runner.extensions:
            raise RuntimeError("Expected an isolated eval session to start without extensions.")

        steps = input if isinstance(input, list) else [{"type": "prompt", "content": input}]
        response: str | None = None
        saw_prompt_step = False
        for step in steps:
            if not isinstance(step, dict):
                raise RuntimeError("Pi eval input steps must be objects.")
            if step.get("type") == "prompt":
                saw_prompt_step = True
                response = await _prompt_agent(session, str(step.get("content") or ""), signal)
            else:
                await _reload_session(
                    session,
                    extension_loader,
                    runtime,
                    options.reload_setup,
                )
        if not saw_prompt_step:
            raise RuntimeError("Pi eval input must include at least one prompt step.")
        output: JsonValue = response
        if options.output is not None:
            output = options.output({"response": response, "session": session})
        transcript = _transcript_events(cast(list[dict[str, Any]], session.get_messages()))
        stats = session.get_session_stats()
        run = HarnessRun(
            output=output,
            events=transcript,
            usage=_build_usage(model, stats),
            timings={"totalMs": int((time.perf_counter() - started) * 1000)},
            artifacts=dict(context.artifacts),
        )
    except BaseException as exc:
        outcome_error = exc

    # 清理阶段：先写 session 快照，再 dispose 并删除临时目录。
    if session_manager is not None:
        try:
            session_file = session_manager.session_path
            if session_file is not None and session_file.exists():
                context.set_artifact(
                    PI_SESSION_SNAPSHOT_ARTIFACT,
                    session_file.read_text(encoding="utf-8"),
                )
        except BaseException as exc:
            if outcome_error is None:
                outcome_error = exc
    if session is not None:
        try:
            await session.dispose()
        except BaseException as exc:
            if outcome_error is None:
                outcome_error = exc
    try:
        shutil.rmtree(root, ignore_errors=True)
    except BaseException as exc:
        if outcome_error is None:
            outcome_error = exc

    if outcome_error is not None:
        raise outcome_error
    assert run is not None
    run.artifacts = dict(context.artifacts)
    return run


def create_pi_coding_agent_harness(
    *,
    name: str = "pi-coding-agent",
    model: dict[str, str] | None = None,
    thinking_level: ThinkingLevel | None = None,
    no_tools: bool | str = False,
    transform_system_prompt: Callable[[str], str] | None = None,
    output: PiCodingAgentOutputFn | None = None,
    runtime: ModelRuntime | None = None,
    workspace_setup: Callable[[Path], None] | None = None,
    reload_setup: Callable[[Path], None] | None = None,
) -> Harness:
    """创建 pi-coding-agent harness（对齐 TS createPiCodingAgentHarness）。"""
    options = PiCodingAgentHarnessOptions(
        name=name,
        model=model,
        thinking_level=thinking_level,
        no_tools=no_tools,
        transform_system_prompt=transform_system_prompt,
        output=output,
        runtime=runtime,
        workspace_setup=workspace_setup,
        reload_setup=reload_setup,
    )

    async def run(input: Any, context: HarnessContext) -> HarnessRun:
        return await run_pi_coding_agent(input, context, options)

    return create_harness(options.name, run)


__all__ = [
    "PiCodingAgentHarnessOptions",
    "PiCodingAgentInput",
    "create_pi_coding_agent_harness",
    "resolve_model_selection",
    "run_pi_coding_agent",
]
