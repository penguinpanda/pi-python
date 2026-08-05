"""评测 harness（对齐 TS packages/evals/src/pi-harness.ts）。

默认使用 faux provider（零网络、可脚本化响应）；设置 PI_PROVIDER /
PI_MODEL 可切换到真实模型。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, cast

from pi_agent import Agent, AgentOptions
from pi_ai import Models
from pi_ai.providers.faux import faux_provider

from pi_coding_agent._session import AgentSession
from pi_coding_agent._session_manager import SessionManager
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime


def resolve_model_selection(
    explicit: dict | None = None,
    environment: dict | None = None,
) -> dict:
    """解析模型选择：显式 > PI_PROVIDER/PI_MODEL 环境变量。"""
    env = environment if environment is not None else os.environ
    provider = (explicit or {}).get("provider") or env.get("PI_PROVIDER")
    model_id = (explicit or {}).get("id") or env.get("PI_MODEL")
    if not provider or not model_id:
        raise ValueError("Select a harness model explicitly or set both PI_PROVIDER and PI_MODEL.")
    return {"provider": provider.strip(), "id": model_id.strip()}


@dataclass(slots=True)
class EvalResult:
    """一次 eval 运行的结果。"""

    output: str
    errors: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    duration_ms: int = 0


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _transcript_events(messages: list[dict]) -> list[dict]:
    events: list[dict] = []
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


def _aggregate_usage(messages: list[dict]) -> dict:
    totals: dict[str, Any] = {
        "provider": None,
        "model": None,
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": 0.0,
    }
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage") or {}
        totals["provider"] = message.get("provider")
        totals["model"] = message.get("model")
        totals["input"] += int(usage.get("input") or 0)
        totals["output"] += int(usage.get("output") or 0)
        totals["cacheRead"] += int(usage.get("cache_read") or 0)
        totals["cacheWrite"] += int(usage.get("cache_write") or 0)
        totals["totalTokens"] += int(usage.get("total_tokens") or 0)
        cost = (usage.get("cost") or {}).get("total") or 0
        totals["cost"] += float(cost)
    return totals


class PiCodingAgentHarness:
    """pi-coding-agent 评测 harness。"""

    def __init__(
        self,
        *,
        model: dict | None = None,
        no_tools: bool = False,
        session_factory: Callable[[], AgentSession] | None = None,
        transform_system_prompt: Callable[[str], str] | None = None,
        output_fn: Callable[[dict], Any] | None = None,
        runtime: ModelRuntime | None = None,
    ) -> None:
        self._model_selection = resolve_model_selection(model)
        self.no_tools = no_tools
        self._session_factory = session_factory
        self._transform_system_prompt = transform_system_prompt
        self._output_fn = output_fn
        self._runtime = runtime

    def _make_runtime(self) -> ModelRuntime:
        if self._runtime is not None:
            return self._runtime
        store = AuthStorage.in_memory()
        models = Models(credentials=store)
        models.add_provider(faux_provider().provider)
        return ModelRuntime(models, store)

    def _create_session(self) -> AgentSession:
        if self._session_factory is not None:
            return self._session_factory()
        runtime = self._make_runtime()
        model = runtime.get_model(self._model_selection["provider"], self._model_selection["id"])
        if model is None:
            raise ValueError(
                f"Eval model not found: {self._model_selection['provider']}/"
                f"{self._model_selection['id']}"
            )
        system_prompt = "You are a helpful coding assistant."
        if self._transform_system_prompt is not None:
            system_prompt = self._transform_system_prompt(system_prompt)
        agent = Agent(
            AgentOptions(
                system_prompt=system_prompt,
                model=model,
                stream_fn=runtime.stream,
                tools=[] if self.no_tools else None,
            )
        )
        return AgentSession(
            agent=agent,
            session_manager=SessionManager.in_memory(cwd="."),
            cwd=".",
            model=model,
            model_runtime=runtime,
        )

    async def run(
        self,
        input: str | list[dict],
    ) -> EvalResult:
        """运行 prompt/reload 步骤序列（对齐 TS runPiCodingAgent）。"""
        started = time.perf_counter()
        steps = input if isinstance(input, list) else [{"type": "prompt", "content": input}]
        session = self._create_session()
        errors: list[str] = []
        last_output = ""
        try:
            for step in steps:
                step_type = step.get("type", "prompt")
                if step_type == "reload":
                    if session.rebuild_system_prompt is not None:
                        session.rebuild_system_prompt()
                    continue
                text = step.get("content", "")
                previous_count = len(session.get_messages())
                await session.prompt(text)
                assistant_messages = [
                    message
                    for message in session.get_messages()[previous_count:]
                    if message.get("role") == "assistant"
                ]
                if not assistant_messages:
                    errors.append("Agent run completed without an assistant message.")
                    continue
                last_message = assistant_messages[-1]
                stop_reason = last_message.get("stop_reason")
                if stop_reason not in (None, "stop"):
                    errors.append(
                        str(
                            last_message.get("error_message")
                            or f"Agent run ended with unexpected stop reason: {stop_reason}"
                        )
                    )
                last_output = session.get_last_assistant_text() or ""
        except Exception as exc:
            errors.append(str(exc))

        transcript = _transcript_events(cast(list[dict], session.get_messages()))
        usage = _aggregate_usage(cast(list[dict], session.get_messages()))
        artifacts = {
            "sessionId": session.session_id,
            "transcript": transcript,
        }
        if self._output_fn is not None:
            try:
                output = self._output_fn(
                    {
                        "response": last_output,
                        "session": session,
                    }
                )
            except Exception as exc:
                errors.append(f"output fn failed: {exc}")
                output = last_output
        else:
            output = last_output
        await session.dispose()
        return EvalResult(
            output=output,
            errors=errors,
            usage=usage,
            transcript=transcript,
            artifacts=artifacts,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


__all__ = ["EvalResult", "PiCodingAgentHarness", "resolve_model_selection"]
