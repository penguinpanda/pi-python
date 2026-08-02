"""
Faux Provider 单元测试。

覆盖：
    • 辅助函数（faux_text / faux_thinking / faux_tool_call / faux_assistant_message）
    • 脚本化响应队列（set / append / 按序消费 / 耗尽报错）
    • 动态响应工厂（context / state / model 感知）
    • 消息重写（api / provider / model）
    • Usage 估算
    • 流式事件（delta / toolCallDelta / thinkingDelta / done / error）
    • 中止信号（abort）
    • 与 Models 注册表集成
"""

import asyncio
import json
import math

import pytest

from pi_ai import Models
from pi_ai._types import (
    AssistantMessage,
    Context,
    Model,
    StreamOptions,
    Tool,
)
from pi_ai.providers.faux import (
    FAUX_MODEL,
    FauxCore,
    faux_assistant_message,
    faux_provider,
    faux_text,
    faux_thinking,
    faux_tool_call,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "hi"}])


async def _collect(stream) -> list[dict]:
    return [e async for e in stream]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


class TestFauxHelpers:
    """faux_text / faux_thinking / faux_tool_call / faux_assistant_message。"""

    def test_faux_text(self):
        assert faux_text("hi") == {"type": "text", "text": "hi"}

    def test_faux_thinking(self):
        assert faux_thinking("think") == {
            "type": "thinking",
            "thinking": "think",
            "signature": None,
        }

    def test_faux_tool_call_dict_args(self):
        tc = faux_tool_call("echo", {"text": "hi"}, tool_call_id="call-1")
        assert tc["type"] == "toolCall"
        assert tc["toolCallId"] == "call-1"
        assert tc["toolName"] == "echo"
        assert tc["args"] == '{"text": "hi"}'

    def test_faux_tool_call_string_args(self):
        tc = faux_tool_call("echo", '{"a": 1}')
        assert tc["args"] == '{"a": 1}'
        assert tc["toolCallId"].startswith("tool:")

    def test_faux_assistant_message_normalizes_content(self):
        # str → 单个 TextContent
        msg = faux_assistant_message("hi")
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"type": "text", "text": "hi"}]
        assert msg["stopReason"] == "end"
        assert msg["provider"] == "faux"
        assert msg["model"] == "faux-1"

        # 单个 block
        msg2 = faux_assistant_message(faux_text("single"))
        assert msg2["content"] == [{"type": "text", "text": "single"}]

        # block 列表
        msg3 = faux_assistant_message([faux_text("a"), faux_text("b")])
        assert msg3["content"] == [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]


# ---------------------------------------------------------------------------
# Provider 构造
# ---------------------------------------------------------------------------


class TestFauxProviderFactory:
    """faux_provider() 工厂。"""

    def test_default_model(self):
        faux = faux_provider()
        assert faux.provider.id == "faux"
        assert faux.provider.name == "Faux"
        assert faux.provider.auth is None
        assert faux.provider._stream_fn is not None
        assert [m.id for m in faux.models] == ["faux-1"]
        assert faux.get_model() is faux.models[0]
        assert faux.get_model("faux-1") is not None
        assert faux.get_model("nope") is None

    def test_custom_models(self):
        models = [
            Model(id="faux-fast", provider="faux", api="openai-completions", name="Faux Fast"),
            Model(id="faux-thinker", provider="faux", api="openai-completions", name="Faux Thinker", thinking=True),
        ]
        faux = faux_provider(models=models)
        assert [m.id for m in faux.models] == ["faux-fast", "faux-thinker"]
        assert faux.get_model("faux-thinker") is not None
        assert faux.get_model("faux-thinker").thinking is True  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 基本响应
# ---------------------------------------------------------------------------


class TestBasicResponses:
    """complete() / stream() 基础行为。"""

    @pytest.mark.asyncio
    async def test_complete_returns_scripted_text_and_usage(self):
        faux = faux_provider()
        faux.set_responses([faux_assistant_message("hello world")])
        ctx = Context(system="Be concise.", messages=[{"role": "user", "content": "hi there"}])

        msg = await faux.provider.complete(faux.models[0], ctx)

        assert msg["content"] == [{"type": "text", "text": "hello world"}]
        assert msg["usage"]["input"] > 0
        assert msg["usage"]["output"] > 0
        assert msg["usage"]["totalTokens"] == msg["usage"]["input"] + msg["usage"]["output"]
        assert msg["usage"]["cost"]["total"] == 0
        assert faux.call_count == 1

    @pytest.mark.asyncio
    async def test_helper_blocks(self):
        faux = faux_provider()
        faux.set_responses([
            faux_assistant_message(
                [
                    faux_thinking("think"),
                    faux_tool_call("echo", {"text": "hi"}, tool_call_id="call-1"),
                    faux_text("done"),
                ],
                stop_reason="toolCall",
            )
        ])

        msg = await faux.provider.complete(faux.models[0], _context())

        assert msg["content"] == [
            {"type": "thinking", "thinking": "think", "signature": None},
            {"type": "toolCall", "toolCallId": "call-1", "toolName": "echo", "args": '{"text": "hi"}'},
            {"type": "text", "text": "done"},
        ]
        assert msg["stopReason"] == "toolCall"

    @pytest.mark.asyncio
    async def test_rewrites_api_provider_model(self):
        faux = faux_provider(
            provider="faux-provider",
            models=[Model(id="faux-model", provider="faux-provider", api="openai-completions", name="Faux Model")],
        )
        faux.set_responses([faux_assistant_message("hello")])

        msg = await faux.provider.complete(faux.get_model("faux-model"), _context())

        assert msg["api"] == "openai-completions"
        assert msg["provider"] == "faux-provider"
        assert msg["model"] == "faux-model"


# ---------------------------------------------------------------------------
# 响应队列
# ---------------------------------------------------------------------------


class TestResponseQueue:
    """set_responses / append_responses / 按序消费 / 耗尽报错。"""

    @pytest.mark.asyncio
    async def test_consumes_queued_in_order_and_errors_when_exhausted(self):
        faux = faux_provider()
        faux.set_responses([faux_assistant_message("first"), faux_assistant_message("second")])
        model = faux.models[0]
        ctx = _context()

        first = await faux.provider.complete(model, ctx)
        second = await faux.provider.complete(model, ctx)
        exhausted = await faux.provider.complete(model, ctx)

        assert first["content"] == [{"type": "text", "text": "first"}]
        assert second["content"] == [{"type": "text", "text": "second"}]
        assert exhausted["stopReason"] == "error"
        assert exhausted["errorMessage"] == "No more faux responses queued"
        assert faux.get_pending_response_count() == 0
        assert faux.call_count == 3

    @pytest.mark.asyncio
    async def test_set_responses_replaces_and_append_adds(self):
        faux = faux_provider()
        model = faux.models[0]
        ctx = _context()

        faux.set_responses([faux_assistant_message("first")])
        assert faux.get_pending_response_count() == 1
        assert (await faux.provider.complete(model, ctx))["content"][0]["text"] == "first"
        assert faux.get_pending_response_count() == 0

        faux.set_responses([faux_assistant_message("second")])
        assert faux.get_pending_response_count() == 1
        assert (await faux.provider.complete(model, ctx))["content"][0]["text"] == "second"

        faux.append_responses([faux_assistant_message("third"), faux_assistant_message("fourth")])
        assert faux.get_pending_response_count() == 2
        assert (await faux.provider.complete(model, ctx))["content"][0]["text"] == "third"
        assert (await faux.provider.complete(model, ctx))["content"][0]["text"] == "fourth"
        assert faux.get_pending_response_count() == 0


# ---------------------------------------------------------------------------
# 动态响应工厂
# ---------------------------------------------------------------------------


class TestResponseFactory:
    """基于 context / state / model 的动态响应。"""

    @pytest.mark.asyncio
    async def test_async_factory_with_context_and_state(self):
        faux = faux_provider()

        async def factory(context, options, state, model):
            return faux_assistant_message(f"{len(context.messages)}:{state['callCount']}")

        faux.set_responses([factory])

        msg = await faux.provider.complete(faux.models[0], _context())
        assert msg["content"][0]["text"] == "1:1"

    @pytest.mark.asyncio
    async def test_model_aware_factory(self):
        models = [
            Model(id="faux-fast", provider="faux", api="openai-completions", name="Faux Fast", thinking=False),
            Model(id="faux-thinker", provider="faux", api="openai-completions", name="Faux Thinker", thinking=True),
        ]
        faux = faux_provider(models=models)

        async def factory(context, options, state, model):
            return faux_assistant_message(f"{model.id}:{model.thinking}")

        faux.set_responses([factory, factory])

        fast = await faux.provider.complete(faux.get_model("faux-fast"), _context())
        thinker = await faux.provider.complete(faux.get_model("faux-thinker"), _context())

        assert fast["content"][0]["text"] == "faux-fast:False"
        assert thinker["content"][0]["text"] == "faux-thinker:True"

    @pytest.mark.asyncio
    async def test_factory_throws_emits_error(self):
        faux = faux_provider()

        async def factory(context, options, state, model):
            raise RuntimeError("boom")

        faux.set_responses([factory])

        events = await _collect(await faux.provider.stream(faux.models[0], _context()))

        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert events[0]["error"]["stopReason"] == "error"
        assert events[0]["error"]["errorMessage"] == "boom"

    @pytest.mark.asyncio
    async def test_pending_stop_reason_rejected(self):
        faux = faux_provider()
        faux.set_responses([faux_assistant_message("partial", stop_reason="pending")])

        events = await _collect(await faux.provider.stream(faux.models[0], _context()))

        assert not any(e["type"] == "done" for e in events)
        assert events[-1]["type"] == "error"
        assert events[-1]["error"]["stopReason"] == "error"
        assert events[-1]["error"]["errorMessage"] == "Faux response ended without a stop reason"


# ---------------------------------------------------------------------------
# Usage 估算
# ---------------------------------------------------------------------------


class TestUsageEstimation:
    """从序列化上下文估算 prompt / output tokens。"""

    @pytest.mark.asyncio
    async def test_usage_from_serialized_context(self):
        faux = faux_provider()
        faux.set_responses([faux_assistant_message("done")])

        tool = Tool(
            name="echo",
            description="Echo back text",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        context = Context(
            system="sys",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image", "url": None, "data": "abcd", "mediaType": "image/png"},
                    ],
                },
                faux_assistant_message("prior"),
                {
                    "role": "toolResult",
                    "toolCallId": "tool-1",
                    "toolName": "echo",
                    "content": [{"type": "text", "text": "tool out"}],
                },
            ],
            tools=[tool],
        )

        msg = await faux.provider.complete(faux.models[0], context)

        tool_dict = {
            "name": "echo",
            "description": "Echo back text",
            "inputSchema": tool.inputSchema,
        }
        expected_prompt = "\n\n".join([
            "system:sys",
            "user:hello\n[image:image/png:4]",
            "assistant:prior",
            "toolResult:echo\ntool out",
            f"tools:{json.dumps([tool_dict], ensure_ascii=False)}",
        ])
        expected_input = math.ceil(len(expected_prompt) / 4)
        expected_output = math.ceil(len("done") / 4)

        assert msg["usage"]["input"] == expected_input
        assert msg["usage"]["output"] == expected_output
        assert msg["usage"]["cacheRead"] == 0
        assert msg["usage"]["cacheWrite"] == 0
        assert msg["usage"]["totalTokens"] == expected_input + expected_output


# ---------------------------------------------------------------------------
# 流式事件
# ---------------------------------------------------------------------------


class TestStreamingEvents:
    """流式输出事件（delta / toolCallDelta / thinkingDelta / done / error）。"""

    @pytest.mark.asyncio
    async def test_text_deltas_and_done(self):
        faux = faux_provider()
        faux.set_responses([faux_assistant_message("hello world")])

        events = await _collect(await faux.provider.stream(faux.models[0], _context()))

        types = [e["type"] for e in events]
        assert types[-1] == "done"

        deltas = [e for e in events if e["type"] == "delta"]
        assert "".join(e["text"] for e in deltas) == "hello world"

        msg = events[-1]["message"]
        assert msg["content"] == [{"type": "text", "text": "hello world"}]
        assert msg["stopReason"] == "end"

    @pytest.mark.asyncio
    async def test_thinking_deltas(self):
        faux = faux_provider()
        faux.set_responses([faux_assistant_message([faux_thinking("reasoning here")])])

        events = await _collect(await faux.provider.stream(faux.models[0], _context()))

        thinking_deltas = [e for e in events if e["type"] == "thinkingDelta"]
        assert "".join(e["thinking"] for e in thinking_deltas) == "reasoning here"
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_tool_call_deltas_and_done(self):
        faux = faux_provider()
        faux.set_responses([
            faux_assistant_message(
                faux_tool_call("echo", {"text": "hi"}, tool_call_id="call-1"),
                stop_reason="toolCall",
            )
        ])

        events = await _collect(await faux.provider.stream(faux.models[0], _context()))

        tool_deltas = [e for e in events if e["type"] == "toolCallDelta"]
        assert len(tool_deltas) >= 1
        assert "".join(e["argsDelta"] for e in tool_deltas) == '{"text": "hi"}'
        assert tool_deltas[0]["toolCallId"] == "call-1"
        assert tool_deltas[0]["toolName"] == "echo"

        msg = events[-1]["message"]
        assert msg["stopReason"] == "toolCall"
        assert msg["content"] == [
            {"type": "toolCall", "toolCallId": "call-1", "toolName": "echo", "args": '{"text": "hi"}'}
        ]

    @pytest.mark.asyncio
    async def test_error_stop_reason_emits_error_event(self):
        faux = faux_provider()
        faux.set_responses([faux_assistant_message("", stop_reason="error", error_message="oops")])

        events = await _collect(await faux.provider.stream(faux.models[0], _context()))

        assert events[-1]["type"] == "error"
        assert events[-1]["reason"] == "error"
        assert events[-1]["error"]["errorMessage"] == "oops"

    @pytest.mark.asyncio
    async def test_abort_via_signal(self):
        faux = faux_provider(tokens_per_second=100)
        faux.set_responses([faux_assistant_message("x" * 200)])
        signal = asyncio.Event()

        stream = await faux.provider.stream(faux.models[0], _context(), {"signal": signal})

        async def consume():
            return await _collect(stream)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        signal.set()
        events = await task

        assert events[-1]["type"] == "error"
        assert events[-1]["reason"] == "aborted"
        assert events[-1]["error"]["stopReason"] == "aborted"
        assert events[-1]["error"]["errorMessage"] == "Request was aborted"


# ---------------------------------------------------------------------------
# Models 集成
# ---------------------------------------------------------------------------


class TestModelsIntegration:
    """通过 Models 注册表调度 Faux Provider。"""

    @pytest.mark.asyncio
    async def test_models_complete_and_stream(self):
        faux = faux_provider()
        models = Models()
        models.add_provider(faux.provider)
        faux.set_responses([faux_assistant_message("Hello!")])

        model = models.get_model("faux", "faux-1")
        assert model is not None

        msg = await models.complete(model, _context())
        assert msg["content"] == [{"type": "text", "text": "Hello!"}]

        faux.set_responses([faux_assistant_message("again")])
        events = await _collect(await models.stream(model, _context()))
        assert events[-1]["type"] == "done"
        assert events[-1]["message"]["content"] == [{"type": "text", "text": "again"}]
