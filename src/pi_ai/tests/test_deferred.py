"""deferred responses 基础层测试（P1：pi_ai 部分）。"""

from __future__ import annotations

import pytest

from pi_ai import Models
from pi_ai.provider import create_provider
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_ai.types import Context, Model, Tool
from pi_ai.utils.deferred_tools import split_deferred_tools


def _model(model_id: str = "faux-1") -> Model:
    return Model(
        id=model_id,
        provider="faux",
        api="openai-completions",
        name=model_id,
    )


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
    )


class TestSplitDeferredTools:
    def test_no_deferred_when_disabled(self):
        context = Context(messages=[], tools=[_tool("a"), _tool("b")])
        immediate, deferred = split_deferred_tools(context, enabled=False)
        assert [tool.name for tool in immediate] == ["a", "b"]
        assert deferred == {}

    def test_defers_tools_added_by_tool_results(self):
        context = Context(
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "name": "a", "arguments": {}, "id": "c1"}],
                    "timestamp": 1,
                },
                {
                    "role": "toolResult",
                    "tool_call_id": "c1",
                    "tool_name": "a",
                    "content": [{"type": "text", "text": "ok"}],
                    "is_error": False,
                    "timestamp": 2,
                    "added_tool_names": ["b"],
                },
            ],
            tools=[_tool("a"), _tool("b")],
        )
        immediate, deferred = split_deferred_tools(context, enabled=True)
        assert [tool.name for tool in immediate] == ["a"]
        assert list(deferred.keys()) == ["b"]


class TestFauxDeferred:
    @pytest.mark.asyncio
    async def test_fetch_and_cancel(self):
        core = faux_provider()
        models = Models()
        models.add_provider(core.provider)
        model = _model()
        core.set_deferred_response("d1", faux_assistant_message("done"))

        assert models.supports_deferred(model) is True
        handle = {
            "provider": "faux",
            "model_id": model.id,
            "api": "openai-completions",
            "id": "d1",
        }
        message = await models.fetch_deferred(model, handle)
        assert message["content"] == [{"type": "text", "text": "done"}]

        core.set_deferred_response("d2", faux_assistant_message("x"))
        await models.cancel_deferred(
            model,
            {
                "provider": "faux",
                "model_id": model.id,
                "api": "openai-completions",
                "id": "d2",
            },
        )
        assert "d2" in core._cancelled_deferred

    @pytest.mark.asyncio
    async def test_missing_deferred_raises(self):
        core = faux_provider()
        models = Models()
        models.add_provider(core.provider)
        handle = {
            "provider": "faux",
            "model_id": "faux-1",
            "api": "openai-completions",
            "id": "missing",
        }
        with pytest.raises(RuntimeError, match="not found"):
            await models.fetch_deferred(_model(), handle)

    @pytest.mark.asyncio
    async def test_unsupported_provider_raises(self):
        plain_model = Model(
            id="m",
            provider="plain",
            api="openai-completions",
            name="m",
        )
        provider = create_provider(
            id="plain",
            name="Plain",
            auth=None,
            models=[plain_model],
        )
        models = Models()
        models.add_provider(provider)
        assert models.supports_deferred(plain_model) is False
        with pytest.raises(RuntimeError, match="does not support deferred"):
            await models.fetch_deferred(
                plain_model,
                {
                    "provider": "plain",
                    "model_id": "m",
                    "api": "openai-completions",
                    "id": "x",
                },
            )

    @pytest.mark.asyncio
    async def test_streamed_message_carries_deferred_handle(self):
        core = faux_provider()
        handle = {
            "provider": "faux",
            "model_id": "faux-1",
            "api": "openai-completions",
            "id": "d1",
        }
        core.set_responses(
            [
                {
                    **faux_assistant_message("pending"),
                    "stop_reason": "deferred",
                    "deferred": handle,
                }
            ]
        )
        models = Models()
        models.add_provider(core.provider)
        model = _model()
        message = await models.complete_simple(model, Context(messages=[]))
        assert message["stop_reason"] == "deferred"
        assert message["deferred"] == handle
