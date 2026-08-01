"""
Integration tests for real API calls (requires API keys).

These tests are skipped when no API key is available.
Set OPENAI_API_KEY or DEEPSEEK_API_KEY to run.
"""

import os
import sys

import pytest

# Skip all integration tests if no API keys are set
pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")),
    reason="No API keys configured. Set OPENAI_API_KEY or DEEPSEEK_API_KEY.",
)


@pytest.mark.asyncio
async def test_deepseek_chat_stream():
    """Test DeepSeek streaming with a simple prompt."""
    from pi_ai import create_default_models, Context

    models = create_default_models()
    model = models.get_model("deepseek", "deepseek-chat")
    assert model is not None

    context = Context(
        messages=[
            {"role": "user", "content": "Say exactly: hello world"},
        ]
    )

    events = []
    async for event in await models.stream(model, context, {"maxTokens": 50}):
        events.append(event)

    # Should have at least a delta and a done event
    assert len(events) >= 2
    assert events[-1]["type"] == "done"
    msg = events[-1]["message"]
    assert msg["role"] == "assistant"
    assert msg["stopReason"] in ("end", "length")


@pytest.mark.asyncio
async def test_deepseek_chat_complete():
    """Test DeepSeek non-streaming completion."""
    from pi_ai import create_default_models, Context

    models = create_default_models()
    model = models.get_model("deepseek", "deepseek-chat")
    assert model is not None

    context = Context(
        messages=[
            {"role": "user", "content": 'Reply with just the word "ok"'},
        ]
    )

    msg = await models.complete(model, context, {"maxTokens": 20})
    assert msg["role"] == "assistant"
    assert len(msg["content"]) > 0
    text = "".join(
        block["text"] for block in msg["content"] if block["type"] == "text"
    )
    assert "ok" in text.lower()


@pytest.mark.asyncio
async def test_openai_chat_stream():
    """Test OpenAI streaming with a simple prompt."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    from pi_ai import create_default_models, Context

    models = create_default_models()
    model = models.get_model("openai", "gpt-4o-mini")
    assert model is not None

    context = Context(
        messages=[
            {"role": "user", "content": "Say exactly: hello world"},
        ]
    )

    events = []
    async for event in await models.stream(model, context, {"maxTokens": 50}):
        events.append(event)

    assert len(events) >= 2
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_tool_calling():
    """Test DeepSeek tool calling."""
    from pi_ai import create_default_models, Context, Tool

    models = create_default_models()
    model = models.get_model("deepseek", "deepseek-chat")
    assert model is not None

    tool = Tool(
        name="get_weather",
        description="Get the weather for a city",
        inputSchema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    )

    context = Context(
        messages=[
            {"role": "user", "content": "What is the weather in Beijing?"},
        ],
        tools=[tool],
    )

    msg = await models.complete(model, context, {"maxTokens": 100})
    assert msg["role"] == "assistant"

    # Check for tool calls in content
    tool_calls = [
        block for block in msg["content"] if block["type"] == "toolCall"
    ]
    if tool_calls:
        tc = tool_calls[0]
        assert tc["toolName"] == "get_weather"
        assert "Beijing" in tc["args"]
