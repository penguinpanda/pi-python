"""
Unit tests for _shared.py — shared helper functions.
"""

from pi_ai._types import (
    Message,
    Model,
    Tool,
)
from pi_ai.api._shared import (
    build_error_message,
    empty_usage,
    extract_text,
    parse_tool_arguments,
    to_openai_messages,
    to_openai_tools,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_model(
    model_id: str = "test-model",
    provider: str = "test-provider",
    api: str = "openai-completions",
    supports_images: bool = False,
) -> Model:
    return Model(
        id=model_id,
        provider=provider,
        api=api,
        name=model_id,
        input=["text"] + (["image"] if supports_images else []),
        output=["text"],
    )


# ---------------------------------------------------------------------------
# to_openai_messages
# ---------------------------------------------------------------------------


class TestToOpenaiMessages:
    """Message format conversion: SDK → OpenAI Chat Completions."""

    def test_system_message(self):
        messages: list[Message] = [
            {"role": "system", "content": "You are helpful."}  # type: ignore[typeddict-unknown-key]
        ]
        result = to_openai_messages(messages, _make_model())
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_user_message_string(self):
        messages: list[Message] = [
            {"role": "user", "content": "Hello"}  # type: ignore[typeddict-unknown-key]
        ]
        result = to_openai_messages(messages, _make_model())
        assert result == [{"role": "user", "content": "Hello"}]

    def test_user_message_multimodal(self):
        messages: list[Message] = [
            {  # type: ignore[typeddict-unknown-key]
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "url": "https://example.com/img.png",
                        "data": None,
                        "mime_type": None,
                    },
                ],
            }
        ]
        model = _make_model(supports_images=True)
        result = to_openai_messages(messages, model)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        parts = result[0]["content"]
        text_part = next(p for p in parts if p["type"] == "text")
        assert text_part["text"] == "What is this?"
        image_part = next(p for p in parts if p["type"] == "image_url")
        assert image_part["image_url"]["url"] == "https://example.com/img.png"

    def test_user_message_image_base64_data(self):
        messages: list[Message] = [
            {  # type: ignore[typeddict-unknown-key]
                "role": "user",
                "content": [
                    {"type": "image", "url": None, "data": "abc123", "mime_type": "image/jpeg"},
                ],
            }
        ]
        model = _make_model(supports_images=True)
        result = to_openai_messages(messages, model)
        parts = result[0]["content"]
        image_part = next(p for p in parts if p["type"] == "image_url")
        assert image_part["image_url"]["url"] == "data:image/jpeg;base64,abc123"

    def test_user_message_image_skipped_when_model_no_images(self):
        messages: list[Message] = [
            {  # type: ignore[typeddict-unknown-key]
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {
                        "type": "image",
                        "url": "https://example.com/img.png",
                        "data": None,
                        "mime_type": None,
                    },
                ],
            }
        ]
        model = _make_model(supports_images=False)
        result = to_openai_messages(messages, model)
        # image should be filtered out; only text remains
        assert result[0]["content"] == "Hi"

    def test_user_message_single_text_optimized(self):
        messages: list[Message] = [
            {  # type: ignore[typeddict-unknown-key]
                "role": "user",
                "content": [
                    {"type": "text", "text": "Just text"},
                ],
            }
        ]
        result = to_openai_messages(messages, _make_model())
        # single text part → optimized to plain string content
        assert result[0]["content"] == "Just text"

    def test_assistant_message_text(self):
        messages: list[Message] = [
            {  # type: ignore[typeddict-unknown-key]
                "role": "assistant",
                "content": [{"type": "text", "text": "I can help."}],
                "api": "openai-completions",
                "provider": "test",
                "model": "test-model",
            }
        ]
        result = to_openai_messages(messages, _make_model())
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "I can help."

    def test_assistant_message_with_tool_calls(self):
        messages: list[Message] = [
            {  # type: ignore[typeddict-unknown-key]
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {
                        "type": "toolCall",
                        "id": "call_1",
                        "name": "search",
                        "arguments": {"q": "test"},
                    },
                ],
                "api": "openai-completions",
                "provider": "test",
                "model": "test-model",
            }
        ]
        result = to_openai_messages(messages, _make_model())
        oai_msg = result[0]
        assert oai_msg["content"] == "Let me search."
        assert len(oai_msg["tool_calls"]) == 1
        tc = oai_msg["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == '{"q": "test"}'

    def test_assistant_thinking_skipped(self):
        messages: list[Message] = [
            {  # type: ignore[typeddict-unknown-key]
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Hmm..."},
                    {"type": "text", "text": "Answer."},
                ],
                "api": "openai-completions",
                "provider": "test",
                "model": "test-model",
            }
        ]
        result = to_openai_messages(messages, _make_model())
        # thinking should be skipped; only text remains
        assert result[0]["content"] == "Answer."
        assert "tool_calls" not in result[0]

    def test_tool_result_message(self):
        messages: list[Message] = [
            {  # type: ignore[typeddict-unknown-key]
                "role": "toolResult",
                "tool_call_id": "call_1",
                "tool_name": "search",
                "content": [{"type": "text", "text": "42 results found."}],
            }
        ]
        result = to_openai_messages(messages, _make_model())
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "42 results found."

    def test_tool_result_message_keeps_images_for_vision_models(self):
        """视觉模型经 completions 时，toolResult 中的图片不得被丢弃。"""
        messages: list[Message] = [
            {  # type: ignore[typeddict-unknown-key]
                "role": "toolResult",
                "tool_call_id": "call_1",
                "tool_name": "vision",
                "content": [
                    {"type": "text", "text": "screenshot:"},
                    {"type": "image", "url": None, "data": "aGVsbG8=", "mime_type": "image/png"},
                ],
            }
        ]
        vision_model = Model(
            id="gpt-5-chat-latest",
            provider="openai",
            api="openai-completions",
            input=["text", "image"],
            output=["text"],
        )
        result = to_openai_messages(messages, vision_model)
        assert result[0]["role"] == "tool"
        assert result[0]["content"] == [
            {"type": "text", "text": "screenshot:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ]

    def test_multiple_messages(self):
        messages: list[Message] = [
            {"role": "system", "content": "System prompt"},  # type: ignore[typeddict-unknown-key]
            {"role": "user", "content": "User question"},  # type: ignore[typeddict-unknown-key]
        ]
        result = to_openai_messages(messages, _make_model())
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"

    def test_agent_message_skipped_safely(self):
        """AgentMessage（未知 role）不崩溃；由调用方自定义转换。"""
        messages: list[Message] = [
            {"role": "user", "content": "Hi"},  # type: ignore[typeddict-unknown-key]
            {
                "role": "planner",
                "content": "Plan: search then answer",
            },  # type: ignore[typeddict-unknown-key]
        ]
        result = to_openai_messages(messages, _make_model())
        # 未知 role 安全跳过，不抛异常
        assert result == [{"role": "user", "content": "Hi"}]


# ---------------------------------------------------------------------------
# to_openai_tools
# ---------------------------------------------------------------------------


class TestToOpenaiTools:
    """Tool schema conversion: SDK Tool → OpenAI function tool."""

    def test_empty_tools(self):
        assert to_openai_tools([]) == []

    def test_single_tool(self):
        tools = [
            Tool(
                name="search",
                description="Search the web",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]
        result = to_openai_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        fn = result[0]["function"]
        assert fn["name"] == "search"
        assert fn["description"] == "Search the web"
        assert fn["parameters"] == {"type": "object", "properties": {"q": {"type": "string"}}}

    def test_multiple_tools(self):
        tools = [
            Tool(name="t1", description="d1", input_schema={}),
            Tool(name="t2", description="d2", input_schema={}),
        ]
        result = to_openai_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "t1"
        assert result[1]["function"]["name"] == "t2"


# ---------------------------------------------------------------------------
# empty_usage
# ---------------------------------------------------------------------------


class TestEmptyUsage:
    """Construct zeroed Usage."""

    def test_all_zero(self):
        usage = empty_usage()
        assert usage["input"] == 0
        assert usage["output"] == 0
        assert usage["cache_read"] == 0
        assert usage["cache_write"] == 0
        assert usage["total_tokens"] == 0

    def test_cost_dict(self):
        usage = empty_usage()
        assert "cost" in usage
        cost = usage["cost"]
        assert cost["input"] == 0
        assert cost["output"] == 0
        assert cost["cache_read"] == 0
        assert cost["cache_write"] == 0
        assert cost["total"] == 0


# ---------------------------------------------------------------------------
# build_error_message
# ---------------------------------------------------------------------------


class TestBuildErrorMessage:
    """Error AssistantMessage construction."""

    def test_structure(self):
        model = _make_model("gpt-4o", "openai", "openai-responses")
        exc = ValueError("API key missing")
        msg = build_error_message(model, exc)
        assert msg["role"] == "assistant"
        assert msg["content"] == []
        assert msg["api"] == "openai-responses"
        assert msg["provider"] == "openai"
        assert msg["model"] == "gpt-4o"
        assert msg["stop_reason"] == "error"
        assert msg["error_message"] == "API key missing"
        assert msg["usage"]["input"] == 0
        assert msg["timestamp"] > 0

    def test_different_exception_types(self):
        model = _make_model()
        for exc in [RuntimeError("err"), ConnectionError("conn"), TimeoutError("timeout")]:
            msg = build_error_message(model, exc)
            assert msg["error_message"] == str(exc)


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------


class TestExtractText:
    """Pure-text extraction from ContentBlock list."""

    def test_text_only(self):
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert extract_text(content) == "Hello\nWorld"

    def test_thinking_only(self):
        content = [
            {"type": "thinking", "thinking": "Let me think..."},
        ]
        assert extract_text(content) == "Let me think..."

    def test_mixed(self):
        content = [
            {"type": "thinking", "thinking": "Hmm"},
            {"type": "text", "text": "Result"},
            {"type": "thinking", "thinking": "Done"},
        ]
        assert extract_text(content) == "Hmm\nResult\nDone"

    def test_empty_list(self):
        assert extract_text([]) == ""

    def test_tool_call_ignored(self):
        content = [
            {"type": "toolCall", "id": "c1", "name": "search", "arguments": {}},
            {"type": "text", "text": "Answer"},
        ]
        assert extract_text(content) == "Answer"


# ---------------------------------------------------------------------------
# parse_tool_arguments
# ---------------------------------------------------------------------------


class TestParseToolArguments:
    """Streaming tool-call raw JSON parsing."""

    def test_empty_string(self):
        assert parse_tool_arguments("") == {}

    def test_whitespace_only(self):
        assert parse_tool_arguments("   ") == {}

    def test_valid_object(self):
        assert parse_tool_arguments('{"q":"test"}') == {"q": "test"}

    def test_non_object_json(self):
        assert parse_tool_arguments("[1,2,3]") == {"value": [1, 2, 3]}

    def test_invalid_json(self):
        assert parse_tool_arguments('{"q":') == {"_error": "Invalid JSON arguments"}
