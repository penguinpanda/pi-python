"""
Unit tests for responses.py — Responses API helpers.
"""

import json
from pi_ai._types import Context, Model, Tool
from pi_ai.api.responses import _to_responses_input


def _make_model(model_id="gpt-4o", provider="openai", api="openai-responses"):
    return Model(
        id=model_id, provider=provider, api=api, name=model_id, input=["text"], output=["text"]
    )


def _make_deepseek_responses_model() -> Model:
    return Model(
        id="deepseek-v4-flash",
        provider="deepseek",
        api="openai-responses",
        name="DeepSeek V4 Flash",
        input=["text"],
        output=["text"],
        reasoning=True,
        thinking_level_map={"high": "high", "max": "max"},
        compat={
            "supportsWebSearch": True,
            "supportsExplicitPromptCacheMode": False,
            "supportsLongCacheRetention": False,
        },
    )


class TestToResponsesInput:
    def test_system_message(self):
        # 顶层 system prompt 由 requests.create 的 instructions 承载，不进 input。
        result = _to_responses_input([])
        assert result == []

    def test_user_string(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = _to_responses_input(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_user_multimodal_text_and_image_url(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe:"},
                    {
                        "type": "image",
                        "url": "https://example.com/pic.png",
                        "data": None,
                        "mime_type": None,
                    },
                ],
            }
        ]
        result = _to_responses_input(messages)
        parts = result[0]["content"]
        assert parts[0] == {"type": "input_text", "text": "Describe:"}
        assert parts[1] == {"type": "input_image", "image_url": "https://example.com/pic.png"}

    def test_user_image_base64(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": None, "data": "abc123", "mime_type": "image/jpeg"},
                ],
            }
        ]
        result = _to_responses_input(messages)
        img = result[0]["content"][0]
        assert img["image_url"] == "data:image/jpeg;base64,abc123"

    def test_assistant_message(self):
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Answer"}],
                "api": "openai-responses",
                "provider": "openai",
                "model": "gpt-4o",
            }
        ]
        result = _to_responses_input(messages)
        # 文本块展开为独立的 message item（含 fallback id）。
        assert result[0] == {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Answer", "annotations": []}],
            "status": "completed",
            "id": "msg_pi_0",
        }

    def test_tool_result_message(self):
        messages = [
            {
                "role": "toolResult",
                "tool_call_id": "call_1",
                "tool_name": "search",
                "content": [{"type": "text", "text": "42 results"}],
            }
        ]
        result = _to_responses_input(messages)
        assert result[0]["type"] == "function_call_output"
        assert result[0]["call_id"] == "call_1"
        assert result[0]["output"] == "42 results"

    def test_system_in_messages(self):
        messages = [{"role": "system", "content": "Mid note"}]
        result = _to_responses_input(messages)
        assert result == [{"role": "system", "content": "Mid note"}]

    def test_combined_system_and_messages(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = _to_responses_input(messages)
        assert result == [{"role": "user", "content": "Hi"}]

    def test_user_image_filtered_when_model_no_images(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {
                        "type": "image",
                        "url": "https://example.com/pic.png",
                        "data": None,
                        "mime_type": None,
                    },
                ],
            }
        ]
        # _make_model() default input=['text'] — no image capability
        result = _to_responses_input(messages, _make_model())
        parts = result[0]["content"]
        # image should be filtered out; only text remains
        assert len(parts) == 1
        assert parts[0] == {"type": "input_text", "text": "Hi"}

    def test_user_image_kept_when_model_supports_images(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://example.com/pic.png",
                        "data": None,
                        "mime_type": None,
                    },
                ],
            }
        ]
        model = _make_model()
        model.input = ["text", "image"]
        result = _to_responses_input(messages, model)
        img = result[0]["content"][0]
        assert img["type"] == "input_image"
        assert img["image_url"] == "https://example.com/pic.png"


class TestResponsesNormalizeToolCallId:
    """Responses 系 fc_ item id 规范化（normalize_responses_tool_call_id）。"""

    def test_pipe_id_kept_for_allowed_provider(self):
        from pi_ai.api.transform_messages import normalize_responses_tool_call_id

        model = _make_model(model_id="gpt-5", provider="openai", api="openai-responses")
        source = {
            "role": "assistant",
            "provider": "openai",
            "api": "openai-responses",
            "model": "gpt-5",
        }
        result = normalize_responses_tool_call_id("call_1|fc_abc", model, source)
        assert result == "call_1|fc_abc"

    def test_foreign_item_id_hashed_to_fc(self):
        from pi_ai.api.transform_messages import normalize_responses_tool_call_id

        model = _make_model(model_id="gpt-5", provider="openai", api="openai-responses")
        # 跨模型（source 为 completions provider）→ item id 用 short_hash 重建。
        source = {
            "role": "assistant",
            "provider": "deepseek",
            "api": "openai-completions",
            "model": "deepseek-chat",
        }
        result = normalize_responses_tool_call_id("call_1|" + "x" * 200, model, source)
        call_id, _, item_id = result.partition("|")
        assert call_id == "call_1"
        assert item_id.startswith("fc_")
        assert len(item_id) <= 64

    def test_pipe_id_kept_for_deepseek(self):
        from pi_ai.api.transform_messages import normalize_responses_tool_call_id

        model = _make_model(
            model_id="deepseek-v4-flash", provider="deepseek", api="openai-responses"
        )
        source = {
            "role": "assistant",
            "provider": "deepseek",
            "api": "openai-responses",
            "model": "deepseek-v4-flash",
        }
        result = normalize_responses_tool_call_id("call_1|fc_abc", model, source)
        assert result == "call_1|fc_abc"

    def test_non_pipe_id_normalized_single_part(self):
        from pi_ai.api.transform_messages import normalize_responses_tool_call_id

        model = _make_model(provider="openai", api="openai-responses")
        source = {
            "role": "assistant",
            "provider": "openai",
            "api": "openai-responses",
            "model": "gpt-5",
        }
        result = normalize_responses_tool_call_id("call+1/foo!", model, source)
        assert result == "call_1_foo"

    def test_disallowed_provider_degrades_to_single_part(self):
        from pi_ai.api.transform_messages import normalize_responses_tool_call_id

        model = _make_model(provider="qwen", api="openai-completions")
        source = {
            "role": "assistant",
            "provider": "openai",
            "api": "openai-responses",
            "model": "gpt-5",
        }
        result = normalize_responses_tool_call_id("call_1|fc_abc", model, source)
        assert result == "call_1_fc_abc"


class TestResponsesReasoningReplay:
    """Responses 侧 reasoning / text / function_call 历史回放。"""

    def test_thinking_signature_replayed_as_reasoning_item(self):
        model = _make_model()
        reasoning_item = {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "think"}],
            "encrypted_content": "enc",
        }
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "think",
                        "thinking_signature": json.dumps(reasoning_item),
                    },
                    {"type": "text", "text": "Answer"},
                ],
                "api": "openai-responses",
                "provider": "openai",
                "model": "gpt-4o",
            }
        ]
        result = _to_responses_input(messages, model)
        # thinking → 独立 reasoning item（原样回放）。
        assert result[0] == reasoning_item
        # text → message item。
        assert result[1]["type"] == "message"
        assert result[1]["content"] == [
            {"type": "output_text", "text": "Answer", "annotations": []}
        ]

    def test_thinking_without_signature_skipped(self):
        model = _make_model()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "no signature"},
                ],
                "api": "openai-responses",
                "provider": "openai",
                "model": "gpt-4o",
            }
        ]
        result = _to_responses_input(messages, model)
        assert result == []

    def test_text_signature_id_and_phase_replayed(self):
        model = _make_model()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Answer",
                        "text_signature": '{"v":1,"id":"msg_abc","phase":"final_answer"}',
                    },
                ],
                "api": "openai-responses",
                "provider": "openai",
                "model": "gpt-4o",
            }
        ]
        result = _to_responses_input(messages, model)
        assert result[0]["id"] == "msg_abc"
        assert result[0]["phase"] == "final_answer"

    def test_text_signature_long_id_hashed(self):
        model = _make_model()
        long_id = "m" * 100
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Answer", "text_signature": long_id},
                ],
                "api": "openai-responses",
                "provider": "openai",
                "model": "gpt-4o",
            }
        ]
        result = _to_responses_input(messages, model)
        assert result[0]["id"].startswith("msg_")
        assert len(result[0]["id"]) < 20

    def test_function_call_item_id_kept_same_model(self):
        model = _make_model(model_id="gpt-4o", provider="openai", api="openai-responses")
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call_1|fc_123",
                        "name": "bash",
                        "raw_arguments": "",
                        "arguments": {"command": "ls"},
                    },
                ],
                "api": "openai-responses",
                "provider": "openai",
                "model": "gpt-4o",
            }
        ]
        result = _to_responses_input(messages, model)
        assert result[0] == {
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_1",
            "name": "bash",
            "arguments": '{"command": "ls"}',
        }

    def test_function_call_item_id_dropped_different_model(self):
        # 同 provider/api、不同 model → fc_ item id 被省略（避开 pairing 校验）。
        model = _make_model(model_id="gpt-5", provider="openai", api="openai-responses")
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call_1|fc_123",
                        "name": "bash",
                        "raw_arguments": "",
                        "arguments": {},
                    },
                ],
                "api": "openai-responses",
                "provider": "openai",
                "model": "gpt-4o",
            }
        ]
        result = _to_responses_input(messages, model)
        assert result[0] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "bash",
            "arguments": "{}",
        }

    def test_tool_result_pipe_call_id_split(self):
        model = _make_model()
        messages = [
            {
                "role": "toolResult",
                "tool_call_id": "call_1|fc_123",
                "tool_name": "bash",
                "content": [{"type": "text", "text": "out"}],
                "is_error": False,
            }
        ]
        result = _to_responses_input(messages, model)
        assert result[0] == {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "out",
        }

    def test_tool_result_empty_output_placeholder(self):
        model = _make_model()
        messages = [
            {
                "role": "toolResult",
                "tool_call_id": "call_1",
                "tool_name": "bash",
                "content": [],
                "is_error": False,
            }
        ]
        result = _to_responses_input(messages, model)
        assert result[0]["output"] == "(no tool output)"


# ===========================================================================
# Responses API 流式主循环
#
# 通过 patch 掉 _create_client 返回 mock 客户端，
# 完全离线测试 Responses Event → SDK Event 的适配逻辑。
#
# 注意：responses_stream() 会立即返回并后台调度 _run()，
# 因此必须在 patch 生效期间同时完成流的创建与消费。
# ===========================================================================

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pi_ai.api.responses import _create_client, responses_stream


def _async_iter(items):
    """将一个列表包装成异步可迭代对象（模拟 Responses Event 流）。"""

    async def gen():
        for item in items:
            yield item

    return gen()


def _event(event_type, **kwargs):
    """构造一个假的 Responses Event。"""
    return SimpleNamespace(type=event_type, **kwargs)


def _mock_client(events):
    """构造一个 mock 的 AsyncOpenAI 客户端，responses.create() 返回事件流。"""
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_async_iter(events))
    return client


async def _collect_events(
    model, context, client, options=None, base_url="https://api.openai.com/v1"
):
    """在 patch 生效期间创建并消费流，返回 (events, stream)。"""
    with patch("pi_ai.api.responses._create_client", return_value=client):
        stream = await responses_stream(model, context, "sk-test", base_url, options)
        events = [e async for e in stream]
        return events, stream


class TestResponsesCreateClient:
    """responses.py 的 _create_client() 客户端工厂。"""

    def test_no_base_url_not_passed(self):
        with patch("pi_ai.api.responses.AsyncOpenAI") as mock_openai:
            _create_client("sk-test", base_url="")

        mock_openai.assert_called_once()
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["api_key"] == "sk-test"
        assert "base_url" not in kwargs
        assert kwargs["max_retries"] == 2

    def test_base_url_stripped(self):
        with patch("pi_ai.api.responses.AsyncOpenAI") as mock_openai:
            _create_client("sk-test", base_url="https://api.openai.com/v1/")

        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"] == "https://api.openai.com/v1"

    def test_default_timeout(self):
        with patch("pi_ai.api.responses.AsyncOpenAI") as mock_openai:
            _create_client("sk-test", base_url="https://api.openai.com/v1")

        kwargs = mock_openai.call_args.kwargs
        assert isinstance(kwargs["timeout"], httpx.Timeout)
        assert kwargs["timeout"].connect == 180.0

    def test_custom_max_retries(self):
        with patch("pi_ai.api.responses.AsyncOpenAI") as mock_openai:
            _create_client("sk-test", base_url="https://api.openai.com/v1", max_retries=5)

        kwargs = mock_openai.call_args.kwargs
        assert kwargs["max_retries"] == 5


class TestResponsesStream:
    """responses_stream() 流式主循环。"""

    @pytest.mark.asyncio
    async def test_text_delta(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.output_text.delta", delta="Hello"),
            _event("response.output_text.delta", delta=" world"),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="Hello world",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == [
            "start",
            "text_start",
            "text_delta",
            "text_delta",
            "text_end",
            "done",
        ]
        assert collected[2]["delta"] == "Hello"
        assert collected[3]["delta"] == " world"

        msg = collected[-1]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"type": "text", "text": "Hello world"}]
        assert msg["stop_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_thinking_events(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event(
                "response.reasoning_summary_part.added",
                part=SimpleNamespace(
                    type="summary_text",
                    text="Let me think",
                ),
            ),
            _event("response.reasoning_text.delta", delta=" step by step"),
            _event("response.output_text.delta", delta="Answer"),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="Answer",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == [
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_end",
            "done",
        ]
        thinking = [e for e in collected if e["type"] == "thinking_delta"]
        assert [t["delta"] for t in thinking] == ["Let me think", " step by step"]

        msg = collected[-1]["message"]
        # Thinking 块在 Text 块之前。
        assert msg["content"] == [
            {"type": "thinking", "thinking": "Let me think step by step"},
            {"type": "text", "text": "Answer"},
        ]

    @pytest.mark.asyncio
    async def test_thinking_summary_non_summary_ignored(self):
        """summary part 类型不是 summary_text 时应被忽略。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event(
                "response.reasoning_summary_part.added",
                part=SimpleNamespace(
                    type="other",
                    text="ignored",
                ),
            ),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["start", "done"]
        assert collected[-1]["message"]["content"] == []

    @pytest.mark.asyncio
    async def test_tool_call_flow(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "weather?"}])
        events = [
            _event(
                "response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="get_weather",
                ),
            ),
            _event("response.function_call_arguments.delta", delta='{"city":'),
            _event("response.function_call_arguments.delta", delta='"Beijing"}'),
            _event("response.function_call_arguments.done"),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == [
            "start",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
            "done",
        ]
        tool_deltas = [e for e in collected if e["type"] == "toolcall_delta"]
        assert len(tool_deltas) == 2
        assert tool_deltas[0]["delta"] == '{"city":'
        assert tool_deltas[1]["delta"] == '"Beijing"}'

        msg = collected[-1]["message"]
        assert msg["content"] == [
            {
                "type": "toolCall",
                "id": "call_1",
                "name": "get_weather",
                "raw_arguments": '{"city":"Beijing"}',
                "arguments": {"city": "Beijing"},
            }
        ]

    @pytest.mark.asyncio
    async def test_output_item_non_function_ignored(self):
        """output_item.added 但 item 不是 function_call 时应被忽略。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event(
                "response.output_item.added",
                item=SimpleNamespace(
                    type="message",
                    call_id="",
                    name="",
                ),
            ),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["start", "done"]
        assert collected[-1]["message"]["content"] == []

    @pytest.mark.asyncio
    async def test_completed_usage(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="Hi",
                    usage=SimpleNamespace(
                        input_tokens=7,
                        output_tokens=3,
                        total_tokens=10,
                        input_tokens_details=SimpleNamespace(
                            cached_tokens=4,
                            cache_write_tokens=2,
                        ),
                        output_tokens_details=SimpleNamespace(reasoning_tokens=1),
                    ),
                ),
            ),
        ]
        client = _mock_client(events)

        collected, stream = await _collect_events(model, context, client)
        msg = await stream.result()
        assert msg["content"] == [{"type": "text", "text": "Hi"}]
        assert msg["usage"] == {
            # OpenAI 的 input_tokens 已包含缓存 token：扣减后不再双重计费。
            "input": 1,
            "output": 3,
            "cache_read": 4,
            "cache_write": 2,
            "total_tokens": 10,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
            "reasoning": 1,
        }

    @pytest.mark.asyncio
    async def test_completed_output_text_no_duplication_after_tool_call(self):
        """文本→toolCall→文本 的多段响应不得把整段 output_text 重复进首个文本块。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        fc_item = SimpleNamespace(
            type="function_call",
            call_id="call_1",
            id="fc_abc",
            name="tool_a",
            arguments="{}",
        )
        events = [
            _event("response.output_text.delta", delta="Before "),
            _event("response.output_item.added", item=fc_item),
            _event("response.function_call_arguments.delta", delta='{"q":'),
            _event("response.function_call_arguments.done"),
            _event("response.output_item.done", item=fc_item),
            _event("response.output_text.delta", delta="After"),
            _event(
                "response.completed",
                response=SimpleNamespace(output_text="Before After", usage=None),
            ),
        ]
        client = _mock_client(events)

        collected, stream = await _collect_events(model, context, client)
        msg = await stream.result()
        assert [b["type"] for b in msg["content"]] == ["text", "toolCall", "text"]
        assert msg["content"][0]["text"] == "Before "
        assert msg["content"][2]["text"] == "After"

    @pytest.mark.asyncio
    async def test_completed_no_response(self):
        """completed 事件没有 response 对象时不应崩溃。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [_event("response.completed", response=None)]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["start", "done"]
        msg = collected[-1]["message"]
        assert msg["content"] == []
        assert msg["usage"]["total_tokens"] == 0

    @pytest.mark.asyncio
    async def test_output_text_done(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.output_text.done", text="Hello"),
            _event(
                "response.completed",
                response=SimpleNamespace(output_text="", usage=None),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        msg = collected[-1]["message"]
        assert msg["content"] == [{"type": "text", "text": "Hello"}]

    @pytest.mark.asyncio
    async def test_reasoning_text_done(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.reasoning_text.done", text="Let me think"),
            _event(
                "response.completed",
                response=SimpleNamespace(output_text="", usage=None),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        msg = collected[-1]["message"]
        assert msg["content"] == [{"type": "thinking", "thinking": "Let me think"}]

    @pytest.mark.asyncio
    async def test_incomplete_max_output_tokens(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event(
                "response.incomplete",
                response=SimpleNamespace(
                    output_text="partial",
                    usage=SimpleNamespace(
                        input_tokens=7,
                        output_tokens=3,
                        total_tokens=10,
                        input_tokens_details=None,
                        output_tokens_details=None,
                    ),
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert collected[-1]["type"] == "done"
        assert collected[-1]["reason"] == "length"
        assert collected[-1]["message"]["stop_reason"] == "length"
        assert collected[-1]["message"]["usage"]["total_tokens"] == 10

    @pytest.mark.asyncio
    async def test_failed_event(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event(
                "response.failed",
                response=SimpleNamespace(
                    output_text="",
                    usage=None,
                    error=SimpleNamespace(message="boom"),
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["start", "error"]
        assert collected[-1]["error"]["error_message"] == "boom"

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([])

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["start", "done"]
        assert collected[-1]["message"]["content"] == []

    @pytest.mark.asyncio
    async def test_request_kwargs(self):
        model = _make_model()
        tool = Tool(
            name="get_weather",
            description="Get weather",
            input_schema={"type": "object", "properties": {}},
        )
        context = Context(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[tool],
            system_prompt="Be helpful",
        )
        options = {"temperature": 0.5, "max_tokens": 100}
        client = _mock_client([_event("response.completed", response=None)])
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(
                model, context, "sk-test", "https://api.openai.com/v1", options
            )
            [e async for e in stream]

        create = client.responses.create
        create.assert_called_once()
        kwargs = create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["stream"] is True
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_output_tokens"] == 100
        assert kwargs["instructions"] == "Be helpful"
        # System prompt 走 instructions，不进 input。
        assert kwargs["input"] == [{"role": "user", "content": "Hi"}]
        assert kwargs["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    @pytest.mark.asyncio
    async def test_deepseek_request_kwargs_default_web_search(self):
        model = _make_deepseek_responses_model()
        tool = Tool(
            name="get_weather",
            description="Get weather",
            input_schema={"type": "object", "properties": {}},
        )
        context = Context(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[tool],
            system_prompt="Be helpful",
        )
        options = {
            "reasoning": "high",
            "max_tokens": 100,
            "session_id": "s-123",
            "cache_retention": "long",
        }
        client = _mock_client([_event("response.completed", response=None)])
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(
                model, context, "sk-test", "https://api.deepseek.com", options
            )
            [e async for e in stream]

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["instructions"] == "Be helpful"
        assert kwargs["input"] == [{"role": "user", "content": "Hi"}]
        assert kwargs["max_output_tokens"] == 100
        assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
        assert kwargs["include"] == ["reasoning.encrypted_content"]
        assert kwargs["store"] is False
        assert kwargs["tools"] == [
            {"type": "web_search"},
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        ]
        assert "prompt_cache_key" not in kwargs
        assert "prompt_cache_retention" not in kwargs

    @pytest.mark.asyncio
    async def test_deepseek_web_search_explicit_off(self):
        model = _make_deepseek_responses_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_event("response.completed", response=None)])
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(
                model,
                context,
                "sk-test",
                "https://api.deepseek.com",
                {"web_search": False},
            )
            [e async for e in stream]

        kwargs = client.responses.create.call_args.kwargs
        assert "tools" not in kwargs or kwargs["tools"] == []

    @pytest.mark.asyncio
    async def test_options_max_retries_timeout_forwarded(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_event("response.completed", response=None)])

        captured: dict[str, object] = {}

        def _spy(
            api_key: str,
            base_url: str = "",
            *,
            timeout: float = 180.0,
            max_retries: int = 2,
            headers=None,
        ):
            captured["timeout"] = timeout
            captured["max_retries"] = max_retries
            return client

        with patch("pi_ai.api.responses._create_client", side_effect=_spy):
            stream = await responses_stream(
                model,
                context,
                "sk-test",
                "https://api.openai.com/v1",
                options={"max_retries": 5, "timeout_ms": 45000},
            )
            [e async for e in stream]

        assert captured["max_retries"] == 5
        assert captured["timeout"] == 45.0

    @pytest.mark.asyncio
    async def test_options_defaults_when_absent(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_event("response.completed", response=None)])

        captured: dict[str, object] = {}

        def _spy(
            api_key: str,
            base_url: str = "",
            *,
            timeout: float = 180.0,
            max_retries: int = 2,
            headers=None,
        ):
            captured["timeout"] = timeout
            captured["max_retries"] = max_retries
            return client

        with patch("pi_ai.api.responses._create_client", side_effect=_spy):
            stream = await responses_stream(
                model,
                context,
                "sk-test",
                "https://api.openai.com/v1",
            )
            [e async for e in stream]

        assert captured["max_retries"] == 2
        assert captured["timeout"] == 180.0

    @pytest.mark.asyncio
    async def test_error_event(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=RuntimeError("boom"))

        collected, stream = await _collect_events(model, context, client)
        assert collected[-1]["type"] == "error"
        assert collected[-1]["reason"] == "error"
        err = collected[-1]["error"]
        assert err["role"] == "assistant"
        assert err["error_message"] == "boom"
        assert err["stop_reason"] == "error"

        # result() 返回携带错误的 AssistantMessage。
        msg = await stream.result()
        assert msg["error_message"] == "boom"

    @pytest.mark.asyncio
    async def test_cancelled_error(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=asyncio.CancelledError())
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(model, context, "sk-test", "https://api.openai.com/v1")
            # 让后台协程有机会执行。
            await asyncio.sleep(0.01)
            with pytest.raises(asyncio.CancelledError):
                await stream.result()


class TestResponsesStreamReasoningCapture:
    """输出侧：reasoning item 捕获到 thinking_signature、工具调用双段 ID。"""

    @pytest.mark.asyncio
    async def test_output_item_done_captures_thinking_signature(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        reasoning_item = SimpleNamespace(
            type="reasoning",
            id="rs_1",
            summary=[SimpleNamespace(type="summary_text", text="Let me think")],
            content=[],
            encrypted_content="encrypted-data",
        )
        events = [
            _event(
                "response.reasoning_summary_part.added",
                part=SimpleNamespace(
                    type="summary_text",
                    text="Let me think",
                ),
            ),
            _event("response.output_item.done", item=reasoning_item),
            _event("response.output_text.delta", delta="Answer"),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="Answer",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        msg = collected[-1]["message"]
        thinking = msg["content"][0]
        assert thinking["type"] == "thinking"
        assert thinking["thinking"] == "Let me think"
        # thinking_signature 存有完整 reasoning item，供后续轮次回放。
        parsed = json.loads(thinking["thinking_signature"])
        assert parsed["id"] == "rs_1"
        assert parsed["type"] == "reasoning"
        assert parsed["encrypted_content"] == "encrypted-data"
        assert parsed["summary"][0]["text"] == "Let me think"

    @pytest.mark.asyncio
    async def test_tool_call_stores_pipe_id(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "weather?"}])
        events = [
            _event(
                "response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    id="fc_abc",
                    name="get_weather",
                ),
            ),
            _event("response.function_call_arguments.delta", delta='{"city":'),
            _event("response.function_call_arguments.delta", delta='"Beijing"}'),
            _event("response.function_call_arguments.done"),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        msg = collected[-1]["message"]
        # 输出侧保存完整双段 ID（call_id|fc_item_id）。
        assert msg["content"][0]["id"] == "call_1|fc_abc"

    @pytest.mark.asyncio
    async def test_tool_call_without_item_id_keeps_call_id(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "weather?"}])
        events = [
            _event(
                "response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    name="get_weather",
                ),
            ),
            _event("response.function_call_arguments.delta", delta="{}"),
            _event("response.function_call_arguments.done"),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        msg = collected[-1]["message"]
        assert msg["content"][0]["id"] == "call_1"

    @pytest.mark.asyncio
    async def test_text_before_tool_call_not_duplicated_on_completed(self):
        """toolCall 后的 response.completed 不应把 output_text 再追加一次。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.output_text.delta", delta="Let me check"),
            _event(
                "response.output_item.added",
                item=SimpleNamespace(
                    type="function_call",
                    call_id="call_1",
                    id="fc_abc",
                    name="bash",
                ),
            ),
            _event("response.function_call_arguments.delta", delta="{}"),
            _event("response.function_call_arguments.done"),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="Let me check",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        msg = collected[-1]["message"]
        assert [b.get("type") for b in msg["content"]] == ["text", "toolCall"]
        assert len([b for b in msg["content"] if b.get("type") == "text"]) == 1


class TestResponsesWebSearch:
    """DeepSeek Responses 服务端 web_search 捕获与 stateless 回放。"""

    @pytest.mark.asyncio
    async def test_capture_and_replay(self):
        model = _make_deepseek_responses_model()
        context = Context(messages=[{"role": "user", "content": "search?"}])
        events = [
            _event(
                "response.output_item.done",
                item=SimpleNamespace(
                    type="web_search_call",
                    id="ws_1",
                    status="completed",
                ),
            ),
            _event("response.output_text.delta", delta="result"),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="result",
                    usage=None,
                ),
            ),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(
            model,
            context,
            client,
            base_url="https://api.deepseek.com",
        )
        msg = collected[-1]["message"]
        assert msg.get("responses_items") == [
            {"type": "web_search_call", "id": "ws_1", "status": "completed"}
        ]

        replayed = _to_responses_input([msg], model)
        assert replayed[0] == {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
        }
        assert replayed[1]["type"] == "message"

        without_replay = _to_responses_input(
            [msg],
            model,
            replay_web_search_items=False,
        )
        assert all(item["type"] != "web_search_call" for item in without_replay)

    @pytest.mark.asyncio
    async def test_reasoning_item_content_backfilled_from_deltas(self):
        """DeepSeek 流式 reasoning_text 需在回放项中补全 content。"""
        model = _make_deepseek_responses_model()
        context = Context(messages=[{"role": "user", "content": "think?"}])
        events = [
            _event("response.reasoning_text.delta", delta="first "),
            _event("response.reasoning_text.delta", delta="second"),
            _event(
                "response.output_item.done",
                item=SimpleNamespace(
                    type="reasoning",
                    id="rs_1",
                    summary=[SimpleNamespace(type="summary_text", text="sum")],
                    content=[],
                ),
            ),
            _event("response.output_text.delta", delta="answer"),
            _event(
                "response.completed",
                response=SimpleNamespace(output_text="answer", usage=None),
            ),
        ]
        client = _mock_client(events)
        collected, _ = await _collect_events(
            model,
            context,
            client,
            base_url="https://api.deepseek.com",
        )
        msg = collected[-1]["message"]
        thinking_block = next(block for block in msg["content"] if block["type"] == "thinking")
        replayed = _to_responses_input([msg], model)
        reasoning_item = replayed[0]
        assert reasoning_item["type"] == "reasoning"
        assert reasoning_item["content"] == [
            {"type": "reasoning_text", "text": "first second", "annotations": []}
        ]
        assert thinking_block["thinking"] == "sum"


class TestResponsesStatelessAndEvents:
    @pytest.mark.asyncio
    async def test_store_false_include_and_reasoning_summary(self):
        model = _make_deepseek_responses_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_event("response.completed", response=None)])
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(
                model,
                context,
                "sk-test",
                "https://api.deepseek.com",
                {"reasoning": "high", "reasoning_summary": "concise"},
            )
            [e async for e in stream]
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["store"] is False
        assert kwargs["include"] == ["reasoning.encrypted_content"]
        assert kwargs["reasoning"] == {"effort": "high", "summary": "concise"}

    @pytest.mark.asyncio
    async def test_reasoning_off_no_include(self):
        model = _make_deepseek_responses_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_event("response.completed", response=None)])
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(
                model,
                context,
                "sk-test",
                "https://api.deepseek.com",
                {"reasoning": "off"},
            )
            [e async for e in stream]
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["store"] is False
        assert "include" not in kwargs
        assert kwargs["reasoning"] == {"effort": "none"}

    @pytest.mark.asyncio
    async def test_xai_requests_encrypted_content(self):
        model = _make_model(model_id="grok", provider="xai")
        model.reasoning = True
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_event("response.completed", response=None)])
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(model, context, "sk-test", "https://api.x.ai")
            [e async for e in stream]
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["include"] == ["reasoning.encrypted_content"]

    @pytest.mark.asyncio
    async def test_include_system_prompt_false_omits_instructions(self):
        model = _make_model()
        context = Context(
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="Be helpful",
        )
        client = _mock_client([_event("response.completed", response=None)])
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(
                model,
                context,
                "sk-test",
                "https://api.test.com",
                {"include_system_prompt": False},
            )
            [e async for e in stream]
        kwargs = client.responses.create.call_args.kwargs
        assert "instructions" not in kwargs

    @pytest.mark.asyncio
    async def test_refusal_delta_accumulates_text(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.refusal.delta", delta="I cannot"),
            _event("response.refusal.delta", delta=" help"),
            _event(
                "response.completed",
                response=SimpleNamespace(output_text="I cannot help", usage=None),
            ),
        ]
        client = _mock_client(events)
        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == [
            "start",
            "text_start",
            "text_delta",
            "text_delta",
            "text_end",
            "done",
        ]
        assert collected[-1]["message"]["content"][0]["text"] == "I cannot help"

    @pytest.mark.asyncio
    async def test_message_text_signature_captured_and_replayed(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.output_text.delta", delta="answer"),
            _event(
                "response.output_item.done",
                item=SimpleNamespace(
                    type="message",
                    id="msg_1",
                    phase="final_answer",
                ),
            ),
            _event(
                "response.completed",
                response=SimpleNamespace(output_text="answer", usage=None),
            ),
        ]
        client = _mock_client(events)
        collected, _ = await _collect_events(model, context, client)
        msg = collected[-1]["message"]
        text_block = next(block for block in msg["content"] if block["type"] == "text")
        assert text_block["text_signature"] == json.dumps(
            {"v": 1, "id": "msg_1", "phase": "final_answer"}, ensure_ascii=False
        )
        replayed = _to_responses_input([msg], model)
        assert replayed[0]["type"] == "message"
        assert replayed[0]["id"] == "msg_1"
        assert replayed[0]["phase"] == "final_answer"

    @pytest.mark.asyncio
    async def test_completed_backfills_encrypted_content(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.reasoning_text.delta", delta="think"),
            _event(
                "response.output_item.done",
                item=SimpleNamespace(
                    type="reasoning",
                    id="rs_1",
                    summary=[],
                    content=[],
                ),
            ),
            _event(
                "response.completed",
                response=SimpleNamespace(
                    output_text="",
                    usage=None,
                    output=[
                        SimpleNamespace(
                            type="reasoning",
                            id="rs_1",
                            encrypted_content="enc",
                        )
                    ],
                ),
            ),
        ]
        client = _mock_client(events)
        collected, _ = await _collect_events(model, context, client)
        msg = collected[-1]["message"]
        thinking_block = next(block for block in msg["content"] if block["type"] == "thinking")
        stored = json.loads(thinking_block["thinking_signature"])
        assert stored["encrypted_content"] == "enc"


class TestResponsesDeferredTools:
    @pytest.mark.asyncio
    async def test_deferred_tool_search_items(self):
        from pi_ai import Tool

        model = Model(
            id="gpt-4o",
            provider="openai",
            api="openai-responses",
            name="gpt-4o",
            input=["text"],
            output=["text"],
            compat={"supportsToolSearch": True, "supportsWebSearch": False},
        )
        read_tool = Tool(
            name="read",
            description="Read files",
            input_schema={"type": "object", "properties": {}},
        )
        custom_tool = Tool(
            name="custom",
            description="Custom tool",
            input_schema={"type": "object", "properties": {}},
        )
        context = Context(
            messages=[
                {"role": "user", "content": "use custom"},
                {
                    "role": "toolResult",
                    "tool_call_id": "c1",
                    "content": [{"type": "text", "text": "ok"}],
                    "added_tool_names": ["custom"],
                },
            ],
            tools=[read_tool, custom_tool],
        )
        client = _mock_client([_event("response.completed", response=None)])
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(model, context, "sk-test", "https://api.test.com")
            [e async for e in stream]
        kwargs = client.responses.create.call_args.kwargs
        assert [tool["name"] for tool in kwargs["tools"]] == ["read"]
        input_items = kwargs["input"]
        search_output = next(
            item for item in input_items if item.get("type") == "tool_search_output"
        )
        assert search_output["execution"] == "client"
        assert search_output["tools"][0]["name"] == "custom"
        assert search_output["tools"][0]["defer_loading"] is True
        assert any(item.get("type") == "tool_search_call" for item in input_items)
