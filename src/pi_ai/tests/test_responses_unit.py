"""
Unit tests for responses.py — Responses API helpers.
"""
from pi_ai._types import Context, Message, Model, Tool
from pi_ai.api.responses import _to_responses_input

def _make_model(model_id='gpt-4o', provider='openai', api='openai-responses'):
    return Model(id=model_id, provider=provider, api=api, name=model_id, input=['text'], output=['text'])

class TestToResponsesInput:

    def test_system_message(self):
        result = _to_responses_input([], system='Be helpful')
        assert result == [{'role': 'system', 'content': 'Be helpful'}]

    def test_user_string(self):
        messages = [{'role': 'user', 'content': 'Hello'}]
        result = _to_responses_input(messages, system=None)
        assert result == [{'role': 'user', 'content': 'Hello'}]

    def test_user_multimodal_text_and_image_url(self):
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe:'},
            {'type': 'image', 'url': 'https://example.com/pic.png', 'data': None, 'mediaType': None},
        ]}]
        result = _to_responses_input(messages, system=None)
        parts = result[0]['content']
        assert parts[0] == {'type': 'input_text', 'text': 'Describe:'}
        assert parts[1] == {'type': 'input_image', 'image_url': 'https://example.com/pic.png'}

    def test_user_image_base64(self):
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'url': None, 'data': 'abc123', 'mediaType': 'image/jpeg'},
        ]}]
        result = _to_responses_input(messages, system=None)
        img = result[0]['content'][0]
        assert img['image_url'] == 'data:image/jpeg;base64,abc123'

    def test_assistant_message(self):
        messages = [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'Answer'}],
            'api': 'openai-responses', 'provider': 'openai', 'model': 'gpt-4o'}]
        result = _to_responses_input(messages, system=None)
        assert result[0]['role'] == 'assistant'
        assert result[0]['content'] == [{'type': 'output_text', 'text': 'Answer'}]

    def test_tool_result_message(self):
        messages = [{'role': 'toolResult', 'toolCallId': 'call_1', 'toolName': 'search',
            'content': [{'type': 'text', 'text': '42 results'}]}]
        result = _to_responses_input(messages, system=None)
        assert result[0]['type'] == 'function_call_output'
        assert result[0]['call_id'] == 'call_1'
        assert result[0]['output'] == '42 results'

    def test_system_in_messages(self):
        messages = [{'role': 'system', 'content': 'Mid note'}]
        result = _to_responses_input(messages, system=None)
        assert result == [{'role': 'system', 'content': 'Mid note'}]

    def test_combined_system_and_messages(self):
        messages = [{'role': 'user', 'content': 'Hi'}]
        result = _to_responses_input(messages, system='Top-level system')
        assert len(result) == 2
        assert result[0] == {'role': 'system', 'content': 'Top-level system'}
        assert result[1] == {'role': 'user', 'content': 'Hi'}

    def test_user_image_filtered_when_model_no_images(self):
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Hi'},
            {'type': 'image', 'url': 'https://example.com/pic.png', 'data': None, 'mediaType': None},
        ]}]
        # _make_model() default input=['text'] — no image capability
        result = _to_responses_input(messages, system=None, model=_make_model())
        parts = result[0]['content']
        # image should be filtered out; only text remains
        assert len(parts) == 1
        assert parts[0] == {'type': 'input_text', 'text': 'Hi'}

    def test_user_image_kept_when_model_supports_images(self):
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'url': 'https://example.com/pic.png', 'data': None, 'mediaType': None},
        ]}]
        model = _make_model()
        model.input = ['text', 'image']
        result = _to_responses_input(messages, system=None, model=model)
        img = result[0]['content'][0]
        assert img['type'] == 'input_image'
        assert img['image_url'] == 'https://example.com/pic.png'


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


async def _collect_events(model, context, client, options=None, base_url="https://api.openai.com/v1"):
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


class TestResponsesStream:
    """responses_stream() 流式主循环。"""

    @pytest.mark.asyncio
    async def test_text_delta(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.output_text.delta", delta="Hello"),
            _event("response.output_text.delta", delta=" world"),
            _event("response.completed", response=SimpleNamespace(
                output_text="Hello world", usage=None,
            )),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["delta", "delta", "done"]
        assert collected[0]["text"] == "Hello"
        assert collected[1]["text"] == " world"

        msg = collected[-1]["message"]
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"type": "text", "text": "Hello world"}]
        assert msg["stopReason"] == "stop"

    @pytest.mark.asyncio
    async def test_thinking_events(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.reasoning_summary_part.added", part=SimpleNamespace(
                type="summary_text", text="Let me think",
            )),
            _event("response.reasoning_text.delta", delta=" step by step"),
            _event("response.output_text.delta", delta="Answer"),
            _event("response.completed", response=SimpleNamespace(
                output_text="Answer", usage=None,
            )),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        thinking = [e for e in collected if e["type"] == "thinkingDelta"]
        assert [t["thinking"] for t in thinking] == ["Let me think", " step by step"]

        msg = collected[-1]["message"]
        # Thinking 块在 Text 块之前。
        assert msg["content"] == [
            {"type": "thinking", "thinking": "Let me think step by step", "signature": None},
            {"type": "text", "text": "Answer"},
        ]

    @pytest.mark.asyncio
    async def test_thinking_summary_non_summary_ignored(self):
        """summary part 类型不是 summary_text 时应被忽略。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.reasoning_summary_part.added", part=SimpleNamespace(
                type="other", text="ignored",
            )),
            _event("response.completed", response=SimpleNamespace(
                output_text="", usage=None,
            )),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["done"]
        assert collected[-1]["message"]["content"] == []

    @pytest.mark.asyncio
    async def test_tool_call_flow(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "weather?"}])
        events = [
            _event("response.output_item.added", item=SimpleNamespace(
                type="function_call", call_id="call_1", name="get_weather",
            )),
            _event("response.function_call_arguments.delta", delta='{"city":'),
            _event("response.function_call_arguments.delta", delta='"Beijing"}'),
            _event("response.function_call_arguments.done"),
            _event("response.completed", response=SimpleNamespace(
                output_text="", usage=None,
            )),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        tool_deltas = [e for e in collected if e["type"] == "toolCallDelta"]
        assert len(tool_deltas) == 2
        assert tool_deltas[0] == {
            "type": "toolCallDelta",
            "toolCallId": "call_1",
            "toolName": "get_weather",
            "argsDelta": '{"city":',
        }
        assert tool_deltas[1]["argsDelta"] == '"Beijing"}'

        msg = collected[-1]["message"]
        assert msg["content"] == [{
            "type": "toolCall",
            "toolCallId": "call_1",
            "toolName": "get_weather",
            "args": '{"city":"Beijing"}',
        }]

    @pytest.mark.asyncio
    async def test_output_item_non_function_ignored(self):
        """output_item.added 但 item 不是 function_call 时应被忽略。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.output_item.added", item=SimpleNamespace(
                type="message", call_id="", name="",
            )),
            _event("response.completed", response=SimpleNamespace(
                output_text="", usage=None,
            )),
        ]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["done"]
        assert collected[-1]["message"]["content"] == []

    @pytest.mark.asyncio
    async def test_completed_usage(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [
            _event("response.completed", response=SimpleNamespace(
                output_text="Hi",
                usage=SimpleNamespace(input_tokens=7, output_tokens=3, total_tokens=10),
            )),
        ]
        client = _mock_client(events)

        collected, stream = await _collect_events(model, context, client)
        msg = await stream.result()
        assert msg["content"] == [{"type": "text", "text": "Hi"}]
        assert msg["usage"] == {
            "input": 7, "output": 3, "cacheRead": 0, "cacheWrite": 0,
            "totalTokens": 10,
        }

    @pytest.mark.asyncio
    async def test_completed_no_response(self):
        """completed 事件没有 response 对象时不应崩溃。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        events = [_event("response.completed", response=None)]
        client = _mock_client(events)

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["done"]
        msg = collected[-1]["message"]
        assert msg["content"] == []
        assert msg["usage"]["totalTokens"] == 0

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([])

        collected, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in collected] == ["done"]
        assert collected[-1]["message"]["content"] == []

    @pytest.mark.asyncio
    async def test_request_kwargs(self):
        model = _make_model()
        tool = Tool(
            name="get_weather",
            description="Get weather",
            inputSchema={"type": "object", "properties": {}},
        )
        context = Context(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[tool],
            system="Be helpful",
        )
        options = {"temperature": 0.5, "maxTokens": 100}
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
        assert kwargs["max_tokens"] == 100
        # System 作为第一条 input item。
        assert kwargs["input"][0] == {"role": "system", "content": "Be helpful"}
        assert kwargs["input"][1] == {"role": "user", "content": "Hi"}
        assert kwargs["tools"] == [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

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
        assert err["errorMessage"] == "boom"
        assert err["stopReason"] == "error"

        # result() 返回携带错误的 AssistantMessage。
        msg = await stream.result()
        assert msg["errorMessage"] == "boom"

    @pytest.mark.asyncio
    async def test_cancelled_error(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = MagicMock()
        client.responses.create = AsyncMock(side_effect=asyncio.CancelledError())
        with patch("pi_ai.api.responses._create_client", return_value=client):
            stream = await responses_stream(
                model, context, "sk-test", "https://api.openai.com/v1"
            )
            # 让后台协程有机会执行。
            await asyncio.sleep(0.01)
            with pytest.raises(asyncio.CancelledError):
                await stream.result()
