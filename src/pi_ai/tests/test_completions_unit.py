"""
Unit tests for completions.py — Chat Completions API helpers.
"""

import pytest
from pi_ai.api.completions import _map_stop_reason


class TestMapStopReason:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("stop", "stop"),
            ("end", "stop"),
            ("length", "length"),
            ("tool_calls", "tool_call"),
            ("function_call", "tool_call"),
            ("content_filter", "error"),
            ("network_error", "error"),
            ("some_unknown_reason", "error"),
        ],
    )
    def test_mapping(self, raw, expected):
        assert _map_stop_reason(raw) == expected

    def test_empty_string(self):
        assert _map_stop_reason("") == "stop"

    def test_none_string(self):
        assert _map_stop_reason("None") == "error"


# ===========================================================================
# Chat Completions 流式主循环
#
# 通过 patch 掉 _create_client 返回 mock 客户端，
# 完全离线测试 chunk → event → AssistantMessage 的转换逻辑。
#
# 注意：chat_completions_stream() 会立即返回并后台调度 _run()，
# 因此必须在 patch 生效期间同时完成流的创建与消费。
# ===========================================================================

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from pi_ai._types import Context, Model, Tool
from pi_ai.api.completions import _create_client, chat_completions_stream


def _make_model(
    model_id: str = "deepseek-chat",
    provider: str = "deepseek",
    api: str = "openai-completions",
) -> Model:
    return Model(
        id=model_id,
        provider=provider,
        api=api,
        name=model_id,
        input=["text"],
        output=["text"],
    )


def _async_iter(items):
    """将一个列表包装成异步可迭代对象（模拟 OpenAI streaming response）。"""

    async def gen():
        for item in items:
            yield item

    return gen()


def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """构造一个假的 OpenAI Streaming Chunk。"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _tool_call(index, id_, name, arguments):
    """构造一个假的 delta.tool_calls 元素。"""
    return SimpleNamespace(
        index=index,
        id=id_,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_client(chunks):
    """构造一个 mock 的 AsyncOpenAI 客户端，create() 返回异步可迭代对象。"""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_async_iter(chunks))
    return client


def _empty_usage_dict():
    """与 _shared.empty_usage() 一致的完整空 Usage（含 cost）。"""
    return {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_write": 0,
        "total_tokens": 0,
        "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
    }


async def _collect_events(model, context, client, options=None, base_url="https://api.test.com"):
    """在 patch 生效期间创建并消费流，返回 (events, stream)。"""
    with patch("pi_ai.api.completions._create_client", return_value=client):
        stream = await chat_completions_stream(model, context, "sk-test", base_url, options)
        events = [e async for e in stream]
        return events, stream


class TestCreateClient:
    """_create_client() 客户端工厂。"""

    def test_constructs_openai_client(self):
        with patch("pi_ai.api.completions.AsyncOpenAI") as mock_openai:
            _create_client("sk-test", "https://api.test.com", timeout=30.0)

        mock_openai.assert_called_once()
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["base_url"] == "https://api.test.com"
        assert kwargs["max_retries"] == 2
        assert isinstance(kwargs["timeout"], httpx.Timeout)
        assert kwargs["timeout"].connect == 30.0

    def test_base_url_trailing_slash_stripped(self):
        with patch("pi_ai.api.completions.AsyncOpenAI") as mock_openai:
            _create_client("sk-test", "https://api.test.com///")

        kwargs = mock_openai.call_args.kwargs
        assert kwargs["base_url"] == "https://api.test.com"

    def test_default_timeout(self):
        with patch("pi_ai.api.completions.AsyncOpenAI") as mock_openai:
            _create_client("sk-test", "https://api.test.com")

        kwargs = mock_openai.call_args.kwargs
        assert kwargs["timeout"].connect == 120.0

    def test_custom_max_retries(self):
        with patch("pi_ai.api.completions.AsyncOpenAI") as mock_openai:
            _create_client("sk-test", "https://api.test.com", max_retries=5)

        kwargs = mock_openai.call_args.kwargs
        assert kwargs["max_retries"] == 5


class TestCompletionsStream:
    """chat_completions_stream() 流式主循环。"""

    @pytest.mark.asyncio
    async def test_text_stream(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        chunks = [
            _chunk(content="Hello", finish_reason=None),
            _chunk(content=" world", finish_reason="stop"),
        ]
        client = _mock_client(chunks)

        events, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in events] == [
            "start",
            "text_start",
            "text_delta",
            "text_delta",
            "text_end",
            "done",
        ]
        assert events[2]["delta"] == "Hello"
        assert events[3]["delta"] == " world"

        msg = events[-1]["message"]
        assert msg["role"] == "assistant"
        # 连续的 text delta 累积到同一个 TextContent 块。
        assert msg["content"] == [
            {"type": "text", "text": "Hello world"},
        ]
        assert msg["stop_reason"] == "stop"
        assert msg["model"] == "deepseek-chat"
        assert msg["provider"] == "deepseek"
        assert msg["api"] == "openai-completions"
        assert msg["usage"] == _empty_usage_dict()

    @pytest.mark.asyncio
    async def test_usage_chunk_without_choices_is_captured(self):
        """回归：收尾 chunk 只有 usage、没有 choices（如 DashScope）时 usage 不能被跳过。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=5),
        )
        chunks = [
            _chunk(content="Hello", finish_reason="stop"),
            SimpleNamespace(choices=[], usage=usage),
        ]
        client = _mock_client(chunks)

        events, _ = await _collect_events(model, context, client)
        msg = events[-1]["message"]
        assert msg["usage"]["input"] == 100
        assert msg["usage"]["output"] == 20
        assert msg["usage"]["total_tokens"] == 120
        assert msg["usage"]["cache_read"] == 5

    @pytest.mark.asyncio
    async def test_result_returns_message(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        chunks = [_chunk(content="ok", finish_reason="stop")]
        client = _mock_client(chunks)

        events, stream = await _collect_events(model, context, client)
        msg = await stream.result()
        assert msg["role"] == "assistant"
        assert msg["content"] == [{"type": "text", "text": "ok"}]
        assert events[-1]["type"] == "done"

    @pytest.mark.asyncio
    async def test_options_max_retries_timeout_forwarded(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        chunks = [_chunk(content="ok", finish_reason="stop")]
        client = _mock_client(chunks)

        captured: dict[str, object] = {}

        def _spy(
            api_key: str,
            base_url: str,
            *,
            timeout: float = 120.0,
            max_retries: int = 2,
            headers=None,
        ):
            captured["timeout"] = timeout
            captured["max_retries"] = max_retries
            return client

        with patch("pi_ai.api.completions._create_client", side_effect=_spy):
            stream = await chat_completions_stream(
                model,
                context,
                "sk-test",
                "https://api.test.com",
                options={"max_retries": 5, "timeout_ms": 30000},
            )
            events = [e async for e in stream]

        assert events[-1]["type"] == "done"
        assert captured["max_retries"] == 5
        assert captured["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_options_defaults_when_absent(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        chunks = [_chunk(content="ok", finish_reason="stop")]
        client = _mock_client(chunks)

        captured: dict[str, object] = {}

        def _spy(
            api_key: str,
            base_url: str,
            *,
            timeout: float = 120.0,
            max_retries: int = 2,
            headers=None,
        ):
            captured["timeout"] = timeout
            captured["max_retries"] = max_retries
            return client

        with patch("pi_ai.api.completions._create_client", side_effect=_spy):
            stream = await chat_completions_stream(
                model,
                context,
                "sk-test",
                "https://api.test.com",
            )
            events = [e async for e in stream]

        assert events[-1]["type"] == "done"
        assert captured["max_retries"] == 2
        assert captured["timeout"] == 120.0

    @pytest.mark.asyncio
    async def test_no_choices_chunk_skipped(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        # choices 为空的 chunk 应被跳过，不产生事件也不崩溃。
        chunks = [SimpleNamespace(choices=[], usage=None)]
        client = _mock_client(chunks)

        events, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in events] == ["start", "done"]
        assert events[-1]["message"]["content"] == []

    @pytest.mark.asyncio
    async def test_delta_none_skipped(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        # delta 为 None 的 chunk 应被跳过。
        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=None, finish_reason=None)],
                usage=None,
            )
        ]
        client = _mock_client(chunks)

        events, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in events] == ["start", "done"]

    @pytest.mark.asyncio
    async def test_tool_call_accumulation(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "weather?"}])
        chunks = [
            _chunk(
                tool_calls=[_tool_call(0, "call_1", "get_weather", '{"city":')],
                finish_reason=None,
            ),
            _chunk(
                tool_calls=[_tool_call(0, None, None, '"Beijing"}')],
                finish_reason="tool_calls",
            ),
        ]
        client = _mock_client(chunks)

        events, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in events] == [
            "start",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
            "done",
        ]
        tool_deltas = [e for e in events if e["type"] == "toolcall_delta"]
        assert len(tool_deltas) == 2
        assert tool_deltas[0]["delta"] == '{"city":'
        assert tool_deltas[1]["delta"] == '"Beijing"}'

        msg = events[-1]["message"]
        assert msg["stop_reason"] == "tool_call"
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
    async def test_usage_extraction(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
        )
        chunks = [_chunk(content="Hi", finish_reason="stop", usage=usage)]
        client = _mock_client(chunks)

        events, stream = await _collect_events(model, context, client)
        msg = await stream.result()
        assert msg["usage"] == {
            "input": 10,
            "output": 5,
            "cache_read": 3,
            "cache_write": 0,
            "total_tokens": 15,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        }

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
            system_prompt="You are helpful",
        )
        options = {"temperature": 0.5, "max_tokens": 100}
        client = _mock_client([_chunk(content="Hi", finish_reason="stop")])
        with patch("pi_ai.api.completions._create_client", return_value=client):
            stream = await chat_completions_stream(
                model, context, "sk-test", "https://api.test.com", options
            )
            [e async for e in stream]

        create = client.chat.completions.create
        create.assert_called_once()
        kwargs = create.call_args.kwargs
        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 100
        # System Prompt 作为第一条 message。
        assert kwargs["messages"][0] == {"role": "system", "content": "You are helpful"}
        assert kwargs["messages"][1] == {"role": "user", "content": "Hi"}
        # tools 已转换为 OpenAI Tool Schema。
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    @pytest.mark.asyncio
    async def test_no_options_no_tools(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_chunk(content="Hi", finish_reason="stop")])
        with patch("pi_ai.api.completions._create_client", return_value=client):
            stream = await chat_completions_stream(
                model, context, "sk-test", "https://api.test.com"
            )
            [e async for e in stream]

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "temperature" not in kwargs
        # 对齐 TS buildBaseOptions：max_tokens 始终发送收敛后的值
        # （context_window=0 → 不收敛，返回模型默认 max_tokens 4096）。
        assert kwargs["max_tokens"] == 4096
        assert "tools" not in kwargs

    @pytest.mark.asyncio
    async def test_max_tokens_clamped_to_context(self):
        """max_tokens 被收敛到模型上下文窗口内（预留安全余量）。"""
        model = _make_model()
        model.context_window = 10_000
        model.max_tokens = 8_000
        # "x"*4000 → 估算 1000 tokens
        context = Context(messages=[{"role": "user", "content": "x" * 4_000}])
        client = _mock_client([_chunk(content="Hi", finish_reason="stop")])
        with patch("pi_ai.api.completions._create_client", return_value=client):
            stream = await chat_completions_stream(
                model,
                context,
                "sk-test",
                "https://api.test.com",
                {"max_tokens": 8_000},
            )
            [e async for e in stream]

        kwargs = client.chat.completions.create.call_args.kwargs
        # available = 10000 - 1000 - 4096 = 4904
        assert kwargs["max_tokens"] == 4_904

    @pytest.mark.asyncio
    async def test_error_event(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))

        events, stream = await _collect_events(model, context, client)
        assert events[-1]["type"] == "error"
        assert events[-1]["reason"] == "error"
        err = events[-1]["error"]
        assert err["role"] == "assistant"
        assert err["error_message"] == "boom"
        assert err["stop_reason"] == "error"
        assert err["content"] == []

        # result() 返回携带错误的 AssistantMessage。
        msg = await stream.result()
        assert msg["error_message"] == "boom"

    @pytest.mark.asyncio
    async def test_cancelled_error(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=asyncio.CancelledError())
        with patch("pi_ai.api.completions._create_client", return_value=client):
            stream = await chat_completions_stream(
                model, context, "sk-test", "https://api.test.com"
            )
            # 让后台协程有机会执行。
            await asyncio.sleep(0.01)
            with pytest.raises(asyncio.CancelledError):
                await stream.result()
