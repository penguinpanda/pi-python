"""
Unit tests for completions.py — Chat Completions API helpers.
"""

from __future__ import annotations

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

from pi_ai import ModelCost
from pi_ai._types import Context, Model, Tool
from pi_ai.api.completions import _create_client, chat_completions_stream
from pi_ai.api._shared import to_openai_messages


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


def _chunk(
    content=None,
    tool_calls=None,
    finish_reason=None,
    usage=None,
    reasoning_content=None,
    reasoning=None,
):
    """构造一个假的 OpenAI Streaming Chunk。"""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=reasoning_content,
                    reasoning=reasoning,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def _make_deepseek_v4_model() -> Model:
    """deepseek-v4-pro 元数据（Completions 路径，flash 已切到 Responses）。"""
    return Model(
        id="deepseek-v4-pro",
        provider="deepseek",
        api="openai-completions",
        name="DeepSeek V4 Pro",
        input=["text"],
        output=["text"],
        reasoning=True,
        thinking_level_map={
            "minimal": None,
            "low": None,
            "medium": None,
            "high": "high",
            "max": "max",
        },
        compat={
            "supportsStore": False,
            "supportsDeveloperRole": False,
            "requiresReasoningContentOnAssistantMessages": True,
            "thinkingFormat": "deepseek",
        },
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
        # 缓存命中 token 从 input 中扣除，避免双重计费。
        assert msg["usage"]["input"] == 95
        assert msg["usage"]["output"] == 20
        assert msg["usage"]["total_tokens"] == 120
        assert msg["usage"]["cache_read"] == 5

    @pytest.mark.asyncio
    async def test_ollama_reasoning_field_emits_thinking_events(self):
        """ollama/qwen3 流式 reasoning 字段应进入 thinking 块（对齐 reasoning_content）。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        chunks = [
            _chunk(reasoning="Think", finish_reason=None),
            _chunk(reasoning=" more", finish_reason=None),
            _chunk(content="Answer", finish_reason="stop"),
        ]
        client = _mock_client(chunks)

        events, _ = await _collect_events(model, context, client)
        assert [e["type"] for e in events] == [
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
        msg = events[-1]["message"]
        assert msg["content"] == [
            {"type": "thinking", "thinking": "Think more"},
            {"type": "text", "text": "Answer"},
        ]

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
    async def test_parallel_tool_calls_interleaved(self):
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "two tools?"}])
        # 标准 OpenAI 兼容流：首个 chunk 带两个调用（id + index），
        # 后续参数增量 chunk 只带 index（id 为 None），且交错到达。
        chunks = [
            _chunk(
                tool_calls=[
                    _tool_call(0, "call_0", "tool_a", '{"x":'),
                    _tool_call(1, "call_1", "tool_b", '{"y":'),
                ],
                finish_reason=None,
            ),
            _chunk(tool_calls=[_tool_call(0, None, None, "1}")], finish_reason=None),
            _chunk(tool_calls=[_tool_call(1, None, None, "2}")], finish_reason=None),
            _chunk(tool_calls=[_tool_call(0, None, None, "")], finish_reason="tool_calls"),
        ]
        client = _mock_client(chunks)

        events, _ = await _collect_events(model, context, client)
        msg = events[-1]["message"]
        assert msg["stop_reason"] == "tool_call"
        assert [b["type"] for b in msg["content"]] == ["toolCall", "toolCall"]
        blocks = {b["id"]: b for b in msg["content"]}
        assert blocks["call_0"]["raw_arguments"] == '{"x":1}'
        assert blocks["call_0"]["arguments"] == {"x": 1}
        assert blocks["call_1"]["raw_arguments"] == '{"y":2}'
        assert blocks["call_1"]["arguments"] == {"y": 2}

        # 事件流：每个调用一次 start，流末统一 end。
        types = [e["type"] for e in events]
        assert types.count("toolcall_start") == 2
        assert types.count("toolcall_end") == 2
        assert [e["content_index"] for e in events if e["type"] == "toolcall_end"] == [0, 1]

    @pytest.mark.asyncio
    async def test_mixed_index_protocol_no_state_collision(self):
        """先无 index 后有 index=0 的混合协议：两个调用不得合并/吞并。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "two tools?"}])
        chunks = [
            _chunk(
                tool_calls=[_tool_call(None, "call_1", "tool_a", '{"x":')],
                finish_reason=None,
            ),
            _chunk(
                tool_calls=[_tool_call(0, "call_2", "tool_b", '{"y":')],
                finish_reason=None,
            ),
            _chunk(tool_calls=[_tool_call(None, "call_1", None, "1}")], finish_reason=None),
            _chunk(tool_calls=[_tool_call(0, None, None, "2}")], finish_reason="tool_calls"),
        ]
        client = _mock_client(chunks)

        events, _ = await _collect_events(model, context, client)
        msg = events[-1]["message"]
        blocks = [b for b in msg["content"] if b["type"] == "toolCall"]
        assert len(blocks) == 2
        by_id = {b["id"]: b for b in blocks}
        assert by_id["call_1"]["raw_arguments"] == '{"x":1}'
        assert by_id["call_1"]["arguments"] == {"x": 1}
        assert by_id["call_2"]["raw_arguments"] == '{"y":2}'
        assert by_id["call_2"]["arguments"] == {"y": 2}

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
            "input": 7,
            "output": 5,
            "cache_read": 3,
            "cache_write": 0,
            "total_tokens": 15,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        }

    @pytest.mark.asyncio
    async def test_deepseek_prompt_cache_hit_tokens_usage(self):
        """DeepSeek 顶层 prompt_cache_hit_tokens 计入 cache_read 且计费正确。"""
        model = _make_model()
        model.cost = ModelCost(input=0.14, output=0.28, cache_read=0.0028, cache_write=0.0)
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        usage = SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=100,
            total_tokens=1100,
            prompt_cache_hit_tokens=800,
            prompt_cache_miss_tokens=150,
        )
        chunks = [_chunk(content="Hi", finish_reason="stop", usage=usage)]
        client = _mock_client(chunks)

        events, stream = await _collect_events(model, context, client)
        msg = await stream.result()
        parsed = msg["usage"]
        # input 按 prompt_tokens - hit 计算（对齐 TS，不依赖 miss 字段）。
        assert parsed["input"] == 200
        assert parsed["output"] == 100
        assert parsed["cache_read"] == 800
        assert parsed["cache_write"] == 0
        assert parsed["total_tokens"] == 1100
        cost = parsed["cost"]
        assert cost["input"] == pytest.approx(200 * 0.14 / 1_000_000)
        assert cost["cache_read"] == pytest.approx(800 * 0.0028 / 1_000_000)
        assert cost["output"] == pytest.approx(100 * 0.28 / 1_000_000)
        assert cost["total"] == pytest.approx(cost["input"] + cost["cache_read"] + cost["output"])

    @pytest.mark.asyncio
    async def test_cache_write_tokens_split_from_reads(self):
        """cache_write_tokens 单独计数，不混入 cache_read（对齐 TS）。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=5,
            total_tokens=105,
            prompt_tokens_details=SimpleNamespace(cached_tokens=50, cache_write_tokens=30),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
        )
        chunks = [_chunk(content="Hi", finish_reason="stop", usage=usage)]
        client = _mock_client(chunks)

        events, stream = await _collect_events(model, context, client)
        msg = await stream.result()
        assert msg["usage"] == {
            "input": 20,
            "output": 5,
            "cache_read": 50,
            "cache_write": 30,
            "total_tokens": 105,
            "reasoning": 2,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        }

    @pytest.mark.asyncio
    async def test_cached_tokens_zero_takes_precedence_over_hit_tokens(self):
        """prompt_tokens_details.cached_tokens 存在（含 0）时优先于 DeepSeek 字段。"""
        model = _make_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            prompt_cache_hit_tokens=8,
        )
        chunks = [_chunk(content="Hi", finish_reason="stop", usage=usage)]
        client = _mock_client(chunks)

        events, stream = await _collect_events(model, context, client)
        msg = await stream.result()
        assert msg["usage"]["input"] == 10
        assert msg["usage"]["cache_read"] == 0

    @pytest.mark.asyncio
    async def test_deepseek_reasoning_effort_forwarded(self):
        """DeepSeek V4：reasoning 选项翻译为 thinking + reasoning_effort。"""
        model = _make_deepseek_v4_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])

        _, _ = await _collect_events(model, context, client, options={"reasoning": "high"})
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
        assert kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_deepseek_thinking_disabled_for_off(self):
        """DeepSeek V4：off 翻译为 thinking.type=disabled，不带 reasoning_effort。"""
        model = _make_deepseek_v4_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])

        _, _ = await _collect_events(model, context, client, options={"reasoning": "off"})
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in kwargs

    @pytest.mark.asyncio
    async def test_deepseek_thinking_disabled_when_no_effort(self):
        """DeepSeek V4：未指定 effort 且 map 无 off 时显式 thinking.type=disabled。"""
        model = _make_deepseek_v4_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])

        _, _ = await _collect_events(model, context, client)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in kwargs

    @pytest.mark.asyncio
    async def test_deepseek_off_null_skips_disabled(self):
        """DeepSeek V4：map 把 off 声明为 None 时不发送 thinking.type=disabled。"""
        model = _make_deepseek_v4_model()
        model.thinking_level_map = {"off": None}
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])

        _, _ = await _collect_events(model, context, client)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "thinking" not in kwargs
        assert "thinking" not in (kwargs.get("extra_body") or {})
        assert "reasoning_effort" not in kwargs

    @pytest.mark.parametrize(
        "level,expected",
        [
            ("minimal", "minimal"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "xhigh"),
            ("max", "max"),
        ],
    )
    @pytest.mark.asyncio
    async def test_deepseek_effort_passthrough(self, level, expected):
        """DeepSeek V4：map 中缺失或为 None 的级别按原值透传（对齐 TS）。"""
        model = _make_deepseek_v4_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])

        _, _ = await _collect_events(model, context, client, options={"reasoning": level})
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["reasoning_effort"] == expected

    @pytest.mark.asyncio
    async def test_reasoning_content_parsed_to_thinking_blocks(self):
        """DeepSeek delta.reasoning_content 解析为 thinking 块，且顺序在 text 之前。"""
        model = _make_deepseek_v4_model()
        context = Context(messages=[{"role": "user", "content": "Hi"}])
        chunks = [
            _chunk(reasoning_content="Let me think", finish_reason=None),
            _chunk(reasoning_content=" deeply", finish_reason=None),
            _chunk(content="Answer.", finish_reason="stop"),
        ]
        client = _mock_client(chunks)

        events, stream = await _collect_events(model, context, client)
        assert [e["type"] for e in events] == [
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
        msg = await stream.result()
        assert msg["content"] == [
            {"type": "thinking", "thinking": "Let me think deeply"},
            {"type": "text", "text": "Answer."},
        ]

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


class TestOpenaiMessagesReasoningContent:
    """to_openai_messages 对 DeepSeek reasoning_content 的回传。"""

    def _assistant(self) -> dict:
        return {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "reasoning text"},
                {"type": "text", "text": "answer"},
            ],
        }

    def test_deepseek_roundtrips_reasoning_content(self):
        model = _make_deepseek_v4_model()
        messages = [self._assistant()]
        result = to_openai_messages(messages, model)  # type: ignore[arg-type]
        assert result[0]["reasoning_content"] == "reasoning text"
        assert result[0]["content"] == "answer"

    def test_other_models_skip_thinking_blocks(self):
        model = _make_model()
        messages = [self._assistant()]
        result = to_openai_messages(messages, model)  # type: ignore[arg-type]
        assert "reasoning_content" not in result[0]
        assert result[0]["content"] == "answer"


class TestThinkingFormatMatrix:
    """thinkingFormat 矩阵（zai/qwen/openrouter/string-thinking 等，对齐 TS）。"""

    def _model(self, thinking_format: str, **compat) -> Model:
        return Model(
            id=f"m-{thinking_format}",
            provider="test",
            api="openai-completions",
            name="T",
            input=["text"],
            output=["text"],
            reasoning=True,
            thinking_level_map={"off": "off", "high": "high"},
            compat={"thinkingFormat": thinking_format, **compat},
        )

    @pytest.mark.asyncio
    async def test_vllm_budget_xhigh_max_clamp_to_high(self):
        """xhigh/max 收敛到 high 查表（对齐 TS clampReasoning），仍发预算。"""
        model = self._model("qwen", supportsThinkingTokenBudget=True)
        model.max_tokens = 32000
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])
        for level in ("xhigh", "max"):
            _, _ = await _collect_events(
                model,
                Context(messages=[{"role": "user", "content": "Hi"}]),
                client,
                options={"reasoning": level},
            )
            kwargs = client.chat.completions.create.call_args.kwargs
            assert kwargs["extra_body"]["thinking_token_budget"] == 16384

    @pytest.mark.asyncio
    async def test_vllm_budget_ceiling_uses_nullish_semantics(self):
        """ceiling 取 max_tokens ?? max_completion_tokens ?? model.maxTokens。"""
        model = self._model("qwen", supportsThinkingTokenBudget=True)
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])
        _, _ = await _collect_events(
            model,
            Context(messages=[{"role": "user", "content": "Hi"}]),
            client,
            options={"reasoning": "high", "max_tokens": 2000},
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        # ceiling=2000 → budget = min(16384, 2000 - 1024) = 976
        assert kwargs["extra_body"]["thinking_token_budget"] == 976

    @pytest.mark.asyncio
    async def test_vllm_budget_not_sent_when_level_missing(self):
        """budget 表查不到且无 clamp 路径时不发送（off 由 _thinking_on 排除）。"""
        model = self._model("qwen", supportsThinkingTokenBudget=True)
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])
        _, _ = await _collect_events(
            model,
            Context(messages=[{"role": "user", "content": "Hi"}]),
            client,
            options={"reasoning": "off"},
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert "thinking_token_budget" not in (kwargs.get("extra_body") or {})

    @pytest.mark.asyncio
    async def test_zai_thinking_enabled(self):
        model = self._model("zai", supportsReasoningEffort=True)
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])
        _, _ = await _collect_events(
            model,
            Context(messages=[{"role": "user", "content": "Hi"}]),
            client,
            options={"reasoning": "high"},
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled", "clear_thinking": False}
        assert kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_qwen_enable_thinking_flag(self):
        model = self._model("qwen", supportsReasoningEffort=True)
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])
        _, _ = await _collect_events(
            model,
            Context(messages=[{"role": "user", "content": "Hi"}]),
            client,
            options={"reasoning": "off"},
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_openrouter_nested_reasoning(self):
        model = self._model("openrouter")
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])
        _, _ = await _collect_events(
            model,
            Context(messages=[{"role": "user", "content": "Hi"}]),
            client,
            options={"reasoning": "high"},
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["reasoning"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_string_thinking_passthrough(self):
        model = self._model("string-thinking")
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])
        _, _ = await _collect_events(
            model,
            Context(messages=[{"role": "user", "content": "Hi"}]),
            client,
            options={"reasoning": "high"},
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["thinking"] == "high"

    @pytest.mark.asyncio
    async def test_together_reasoning_enabled_flag(self):
        model = self._model("together", supportsReasoningEffort=True)
        client = _mock_client([_chunk(content="ok", finish_reason="stop")])
        _, _ = await _collect_events(
            model,
            Context(messages=[{"role": "user", "content": "Hi"}]),
            client,
            options={"reasoning": "high"},
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["reasoning"] == {"enabled": True}
        assert kwargs["reasoning_effort"] == "high"
