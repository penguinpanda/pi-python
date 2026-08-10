"""pi_ai.utils.prompt_cache 单元测试 + completions/responses 集成测试。

覆盖：
    - clamp_openai_prompt_cache_key（64 字符截断）
    - resolve_cache_retention（env / PI_CACHE_RETENTION 回退）
    - completions.py 请求 kwargs（prompt_cache_key / prompt_cache_retention）
    - responses.py 请求 kwargs
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from pi_ai._types import Context, Model
from pi_ai.utils.prompt_cache import (
    OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH,
    clamp_openai_prompt_cache_key,
    resolve_cache_retention,
)


# ============================================================================
# clamp_openai_prompt_cache_key
# ============================================================================


class TestClampOpenAIPromptCacheKey:
    def test_none_returns_none(self):
        assert clamp_openai_prompt_cache_key(None) is None

    def test_short_key_unchanged(self):
        assert clamp_openai_prompt_cache_key("session-123") == "session-123"

    def test_exactly_64_unchanged(self):
        key = "x" * OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH
        assert clamp_openai_prompt_cache_key(key) == key

    def test_long_key_clamped(self):
        key = "x" * 100
        result = clamp_openai_prompt_cache_key(key)
        assert len(result) == 64
        assert result == "x" * 64

    def test_multibyte_clamped_by_code_point(self):
        # 中文按码位截断（不切断多字节字符）
        key = "中" * 100
        result = clamp_openai_prompt_cache_key(key)
        assert len(result) == 64
        assert result == "中" * 64


# ============================================================================
# resolve_cache_retention
# ============================================================================


class TestResolveCacheRetention:
    def test_default_short(self):
        assert resolve_cache_retention() == "short"

    def test_explicit_wins(self):
        assert resolve_cache_retention("none") == "none"
        assert resolve_cache_retention("long") == "long"

    def test_provider_env_long(self):
        assert resolve_cache_retention(None, {"PI_CACHE_RETENTION": "long"}) == "long"

    def test_provider_env_ignored_for_other_values(self):
        assert resolve_cache_retention(None, {"PI_CACHE_RETENTION": "bogus"}) == "short"

    def test_os_environ_long(self, monkeypatch):
        monkeypatch.setenv("PI_CACHE_RETENTION", "long")
        assert resolve_cache_retention() == "long"

    def test_provider_env_precedes_os_environ(self, monkeypatch):
        monkeypatch.setenv("PI_CACHE_RETENTION", "long")
        assert resolve_cache_retention(None, {"PI_CACHE_RETENTION": "short"}) == "short"


# ============================================================================
# 通用 mock 辅助（completions / responses）
# ============================================================================


def _async_iter(items):
    async def gen():
        for item in items:
            yield item

    return gen()


def _completions_chunk(content=None, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=None,
    )


def _responses_event(event_type, **kwargs):
    return SimpleNamespace(type=event_type, **kwargs)


def _completions_model() -> Model:
    return Model(
        id="deepseek-chat",
        provider="deepseek",
        api="openai-completions",
        name="deepseek-chat",
        input=["text"],
        output=["text"],
    )


def _responses_model() -> Model:
    return Model(
        id="gpt-4o",
        provider="openai",
        api="openai-responses",
        name="gpt-4o",
        input=["text"],
        output=["text"],
    )


def _context() -> Context:
    return Context(messages=[{"role": "user", "content": "Hi"}])


async def _collect_completions(model, client, options=None, base_url="https://api.test.com"):
    from pi_ai.api.completions import chat_completions_stream

    with patch("pi_ai.api.completions._create_client", return_value=client):
        stream = await chat_completions_stream(model, _context(), "sk-test", base_url, options)
        [e async for e in stream]


async def _collect_responses(model, client, options=None, base_url="https://api.openai.com/v1"):
    from pi_ai.api.responses import responses_stream

    with patch("pi_ai.api.responses._create_client", return_value=client):
        stream = await responses_stream(model, _context(), "sk-test", base_url, options)
        [e async for e in stream]


# ============================================================================
# completions.py 集成
# ============================================================================


class TestCompletionsPromptCache:
    def _run(self, options=None, base_url="https://api.test.com", compat=None):
        """同步包装：运行请求并返回 create 的 kwargs。"""
        model = _completions_model()
        if compat is not None:
            model.compat = compat
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=_async_iter([_completions_chunk("Hi", "stop")])
        )
        asyncio.run(_collect_completions(model, client, options, base_url))
        return client.chat.completions.create.call_args.kwargs

    def test_non_openai_default_short_no_key(self):
        kwargs = self._run()
        assert "prompt_cache_key" not in kwargs
        assert "prompt_cache_retention" not in kwargs

    def test_openai_short_sends_key_no_retention(self):
        kwargs = self._run(
            options={"session_id": "s-123", "cache_retention": "short"},
            base_url="https://api.openai.com/v1",
        )
        assert kwargs["prompt_cache_key"] == "s-123"
        assert "prompt_cache_retention" not in kwargs

    def test_openai_none_no_key(self):
        kwargs = self._run(
            options={"session_id": "s-123", "cache_retention": "none"},
            base_url="https://api.openai.com/v1",
        )
        assert "prompt_cache_key" not in kwargs

    def test_non_openai_long_sends_key_and_retention(self):
        kwargs = self._run(
            options={"session_id": "s-123", "cache_retention": "long"},
            base_url="https://api.deepseek.com",
        )
        assert kwargs["prompt_cache_key"] == "s-123"
        assert kwargs["prompt_cache_retention"] == "24h"

    def test_long_key_clamped_to_64(self):
        kwargs = self._run(
            options={"session_id": "x" * 100, "cache_retention": "long"},
            base_url="https://api.deepseek.com",
        )
        assert len(kwargs["prompt_cache_key"]) == 64

    def test_long_without_support_no_key(self):
        kwargs = self._run(
            options={"session_id": "s-123", "cache_retention": "long"},
            base_url="https://api.deepseek.com",
            compat={"supportsLongCacheRetention": False},
        )
        assert "prompt_cache_key" not in kwargs
        assert "prompt_cache_retention" not in kwargs

    def test_env_long_enables_key_for_openai(self, monkeypatch):
        monkeypatch.setenv("PI_CACHE_RETENTION", "long")
        kwargs = self._run(
            options={"session_id": "s-123"},
            base_url="https://api.openai.com/v1",
        )
        assert kwargs["prompt_cache_key"] == "s-123"


# ============================================================================
# responses.py 集成
# ============================================================================


class TestResponsesPromptCache:
    def _run(self, options=None, compat=None):
        model = _responses_model()
        if compat is not None:
            model.compat = compat
        client = MagicMock()
        client.responses.create = AsyncMock(
            return_value=_async_iter(
                [
                    _responses_event(
                        "response.completed",
                        response=SimpleNamespace(output_text="Hi", usage=None),
                    ),
                ]
            )
        )
        asyncio.run(_collect_responses(model, client, options))
        return client.responses.create.call_args.kwargs

    def test_default_short_sends_key_no_retention(self):
        kwargs = self._run(options={"session_id": "s-123"})
        assert kwargs["prompt_cache_key"] == "s-123"
        assert "prompt_cache_retention" not in kwargs

    def test_long_sends_key_and_retention(self):
        kwargs = self._run(options={"session_id": "s-123", "cache_retention": "long"})
        assert kwargs["prompt_cache_key"] == "s-123"
        assert kwargs["prompt_cache_retention"] == "24h"

    def test_none_no_key(self):
        kwargs = self._run(options={"session_id": "s-123", "cache_retention": "none"})
        assert "prompt_cache_key" not in kwargs
        assert "prompt_cache_retention" not in kwargs

    def test_long_without_support_no_retention_but_key(self):
        kwargs = self._run(
            options={"session_id": "s-123", "cache_retention": "long"},
            compat={"supportsLongCacheRetention": False},
        )
        # Responses：key 在 != none 时发送；retention 仅 long+支持
        assert kwargs["prompt_cache_key"] == "s-123"
        assert "prompt_cache_retention" not in kwargs

    def test_deepseek_explicit_prompt_cache_disabled(self):
        kwargs = self._run(
            options={"session_id": "s-123", "cache_retention": "long"},
            compat={
                "supportsExplicitPromptCacheMode": False,
                "supportsLongCacheRetention": False,
            },
        )
        assert "prompt_cache_key" not in kwargs
        assert "prompt_cache_retention" not in kwargs
