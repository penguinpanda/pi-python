"""Ollama 真实联调集成测试（gated）。

本机 Ollama 不可用时（127.0.0.1:11434 无响应）自动跳过，
不影响日常测试。Ollama 运行时才会真正调用本地模型。

验证 pi_ai 层的 Ollama Provider 端到端：
    - complete() 非流式调用
    - stream() 流式调用（agent loop 使用的路径）
"""

from __future__ import annotations

import httpx
import pytest

from pi_ai import Context
from pi_ai.providers.ollama import ollama_provider

OLLAMA_BASE = "http://127.0.0.1:11434"


def _ollama_available() -> bool:
    """探测本机 Ollama 是否可用。"""
    try:
        resp = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=1.0)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama is not running locally (127.0.0.1:11434)",
)


def _extract_text(result) -> str:
    return "".join(
        block["text"]
        for block in result["content"]
        if block["type"] == "text"
    )


class TestOllamaLive:
    """真实 Ollama 联调（gated）。"""

    async def test_complete_returns_text(self):
        provider = ollama_provider()
        model = provider.get_models()[0]

        result = await provider.complete(
            model,
            Context(messages=[{"role": "user", "content": "Say exactly: ok"}]),
        )

        assert result["stopReason"] == "stop"
        assert _extract_text(result).strip() == "ok"

    async def test_stream_yields_deltas(self):
        provider = ollama_provider()
        model = provider.get_models()[0]

        stream = await provider.stream(
            model,
            Context(messages=[{"role": "user", "content": "Say exactly: ok"}]),
        )

        deltas: list[str] = []
        async for event in stream:
            if event.get("type") == "delta":
                deltas.append(event["text"])

        assert len(deltas) > 0
        assert "".join(deltas).strip() == "ok"
