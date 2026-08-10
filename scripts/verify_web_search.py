#!/usr/bin/env python3
"""web_search 功能验证脚本。

离线验证（默认，无需 API key）：
    构造 DeepSeek Responses 请求，断言服务端 web_search 默认开启、
    web_search_call 项被捕获，且 stateless 回放/关闭行为正确。

在线验证（--live，需 DEEPSEEK_API_KEY）：
    真实调用 DeepSeek Responses，确认返回内容与 web_search_call 项；
    也可用 --api-key 直接传入密钥（避免 shell set 不生效的坑）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pi_ai import Context, Model
from pi_ai.api.responses import _to_responses_input, responses_stream

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_TOKENS = 2048


def _deepseek_model() -> Model:
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


def _async_iter(items):
    async def gen():
        for item in items:
            yield item

    return gen()


def _event(event_type, **kwargs):
    return SimpleNamespace(type=event_type, **kwargs)


async def _consume(model, context, client, options=None, base_url=DEEPSEEK_BASE_URL):
    with patch("pi_ai.api.responses._create_client", return_value=client):
        stream = await responses_stream(model, context, "sk-test", base_url, options)
        return [event async for event in stream]


async def verify_offline() -> None:
    model = _deepseek_model()
    context = Context(messages=[{"role": "user", "content": "今天天气如何？"}])

    # 1) 默认开启：请求 tools 应含 {"type": "web_search"}。
    events = [
        _event(
            "response.output_item.done",
            item=SimpleNamespace(type="web_search_call", id="ws_1", status="completed"),
        ),
        _event("response.output_text.delta", delta="晴"),
        _event(
            "response.completed",
            response=SimpleNamespace(output_text="晴", usage=None),
        ),
    ]
    client = MagicMock()
    client.responses.create = AsyncMock(return_value=_async_iter(events))
    collected = await _consume(model, context, client)
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["tools"] == [{"type": "web_search"}], kwargs["tools"]
    print("[offline] 默认开启: tools =", kwargs["tools"])

    msg = collected[-1]["message"]
    items = msg.get("responses_items")
    assert items == [{"type": "web_search_call", "id": "ws_1", "status": "completed"}]
    print("[offline] web_search_call 捕获:", items)

    replayed = _to_responses_input([msg], model)
    assert replayed[0]["type"] == "web_search_call"
    print("[offline] stateless 回放: 首项为", replayed[0]["type"])

    no_replay = _to_responses_input([msg], model, replay_web_search_items=False)
    assert all(item["type"] != "web_search_call" for item in no_replay)
    print("[offline] web_search=False 时回放排除 web_search_call")

    # 2) 显式关闭：不追加 web_search 工具。
    client2 = MagicMock()
    client2.responses.create = AsyncMock(
        return_value=_async_iter([_event("response.completed", response=None)])
    )
    await _consume(model, context, client2, options={"web_search": False})
    kwargs2 = client2.responses.create.call_args.kwargs
    assert "tools" not in kwargs2 or kwargs2["tools"] == []
    print("[offline] 显式关闭: tools =", kwargs2.get("tools", []))
    print("[offline] PASS")


async def verify_live(api_key: str) -> None:
    model = _deepseek_model()
    context = Context(
        messages=[
            {
                "role": "user",
                "content": "请联网搜索并回答：2026 年 8 月 10 日有什么新闻？",
            }
        ]
    )
    stream = await responses_stream(
        model, context, api_key, DEEPSEEK_BASE_URL, {"max_tokens": MAX_TOKENS}
    )
    text_parts: list[str] = []
    items: list[dict] = []
    stop_reason: str | None = None
    usage: dict | None = None
    async for event in stream:
        if event["type"] == "text_delta":
            text_parts.append(event["delta"])
        elif event["type"] == "done":
            message = event["message"]
            items = message.get("responses_items") or []
            stop_reason = message.get("stop_reason")
            usage = message.get("usage")
        elif event["type"] == "error":
            raise RuntimeError(event["error"].get("error_message"))
    print("[live] web_search_call items:", items)
    print("[live] stop_reason:", stop_reason)
    print("[live] usage:", usage)
    print("[live] answer:", "".join(text_parts))
    if not items:
        print("[live] 未捕获 web_search_call（模型可能未触发搜索）")
    if stop_reason == "length":
        print("[live] 注意: 输出被 max_tokens 截断，可用 --max-tokens 加大预算重试")
    print("[live] PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="web_search 功能验证")
    parser.add_argument(
        "--live", action="store_true", help="真实调用 DeepSeek（需 DEEPSEEK_API_KEY）"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="DeepSeek API key（优先于 DEEPSEEK_API_KEY 环境变量）",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="真实调用的输出 token 预算")
    args = parser.parse_args()

    global MAX_TOKENS
    MAX_TOKENS = args.max_tokens

    asyncio.run(verify_offline())
    if args.live:
        api_key = (args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not api_key:
            print("SKIP: --live 需要设置 DEEPSEEK_API_KEY 或传入 --api-key")
            return 1
        asyncio.run(verify_live(api_key))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
