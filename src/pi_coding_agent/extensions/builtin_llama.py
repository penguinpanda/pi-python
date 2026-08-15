"""llama.cpp 内置扩展（对齐 TS packages/coding-agent/src/extensions/llama）。

注册 `llama.cpp` provider（OpenAI 兼容 `/v1`），并提供 `/llama` 命令列出
llama.cpp router 模型目录。完整下载/加载 UI 保持为 TS 独有功能。
"""

from __future__ import annotations

import os

from .types import ExtensionAPI

LLAMA_PROVIDER_ID = "llama.cpp"
DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8080"


def _normalize_server_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    return normalized or DEFAULT_LLAMA_SERVER_URL


def _inference_url(server_url: str) -> str:
    return f"{_normalize_server_url(server_url)}/v1"


def create_extension(pi: ExtensionAPI):
    server_url = _normalize_server_url(os.environ.get("LLAMA_BASE_URL") or "")
    api_key = os.environ.get("LLAMA_API_KEY") or "local"
    pi.register_provider(
        LLAMA_PROVIDER_ID,
        {
            "name": "llama.cpp",
            "api": "openai-completions",
            "base_url": _inference_url(server_url),
            "api_key": api_key,
            "models": [
                {
                    "id": "llama3",
                    "name": "Llama 3",
                    "reasoning": False,
                }
            ],
        },
    )

    async def _llama_command(ctx, args: str) -> None:
        if ctx.mode != "tui":
            ctx.ui.notify("/llama is available in interactive mode", "warning")
            return
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{_normalize_server_url(server_url)}/models",
                    headers={"Authorization": f"Bearer {api_key}"} if api_key != "local" else {},
                )
                response.raise_for_status()
                payload = response.json()
            models = [
                str(item.get("id"))
                for item in payload.get("data", [])
                if isinstance(item, dict) and item.get("id")
            ]
            ctx.ui.notify(
                f"llama.cpp models: {', '.join(models) or 'no models loaded'}",
                "info",
            )
        except Exception as exc:
            ctx.ui.notify(f"Could not connect to llama.cpp: {exc}", "error")

    pi.register_command(
        "llama",
        {
            "description": "List llama.cpp router models",
            "handler": _llama_command,
        },
    )


__all__ = [
    "LLAMA_PROVIDER_ID",
    "DEFAULT_LLAMA_SERVER_URL",
    "create_extension",
]
