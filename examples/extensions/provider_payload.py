"""Provider payload extension - log / override provider requests.

Python port of provider-payload.ts。before_provider_request 事件带
stream_options（对齐 TS 的 event.payload）；after_provider_response 事件
带 response（stop_reason / usage）。
"""

import json
from pathlib import Path

from pi_coding_agent import ExtensionAPI


def _log_path(ctx) -> Path:
    return Path(ctx.cwd) / ".pi" / "provider-payload.log"


def _append(ctx, text: str) -> None:
    path = _log_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def create_extension(pi: ExtensionAPI):
    async def on_before_provider_request(event, ctx):
        _append(
            ctx, json.dumps(event.get("stream_options", {}), ensure_ascii=False, indent=2) + "\n\n"
        )
        # 可选：替换请求选项（例如强制 temperature）
        # return {"stream_options": {**event["stream_options"], "temperature": 0}}

    async def on_after_provider_response(event, ctx):
        response = event.get("response") or {}
        usage = response.get("usage") or {}
        _append(ctx, f"[{response.get('stop_reason', '?')}] {json.dumps(usage)}\n\n")

    pi.on("before_provider_request", on_before_provider_request)
    pi.on("after_provider_response", on_after_provider_response)
