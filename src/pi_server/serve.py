"""stdio 常驻服务：stdin 读 JSONL 请求，stdout 写 JSONL 消息。"""

from __future__ import annotations

import asyncio
import sys

from pi_protocol.framing import encode_frame


async def run_stdio_server(server) -> int:
    """阻塞运行直到 stdin EOF。"""
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        try:
            messages = await server.handle_line(line)
        except Exception as exc:
            messages = [
                {
                    "type": "hello_error",
                    "error": {"code": "invalid_request", "message": str(exc)},
                }
            ]
        for message in messages:
            sys.stdout.write(encode_frame(message))
            sys.stdout.flush()
    return 0


__all__ = ["run_stdio_server"]
