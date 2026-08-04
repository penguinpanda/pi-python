"""python -m pi_server — 启动常驻服务（stdio JSONL）。"""

from __future__ import annotations

import asyncio
import os

from pi_ai import create_default_models

from pi_coding_agent._config import get_agent_dir
from pi_coding_agent.model_runtime import ModelRuntime

from .handler import PiServer
from .serve import run_stdio_server


async def _main() -> int:
    runtime = await ModelRuntime.create(
        providers=create_default_models().get_providers(),
        auth_path=str(get_agent_dir() / "auth.json"),
        models_path=str(get_agent_dir() / "models.json"),
        allow_model_network=True,
        model_refresh_timeout_ms=15000,
    )
    server = PiServer(
        model_runtime=runtime,
        token=os.environ.get("PI_SERVER_TOKEN") or None,
    )
    return await run_stdio_server(server)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
