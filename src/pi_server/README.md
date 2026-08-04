# pi-server — 常驻 pi 服务

对齐 TS [packages/server](https://github.com/earendil-works/pi-mono/tree/main/packages/server)：
asyncio 常驻进程，基于 protocol v2 提供 attach/detach、快照推送与命令分发。

## 内容

- `handler.py` — `PiServer`：会话注册表 + 命令分发（list/create/attach/detach/prompt/
  steer/abort/set_model/set_thinking）+ 快照构建与推送；`ServerSession` 包装单个
  `AgentSession`（事件订阅 → revision / updatedAt 递增）。
- `serve.py` — `run_stdio_server`：stdin 读 JSONL 请求，stdout 写 JSONL 消息。
- `__main__.py` — `python -m pi_server` 常驻入口（ModelRuntime + `PI_SERVER_TOKEN`）。

## 用法

```bash
# 启动（stdio JSONL）
PI_SERVER_TOKEN=... python -m pi_server
```

客户端协议（`pi_protocol`）：

```
client → hello（version + token）
server → hello（version + connectionId + ServerSnapshot）
client → request {id, request: Command}
server → response {id, ok, result|error} + event（server_snapshot / session_snapshot）
```

程序化使用：

```python
import asyncio
from pi_server import PiServer


async def main():
    server = PiServer(session_factory=lambda cwd: my_agent_session(cwd))
    await server.handle_line('{"type":"hello","version":2,"token":""}\n')
    messages = await server.handle_line(
        '{"type":"request","id":"1","request":{"command":"create","cwd":"."}}\n'
    )
    # messages[0] = response；后续为 server_snapshot / session_snapshot 事件


asyncio.run(main())
```

## 测试

```bash
uv run pytest src/pi_server/tests/ -v
```

覆盖 hello（版本/令牌）、命令分发、错误响应与子进程 spawn 验收
（spawn server → create/attach/prompt → 收到 `session_snapshot` 事件）。

## 许可

MIT
