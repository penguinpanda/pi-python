# pi-protocol — protocol v2 线协议
[English](README.en.md) | [中文](README.md)

对齐 TS [packages/protocol](https://github.com/earendil-works/pi-mono/tree/main/packages/protocol)，
定义 pi server 与 client 之间的线协议（`PROTOCOL_VERSION = 2`）。

## 内容

- `schemas.py` — pydantic v2 严格 schema（`extra="forbid"`）：
  - 模型：`ModelRef` / `ModelMetadata` / `ModelCost`
  - 内容：`TextContent` / `ThinkingContent` / `ImageContent` / `ToolCallContent`
  - 消息：`UserTranscriptItem` / `AssistantTranscriptItem` / `ToolTranscriptItem`
  - 进度：`TranscriptProgress`（item_started / assistant_delta / item_updated / item_finished）
  - 快照：`SessionSummary` / `SessionSnapshot` / `ServerSnapshot`
  - 命令：`list` / `create` / `attach` / `detach` / `prompt` / `steer` / `abort` /
    `set_model` / `set_thinking`
  - 结果：每个命令对应的 `*Result`（session 快照或 sessionId）
  - 信封：`ClientHello` / `RequestEnvelope` / `ServerHello` / `ServerHelloError` /
    `ResponseEnvelope` / `EventEnvelope` / `ServerEvent`
- `framing.py` — JSONL framing：`encode_frame` / `decode_frame` / `iter_frames`，
  以及 `parse_client_message` / `parse_server_message`（TypeAdapter 校验）。

## 用法

```python
from pi_protocol import encode_frame, parse_server_message
from pi_protocol.schemas import PROTOCOL_VERSION, ServerHello, ServerSnapshot

hello = ServerHello(
    type="hello",
    version=PROTOCOL_VERSION,
    connectionId="c1",
    snapshot=ServerSnapshot(
        serverId="srv",
        protocolVersion=PROTOCOL_VERSION,
        revision=0,
        sessions=[],
        models=[],
    ),
)
line = encode_frame(hello)  # JSON 行（含 \n）
parsed = parse_server_message({})  # 解析并校验服务端消息
```

## 测试

```bash
uv run pytest src/pi_protocol/tests/ -v
```

覆盖：全部命令/结果/信封的往返编解码、未知字段拒绝、版本字面量、framing 跳过空行。

## 许可

MIT
