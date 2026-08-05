"""JSONL framing（每行一个 JSON 消息，`\n` 分隔）。"""

from __future__ import annotations

import json
from typing import Iterator

from pydantic import TypeAdapter

from .schemas import ClientMessage, ServerMessage

_CLIENT_MESSAGE_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)
_SERVER_MESSAGE_ADAPTER: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def encode_frame(message) -> str:
    """消息 → JSON 行（含结尾换行）。BaseModel 或 dict 均可。"""
    if hasattr(message, "model_dump"):
        data = message.model_dump(mode="json")
    elif isinstance(message, dict):
        data = message
    else:
        raise TypeError(f"Unsupported message type: {type(message)!r}")
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"


def decode_frame(line: str) -> dict | None:
    """JSON 行 → dict；空行返回 None。"""
    stripped = line.strip()
    if not stripped:
        return None
    return json.loads(stripped)


def iter_frames(text: str) -> Iterator[dict]:
    """按行迭代消息 dict（跳过空行）。"""
    for line in text.split("\n"):
        frame = decode_frame(line)
        if frame is not None:
            yield frame


def parse_client_message(data: dict) -> ClientMessage:
    """校验并解析客户端消息（hello / request）。"""
    return _CLIENT_MESSAGE_ADAPTER.validate_python(data)


def parse_server_message(data: dict) -> ServerMessage:
    """校验并解析服务端消息（hello / hello_error / response / event）。"""
    return _SERVER_MESSAGE_ADAPTER.validate_python(data)


__all__ = [
    "encode_frame",
    "decode_frame",
    "iter_frames",
    "parse_client_message",
    "parse_server_message",
]
