"""JSONL framing 测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pi_protocol.framing import (
    decode_frame,
    encode_frame,
    iter_frames,
    parse_client_message,
    parse_server_message,
)
from pi_protocol.schemas import (
    PROTOCOL_VERSION,
    ClientHello,
    ListCommand,
    ProtocolError,
    RequestEnvelope,
    ResponseEnvelope,
    ServerHello,
    ServerSnapshot,
)


class TestFrames:
    def test_encode_decode_round_trip(self):
        message = ClientHello(type="hello", version=2, token="abc")
        line = encode_frame(message)
        assert line.endswith("\n")
        assert decode_frame(line) == message.model_dump(mode="json")

    def test_dict_encoding(self):
        line = encode_frame({"type": "hello", "version": 2, "token": "x"})
        assert decode_frame(line)["token"] == "x"

    def test_iter_frames_skips_blanks(self):
        text = '{"a":1}\n\n{"b":2}\n'
        frames = list(iter_frames(text))
        assert frames == [{"a": 1}, {"b": 2}]

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            encode_frame(42)


class TestParsing:
    def test_parse_client_hello(self):
        message = parse_client_message({"type": "hello", "version": 2, "token": "abc"})
        assert isinstance(message, ClientHello)

    def test_parse_client_request(self):
        message = parse_client_message(
            {
                "type": "request",
                "id": "q1",
                "request": {"command": "list"},
            }
        )
        assert isinstance(message, RequestEnvelope)
        assert isinstance(message.request, ListCommand)

    def test_parse_client_invalid(self):
        with pytest.raises(ValidationError):
            parse_client_message({"type": "bogus"})

    def test_parse_server_hello(self):
        message = parse_server_message(
            {
                "type": "hello",
                "version": PROTOCOL_VERSION,
                "connectionId": "c1",
                "snapshot": {
                    "serverId": "srv",
                    "protocolVersion": PROTOCOL_VERSION,
                    "revision": 0,
                    "sessions": [],
                    "models": [],
                },
            }
        )
        assert isinstance(message, ServerHello)
        assert isinstance(message.snapshot, ServerSnapshot)

    def test_parse_server_response(self):
        message = parse_server_message(
            {
                "type": "response",
                "id": "r1",
                "ok": False,
                "error": {"code": "not_found", "message": "missing"},
            }
        )
        assert isinstance(message, ResponseEnvelope)
        assert isinstance(message.error, ProtocolError)

    def test_parse_server_event(self):
        message = parse_server_message(
            {
                "type": "event",
                "event": {
                    "type": "session_removed",
                    "sessionId": "s1",
                },
            }
        )
        assert message.type == "event"
        assert message.event.sessionId == "s1"
