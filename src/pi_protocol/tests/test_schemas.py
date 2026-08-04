"""protocol v2 schema 测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pi_protocol.schemas import (
    PROTOCOL_VERSION,
    AssistantTranscriptItem,
    ClientHello,
    CreateCommand,
    CreateResult,
    ListCommand,
    ListResult,
    ModelCost,
    ModelMetadata,
    ModelRef,
    PromptCommand,
    ProtocolError,
    RequestEnvelope,
    ResponseEnvelope,
    ServerHello,
    ServerSnapshot,
    SessionProgressEvent,
    SessionSnapshot,
    SessionSummary,
    TextContent,
    UserTranscriptItem,
    parse_command,
    parse_result,
)


def _model_ref() -> ModelRef:
    return ModelRef(provider="faux", id="faux-1")


def _model_metadata() -> ModelMetadata:
    return ModelMetadata(
        provider="faux",
        id="faux-1",
        name="Faux 1",
        api="openai-completions",
        reasoning=False,
        input=["text"],
        contextWindow=128000,
        maxTokens=8192,
        cost=ModelCost(input=0.1, output=0.2, cacheRead=0.05, cacheWrite=0.15),
        supportedThinkingLevels=["off"],
        authenticated=True,
    )


def _user_item() -> UserTranscriptItem:
    return UserTranscriptItem(
        id="u1",
        role="user",
        content=[TextContent(type="text", text="hello")],
        timestamp=1000,
    )


def _assistant_item() -> AssistantTranscriptItem:
    return AssistantTranscriptItem(
        id="a1",
        role="assistant",
        content=[TextContent(type="text", text="hi")],
        status="complete",
        model=_model_ref(),
        timestamp=2000,
    )


def _snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        id="s1",
        name="test",
        cwd="/tmp/proj",
        createdAt=1000,
        updatedAt=2000,
        phase="idle",
        model=_model_ref(),
        thinkingLevel="off",
        attached=True,
        locked=False,
        revision=3,
        transcript=[_user_item(), _assistant_item()],
        queuedSteer=[],
        queuedSteerCount=0,
    )


class TestRoundTrip:
    def test_session_snapshot_round_trip(self):
        snapshot = _snapshot()
        restored = SessionSnapshot.model_validate(snapshot.model_dump(mode="json"))
        assert restored == snapshot

    def test_command_round_trip(self):
        for command in (
            ListCommand(command="list"),
            CreateCommand(command="create", cwd="/tmp", model=_model_ref()),
            PromptCommand(command="prompt", sessionId="s1", text="hi"),
        ):
            restored = parse_command(command.model_dump(mode="json"))
            assert restored == command

    def test_result_round_trip(self):
        result = CreateResult(command="create", session=_snapshot())
        restored = parse_result(result.model_dump(mode="json"))
        assert restored == result

    def test_server_snapshot_round_trip(self):
        summary_data = {
            key: _snapshot().model_dump(mode="json")[key]
            for key in (
                "id",
                "name",
                "cwd",
                "createdAt",
                "updatedAt",
                "phase",
                "model",
                "thinkingLevel",
                "attached",
                "locked",
            )
        }
        server = ServerSnapshot(
            serverId="server-1",
            protocolVersion=PROTOCOL_VERSION,
            revision=1,
            sessions=[SessionSummary.model_validate(summary_data)],
            models=[_model_metadata()],
        )
        restored = ServerSnapshot.model_validate(server.model_dump(mode="json"))
        assert restored == server

    def test_response_envelope_round_trip(self):
        ok = ResponseEnvelope(
            type="response",
            id="r1",
            ok=True,
            result=ListResult(command="list", sessions=[]),
        )
        assert ResponseEnvelope.model_validate(ok.model_dump(mode="json")) == ok
        err = ResponseEnvelope(
            type="response",
            id="r2",
            ok=False,
            error=ProtocolError(code="not_found", message="missing"),
        )
        assert ResponseEnvelope.model_validate(err.model_dump(mode="json")) == err


class TestValidation:
    def test_unknown_command_rejected(self):
        with pytest.raises(ValidationError):
            parse_command({"command": "explode"})

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ListCommand.model_validate({"command": "list", "extra": 1})

    def test_ok_response_requires_result(self):
        with pytest.raises(ValidationError):
            ResponseEnvelope(type="response", id="r1", ok=True)

    def test_error_response_requires_error(self):
        with pytest.raises(ValidationError):
            ResponseEnvelope(type="response", id="r2", ok=False)

    def test_protocol_version_literal(self):
        with pytest.raises(ValidationError):
            ServerSnapshot(
                serverId="s",
                protocolVersion=1,
                revision=0,
                sessions=[],
                models=[],
            )

    def test_client_hello(self):
        hello = ClientHello(type="hello", version=2, token="abc")
        assert hello.version == 2

    def test_request_envelope(self):
        envelope = RequestEnvelope(
            type="request",
            id="q1",
            request=PromptCommand(command="prompt", sessionId="s1", text="hi"),
        )
        restored = RequestEnvelope.model_validate(envelope.model_dump(mode="json"))
        assert restored.request.sessionId == "s1"

    def test_server_hello(self):
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
        assert hello.version == 2

    def test_session_progress_event(self):
        event = SessionProgressEvent(
            type="session_progress",
            sessionId="s1",
            progress={
                "type": "assistant_delta",
                "messageId": "a1",
                "contentIndex": 0,
                "kind": "text",
                "delta": "hel",
            },
        )
        assert event.progress.delta == "hel"
