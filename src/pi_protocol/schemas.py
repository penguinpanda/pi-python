"""protocol v2 schema（对齐 TS packages/protocol/src/schemas.ts）。

用 pydantic v2 表达 TypeBox 的 StrictObject 语义：
- 未知字段拒绝（extra="forbid"）；
- Literal 字段自动做联合判别（command / type / role / status）。
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

PROTOCOL_VERSION = 2

Id = Annotated[str, Field(min_length=1)]
Timestamp = Annotated[int, Field(ge=0)]

_STRICT = ConfigDict(extra="forbid")

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
SessionPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]
StopReason = Literal["stop", "length", "tool_use", "error", "aborted"]
AssistantStatus = Literal["streaming", "complete", "error", "aborted"]
ToolStatus = Literal["running", "complete", "error"]


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


class ModelRef(BaseModel):
    model_config = _STRICT

    provider: Id
    id: Id


class ModelCost(BaseModel):
    model_config = _STRICT

    input: Annotated[float, Field(ge=0)]
    output: Annotated[float, Field(ge=0)]
    cacheRead: Annotated[float, Field(ge=0)]
    cacheWrite: Annotated[float, Field(ge=0)]


class ModelMetadata(BaseModel):
    model_config = _STRICT

    provider: Id
    id: Id
    name: Annotated[str, Field(min_length=1)]
    api: Id
    reasoning: bool
    input: list[Literal["text", "image"]]
    contextWindow: Annotated[int, Field(ge=1)]
    maxTokens: Annotated[int, Field(ge=1)]
    cost: ModelCost
    supportedThinkingLevels: Annotated[list[ThinkingLevel], Field(min_length=1)]
    authenticated: bool


# ---------------------------------------------------------------------------
# 内容
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    model_config = _STRICT

    type: Literal["text"]
    text: str


class ThinkingContent(BaseModel):
    model_config = _STRICT

    type: Literal["thinking"]
    thinking: str
    redacted: bool | None = None


class ImageContent(BaseModel):
    model_config = _STRICT

    type: Literal["image"]
    data: str
    mimeType: Annotated[str, Field(min_length=1)]


class ToolCallContent(BaseModel):
    model_config = _STRICT

    type: Literal["tool_call"]
    toolCallId: Id
    toolName: Id
    input: JsonValue


UserContent = Union[TextContent, ImageContent]
AssistantContent = Union[TextContent, ThinkingContent, ToolCallContent]
ToolContent = Union[TextContent, ImageContent]


class UsageCost(BaseModel):
    model_config = _STRICT

    input: Annotated[float, Field(ge=0)]
    output: Annotated[float, Field(ge=0)]
    cacheRead: Annotated[float, Field(ge=0)]
    cacheWrite: Annotated[float, Field(ge=0)]
    total: Annotated[float, Field(ge=0)]


class Usage(BaseModel):
    model_config = _STRICT

    input: Annotated[int, Field(ge=0)]
    output: Annotated[int, Field(ge=0)]
    cacheRead: Annotated[int, Field(ge=0)]
    cacheWrite: Annotated[int, Field(ge=0)]
    reasoning: Annotated[int, Field(ge=0)] | None = None
    totalTokens: Annotated[int, Field(ge=0)]
    cost: UsageCost


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


class UserTranscriptItem(BaseModel):
    model_config = _STRICT

    id: Id
    role: Literal["user"]
    content: list[UserContent]
    timestamp: Timestamp


class AssistantTranscriptItem(BaseModel):
    model_config = _STRICT

    id: Id
    role: Literal["assistant"]
    content: list[AssistantContent]
    status: AssistantStatus
    model: ModelRef
    responseModel: Annotated[str, Field(min_length=1)] | None = None
    usage: Usage | None = None
    stopReason: StopReason | None = None
    errorMessage: str | None = None
    timestamp: Timestamp


class ToolTranscriptItem(BaseModel):
    model_config = _STRICT

    id: Id
    role: Literal["tool"]
    toolCallId: Id
    toolName: Id
    input: JsonValue
    content: list[ToolContent]
    details: JsonValue | None = None
    status: ToolStatus
    isError: bool
    usage: Usage | None = None
    timestamp: Timestamp


TranscriptItem = Union[UserTranscriptItem, AssistantTranscriptItem, ToolTranscriptItem]


class ItemStartedProgress(BaseModel):
    model_config = _STRICT

    type: Literal["item_started"]
    item: TranscriptItem


class AssistantDeltaProgress(BaseModel):
    model_config = _STRICT

    type: Literal["assistant_delta"]
    messageId: Id
    contentIndex: Annotated[int, Field(ge=0)]
    kind: Literal["text", "thinking", "tool_call"]
    delta: str


class ItemUpdatedProgress(BaseModel):
    model_config = _STRICT

    type: Literal["item_updated"]
    item: Union[AssistantTranscriptItem, ToolTranscriptItem]


class ItemFinishedProgress(BaseModel):
    model_config = _STRICT

    type: Literal["item_finished"]
    item: Union[AssistantTranscriptItem, ToolTranscriptItem]


TranscriptProgress = Union[
    ItemStartedProgress,
    AssistantDeltaProgress,
    ItemUpdatedProgress,
    ItemFinishedProgress,
]


# ---------------------------------------------------------------------------
# 会话快照
# ---------------------------------------------------------------------------


class _SessionSummaryBase(BaseModel):
    model_config = _STRICT

    id: Id
    name: str | None = None
    cwd: Annotated[str, Field(min_length=1)]
    createdAt: Timestamp
    updatedAt: Timestamp
    phase: SessionPhase
    model: ModelRef
    thinkingLevel: ThinkingLevel
    attached: bool
    locked: bool


class SessionSummary(_SessionSummaryBase):
    pass


class SessionSnapshot(_SessionSummaryBase):
    revision: Annotated[int, Field(ge=0)]
    transcript: list[TranscriptItem]
    queuedSteer: list[UserTranscriptItem]
    queuedSteerCount: Annotated[int, Field(ge=0)]


class ServerSnapshot(BaseModel):
    model_config = _STRICT

    serverId: Id
    protocolVersion: Literal[2]
    revision: Annotated[int, Field(ge=0)]
    sessions: list[SessionSummary]
    models: list[ModelMetadata]


# ---------------------------------------------------------------------------
# 错误
# ---------------------------------------------------------------------------


ProtocolErrorCode = Literal[
    "auth", "version", "busy", "session_locked", "not_found", "invalid_request"
]


class ProtocolError(BaseModel):
    model_config = _STRICT

    code: ProtocolErrorCode
    message: str
    details: JsonValue | None = None


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------


class ListCommand(BaseModel):
    model_config = _STRICT

    command: Literal["list"]


class CreateCommand(BaseModel):
    model_config = _STRICT

    command: Literal["create"]
    cwd: Annotated[str, Field(min_length=1)] | None = None
    name: str | None = None
    model: ModelRef | None = None
    thinkingLevel: ThinkingLevel | None = None


class AttachCommand(BaseModel):
    model_config = _STRICT

    command: Literal["attach"]
    sessionId: Id


class DetachCommand(BaseModel):
    model_config = _STRICT

    command: Literal["detach"]
    sessionId: Id


class _PromptPayload(BaseModel):
    model_config = _STRICT

    sessionId: Id
    text: str


class PromptCommand(_PromptPayload):
    command: Literal["prompt"]


class SteerCommand(_PromptPayload):
    command: Literal["steer"]


class AbortCommand(BaseModel):
    model_config = _STRICT

    command: Literal["abort"]
    sessionId: Id


class SetModelCommand(BaseModel):
    model_config = _STRICT

    command: Literal["set_model"]
    sessionId: Id
    model: ModelRef


class SetThinkingCommand(BaseModel):
    model_config = _STRICT

    command: Literal["set_thinking"]
    sessionId: Id
    thinkingLevel: ThinkingLevel


Command = Union[
    ListCommand,
    CreateCommand,
    AttachCommand,
    DetachCommand,
    PromptCommand,
    SteerCommand,
    AbortCommand,
    SetModelCommand,
    SetThinkingCommand,
]
CommandName = Literal[
    "list",
    "create",
    "attach",
    "detach",
    "prompt",
    "steer",
    "abort",
    "set_model",
    "set_thinking",
]


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


class _SessionResult(BaseModel):
    model_config = _STRICT

    session: SessionSnapshot


class CreateResult(_SessionResult):
    command: Literal["create"]


class AttachResult(_SessionResult):
    command: Literal["attach"]


class PromptResult(_SessionResult):
    command: Literal["prompt"]


class SteerResult(_SessionResult):
    command: Literal["steer"]


class AbortResult(_SessionResult):
    command: Literal["abort"]


class SetModelResult(_SessionResult):
    command: Literal["set_model"]


class SetThinkingResult(_SessionResult):
    command: Literal["set_thinking"]


class ListResult(BaseModel):
    model_config = _STRICT

    command: Literal["list"]
    sessions: list[SessionSummary]


class DetachResult(BaseModel):
    model_config = _STRICT

    command: Literal["detach"]
    sessionId: Id


CommandResult = Union[
    ListResult,
    CreateResult,
    AttachResult,
    DetachResult,
    PromptResult,
    SteerResult,
    AbortResult,
    SetModelResult,
    SetThinkingResult,
]


def parse_command(data: dict) -> Command:
    """按 command 字面量解析命令。"""
    return TypeAdapter(Command).validate_python(data)


def parse_result(data: dict) -> CommandResult:
    """按 command 字面量解析命令结果。"""
    return TypeAdapter(CommandResult).validate_python(data)


# ---------------------------------------------------------------------------
# 信封
# ---------------------------------------------------------------------------


class ClientHello(BaseModel):
    model_config = _STRICT

    type: Literal["hello"]
    version: Annotated[int, Field(ge=0)]
    token: Annotated[str, Field(min_length=1)]


class RequestEnvelope(BaseModel):
    model_config = _STRICT

    type: Literal["request"]
    id: Id
    request: Command


ClientMessage = Union[ClientHello, RequestEnvelope]


class ServerSnapshotEvent(BaseModel):
    model_config = _STRICT

    type: Literal["server_snapshot"]
    snapshot: ServerSnapshot


class SessionSnapshotEvent(BaseModel):
    model_config = _STRICT

    type: Literal["session_snapshot"]
    snapshot: SessionSnapshot


class SessionProgressEvent(BaseModel):
    model_config = _STRICT

    type: Literal["session_progress"]
    sessionId: Id
    progress: TranscriptProgress


class SessionRemovedEvent(BaseModel):
    model_config = _STRICT

    type: Literal["session_removed"]
    sessionId: Id


ServerEvent = Union[
    ServerSnapshotEvent,
    SessionSnapshotEvent,
    SessionProgressEvent,
    SessionRemovedEvent,
]


class ServerHello(BaseModel):
    model_config = _STRICT

    type: Literal["hello"]
    version: Literal[2]
    connectionId: Id
    snapshot: ServerSnapshot


class ServerHelloError(BaseModel):
    model_config = _STRICT

    type: Literal["hello_error"]
    error: ProtocolError


class ResponseEnvelope(BaseModel):
    model_config = _STRICT

    type: Literal["response"]
    id: Id
    ok: bool
    result: CommandResult | None = None
    error: ProtocolError | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> "ResponseEnvelope":
        if self.ok and self.result is None:
            raise ValueError("ok response requires a result")
        if not self.ok and self.error is None:
            raise ValueError("error response requires an error")
        return self


class EventEnvelope(BaseModel):
    model_config = _STRICT

    type: Literal["event"]
    event: ServerEvent


ServerMessage = Union[ServerHello, ServerHelloError, ResponseEnvelope, EventEnvelope]


__all__ = [
    "PROTOCOL_VERSION",
    "AbortCommand",
    "AbortResult",
    "AssistantDeltaProgress",
    "AssistantTranscriptItem",
    "AttachCommand",
    "AttachResult",
    "ClientHello",
    "ClientMessage",
    "Command",
    "CommandName",
    "CommandResult",
    "CreateCommand",
    "CreateResult",
    "DetachCommand",
    "DetachResult",
    "EventEnvelope",
    "ImageContent",
    "ItemFinishedProgress",
    "ItemStartedProgress",
    "ItemUpdatedProgress",
    "ListCommand",
    "ListResult",
    "ModelCost",
    "ModelMetadata",
    "ModelRef",
    "PromptCommand",
    "PromptResult",
    "ProtocolError",
    "ProtocolErrorCode",
    "RequestEnvelope",
    "ResponseEnvelope",
    "ServerEvent",
    "ServerHello",
    "ServerHelloError",
    "ServerMessage",
    "ServerSnapshot",
    "SessionPhase",
    "SessionProgressEvent",
    "SessionRemovedEvent",
    "SessionSnapshot",
    "SessionSummary",
    "SetModelCommand",
    "SetModelResult",
    "SetThinkingCommand",
    "SetThinkingResult",
    "SteerCommand",
    "SteerResult",
    "StopReason",
    "TextContent",
    "ThinkingContent",
    "ThinkingLevel",
    "ToolCallContent",
    "ToolTranscriptItem",
    "TranscriptItem",
    "TranscriptProgress",
    "Usage",
    "UsageCost",
    "UserTranscriptItem",
]
