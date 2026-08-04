"""服务端会话处理器：命令分发、快照构建、事件推送。"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable

from pi_agent import Agent, AgentOptions
from pi_protocol.schemas import (
    PROTOCOL_VERSION,
    AbortCommand,
    AbortResult,
    AssistantTranscriptItem,
    AttachCommand,
    AttachResult,
    ClientHello,
    Command,
    CreateCommand,
    CreateResult,
    DetachCommand,
    DetachResult,
    EventEnvelope,
    ListCommand,
    ListResult,
    ModelCost,
    ModelMetadata,
    ModelRef,
    PromptCommand,
    PromptResult,
    ProtocolError,
    RequestEnvelope,
    ResponseEnvelope,
    ServerHello,
    ServerHelloError,
    ServerSnapshot,
    ServerSnapshotEvent,
    SessionSnapshot,
    SessionSummary,
    SetModelCommand,
    SetModelResult,
    SetThinkingCommand,
    SetThinkingResult,
    SessionSnapshotEvent,
    SteerCommand,
    SteerResult,
    ToolTranscriptItem,
    Usage,
    UsageCost,
    UserTranscriptItem,
)

_STOP_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "tool_use": "tool_use",
    "error": "error",
    "aborted": "aborted",
}


class ProtocolException(Exception):
    """内部协议错误（响应时转换为 ProtocolError 模型）。"""

    def __init__(self, code: str, message: str, details=None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id() -> str:
    return uuid.uuid4().hex


def _content_to_protocol(blocks: Any) -> list[dict]:
    """pi 内容块 → protocol 内容 dict。"""
    if isinstance(blocks, str):
        return [{"type": "text", "text": blocks}]
    result: list[dict] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "image":
            result.append(
                {
                    "type": "image",
                    "data": block.get("data") or block.get("url") or "",
                    "mimeType": block.get("mimeType") or block.get("mime_type") or "image/png",
                }
            )
        elif block_type == "thinking":
            result.append(
                {
                    "type": "thinking",
                    "thinking": block.get("thinking", ""),
                    **({"redacted": True} if block.get("redacted") else {}),
                }
            )
        elif block_type == "toolCall":
            result.append(
                {
                    "type": "tool_call",
                    "toolCallId": block.get("id", ""),
                    "toolName": block.get("name", ""),
                    "input": block.get("arguments"),
                }
            )
    return result


def _usage_to_protocol(usage: Any) -> Usage | None:
    if not usage:
        return None
    cost = usage.get("cost") or {}
    return Usage(
        input=int(usage.get("input") or 0),
        output=int(usage.get("output") or 0),
        cacheRead=int(usage.get("cache_read") or 0),
        cacheWrite=int(usage.get("cache_write") or 0),
        reasoning=(int(usage["reasoning"]) if usage.get("reasoning") is not None else None),
        totalTokens=int(usage.get("total_tokens") or 0),
        cost=UsageCost(
            input=float(cost.get("input") or 0),
            output=float(cost.get("output") or 0),
            cacheRead=float(cost.get("cacheRead") or 0),
            cacheWrite=float(cost.get("cacheWrite") or 0),
            total=float(cost.get("total") or 0),
        ),
    )


def _transcript_item(message: dict) -> dict | None:
    role = message.get("role")
    timestamp = int(message.get("timestamp") or _now_ms())
    if role == "user":
        return UserTranscriptItem(
            id=str(message.get("id") or _new_id()),
            role="user",
            content=_content_to_protocol(message.get("content")),
            timestamp=timestamp,
        ).model_dump(mode="json")
    if role == "assistant":
        model = message.get("model") or ""
        provider = message.get("provider") or ""
        stop_reason = _STOP_REASON_MAP.get(message.get("stop_reason"))
        return AssistantTranscriptItem(
            id=str(message.get("id") or _new_id()),
            role="assistant",
            content=_content_to_protocol(message.get("content")),
            status="error" if message.get("stop_reason") == "error" else "complete",
            model=ModelRef(provider=provider or "?", id=model or "?"),
            usage=_usage_to_protocol(message.get("usage")),
            stopReason=stop_reason,
            errorMessage=message.get("error_message"),
            timestamp=timestamp,
        ).model_dump(mode="json")
    if role == "toolResult":
        return ToolTranscriptItem(
            id=str(message.get("id") or _new_id()),
            role="tool",
            toolCallId=str(message.get("tool_call_id") or _new_id()),
            toolName=str(message.get("tool_name") or "?"),
            input=message.get("tool_input"),
            content=_content_to_protocol(message.get("content")),
            status="error" if message.get("is_error") else "complete",
            isError=bool(message.get("is_error")),
            usage=_usage_to_protocol(message.get("usage")),
            timestamp=timestamp,
        ).model_dump(mode="json")
    return None


def _model_ref(model) -> ModelRef:
    return ModelRef(provider=model.provider, id=model.id)


def _model_metadata(model) -> ModelMetadata:
    cost = getattr(model, "cost", None)
    model_cost = ModelCost(
        input=float(getattr(cost, "input", 0) or 0),
        output=float(getattr(cost, "output", 0) or 0),
        cacheRead=float(getattr(cost, "cache_read", 0) or 0),
        cacheWrite=float(getattr(cost, "cache_write", 0) or 0),
    )
    return ModelMetadata(
        provider=model.provider,
        id=model.id,
        name=getattr(model, "name", None) or model.id,
        api=getattr(model, "api", None) or "openai-completions",
        reasoning=bool(getattr(model, "reasoning", False)),
        input=list(getattr(model, "input", ["text"]) or ["text"]),
        contextWindow=int(getattr(model, "context_window", 128000) or 128000),
        maxTokens=int(getattr(model, "max_tokens", 16384) or 16384),
        cost=model_cost,
        supportedThinkingLevels=["off", "low", "medium", "high"] if model.reasoning else ["off"],
        authenticated=True,
    )


class ServerSession:
    """单个会话的服务端包装：快照 + 事件订阅 + 命令执行。"""

    def __init__(self, session, *, attached: bool = False) -> None:
        self.session = session
        self.attached = attached
        self.locked = False
        self.created_at = _now_ms()
        self.updated_at = _now_ms()
        self.revision = 0
        self._unsubscribe = session.subscribe(self._on_event)

    def _on_event(self, event: dict) -> None:
        if event.get("type") in ("message_end", "agent_settled"):
            self.updated_at = _now_ms()
            self.revision += 1

    def dispose(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def summary(self) -> SessionSummary:
        session = self.session
        model = session.model
        return SessionSummary(
            id=session.session_id,
            name=session.session_name,
            cwd=session.cwd,
            createdAt=self.created_at,
            updatedAt=self.updated_at,
            phase="turn" if session.is_streaming else "idle",
            model=_model_ref(model) if model is not None else ModelRef(provider="?", id="?"),
            thinkingLevel=session.thinking_level,
            attached=self.attached,
            locked=self.locked,
        )

    def snapshot(self) -> SessionSnapshot:
        session = self.session
        transcript = [
            item
            for message in session.get_messages()
            if (item := _transcript_item(message)) is not None
        ]
        return SessionSnapshot(
            **self.summary().model_dump(mode="json"),
            revision=self.revision,
            transcript=transcript,
            queuedSteer=[],
            queuedSteerCount=session.pending_message_count,
        )

    async def prompt(self, text: str) -> SessionSnapshot:
        await self.session.prompt(text)
        self.updated_at = _now_ms()
        self.revision += 1
        return self.snapshot()


class PiServer:
    """常驻服务：会话注册表 + 命令分发 + 快照推送。"""

    def __init__(
        self,
        *,
        model_runtime=None,
        session_factory: Callable[[str], Any] | None = None,
        token: str | None = None,
        store=None,
    ) -> None:
        self.model_runtime = model_runtime
        self._session_factory = session_factory
        self.token = token
        self.store = store
        self.server_id = _new_id()
        self.revision = 0
        self.sessions: dict[str, ServerSession] = {}

    # ------------------------------------------------------------------
    # 消息入口
    # ------------------------------------------------------------------

    async def handle_line(self, line: str) -> list[dict]:
        """处理一行 JSONL 客户端消息，返回待发送的服务端消息。"""
        from pi_protocol.framing import decode_frame, parse_client_message

        data = decode_frame(line)
        if data is None:
            return []
        try:
            message = parse_client_message(data)
        except Exception as exc:
            return [
                ServerHelloError(
                    type="hello_error",
                    error=ProtocolError(code="invalid_request", message=f"Invalid message: {exc}"),
                ).model_dump(mode="json")
            ]
        return await self.handle_message(message)

    async def handle_message(self, message) -> list[dict]:
        if isinstance(message, ClientHello):
            return await self._handle_hello(message)
        if isinstance(message, RequestEnvelope):
            return await self._handle_request(message)
        return [
            ServerHelloError(
                type="hello_error",
                error=ProtocolError(code="invalid_request", message="Unknown message type"),
            ).model_dump(mode="json")
        ]

    async def _handle_hello(self, hello: ClientHello) -> list[dict]:
        if hello.version != PROTOCOL_VERSION:
            return [
                ServerHelloError(
                    type="hello_error",
                    error=ProtocolError(
                        code="version",
                        message=f"Unsupported protocol version: {hello.version}",
                    ),
                ).model_dump(mode="json")
            ]
        if self.token is not None and hello.token != self.token:
            return [
                ServerHelloError(
                    type="hello_error",
                    error=ProtocolError(code="auth", message="Invalid token"),
                ).model_dump(mode="json")
            ]
        return [
            ServerHello(
                type="hello",
                version=PROTOCOL_VERSION,
                connectionId=_new_id(),
                snapshot=self.server_snapshot(),
            ).model_dump(mode="json")
        ]

    async def _handle_request(self, envelope: RequestEnvelope) -> list[dict]:
        command = envelope.request
        try:
            result, events = await self._dispatch(command)
            response = ResponseEnvelope(
                type="response",
                id=envelope.id,
                ok=True,
                result=result,
            ).model_dump(mode="json")
        except ProtocolException as error:
            response = ResponseEnvelope(
                type="response",
                id=envelope.id,
                ok=False,
                error=ProtocolError(code=error.code, message=error.message, details=error.details),
            ).model_dump(mode="json")
            events: list[dict] = []
        return [response, *events]

    # ------------------------------------------------------------------
    # 命令分发
    # ------------------------------------------------------------------

    async def _dispatch(self, command: Command):
        if isinstance(command, ListCommand):
            return self._cmd_list(), []
        if isinstance(command, CreateCommand):
            return await self._cmd_create(command)
        if isinstance(command, AttachCommand):
            return await self._cmd_attach(command)
        if isinstance(command, DetachCommand):
            return await self._cmd_detach(command)
        if isinstance(command, PromptCommand):
            return await self._cmd_prompt(command)
        if isinstance(command, SteerCommand):
            return await self._cmd_steer(command)
        if isinstance(command, AbortCommand):
            return await self._cmd_abort(command)
        if isinstance(command, SetModelCommand):
            return await self._cmd_set_model(command)
        if isinstance(command, SetThinkingCommand):
            return await self._cmd_set_thinking(command)
        raise ProtocolException("invalid_request", "Unknown command")

    def _cmd_list(self) -> ListResult:
        return ListResult(
            command="list",
            sessions=[session.summary() for session in self.sessions.values()],
        )

    async def _cmd_create(self, command: CreateCommand):
        cwd = command.cwd or "."
        if self._session_factory is not None:
            session = self._session_factory(cwd)
        else:
            session = await self._create_default_session(cwd)
        server_session = ServerSession(session, attached=True)
        self.sessions[server_session.session.session_id] = server_session
        snapshot = server_session.snapshot()
        result = CreateResult(command="create", session=snapshot)
        events = self._session_snapshot_events(snapshot)
        return result, events

    async def _create_default_session(self, cwd: str):
        runtime = self.model_runtime
        if runtime is None:
            raise ProtocolException("invalid_request", "No model runtime configured")
        available = await runtime.get_available()
        if not available:
            raise ProtocolException("invalid_request", "No available models")
        model = available[0]
        agent = Agent(
            AgentOptions(
                system_prompt="You are a helpful coding assistant.",
                model=model,
                stream_fn=runtime.stream,
            )
        )
        from pi_coding_agent._session import AgentSession
        from pi_coding_agent._session_manager import SessionManager

        return AgentSession(
            agent=agent,
            session_manager=SessionManager.in_memory(cwd=cwd),
            cwd=cwd,
            model=model,
            model_runtime=runtime,
        )

    def _get(self, session_id: str) -> ServerSession:
        server_session = self.sessions.get(session_id)
        if server_session is None:
            raise ProtocolException("not_found", f"Session not found: {session_id}")
        return server_session

    async def _cmd_attach(self, command: AttachCommand):
        server_session = self._get(command.sessionId)
        server_session.attached = True
        snapshot = server_session.snapshot()
        return AttachResult(command="attach", session=snapshot), self._session_snapshot_events(
            snapshot
        )

    async def _cmd_detach(self, command: DetachCommand):
        server_session = self._get(command.sessionId)
        server_session.attached = False
        return DetachResult(command="detach", sessionId=command.sessionId), []

    async def _cmd_prompt(self, command: PromptCommand):
        server_session = self._get(command.sessionId)
        if server_session.locked:
            raise ProtocolException("session_locked", "Session is locked")
        snapshot = await server_session.prompt(command.text)
        return PromptResult(command="prompt", session=snapshot), self._session_snapshot_events(
            snapshot
        )

    async def _cmd_steer(self, command: SteerCommand):
        server_session = self._get(command.sessionId)
        server_session.session.steer(command.text)
        snapshot = server_session.snapshot()
        return SteerResult(command="steer", session=snapshot), self._session_snapshot_events(
            snapshot
        )

    async def _cmd_abort(self, command: AbortCommand):
        server_session = self._get(command.sessionId)
        await server_session.session.abort()
        snapshot = server_session.snapshot()
        return AbortResult(command="abort", session=snapshot), self._session_snapshot_events(
            snapshot
        )

    async def _cmd_set_model(self, command: SetModelCommand):
        server_session = self._get(command.sessionId)
        model = None
        if self.model_runtime is not None:
            model = self.model_runtime.get_model(command.model.provider, command.model.id)
        if model is None:
            raise ProtocolException(
                "not_found",
                f"Model not found: {command.model.provider}/{command.model.id}",
            )
        await server_session.session.set_model(model)
        snapshot = server_session.snapshot()
        return SetModelResult(command="set_model", session=snapshot), self._session_snapshot_events(
            snapshot
        )

    async def _cmd_set_thinking(self, command: SetThinkingCommand):
        server_session = self._get(command.sessionId)
        server_session.session.set_thinking_level(command.thinkingLevel)
        snapshot = server_session.snapshot()
        return SetThinkingResult(
            command="set_thinking", session=snapshot
        ), self._session_snapshot_events(snapshot)

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------

    def server_snapshot(self) -> ServerSnapshot:
        models: list[ModelMetadata] = []
        if self.model_runtime is not None:
            for model in self.model_runtime.get_models():
                try:
                    models.append(_model_metadata(model))
                except Exception:
                    continue
        return ServerSnapshot(
            serverId=self.server_id,
            protocolVersion=PROTOCOL_VERSION,
            revision=self.revision,
            sessions=[session.summary() for session in self.sessions.values()],
            models=models,
        )

    def _session_snapshot_events(self, snapshot: SessionSnapshot) -> list[dict]:
        self.revision += 1
        return [
            EventEnvelope(
                type="event",
                event=ServerSnapshotEvent(type="server_snapshot", snapshot=self.server_snapshot()),
            ).model_dump(mode="json"),
            EventEnvelope(
                type="event",
                event=SessionSnapshotEvent(
                    type="session_snapshot",
                    snapshot=snapshot,
                ),
            ).model_dump(mode="json"),
        ]


__all__ = ["PiServer", "ServerSession"]
