"""RPC 模式（对齐 TS modes/rpc/rpc-mode.ts）。

协议：

- 命令（stdin）：JSON 对象，`type` 字段 + 可选 `id`；
- 响应（stdout）：`{"id", "type": "response", "command", "success", data|error}`；
- 事件（stdout）：AgentSession 事件逐条转发；
- 扩展 UI：`extension_ui_request` 请求 + `extension_ui_response` 响应。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from dataclasses import asdict
from typing import Any, Callable

from pi_ai.models.models_store import model_to_dict

from .._session import AgentSession
from ..model_runtime import ModelRuntime
from ..tools._bash import _run_command
from .jsonl import serialize_json_line
from .rpc_types import error_response, success_response


# ---------------------------------------------------------------------------
# 扩展 UI 上下文（2.2）
# ---------------------------------------------------------------------------


class RpcUiContext:
    """RPC 模式下的 UIContext：UI 操作转为 extension_ui_request 转发。"""

    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self._emit = emit
        self._pending: dict[str, asyncio.Future] = {}
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"ui_{self._counter}"

    async def _request(
        self,
        request: dict[str, Any],
        *,
        default: Any,
        parse: Callable[[dict[str, Any]], Any],
        timeout: float | None = None,
    ) -> Any:
        request_id = self._new_id()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[request_id] = future
        try:
            self._emit({"type": "extension_ui_request", "id": request_id, **request})
            if timeout is not None:
                try:
                    response = await asyncio.wait_for(future, timeout=timeout)
                except asyncio.TimeoutError:
                    return default
            else:
                response = await future
            return parse(response)
        finally:
            self._pending.pop(request_id, None)

    async def select(
        self,
        title: str,
        options: list[str],
        timeout: float | None = None,
    ) -> str | None:
        return await self._request(
            {"method": "select", "title": title, "options": list(options)},
            default=None,
            parse=lambda response: (
                response.get("value") if "value" in response else None
            ),
            timeout=timeout,
        )

    async def confirm(
        self,
        title: str,
        message: str,
        timeout: float | None = None,
    ) -> bool:
        return await self._request(
            {"method": "confirm", "title": title, "message": message},
            default=False,
            parse=lambda response: (
                response.get("confirmed")
                if "confirmed" in response
                else False
            ),
            timeout=timeout,
        )

    async def input(
        self,
        title: str,
        placeholder: str | None = None,
        timeout: float | None = None,
    ) -> str | None:
        return await self._request(
            {"method": "input", "title": title, "placeholder": placeholder},
            default=None,
            parse=lambda response: (
                response.get("value") if "value" in response else None
            ),
            timeout=timeout,
        )

    async def editor(self, title: str, prefill: str | None = None) -> str | None:
        return await self._request(
            {"method": "editor", "title": title, "prefill": prefill},
            default=None,
            parse=lambda response: (
                response.get("value") if "value" in response else None
            ),
        )

    def notify(self, message: str, notify_type: str | None = None) -> None:
        """fire-and-forget 通知（无需响应）。"""
        self._emit({
            "type": "extension_ui_request",
            "id": self._new_id(),
            "method": "notify",
            "message": message,
            "notifyType": notify_type,
        })

    def set_status(self, key: str, text: str | None) -> None:
        self._emit({
            "type": "extension_ui_request",
            "id": self._new_id(),
            "method": "setStatus",
            "statusKey": key,
            "statusText": text,
        })

    def set_title(self, title: str) -> None:
        self._emit({
            "type": "extension_ui_request",
            "id": self._new_id(),
            "method": "setTitle",
            "title": title,
        })

    def set_editor_text(self, text: str) -> None:
        self._emit({
            "type": "extension_ui_request",
            "id": self._new_id(),
            "method": "set_editor_text",
            "text": text,
        })

    def resolve_response(self, response: dict[str, Any]) -> None:
        """处理客户端返回的 extension_ui_response。"""
        request_id = response.get("id")
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(response)

    def has_pending(self) -> bool:
        return bool(self._pending)


# ---------------------------------------------------------------------------
# RPC 消息处理器（2.1）
# ---------------------------------------------------------------------------


class RpcMessageHandler:
    """RPC 命令分发器。"""

    def __init__(
        self,
        session: AgentSession,
        model_runtime: ModelRuntime,
        *,
        ui_context: RpcUiContext | None = None,
        session_factory: Callable[[], AgentSession] | None = None,
    ) -> None:
        self.session = session
        self.model_runtime = model_runtime
        self.ui_context = ui_context or RpcUiContext(emit=lambda _obj: None)
        # new_session 用：创建全新 AgentSession 的工厂（含新 SessionManager/Agent）。
        self.session_factory = session_factory
        self.created_sessions: list[AgentSession] = []
        # 后台 prompt 任务（命令循环不阻塞）。
        self._prompt_tasks: set[asyncio.Task] = set()

        self._handlers: dict[str, Callable[[dict, str | None], Any]] = {
            "prompt": self._handle_prompt,
            "steer": self._handle_steer,
            "follow_up": self._handle_follow_up,
            "abort": self._handle_abort,
            "new_session": self._handle_new_session,
            "get_state": self._handle_get_state,
            "set_model": self._handle_set_model,
            "cycle_model": self._handle_cycle_model,
            "get_available_models": self._handle_get_available_models,
            "set_thinking_level": self._handle_set_thinking_level,
            "cycle_thinking_level": self._handle_cycle_thinking_level,
            "get_available_thinking_levels": self._handle_get_available_thinking_levels,
            "set_steering_mode": self._handle_set_steering_mode,
            "set_follow_up_mode": self._handle_set_follow_up_mode,
            "compact": self._handle_compact,
            "set_auto_compaction": self._handle_set_auto_compaction,
            "set_auto_retry": self._handle_set_auto_retry,
            "abort_retry": self._handle_abort_retry,
            "bash": self._handle_bash,
            "abort_bash": self._handle_abort_bash,
            "get_session_stats": self._handle_get_session_stats,
            "export_html": self._handle_export_html,
            "switch_session": self._handle_not_implemented,
            "fork": self._handle_not_implemented,
            "clone": self._handle_not_implemented,
            "get_fork_messages": self._handle_not_implemented,
            "get_tree": self._handle_not_implemented,
            "get_entries": self._handle_get_entries,
            "get_last_assistant_text": self._handle_get_last_assistant_text,
            "set_session_name": self._handle_set_session_name,
            "get_messages": self._handle_get_messages,
            "get_commands": self._handle_get_commands,
        }

    async def handle_command(self, cmd: dict) -> dict | None:
        """分发命令，返回响应 dict（None 表示无响应）。"""
        command_type = cmd.get("type")
        command_id = cmd.get("id")
        if not isinstance(command_type, str) or not command_type:
            return error_response(command_id, "parse", "Command is missing a type")
        handler = self._handlers.get(command_type)
        if handler is None:
            return error_response(command_id, command_type, f"Unknown command: {command_type}")
        try:
            return await handler(cmd, command_id)
        except Exception as exc:
            return error_response(command_id, command_type, str(exc))

    # ------------------------------------------------------------------
    # 提示类
    # ------------------------------------------------------------------

    async def _handle_prompt(self, cmd: dict, command_id: str | None) -> dict:
        message = cmd.get("message")
        if not isinstance(message, str) or not message.strip():
            return error_response(command_id, "prompt", "Message is required")

        async def _run() -> None:
            try:
                await self.session.prompt(message)
            except Exception:
                pass  # 失败通过事件流呈现（stop_reason=error）

        task = asyncio.create_task(_run())
        self._prompt_tasks.add(task)
        task.add_done_callback(self._prompt_tasks.discard)
        return success_response(command_id, "prompt")

    async def _handle_steer(self, cmd: dict, command_id: str | None) -> dict:
        message = cmd.get("message")
        if not isinstance(message, str) or not message.strip():
            return error_response(command_id, "steer", "Message is required")
        self.session.steer(message)
        return success_response(command_id, "steer")

    async def _handle_follow_up(self, cmd: dict, command_id: str | None) -> dict:
        message = cmd.get("message")
        if not isinstance(message, str) or not message.strip():
            return error_response(command_id, "follow_up", "Message is required")
        self.session.follow_up(message)
        return success_response(command_id, "follow_up")

    async def _handle_abort(self, _cmd: dict, command_id: str | None) -> dict:
        await self.session.abort()
        return success_response(command_id, "abort")

    async def _handle_new_session(self, _cmd: dict, command_id: str | None) -> dict:
        if self.session_factory is None:
            return error_response(command_id, "new_session", "Session factory not configured")
        try:
            result = self.session_factory()
            new_session = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            return error_response(command_id, "new_session", str(exc))
        self.created_sessions.append(new_session)
        self.session = new_session
        return success_response(command_id, "new_session", {"cancelled": False})

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------

    async def _handle_get_state(self, _cmd: dict, command_id: str | None) -> dict:
        session = self.session
        state = {
            "model": model_to_dict(session.model) if session.model is not None else None,
            "thinkingLevel": session.thinking_level,
            "isStreaming": session.is_streaming,
            "isCompacting": session.is_compacting,
            "steeringMode": session.steering_mode,
            "followUpMode": session.follow_up_mode,
            "sessionFile": session.session_file,
            "sessionId": session.session_id,
            "sessionName": session.session_name,
            "autoCompactionEnabled": session.auto_compaction_enabled,
            "messageCount": len(session.get_messages()),
            "pendingMessageCount": session.pending_message_count,
        }
        return success_response(command_id, "get_state", state)

    # ------------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------------

    async def _handle_set_model(self, cmd: dict, command_id: str | None) -> dict:
        provider = cmd.get("provider")
        model_id = cmd.get("modelId")
        if not isinstance(provider, str) or not isinstance(model_id, str):
            return error_response(command_id, "set_model", "provider and modelId are required")
        models = await self.model_runtime.get_available()
        model = next(
            (
                candidate
                for candidate in models
                if candidate.provider == provider and candidate.id == model_id
            ),
            None,
        )
        if model is None:
            return error_response(
                command_id, "set_model", f"Model not found: {provider}/{model_id}"
            )
        await self.session.set_model(model)
        return success_response(command_id, "set_model", model_to_dict(model))

    async def _handle_cycle_model(self, _cmd: dict, command_id: str | None) -> dict:
        result = await self.session.cycle_model(1)
        if result is None:
            return success_response(command_id, "cycle_model", None)
        return success_response(
            command_id,
            "cycle_model",
            {
                "model": model_to_dict(result.model),
                "thinkingLevel": result.thinking_level,
                "isScoped": result.is_scoped,
            },
        )

    async def _handle_get_available_models(
        self, _cmd: dict, command_id: str | None
    ) -> dict:
        models = await self.model_runtime.get_available()
        return success_response(
            command_id,
            "get_available_models",
            {"models": [model_to_dict(model) for model in models]},
        )

    # ------------------------------------------------------------------
    # 思维级别
    # ------------------------------------------------------------------

    async def _handle_set_thinking_level(self, cmd: dict, command_id: str | None) -> dict:
        level = cmd.get("level")
        if not isinstance(level, str):
            return error_response(command_id, "set_thinking_level", "level is required")
        self.session.set_thinking_level(level)
        return success_response(command_id, "set_thinking_level")

    async def _handle_cycle_thinking_level(
        self, _cmd: dict, command_id: str | None
    ) -> dict:
        level = self.session.cycle_thinking_level()
        if level is None:
            return success_response(command_id, "cycle_thinking_level", None)
        return success_response(
            command_id, "cycle_thinking_level", {"level": level}
        )

    async def _handle_get_available_thinking_levels(
        self, _cmd: dict, command_id: str | None
    ) -> dict:
        return success_response(
            command_id,
            "get_available_thinking_levels",
            {"levels": self.session.get_available_thinking_levels()},
        )

    # ------------------------------------------------------------------
    # 队列模式
    # ------------------------------------------------------------------

    async def _handle_set_steering_mode(self, cmd: dict, command_id: str | None) -> dict:
        mode = cmd.get("mode")
        if mode not in ("all", "one-at-a-time"):
            return error_response(
                command_id, "set_steering_mode", "mode must be 'all' or 'one-at-a-time'"
            )
        self.session.set_steering_mode(mode)
        return success_response(command_id, "set_steering_mode")

    async def _handle_set_follow_up_mode(self, cmd: dict, command_id: str | None) -> dict:
        mode = cmd.get("mode")
        if mode not in ("all", "one-at-a-time"):
            return error_response(
                command_id, "set_follow_up_mode", "mode must be 'all' or 'one-at-a-time'"
            )
        self.session.set_follow_up_mode(mode)
        return success_response(command_id, "set_follow_up_mode")

    # ------------------------------------------------------------------
    # 压缩 / 重试
    # ------------------------------------------------------------------

    async def _handle_compact(self, cmd: dict, command_id: str | None) -> dict:
        result = await self.session.compact(cmd.get("customInstructions"))
        if result is None:
            return error_response(
                command_id, "compact", "Nothing to compact"
            )
        return success_response(command_id, "compact", asdict(result))

    async def _handle_set_auto_compaction(
        self, cmd: dict, command_id: str | None
    ) -> dict:
        enabled = cmd.get("enabled")
        if not isinstance(enabled, bool):
            return error_response(
                command_id, "set_auto_compaction", "enabled must be a boolean"
            )
        self.session.set_auto_compaction_enabled(enabled)
        return success_response(command_id, "set_auto_compaction")

    async def _handle_set_auto_retry(self, cmd: dict, command_id: str | None) -> dict:
        enabled = cmd.get("enabled")
        if not isinstance(enabled, bool):
            return error_response(
                command_id, "set_auto_retry", "enabled must be a boolean"
            )
        self.session.set_auto_retry_enabled(enabled)
        return success_response(command_id, "set_auto_retry")

    async def _handle_abort_retry(self, _cmd: dict, command_id: str | None) -> dict:
        self.session.abort_retry()
        return success_response(command_id, "abort_retry")

    # ------------------------------------------------------------------
    # Bash
    # ------------------------------------------------------------------

    async def _handle_bash(self, cmd: dict, command_id: str | None) -> dict:
        command = cmd.get("command")
        if not isinstance(command, str) or not command.strip():
            return error_response(command_id, "bash", "Command is required")
        try:
            result = await _run_command(command, self.session.cwd, timeout=120)
        except Exception as exc:
            return error_response(command_id, "bash", str(exc))
        return success_response(command_id, "bash", result)

    async def _handle_abort_bash(self, _cmd: dict, command_id: str | None) -> dict:
        # 当前 bash 同步执行（无跟踪进程），no-op 成功。
        return success_response(command_id, "abort_bash")

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    async def _handle_get_session_stats(
        self, _cmd: dict, command_id: str | None
    ) -> dict:
        return success_response(
            command_id, "get_session_stats", self.session.get_session_stats()
        )

    async def _handle_export_html(self, _cmd: dict, command_id: str | None) -> dict:
        return error_response(
            command_id, "export_html", "export_html is not implemented yet (Phase 7)"
        )

    async def _handle_not_implemented(self, cmd: dict, command_id: str | None) -> dict:
        command_type = cmd.get("type", "command")
        return error_response(
            command_id,
            command_type,
            f"{command_type} is not implemented yet (Phase 6 session DAG)",
        )

    async def _handle_get_entries(self, cmd: dict, command_id: str | None) -> dict:
        session_manager = self.session.session_manager
        entries = session_manager.get_entries()
        since = cmd.get("since")
        if since is not None:
            since_index = next(
                (index for index, entry in enumerate(entries) if entry.get("id") == since),
                -1,
            )
            if since_index == -1:
                return error_response(command_id, "get_entries", f"Entry not found: {since}")
            entries = entries[since_index + 1 :]
        return success_response(
            command_id,
            "get_entries",
            {"entries": entries, "leafId": session_manager.get_leaf_id()},
        )

    async def _handle_get_last_assistant_text(
        self, _cmd: dict, command_id: str | None
    ) -> dict:
        return success_response(
            command_id,
            "get_last_assistant_text",
            {"text": self.session.get_last_assistant_text()},
        )

    async def _handle_set_session_name(self, cmd: dict, command_id: str | None) -> dict:
        name = cmd.get("name")
        if not isinstance(name, str) or not name.strip():
            return error_response(command_id, "set_session_name", "Session name cannot be empty")
        self.session.set_session_name(name.strip())
        return success_response(command_id, "set_session_name")

    async def _handle_get_messages(self, _cmd: dict, command_id: str | None) -> dict:
        return success_response(
            command_id,
            "get_messages",
            {"messages": self.session.get_messages()},
        )

    async def _handle_get_commands(self, _cmd: dict, command_id: str | None) -> dict:
        # 扩展命令 / 提示模板 / skills 尚未接入（Phase 4/5）。
        return success_response(command_id, "get_commands", {"commands": []})


# ---------------------------------------------------------------------------
# RPC 入口（2.4）
# ---------------------------------------------------------------------------


async def run_rpc_mode(
    session: AgentSession,
    model_runtime: ModelRuntime,
    *,
    session_factory: Callable[[], AgentSession] | None = None,
    stdin=None,
    stdout=None,
) -> int:
    """RPC 模式主循环：stdin JSONL → 分发 → stdout JSONL。

    stdin/stdout 缺省为 sys.stdin.buffer / sys.stdout.buffer。
    写入使用线程池（Windows Proactor 的 connect_write_pipe 对
    子进程管道句柄不可靠）；测试可注入 io.BytesIO。
    """
    stdin_stream = stdin if stdin is not None else sys.stdin.buffer
    stdout_stream = stdout if stdout is not None else sys.stdout.buffer

    handler = RpcMessageHandler(session, model_runtime, session_factory=session_factory)
    write_queue: asyncio.Queue[bytes] = asyncio.Queue()

    def output(obj: dict) -> None:
        write_queue.put_nowait(serialize_json_line(obj).encode("utf-8"))

    async def _flush_output() -> None:
        await write_queue.join()

    async def _readline() -> bytes:
        return await asyncio.to_thread(stdin_stream.readline)

    async def _writer_loop() -> None:
        while True:
            data = await write_queue.get()
            try:
                if data is None:
                    return
                await asyncio.to_thread(_sync_write, stdout_stream, data)
            finally:
                write_queue.task_done()

    unsubscribe: Callable[[], None] | None = None

    def rebind() -> None:
        nonlocal unsubscribe
        if unsubscribe is not None:
            unsubscribe()
            unsubscribe = None
        unsubscribe = handler.session.subscribe(lambda event: output(event))

    rebind()

    async def handle_line(line: str) -> None:
        if not line.strip():
            return
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            output(error_response(None, "parse", f"Failed to parse command: {exc}"))
            await _flush_output()
            return
        if not isinstance(parsed, dict):
            output(error_response(None, "parse", "Command must be a JSON object"))
            await _flush_output()
            return

        # 扩展 UI 响应直接交给 UI 上下文。
        if parsed.get("type") == "extension_ui_response":
            handler.ui_context.resolve_response(parsed)
            return

        response = await handler.handle_command(parsed)
        if response is not None:
            output(response)
            await _flush_output()
        # new_session 替换会话后重新绑定事件转发。
        if session is not handler.session:
            rebind()

    writer_task = asyncio.create_task(_writer_loop())
    try:
        while True:
            raw = await _readline()
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace")
            if text.endswith("\r\n"):
                text = text[:-2]
            elif text.endswith("\n"):
                text = text[:-1]
            elif text.endswith("\r"):
                text = text[:-1]
            await handle_line(text)
    finally:
        if handler._prompt_tasks:
            await asyncio.gather(*list(handler._prompt_tasks), return_exceptions=True)
        if unsubscribe is not None:
            unsubscribe()
        for created in list(handler.created_sessions):
            try:
                await created.dispose()
            except Exception:
                pass
        try:
            await handler.session.dispose()
        except Exception:
            pass
        write_queue.put_nowait(None)
        await writer_task
    return 0


def _sync_write(stream, data: bytes) -> None:
    """同步写入流（线程池中执行）。"""
    stream.write(data)
    try:
        stream.flush()
    except Exception:
        pass


__all__ = [
    "RpcUiContext",
    "RpcMessageHandler",
    "run_rpc_mode",
]
