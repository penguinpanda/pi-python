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
from typing import Any, Awaitable, Callable, cast

from pi_ai.models.models_store import model_to_dict
from pi_ai.types.common import ModelThinkingLevel
from pi_ai.types.content import ImageContent

from .._session import AgentSession
from ..model_runtime import ModelRuntime
from .jsonl import serialize_json_line
from .rpc_types import error_response, success_response


def _parse_images(raw: Any) -> list[ImageContent] | None:
    """解析 RPC images 字段 → ImageContent 列表（对齐 TS ImageContent[] 直传）。"""
    if not isinstance(raw, list) or not raw:
        return None
    images: list[ImageContent] = []
    for item in raw:
        if isinstance(item, dict) and item.get("type") == "image":
            images.append(
                ImageContent(
                    type="image",
                    url=item.get("url"),
                    data=item.get("data"),
                    mime_type=item.get("mime_type"),
                )
            )
    return images or None


def _content_text(content) -> str:
    """内容块 → 纯文本（fork 消息 / 树条目用）。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _entry_text(entry: dict) -> str:
    if entry.get("type") != "message":
        return ""
    message = entry.get("message") or {}
    return _content_text(message.get("content"))


def _tree_node_to_dict(node) -> dict:
    return {
        "id": node.id,
        "parentId": node.parent_id,
        "entry": node.entry,
        "label": node.label,
        "children": [_tree_node_to_dict(child) for child in node.children],
    }


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
            parse=lambda response: response.get("value") if "value" in response else None,
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
            parse=lambda response: response.get("confirmed") if "confirmed" in response else False,
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
            parse=lambda response: response.get("value") if "value" in response else None,
            timeout=timeout,
        )

    async def editor(self, title: str, prefill: str | None = None) -> str | None:
        return await self._request(
            {"method": "editor", "title": title, "prefill": prefill},
            default=None,
            parse=lambda response: response.get("value") if "value" in response else None,
        )

    def notify(self, message: str, notify_type: str | None = None) -> None:
        """fire-and-forget 通知（无需响应）。"""
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "notify",
                "message": message,
                "notifyType": notify_type,
            }
        )

    def set_status(self, key: str, text: str | None) -> None:
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setStatus",
                "statusKey": key,
                "statusText": text,
            }
        )

    def set_title(self, title: str) -> None:
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setTitle",
                "title": title,
            }
        )

    def set_editor_text(self, text: str) -> None:
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "set_editor_text",
                "text": text,
            }
        )

    def set_footer(self, text: str | None) -> None:
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setFooter",
                "text": text,
            }
        )

    def set_header(self, text: str | None) -> None:
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setHeader",
                "text": text,
            }
        )

    def set_editor_component(self, component) -> None:
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setEditorComponent",
                "componentType": type(component).__name__,
            }
        )

    def set_widget(self, key: str, lines: list[str] | None, options: dict | None = None) -> None:
        options = options or {}
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setWidget",
                "key": key,
                "lines": list(lines or []),
                "placement": options.get("placement", "aboveEditor"),
            }
        )

    def set_overlay(self, key: str, lines: list[str], options: dict | None = None) -> None:
        options = options or {}
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setOverlay",
                "key": key,
                "lines": list(lines),
                "anchor": options.get("anchor", "top-left"),
                "margin": options.get("margin", 1),
                "animate": bool(options.get("animate", False)),
                "duration": float(options.get("duration", 0.5)),
                "border": options.get("border"),
                "borderColor": options.get("border_color"),
                "title": options.get("title"),
            }
        )

    def set_overlay_component(self, key: str, component, options: dict | None = None) -> None:
        """组件 overlay 无法跨 RPC 传输，只发送类型与布局意图。"""
        options = options or {}
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setOverlayComponent",
                "key": key,
                "componentType": type(component).__name__,
                "anchor": options.get("anchor", "top-left"),
                "margin": options.get("margin", 1),
            }
        )

    def set_overlay_renderer(self, key: str, renderer, options: dict | None = None) -> None:
        """渲染回调无法跨 RPC 传输，只发送意图请求。"""
        options = options or {}
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setOverlayRenderer",
                "key": key,
                "componentType": "renderer",
                "anchor": options.get("anchor", "top-left"),
                "margin": options.get("margin", 1),
            }
        )

    def set_hidden_thinking_label(self, label: str | None = None) -> None:
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setHiddenThinkingLabel",
                "label": label,
            }
        )

    def set_working_message(self, text: str | None = None) -> None:
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setWorkingMessage",
                "text": text,
            }
        )

    def set_theme(self, theme: str | None = None) -> None:
        self._emit(
            {
                "type": "extension_ui_request",
                "id": self._new_id(),
                "method": "setTheme",
                "theme": theme,
            }
        )

    async def custom(self, factory, *, overlay_options=None, on_handle=None):
        """RPC 模式不支持自定义交互 UI（对齐 TS rpc-mode custom()）。"""
        return None

    def resolve_response(self, response: dict[str, Any]) -> None:
        """处理客户端返回的 extension_ui_response。"""
        request_id = response.get("id")
        future = self._pending.get(cast(str, request_id))
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
        session_factory: Callable[..., AgentSession | Awaitable[AgentSession]] | None = None,
        session_rebuilder=None,
        output: Callable[[dict[Any, Any]], None] | None = None,
    ) -> None:
        self.session = session
        self.model_runtime = model_runtime
        self.ui_context = ui_context or RpcUiContext(emit=lambda _obj: None)
        # new_session 用：创建全新 AgentSession 的工厂（含新 SessionManager/Agent）。
        self.session_factory = session_factory
        # fork/clone/switch_session 用：按给定 SessionManager 重建 AgentSession。
        self.session_rebuilder = session_rebuilder
        self.created_sessions: list[AgentSession] = []
        self._disposed_session_ids: set[int] = set()
        # 后台 prompt 任务（命令循环不阻塞）。
        self._prompt_tasks: set[asyncio.Task] = set()
        # 后台 bash 任务，允许命令循环在 bash 运行期间处理 abort_bash。
        self._bash_tasks: set[asyncio.Task] = set()
        # 输出通道（preflight 成功后 prompt 的 success 响应经此发送）。
        self.output = output or (lambda _obj: None)

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
            "switch_session": self._handle_switch_session,
            "fork": self._handle_fork,
            "clone": self._handle_clone,
            "get_fork_messages": self._handle_get_fork_messages,
            "get_tree": self._handle_get_tree,
            "get_entries": self._handle_get_entries,
            "get_last_assistant_text": self._handle_get_last_assistant_text,
            "set_session_name": self._handle_set_session_name,
            "get_messages": self._handle_get_messages,
            "get_commands": self._handle_get_commands,
        }

    async def _dispose_session(self, session: AgentSession) -> None:
        """释放会话；同一会话最多 dispose 一次。"""
        session_id = id(session)
        if session_id in self._disposed_session_ids:
            return
        self._disposed_session_ids.add(session_id)
        try:
            await session.dispose()
        except Exception:
            pass

    async def _replace_session(self, new_session: AgentSession) -> None:
        """替换当前会话，并释放被替换下来的旧会话。"""
        old = self.session
        if old is not None and old is not new_session:
            await self._dispose_session(old)
        self.session = new_session

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

    async def _handle_prompt(self, cmd: dict, command_id: str | None) -> dict | None:
        message = cmd.get("message")
        if not isinstance(message, str) or not message.strip():
            return error_response(command_id, "prompt", "Message is required")
        if self.session.is_streaming:
            return error_response(command_id, "prompt", "Agent is busy streaming")
        images = _parse_images(cmd.get("images"))

        # 对齐 TS prompt preflight：preflight 成功才发 success；
        # 提交失败（未 preflight）发 error；运行失败由事件流呈现。
        preflight_succeeded = False

        def _on_preflight(did_succeed: bool) -> None:
            nonlocal preflight_succeeded
            if did_succeed:
                preflight_succeeded = True
                self.output(success_response(command_id, "prompt"))

        async def _run() -> None:
            try:
                await self.session.prompt(message, images or None, preflight_result=_on_preflight)
            except Exception as exc:
                if not preflight_succeeded:
                    self.output(error_response(command_id, "prompt", str(exc)))

        task = asyncio.create_task(_run())
        self._prompt_tasks.add(task)
        task.add_done_callback(self._prompt_tasks.discard)
        return None

    async def _handle_steer(self, cmd: dict, command_id: str | None) -> dict:
        message = cmd.get("message")
        if not isinstance(message, str) or not message.strip():
            return error_response(command_id, "steer", "Message is required")
        images = _parse_images(cmd.get("images"))
        self.session.steer(message, images)
        return success_response(command_id, "steer")

    async def _handle_follow_up(self, cmd: dict, command_id: str | None) -> dict:
        message = cmd.get("message")
        if not isinstance(message, str) or not message.strip():
            return error_response(command_id, "follow_up", "Message is required")
        images = _parse_images(cmd.get("images"))
        self.session.follow_up(message, images)
        return success_response(command_id, "follow_up")

    async def _handle_abort(self, _cmd: dict, command_id: str | None) -> dict:
        await self.session.abort()
        return success_response(command_id, "abort")

    async def _handle_new_session(self, cmd: dict, command_id: str | None) -> dict:
        if self.session_factory is None:
            return error_response(command_id, "new_session", "Session factory not configured")
        parent_session = cmd.get("parentSession")
        try:
            if isinstance(parent_session, str) and parent_session:
                # parentSession：fork 自父会话（对齐 TS newSession({parentSession})）。
                new_session = await self._fork_from_parent(parent_session)
            else:
                result = self.session_factory()
                new_session = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            return error_response(command_id, "new_session", str(exc))
        self.created_sessions.append(new_session)
        await self._replace_session(new_session)
        return success_response(command_id, "new_session", {"cancelled": False})

    async def _fork_from_parent(self, parent_session_id: str):
        """按父会话 ID fork 出新会话（parentSession lineage）。"""
        from .._session_manager_v4 import list_sessions, open_session_manager

        from .._config import get_sessions_dir

        # 优先当前会话的 sessions 根目录（测试/自定义目录场景）。
        current_manager = getattr(self.session, "session_manager", None)
        sessions_dir = getattr(current_manager, "_sessions_root", None) or get_sessions_dir()
        sessions = await list_sessions(sessions_dir)
        matches = [info for info in sessions if info.session_id == parent_session_id]
        if not matches:
            raise ValueError(f"Parent session not found: {parent_session_id}")
        parent = await open_session_manager(matches[0].path)
        try:
            leaf = parent.get_leaf_id()
            if leaf is None:
                # 父会话无条目：fork 等价于带 parentSessionId 的新会话。
                from .._session_manager_v4 import create_session_manager

                forked = await create_session_manager(
                    parent.cwd,
                    sessions_dir=sessions_dir,
                    parent_session_id=parent.session_id,
                )
            else:
                forked = await parent.fork(leaf)
        finally:
            close = getattr(parent, "close", None)
            if close is not None:
                await close()
        if self.session_factory is None:
            raise RuntimeError("Session factory not configured")
        result = self.session_factory(forked)
        return await result if inspect.isawaitable(result) else result

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

    async def _handle_get_available_models(self, _cmd: dict, command_id: str | None) -> dict:
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
        self.session.set_thinking_level(cast(ModelThinkingLevel, level))
        return success_response(command_id, "set_thinking_level")

    async def _handle_cycle_thinking_level(self, _cmd: dict, command_id: str | None) -> dict:
        level = self.session.cycle_thinking_level()
        if level is None:
            return success_response(command_id, "cycle_thinking_level", None)
        return success_response(command_id, "cycle_thinking_level", {"level": level})

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
        self.session.steering_mode = mode
        return success_response(command_id, "set_steering_mode")

    async def _handle_set_follow_up_mode(self, cmd: dict, command_id: str | None) -> dict:
        mode = cmd.get("mode")
        if mode not in ("all", "one-at-a-time"):
            return error_response(
                command_id, "set_follow_up_mode", "mode must be 'all' or 'one-at-a-time'"
            )
        self.session.follow_up_mode = mode
        return success_response(command_id, "set_follow_up_mode")

    # ------------------------------------------------------------------
    # 压缩 / 重试
    # ------------------------------------------------------------------

    async def _handle_compact(self, cmd: dict, command_id: str | None) -> dict:
        result = await self.session.compact(cmd.get("customInstructions"))
        if result is None:
            return error_response(command_id, "compact", "Nothing to compact")
        return success_response(command_id, "compact", asdict(result))

    async def _handle_set_auto_compaction(self, cmd: dict, command_id: str | None) -> dict:
        enabled = cmd.get("enabled")
        if not isinstance(enabled, bool):
            return error_response(command_id, "set_auto_compaction", "enabled must be a boolean")
        self.session.set_auto_compaction_enabled(enabled)
        return success_response(command_id, "set_auto_compaction")

    async def _handle_set_auto_retry(self, cmd: dict, command_id: str | None) -> dict:
        enabled = cmd.get("enabled")
        if not isinstance(enabled, bool):
            return error_response(command_id, "set_auto_retry", "enabled must be a boolean")
        self.session.set_auto_retry_enabled(enabled)
        return success_response(command_id, "set_auto_retry")

    async def _handle_abort_retry(self, _cmd: dict, command_id: str | None) -> dict:
        self.session.abort_retry()
        return success_response(command_id, "abort_retry")

    # ------------------------------------------------------------------
    # Bash
    # ------------------------------------------------------------------

    async def _handle_bash(self, cmd: dict, command_id: str | None) -> dict | None:
        command = cmd.get("command")
        if not isinstance(command, str) or not command.strip():
            return error_response(command_id, "bash", "Command is required")
        exclude_from_context = bool(cmd.get("excludeFromContext"))

        async def _run() -> None:
            try:
                # 扩展 user_bash 事件（对齐 TS emitUserBash）：扩展可返回 result 直接采用。
                runner = self.session.extension_runner
                if runner is not None:
                    event_results = await runner.emit_event(
                        "user_bash",
                        {
                            "command": command,
                            "excludeFromContext": exclude_from_context,
                            "cwd": self.session.cwd,
                        },
                    )
                    for candidate in reversed(event_results):
                        if isinstance(candidate, dict) and candidate.get("result") is not None:
                            extension_result = candidate["result"]
                            self.session.record_bash_result(
                                command,
                                extension_result,
                                exclude_from_context=exclude_from_context,
                            )
                            self.output(success_response(command_id, "bash", extension_result))
                            return

                # 经 session 执行（对齐 TS session.executeBash：记录历史并进入上下文）。
                result = await self.session.execute_bash(
                    command,
                    exclude_from_context=exclude_from_context,
                    timeout=120,
                )
                self.output(
                    success_response(
                        command_id,
                        "bash",
                        {
                            "output": result.output,
                            "exit_code": result.exit_code,
                            "canceled": result.cancelled,
                        },
                    )
                )
            except Exception as exc:
                self.output(error_response(command_id, "bash", str(exc)))

        task = asyncio.create_task(_run())
        self._bash_tasks.add(task)
        task.add_done_callback(self._bash_tasks.discard)
        return None

    async def _handle_abort_bash(self, _cmd: dict, command_id: str | None) -> dict:
        self.session.abort_bash()
        return success_response(command_id, "abort_bash")

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    async def _handle_get_session_stats(self, _cmd: dict, command_id: str | None) -> dict:
        return success_response(command_id, "get_session_stats", self.session.get_session_stats())

    async def _handle_export_html(self, cmd: dict, command_id: str | None) -> dict:
        from pathlib import Path

        from ..export_html import export_session_to_html

        output_path = cmd.get("outputPath")
        if not isinstance(output_path, str) or not output_path:
            output_path = str(Path(self.session.cwd) / f"session-{self.session.session_id}.html")
        try:
            path = export_session_to_html(
                self.session.session_manager,
                output_path,
                system_prompt=self.session._agent.state.system_prompt,
            )
        except Exception as exc:
            return error_response(command_id, "export_html", str(exc))
        return success_response(command_id, "export_html", {"path": str(path)})

    async def _rebuild_session(self, manager) -> AgentSession:
        if self.session_rebuilder is None:
            raise RuntimeError("Session rebuilder not configured")
        result = self.session_rebuilder(manager)
        return await result if inspect.isawaitable(result) else result

    async def _handle_fork(self, cmd: dict, command_id: str | None) -> dict:
        entry_id = cmd.get("entryId")
        if not isinstance(entry_id, str):
            return error_response(command_id, "fork", "entryId is required")
        manager = self.session.session_manager
        target = manager.get_entry(entry_id)
        if target is None:
            return error_response(command_id, "fork", f"Entry not found: {entry_id}")
        from .._session_manager_v4 import fork_session_manager

        forked = await fork_session_manager(manager, entry_id)
        try:
            new_session = await self._rebuild_session(forked)
        except Exception as exc:
            return error_response(command_id, "fork", str(exc))
        self.created_sessions.append(new_session)
        await self._replace_session(new_session)
        return success_response(
            command_id,
            "fork",
            {"text": _entry_text(cast(dict[Any, Any], target)), "cancelled": False},
        )

    async def _handle_clone(self, _cmd: dict, command_id: str | None) -> dict:
        leaf_id = self.session.session_manager.get_leaf_id()
        if leaf_id is None:
            return error_response(
                command_id, "clone", "Cannot clone session: no current entry selected"
            )
        from .._session_manager_v4 import fork_session_manager

        forked = await fork_session_manager(self.session.session_manager, leaf_id)
        try:
            new_session = await self._rebuild_session(forked)
        except Exception as exc:
            return error_response(command_id, "clone", str(exc))
        self.created_sessions.append(new_session)
        await self._replace_session(new_session)
        return success_response(command_id, "clone", {"cancelled": False})

    async def _handle_switch_session(self, cmd: dict, command_id: str | None) -> dict:
        session_path = cmd.get("sessionPath")
        if not isinstance(session_path, str):
            return error_response(command_id, "switch_session", "sessionPath is required")
        try:
            from .._session_manager_v4 import open_session_manager

            manager = await open_session_manager(session_path, cwd_override=self.session.cwd)
            new_session = await self._rebuild_session(manager)
        except Exception as exc:
            return error_response(command_id, "switch_session", str(exc))
        self.created_sessions.append(new_session)
        await self._replace_session(new_session)
        return success_response(command_id, "switch_session", {"cancelled": False})

    async def _handle_get_tree(self, _cmd: dict, command_id: str | None) -> dict:
        manager = self.session.session_manager
        return success_response(
            command_id,
            "get_tree",
            {
                "tree": [_tree_node_to_dict(node) for node in manager.get_tree()],
                "leafId": manager.get_leaf_id(),
            },
        )

    async def _handle_get_fork_messages(self, _cmd: dict, command_id: str | None) -> dict:
        messages = []
        for raw_entry in self.session.session_manager.get_branch():
            entry = cast(dict[str, Any], raw_entry)
            if entry.get("type") != "message":
                continue
            message = entry.get("message") or {}
            if message.get("role") != "user":
                continue
            text = _content_text(message.get("content"))
            if text:
                messages.append({"entryId": entry.get("id"), "text": text})
        return success_response(command_id, "get_fork_messages", {"messages": messages})

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

    async def _handle_get_last_assistant_text(self, _cmd: dict, command_id: str | None) -> dict:
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
        commands: list[dict] = []
        session = self.session

        # 扩展命令（Phase 5）。
        runner = session.extension_runner
        if runner is not None:
            for command in runner.get_registered_commands():
                commands.append(
                    {
                        "name": command.name,
                        "description": command.description,
                        "source": "extension",
                        "sourceInfo": command.source_info or {},
                    }
                )

        # 提示模板（Phase 4）。
        if session.template_loader is not None:
            for template in session.template_loader.all():
                commands.append(
                    {
                        "name": template.name,
                        "description": template.description,
                        "source": "prompt",
                        "sourceInfo": {"path": template.file_path},
                    }
                )

        # 技能（Phase 4）。
        if session.skill_loader is not None:
            for skill in session.skill_loader.all():
                commands.append(
                    {
                        "name": f"skill:{skill.name}",
                        "description": skill.description,
                        "source": "skill",
                        "sourceInfo": {"path": skill.file_path},
                    }
                )
        return success_response(command_id, "get_commands", {"commands": commands})


# ---------------------------------------------------------------------------
# RPC 入口（2.4）
# ---------------------------------------------------------------------------


async def run_rpc_mode(
    session: AgentSession,
    model_runtime: ModelRuntime,
    *,
    session_factory: Callable[[], AgentSession | Awaitable[AgentSession]] | None = None,
    session_rebuilder=None,
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

    write_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def output(obj: dict[Any, Any]) -> None:
        write_queue.put_nowait(serialize_json_line(obj).encode("utf-8"))

    handler = RpcMessageHandler(
        session,
        model_runtime,
        session_factory=session_factory,
        session_rebuilder=session_rebuilder,
        output=output,
    )

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
        unsubscribe = handler.session.subscribe(lambda event: output(cast(dict[Any, Any], event)))

    rebind()

    # 扩展错误通知（对齐 TS rpc-mode.ts extension_error 输出）。
    extension_error_unsub: Callable[[], None] | None = None

    def rebind_extension_errors() -> None:
        nonlocal extension_error_unsub
        if extension_error_unsub is not None:
            extension_error_unsub()
            extension_error_unsub = None
        runner = handler.session.extension_runner
        if runner is None:
            return
        extension_error_unsub = runner.on_error(
            lambda err: output(
                {
                    "type": "extension_error",
                    "extensionPath": err.extension_path,
                    "event": err.event,
                    "error": err.error,
                }
            )
        )

    rebind_extension_errors()

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
            rebind_extension_errors()

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
        # 客户端断开（stdin EOF）：中止并取消进行中的任务而非无限等待，
        # 长 prompt/bash 不得拖住进程退出。
        try:
            await handler.session.abort()
        except Exception:
            pass
        for task in list(handler._prompt_tasks):
            if not task.done():
                task.cancel()
        if handler._prompt_tasks:
            await asyncio.gather(*list(handler._prompt_tasks), return_exceptions=True)
        for task in list(handler._bash_tasks):
            if not task.done():
                task.cancel()
        if handler._bash_tasks:
            await asyncio.gather(*list(handler._bash_tasks), return_exceptions=True)
        if unsubscribe is not None:
            unsubscribe()
        if extension_error_unsub is not None:
            extension_error_unsub()
        for created in list(handler.created_sessions):
            await handler._dispose_session(created)
        await handler._dispose_session(handler.session)
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
