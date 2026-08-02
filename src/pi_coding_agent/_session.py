"""
AgentSession — coding-agent 中枢编排类

封装 pi_agent.Agent + 工具注入 + 会话持久化 + 事件转发。

用法:
    session = AgentSession(agent, session_manager, cwd, model)
    session.subscribe(lambda event: print(event))
    await session.prompt("read README.md")
    await session.wait_for_idle()
    await session.dispose()
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from pi_agent import Agent, AgentEvent, AgentMessage, AgentTool
from pi_ai import Model
from pi_ai.utils.retry import (
    RetryPolicy,
    compute_backoff_delay,
    is_retryable_error,
)

from ._session_manager import SessionManager
from .tools import create_all_tools


class AgentSession:
    """中枢会话对象 — 连接 Agent、工具、持久化、事件转发。

    最小核心版: 无扩展/压缩/分支摘要。支持重试（agent 内部 + turn 级）。
    """

    def __init__(
        self,
        agent: Agent,
        session_manager: SessionManager,
        cwd: str,
        model: Model,
        *,
        tools_override: list[AgentTool] | None = None,
        # turn 级重试策略。None = 默认启用（enabled=True, max_retries=3）。
        # 传入 RetryPolicy(enabled=False) 可关闭。
        turn_retry_policy: RetryPolicy | None = None,
    ):
        self._agent = agent
        self._session_manager = session_manager
        self._cwd = cwd
        self._model = model
        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._pending_writes: set[asyncio.Task[object]] = set()
        self._turn_retry_policy = turn_retry_policy
        # turn 级重试退避期间的中止信号（None 表示空闲）。
        self._abort: asyncio.Event | None = None

        # 注入编码工具
        tools = tools_override if tools_override is not None else create_all_tools(cwd)
        self._agent.state.tools = tools

        # 恢复已有会话消息历史（打开已有会话时）
        existing_messages = self._session_manager.build_context()
        if existing_messages:
            self._agent.state.messages = existing_messages

        # 订阅 agent 事件 → 持久化 + 转发
        self._unsub_agent: Callable[[], None] | None = self._agent.subscribe(
            self._handle_agent_event
        )

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    async def prompt(self, text: str) -> None:
        """发送用户消息，触发完整的 Agent 循环 + 工具执行。

        阻塞直到 agent 完成本轮（但 wait_for_idle 可以等待事件监听器完成）。

        Agent 内部重试耗尽后，若本轮仍以可重试错误结束，
        自动移除错误消息并用 continue_() 恢复状态机重跑（turn 级重试）。
        """
        self._abort = asyncio.Event()
        try:
            await self._agent.prompt(text)
            await self._retry_failed_turn()
        finally:
            self._abort = None

    async def _retry_failed_turn(self) -> None:
        """turn 级自动重试：移除错误消息 + continue_() 恢复状态机。

        仅当本轮以可重试错误结束时触发；不重发原始请求。

        持久化说明：被移除的错误消息已在 message_end 时写入 JSONL
        （与 TS 一致：错误保留在 session 历史，仅从 agent state 弹出），
        重启后随 build_context() 恢复。
        """
        policy = self._turn_retry_policy
        if policy is None or not policy.enabled or policy.max_retries <= 0:
            return

        for attempt in range(1, policy.max_retries + 1):
            last = self._last_assistant_message()
            if last is None or last.get("stop_reason") != "error":
                return
            error_message = last.get("error_message")
            if not is_retryable_error(error_message):
                return

            # 移除错误消息，使末条消息非 assistant，满足 continue_() 约束
            self._agent.state._messages.pop()

            delay_ms = compute_backoff_delay(
                attempt,
                policy.base_delay_ms,
                max_delay_ms=policy.max_delay_ms,
                jitter=policy.jitter,
            )
            aborted = await self._abortable_sleep(delay_ms / 1000.0)
            if aborted:
                return
            await self._agent.continue_()

    def _last_assistant_message(self) -> AgentMessage | None:
        """返回 state 中最后一条消息（仅当为 assistant 时）。"""
        messages = self._agent.state.messages
        if not messages:
            return None
        last = messages[-1]
        if last.get("role") == "assistant":
            return last
        return None

    async def _abortable_sleep(self, delay_s: float) -> bool:
        """可中止的退避等待（turn 级重试专用）。

        返回 True 表示被中止（应停止重试）。
        """
        if self._abort is None:
            await asyncio.sleep(delay_s)
            return False
        try:
            await asyncio.wait_for(self._abort.wait(), timeout=delay_s)
        except asyncio.TimeoutError:
            return False
        return True

    async def abort(self) -> None:
        """中止当前运行（含 turn 级重试的退避等待）。"""
        if self._abort is not None:
            self._abort.set()
        self._agent.abort()

    async def wait_for_idle(self) -> None:
        """等待当前运行结束（含所有事件监听器完成）。"""
        await self._agent.wait_for_idle()

    def subscribe(
        self, listener: Callable[[AgentEvent], None]
    ) -> Callable[[], None]:
        """订阅 Agent 生命周期事件。返回取消订阅函数。"""
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

        return _unsubscribe

    def get_messages(self) -> list[AgentMessage]:
        """获取当前会话的所有消息。"""
        return self._agent.state.messages

    async def dispose(self) -> None:
        """销毁会话：等待 pending writes → 取消订阅 + 中止运行。"""
        # 等待所有后台持久化写入完成
        if self._pending_writes:
            await asyncio.gather(*self._pending_writes, return_exceptions=True)
            self._pending_writes.clear()
        if self._unsub_agent:
            self._unsub_agent()
            self._unsub_agent = None
        self._agent.abort()
        self._listeners.clear()

    # ------------------------------------------------------------------
    # 内部：事件桥接
    # ------------------------------------------------------------------

    def _handle_agent_event(self, event: AgentEvent) -> None:
        """Agent 事件 → 持久化到 SessionManager + 转发给外部监听器。

        在 message_end 时自动写入 JSONL。
        """
        event_type = event.get("type")

        # message_end → 持久化
        if event_type == "message_end":
            msg = event.get("message")
            if msg is not None:
                # 后台写入 JSONL，跟踪 task 以便 dispose 时等待
                task = asyncio.create_task(
                    self._session_manager.append_message(msg)
                )
                self._pending_writes.add(task)
                task.add_done_callback(self._pending_writes.discard)

        # 转发给所有外部监听器
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass  # 监听器异常不应影响主流程
