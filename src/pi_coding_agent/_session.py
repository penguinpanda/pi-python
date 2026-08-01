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

from ._session_manager import SessionManager
from .tools import create_all_tools


class AgentSession:
    """中枢会话对象 — 连接 Agent、工具、持久化、事件转发。

    最小核心版: 无扩展/压缩/重试/分支摘要。
    """

    def __init__(
        self,
        agent: Agent,
        session_manager: SessionManager,
        cwd: str,
        model: Model,
        *,
        tools_override: list[AgentTool] | None = None,
    ):
        self._agent = agent
        self._session_manager = session_manager
        self._cwd = cwd
        self._model = model
        self._listeners: list[Callable[[AgentEvent], None]] = []
        self._pending_writes: set[asyncio.Task[object]] = set()

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
        """
        await self._agent.prompt(text)

    async def abort(self) -> None:
        """中止当前运行。"""
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
