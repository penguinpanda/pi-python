"""
纯函数 Agent 循环（核心引擎）

====================================================
设计目标
====================================================

这个模块实现一个无状态（stateless）的 Agent Runtime。

它本身不保存：

- 当前对话状态
- 当前任务状态
- 工具状态
- 模型状态

所有状态都通过参数传入：

    context      当前 Agent 快照（system_prompt / messages / tools）
    config       AgentLoopConfig（model / hooks / converter / callbacks）
    emit         事件输出函数（Agent 不知道外部是谁）
    signal       取消信号（asyncio.Event，例如用户点击停止按钮）
    stream_fn    LLM streaming 调用函数

因此：

同样的输入 + 同样的 LLM 返回
=
相同的 Agent 行为

唯一的例外：prepare_next_turn 可替换模型。为让替换在后续 LLM 调用中
生效，config.model 会被就地更新（_run_loop 内），这是有意为之的副作用；
除此之外不修改任何外部对象（context 一律复制）。

====================================================
工作流程总览
====================================================

两个入口函数，按消息来源区分：

    run_agent_loop(prompts, ...)
        新会话：把 prompts 追加到 context.messages 后进入循环。

    run_agent_loop_continue(context, ...)
        从已有 messages 继续，不追加新提示。
        常用于 retry / resume / interrupted recovery。

两者都：

    1. 复制 context（纯函数：绝不修改外部引用）
    2. 发射 agent_start / turn_start（由入口函数发射，先于 prompts 注入；
       run_agent_loop 额外为每条 prompt 发射 message_start / message_end）
    3. 进入 _run_loop 双重嵌套循环

    +------------------------------+
    | 外层：Follow-up 驱动          |
    |                              |
    | 内层退出后检查 follow-up 队列，|
    | 非空则带着新消息重新进入内层   |
    +------------------------------+
                 |
                 v
    +------------------------------+
    | 内层：Tool + Steering        |
    |                              |
    | 注入 pending 消息            |
    | 流式 LLM 推理                |
    | 执行工具调用（并行 / 顺序）   |
    | 轮询 steering 队列           |
    +------------------------------+

    直到停止条件满足，发射 agent_end 并返回最终完整 messages。

停止条件（agent-loop.ts runLoop）：

    1. stop_reason == "error" / "aborted"
    2. should_stop_after_turn 返回 True
    3. 无更多工具调用（含工具结果 terminate）
       且 steering / follow-up 队列均为空

====================================================
核心函数
====================================================

run_agent_loop(prompts, context, config, emit, signal, stream_fn)

    启动一次新的 Agent 会话：prompts 追加到 context.messages，
    然后进入循环。返回最终完整 messages（含历史与新增）。

    例如：

    开始:  [UserMessage("天气?")]
    结束:  [UserMessage("天气?"),
            AssistantMessage(tool_call="weather"),
            ToolResultMessage(result="晴天"),
            AssistantMessage("今天晴天")]

run_agent_loop_continue(context, config, emit, signal, stream_fn)

    从已有上下文继续（重试 / 恢复场景），不追加新提示，
    直接从现有 context 进入循环。

agent_loop(prompts, ...) / agent_loop_continue(context, ...)

    EventStream 包装（agentLoop / agentLoopContinue）。
    返回 EventStream 而非 emit 回调；agent_end 事件即流结束事件，
    await stream.result() 得到 agent_end 携带的完整 messages。

    agent_loop_continue 额外校验：
    - context.messages 非空
    - 最后一条消息 role 不是 assistant

====================================================
双重嵌套循环（_run_loop）
====================================================

外层（Follow-up 驱动）：

    内层循环结束后（无更多工具调用、无 steering 消息），
    检查 follow-up 队列；有后续消息则重新进入内层循环，
    直到队列为空。

内层（Tool + Steering）：

    每一轮：

    1. 注入待处理消息（steering / follow-up），发射 message_start / message_end
    2. 流式获取助手回复（_stream_assistant_response）
    3. 提取 assistant 消息中的 toolCall 内容
    4. 执行工具调用（_execute_tool_calls），得到 ToolResultMessage 列表
    5. 追加工具结果消息到轨迹，发射 message_start / message_end
    6. 发射 turn_end（携带 assistant 消息与工具结果）
    7. prepare_next_turn（可选：替换 context / model）
    8. should_stop_after_turn（可选：提前停止）
    9. 轮询 steering 队列（趁 Agent 还在工作时注入引导消息）

    特殊处理：

    - stop_reason == "error" / "aborted"：
      立即发射 turn_end（空工具结果）与 agent_end 后返回
    - stop_reason == "length"：工具调用参数可能被截断，
      所有工具标记为错误且不执行（_fail_tool_calls_from_truncated）
    - 工具结果 terminate 为 True：本轮后不再继续工具调用
    - 首轮 turn_start 已由入口函数发射，内层不重复发射
    - 取消（signal 被设置）：_check_signal 抛出 CancelledError，
      入口捕获后补发 agent_end 再向上传播

====================================================
流式助手回复（_stream_assistant_response）
====================================================

一次 LLM 推理的固定管线：

    transform_context（可选）  转换 agent messages（仅本次调用）
    convert_to_llm（必须）     转换为 LLM messages
    get_api_key（可选）        按 provider 取 API key
    构建 LLM context           （messages / tools / system_prompt）
    stream_fn 调用             带应用层重试 retry_assistant_call

事件协议（12 事件模型）：

    每个增量事件（text / thinking / toolcall 的 start / delta / end）
    都携带 partial 快照，因此不做本地拼接；message_start 在开始时
    发射，message_update 随增量发射。

    message_end 只对最终结果发射一次：重试的失败尝试不会提交到状态。
    中止（CancelledError）补发 aborted 的 message_end 后向上传播，
    且永不重试；意外异常补发 error 的 message_end 后向上传播（可重试）。

重试（config.retry_policy，默认 enabled=True / max_retries=3）：

    发射 auto_retry_start / auto_retry_end 事件；
    显式传入 RetryPolicy(enabled=False) 可关闭。

====================================================
工具执行管道（四阶段）
====================================================

    +------------------+   +------------------+   +------------------+
    | 准备             |   | 执行             |   | finalize         |
    | _prepare_tool    |-> | _execute_tool    |-> | _finalize_tool   |
    | _call            |   | _call            |   | _call            |
    +------------------+   +------------------+   +------------------+
                                                      |
                                                      v
                                              tool_execution_end
                                              + ToolResultMessage

准备 _prepare_tool_call：

    查找工具 -> 参数校验（validate_arguments，按 input_schema）-> beforeToolCall -> 中止检查

    立即失败（工具未找到 / 参数校验失败 / beforeToolCall block）返回
    _ImmediateToolOutcome，不发 tool_execution_start；错误 ToolResult
    交给 LLM 自纠。中止检查抛 CancelledError 向上传播。

执行 _execute_tool_call：

    before_execute（可选，可替换参数）
    -> tool_execution_start
    -> execute（流式更新经 tool_execution_update 发出，且保证先于
       tool_execution_end）
    -> after_execute（可选，可替换结果）

    execute 抛异常时返回 error 的 outcome（details 含 exception_type）。

finalize _finalize_tool_call：

    afterToolCall 字段级覆盖：
    content / details / usage / is_error / terminate

调度 _execute_tool_calls：

    - config.tool_execution == "sequential"，或
      批次内任一工具 execution_mode == "sequential"
      -> 整批顺序执行（每个工具完成全部阶段才轮到下一个）
    - 其余情况 -> 并行：顺序准备 + 并发执行 +
      按 assistant 原始顺序输出 ToolResultMessage
      （立即失败项在准备循环中当场发 tool_execution_end）

工具生命周期事件：

    tool_execution_start -> tool_execution_update（可多次）-> tool_execution_end

====================================================
事件清单
====================================================

    agent_start / agent_end
    turn_start / turn_end
    message_start / message_update / message_end
    tool_execution_start / tool_execution_update / tool_execution_end
    auto_retry_start / auto_retry_end

====================================================
核心设计思想
====================================================

1. Message 是唯一状态

    Agent 的状态不是 class、不是对象属性，而是：

        messages: list[AgentMessage]

    每轮追加 assistant message 与 tool result，形成完整轨迹。

2. Event 驱动

    Agent 不直接操作 UI，而是 emit(event)。外部可以：
    CLI 显示 / WebSocket 推送 / 日志记录 / 数据库存储。

3. Tool 是副作用边界

    LLM 纯推理，Tool 执行真实世界操作（例如 HTTP 请求），
    因此 Tool 执行被单独隔离。

4. 钩子
    beforeToolCall / afterToolCall / before_execute / after_execute /
    prepare_next_turn / should_stop_after_turn / transform_context /
    convert_to_llm / get_api_key / get_steering_messages /
    get_follow_up_messages
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, cast

from pi_ai.types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Model,
    now_ms,
)
from pi_ai.utils.retry import (
    RetryCallbacks,
    RetryPolicy,
    retry_assistant_call,
)
from pi_ai.utils._background import track_background_task
from pi_ai.utils._event_stream import EventStream
from pi_ai.utils.validation import ValidationError, validate_arguments

from ._types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    AgentEvent,
    AgentEventSink,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
    StreamFn,
    ThinkingLevel,
)


# ============================================================================
# 公共入口
# ============================================================================


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
    stream_fn: StreamFn,
) -> list[AgentMessage]:
    """
    启动一次新的 Agent 会话。

    输入：

    prompts
        用户新输入消息。


    context
        当前 Agent 快照。

        注意:
        不直接修改。


    config
        Agent 配置:

        - model
        - hooks
        - converter
        - callbacks


    emit
        事件输出函数。

        Agent 本身不知道外部是谁。


    signal
        取消信号。

        例如:

        用户点击停止按钮。


    stream_fn

        LLM streaming 调用函数。


    返回:

        最终完整 messages。


    例如:

    开始:

    [
        UserMessage("天气?")
    ]


    结束:

    [
        UserMessage("天气?"),

        AssistantMessage(
            tool_call="weather"
        ),

        ToolResultMessage(
            result="晴天"
        ),

        AssistantMessage(
            "今天晴天"
        )
    ]

    """

    # 不可变：复制上下文并注入新提示（runAgentLoop）
    messages: list[AgentMessage] = list(context.messages)
    tools = list(context.tools) if context.tools else []
    new_messages: list[AgentMessage] = list(prompts)

    for p in prompts:
        messages.append(p)

    inner_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=messages,
        tools=tools,
    )

    # agent_start / turn_start 由外层发射，且先于 prompts 注入。
    await emit({"type": "agent_start"})
    await emit({"type": "turn_start"})
    for p in prompts:
        await emit({"type": "message_start", "message": p})
        await emit({"type": "message_end", "message": p})

    try:
        result = await _run_loop(
            inner_context,
            config,
            emit,
            signal,
            stream_fn,
            new_messages=new_messages,
            first_turn=True,
            shared_messages=messages,
        )
    except asyncio.CancelledError:
        # messages 为共享列表：循环内追加的最新消息（含 aborted assistant
        # 消息，由 _stream_assistant_response 回填）都在其中。
        await emit({"type": "agent_end", "messages": messages})
        raise

    return result


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
    stream_fn: StreamFn,
) -> list[AgentMessage]:
    """从已有上下文继续（重试/恢复场景）。

    不会追加新提示，直接从现有 context 进入循环。
    """
    messages = list(context.messages)
    tools = list(context.tools) if context.tools else []

    inner_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=messages,
        tools=tools,
    )

    # agent_start / turn_start 由外层发射。
    await emit({"type": "agent_start"})
    await emit({"type": "turn_start"})

    try:
        result = await _run_loop(
            inner_context,
            config,
            emit,
            signal,
            stream_fn,
            new_messages=[],
            first_turn=True,
            shared_messages=messages,
        )
    except asyncio.CancelledError:
        # messages 为共享列表：循环内追加的最新消息都在其中。
        await emit({"type": "agent_end", "messages": messages})
        raise

    return result


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    stream_fn: StreamFn,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """agentLoop() EventStream 包装（agent-loop.ts agentLoop()）。

    返回 EventStream 而非 emit 回调；agent_end 事件即流结束事件，
    await stream.result() 得到 agent_end 携带的完整 messages（含历史）。
    """
    stream = _create_agent_stream()

    async def _emit(event: AgentEvent) -> None:
        stream.push(event)

    async def _run() -> None:
        try:
            messages = await run_agent_loop(
                prompts=prompts,
                context=context,
                config=config,
                emit=_emit,
                signal=signal,
                stream_fn=stream_fn,
            )
        except BaseException as exc:
            stream.error(exc)
        else:
            stream.end(messages)

    track_background_task(_run())
    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
    stream_fn: StreamFn,
) -> EventStream[AgentEvent, list[AgentMessage]]:
    """agentLoopContinue() EventStream 包装（agentLoopContinue()）。"""
    if len(context.messages) == 0:
        raise ValueError("Cannot continue: no messages in context")
    if context.messages[-1].get("role") == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = _create_agent_stream()

    async def _emit(event: AgentEvent) -> None:
        stream.push(event)

    async def _run() -> None:
        try:
            messages = await run_agent_loop_continue(
                context=context,
                config=config,
                emit=_emit,
                signal=signal,
                stream_fn=stream_fn,
            )
        except BaseException as exc:
            stream.error(exc)
        else:
            stream.end(messages)

    track_background_task(_run())
    return stream


def _create_agent_stream() -> EventStream[AgentEvent, list[AgentMessage]]:
    """agent_end 即流结束事件；其结果即 agent_end 携带的 messages。"""
    return EventStream(
        is_complete=lambda e: e["type"] == "agent_end",
        extract_result=lambda e: e["messages"],  # type: ignore[return-value]
    )


# ============================================================================
# 内部循环
# ============================================================================


async def _run_loop(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
    stream_fn: StreamFn,
    *,
    new_messages: list[AgentMessage] | None = None,
    first_turn: bool = True,
    shared_messages: list[AgentMessage] | None = None,
) -> list[AgentMessage]:
    """双重嵌套循环（agent-loop.ts runLoop）。

    外层 (Follow-up): 内层循环结束后（无更多工具调用、无 steering 消息），
    检查 follow-up 队列；有后续消息则重新进入内层循环，直到队列为空。

    内层 (Tool + Steering): 工具调用 + steering 注入循环。
    每一轮：注入待处理消息 → LLM 推理 → 执行工具 → 轮询 steering 队列。

    停止条件（：
    1. stop_reason == error/aborted
    2. should_stop_after_turn 返回 True
    3. 无更多工具调用且 steering / follow-up 队列均为空
    """

    # 默认复制 context.messages（Agent Loop 纯函数思想，避免外部引用被修改）；
    # 传入 shared_messages 时使用该列表对象本身（run_agent_loop 传入其局部
    # messages 副本），使取消时 agent_end 能携带循环内追加的最新消息
    # （含 aborted assistant 消息，由 _stream_assistant_response 回填）。
    messages: list[AgentMessage] = (
        shared_messages if shared_messages is not None else list(context.messages)
    )
    tools = list(context.tools) if context.tools else []
    loop_new_messages: list[AgentMessage] = list(new_messages) if new_messages is not None else []
    current_model = config.model
    current_thinking_level = config.thinking_level
    first = first_turn

    # 首轮开始前轮询一次 steering 队列（用户可能在等待期间已 steer()）
    pending_messages: list[AgentMessage] = []
    if config.get_steering_messages is not None:
        pending_messages = await config.get_steering_messages()

    while True:  # ← 外层：Follow-up 驱动
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:  # ← 内层：Tool + Steering
            _check_signal(signal)

            # 首轮 turn_start 已由 run_agent_loop / run_agent_loop_continue 外层发射。
            if first:
                first = False
            else:
                await emit({"type": "turn_start"})

            # -- 注入待处理消息（steering / follow-up）--
            if pending_messages:
                for m in pending_messages:
                    messages.append(m)
                    loop_new_messages.append(m)
                    await emit({"type": "message_start", "message": m})
                    await emit({"type": "message_end", "message": m})
                pending_messages = []

            # -- 构建本轮上下文 --
            turn_context = AgentContext(
                system_prompt=context.system_prompt,
                messages=messages,
                tools=tools,
            )

            # -- 流式获取助手回复 --
            assistant_msg = await _stream_assistant_response(
                turn_context,
                config,
                emit,
                signal,
                stream_fn,
                aborted_sink=messages,
                model=current_model,
                thinking_level=current_thinking_level,
            )
            messages.append(assistant_msg)
            loop_new_messages.append(assistant_msg)

            # 检查错误/中止
            stop_reason = assistant_msg.get("stop_reason", "stop")
            if stop_reason in ("error", "aborted"):
                await emit(
                    {
                        "type": "turn_end",
                        "message": assistant_msg,
                        "tool_results": [],
                    }
                )
                await emit({"type": "agent_end", "messages": loop_new_messages})
                return loop_new_messages

            # -- 提取工具调用 --
            tool_calls: list[ToolCall] = cast(
                list[ToolCall],
                [c for c in assistant_msg.get("content", []) if c.get("type") == "toolCall"],
            )

            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False

            if tool_calls:
                # 截断保护：stop_reason="length" 时参数可能不完整。
                # TS 对每个调用补发 tool_execution_start/end，并让模型重新发起。
                if stop_reason == "length":
                    tool_results = await _fail_tool_calls_from_truncated(tool_calls, emit)
                    has_more_tool_calls = True
                else:
                    tool_context = AgentContext(
                        system_prompt=context.system_prompt,
                        messages=list(messages),
                        tools=tools,
                    )
                    result = await _execute_tool_calls(
                        tool_calls,
                        assistant_msg,
                        tool_context,
                        config,
                        emit,
                        signal,
                    )
                    tool_results = result["messages"]
                    has_more_tool_calls = not result["terminate"]

                # 追加工具结果消息到上下文
                for tr in tool_results:
                    messages.append(tr)
                    loop_new_messages.append(tr)
                    await emit({"type": "message_start", "message": tr})
                    await emit({"type": "message_end", "message": tr})

            # -- turn_end --
            await emit(
                {
                    "type": "turn_end",
                    "message": assistant_msg,
                    "tool_results": tool_results,
                }
            )

            # -- prepare_next_turn --
            if config.prepare_next_turn is not None:
                next_turn_context = AgentContext(
                    system_prompt=context.system_prompt,
                    messages=list(messages),
                    tools=tools,
                )
                prepare_ctx = PrepareNextTurnContext(
                    message=assistant_msg,
                    tool_results=list(tool_results),
                    context=next_turn_context,
                    new_messages=list(loop_new_messages),
                )
                raw_update = config.prepare_next_turn(prepare_ctx)
                if asyncio.iscoroutine(raw_update):
                    update = await raw_update
                else:
                    update = raw_update
                if update is not None:
                    _update = cast(AgentLoopTurnUpdate, update)
                    if _update.context is not None:
                        context = _update.context
                        if shared_messages is not None:
                            # 保持共享引用，确保取消时 agent_end 仍携带最新消息。
                            messages = shared_messages
                            messages[:] = list(context.messages)
                        else:
                            messages = list(context.messages)
                        tools = list(context.tools) if context.tools else []
                    if _update.model is not None:
                        current_model = _update.model
                    if _update.thinking_level is not None:
                        # TS：thinkingLevel === "off" 时 reasoning 置为 undefined。
                        current_thinking_level = _update.thinking_level

            # -- should_stop_after_turn --
            if config.should_stop_after_turn is not None:
                stop_ctx = ShouldStopAfterTurnContext(
                    message=assistant_msg,
                    tool_results=list(tool_results),
                    context=AgentContext(
                        system_prompt=context.system_prompt,
                        messages=messages,
                        tools=tools,
                    ),
                    new_messages=list(loop_new_messages),
                )
                raw_stop = config.should_stop_after_turn(stop_ctx)
                if asyncio.iscoroutine(raw_stop):
                    should_stop = await raw_stop
                else:
                    should_stop = raw_stop
                if should_stop:
                    await emit({"type": "agent_end", "messages": loop_new_messages})
                    return loop_new_messages

            # -- 轮询 steering 队列（趁 agent 还在工作时注入引导消息）--
            if config.get_steering_messages is not None:
                pending_messages = await config.get_steering_messages()

        # Agent 即将停止：检查 follow-up 队列
        if config.get_follow_up_messages is not None:
            follow_up = await config.get_follow_up_messages()
            if follow_up:
                pending_messages = follow_up
                continue

        # 无更多消息，退出
        break

    await emit({"type": "agent_end", "messages": loop_new_messages})
    return loop_new_messages


# ============================================================================
# 流式助手回复
# ============================================================================


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
    stream_fn: StreamFn,
    aborted_sink: list[AgentMessage] | None = None,
    *,
    model: Model | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> AssistantMessage:
    """
    负责一次 LLM 推理过程。


    输入:

    AgentContext

    包含:

        system prompt
        messages
        tools


    输出:

    AssistantMessage


    中间过程:

        LLM token streaming

    会不断 emit:

        message_start

        message_update

        message_end


    """

    # 1. transformContext（可选；TS 第二参数为 AbortSignal）
    agent_messages = list(context.messages)
    if config.transform_context is not None:
        raw_transform = _call_with_signal(config.transform_context, agent_messages, signal)
        if asyncio.iscoroutine(raw_transform):
            raw_transform = await raw_transform
        if raw_transform is not None:
            agent_messages = list(raw_transform)

    # 2. convertToLlm（必须）
    llm_messages = config.convert_to_llm(agent_messages)

    active_model = model or config.model

    # 3. get_api_key（可选，支持同步 / 异步回调，对齐 TS Promise<string|undefined>；
    # 未解析到时回退到 config.api_key）。
    api_key: str | None = None
    if config.get_api_key is not None:
        raw_key = config.get_api_key(active_model.provider)
        if asyncio.iscoroutine(raw_key):
            raw_key = await raw_key
        api_key = raw_key
    if not api_key:
        api_key = config.api_key

    # 4. 构建 LLM context
    from pi_ai import Context as LlmContext

    llm_context = LlmContext(
        messages=llm_messages,
        tools=_tools_to_pi_ai(context.tools or []),
        system_prompt=context.system_prompt,
    )

    # 5. 调用 LLM（带应用层重试）。
    options: SimpleStreamOptions = SimpleStreamOptions()
    if signal is not None:
        options["signal"] = signal
    if api_key is not None:
        options["api_key"] = api_key

    # 提示缓存：会话标识与保留策略透传给 provider（用于 prompt_cache_key）。
    if config.session_id is not None:
        options["session_id"] = config.session_id
    if config.cache_retention is not None:
        options["cache_retention"] = config.cache_retention

    # 推理预算与传输协议（AgentLoopConfig extends SimpleStreamOptions）。
    if config.thinking_budgets is not None:
        options["thinking_budgets"] = config.thinking_budgets
    if config.transport is not None:
        options["transport"] = config.transport
    if config.on_payload is not None:
        options["on_payload"] = config.on_payload
    if config.on_response is not None:
        options["on_response"] = config.on_response
    if config.max_retry_delay_ms is not None:
        options["max_retry_delay_ms"] = config.max_retry_delay_ms
    # TS：thinkingLevel === "off" 映射为 reasoning undefined（不传该键）。
    if thinking_level is not None and thinking_level != "off":
        options["reasoning"] = thinking_level

    # retry_policy 为 None 时不重试（对齐 TS agent-loop：直接调用 streamFunction）。
    retry_policy = config.retry_policy
    if retry_policy is None:
        retry_policy = RetryPolicy(enabled=False, max_retries=0, base_delay_ms=1000)

    async def _produce() -> AssistantMessage:
        """单次 LLM 调用：流式消费并发射 message_start/update。

        正常结束返回最终消息（不发 message_end —— 由外层统一发射，
        避免失败尝试被提交到状态）。
        中止 / 意外异常时补发对应 message_end 后向上传播（保持现状）。
        """
        response = await stream_fn(active_model, llm_context, options)

        # 6. 迭代事件流。
        #
        # 12 事件协议下，每个增量事件都携带 partial 快照，
        # 因此不需要在此自行拼接内容块。
        final_stop_reason: StopReason = "stop"
        final_error_message: str | None = None
        _final_msg: AssistantMessage | None = None  # DoneEvent/ErrorEvent 的完整消息

        # 当前 partial 消息（随增量事件更新）。
        temp_msg: AssistantMessage = {
            "role": "assistant",
            "content": [],
            "api": active_model.api,
            "provider": active_model.provider,
            "model": active_model.id,
            "timestamp": now_ms(),
        }
        added_partial = False

        def _finalize() -> AssistantMessage:
            """构建最终消息：优先使用 DoneEvent/ErrorEvent 的完整消息。"""
            result: AssistantMessage
            if _final_msg is not None:
                result = _final_msg
                # 确保 stop_reason 被正确设置
                if "stop_reason" not in result:
                    result["stop_reason"] = final_stop_reason
            else:
                # 没有 done/error 时，回退到最后一次 partial 快照
                result = temp_msg
                result["stop_reason"] = final_stop_reason
            if final_error_message and "error_message" not in result:
                result["error_message"] = final_error_message
            return result

        try:
            async for event in response:
                _check_signal(signal)

                event_type = event.get("type")

                if event_type == "start":
                    temp_msg = cast(StartEvent, event)["partial"]
                    added_partial = True
                    await emit({"type": "message_start", "message": temp_msg})

                elif event_type in (
                    "text_start",
                    "text_delta",
                    "text_end",
                    "thinking_start",
                    "thinking_delta",
                    "thinking_end",
                    "toolcall_start",
                    "toolcall_delta",
                    "toolcall_end",
                ):
                    partial = cast(AssistantMessage, event.get("partial"))
                    temp_msg = partial
                    await emit(
                        {
                            "type": "message_update",
                            "message": temp_msg,
                            "assistant_message_event": event,
                        }
                    )

                elif event_type == "done":
                    done_event = cast(DoneEvent, event)
                    _final_msg = done_event["message"]
                    final_stop_reason = _final_msg.get("stop_reason", "stop")
                    final_error_message = _final_msg.get("error_message")
                    break

                elif event_type == "error":
                    err_event = cast(ErrorEvent, event)
                    _final_msg = err_event["error"]
                    final_stop_reason = "error"
                    final_error_message = _final_msg.get(
                        "error_message", err_event.get("reason", "Unknown error")
                    )
                    break

        except asyncio.CancelledError:
            # 中止：补发 aborted 的 message_end（保持现状）后向上传播。
            # 中止永不重试（retry_assistant_call 不会捕获）。
            final_stop_reason = "aborted"
            final_error_message = "Aborted"
            result = _finalize()
            if not added_partial:
                await emit({"type": "message_start", "message": result})
            if aborted_sink is not None:
                aborted_sink.append(result)
            await emit({"type": "message_end", "message": result})
            raise
        except Exception as exc:
            # 意外异常：补发 error 的 message_end（保持现状）后向上传播。
            final_stop_reason = "error"
            final_error_message = str(exc)
            result = _finalize()
            if not added_partial:
                await emit({"type": "message_start", "message": result})
            await emit({"type": "message_end", "message": result})
            raise
        else:
            # 如果没有收到 done/error 事件，从 stream.result() 获取最终消息
            if _final_msg is None:
                _final_msg = await response.result()
                if _final_msg is not None:
                    final_stop_reason = _final_msg.get("stop_reason", "stop")
                    final_error_message = _final_msg.get("error_message")
            result = _finalize()
            if not added_partial:
                await emit({"type": "message_start", "message": result})
            return result

    # 重试回调 → AgentEvent（异步 emit）
    async def _on_retry_scheduled(
        attempt: int,
        max_attempts: int,
        delay_ms: float,
        error_message: str,
    ) -> None:
        await emit(
            {
                "type": "auto_retry_start",
                "attempt": attempt,
                "max_attempts": max_attempts,
                "delay_ms": delay_ms,
                "error_message": error_message,
            }
        )

    async def _on_retry_finished(
        success: bool,
        attempt: int,
        final_error: str | None,
    ) -> None:
        await emit(
            {
                "type": "auto_retry_end",
                "success": success,
                "attempt": attempt,
                "final_error": final_error,
            }
        )

    result = await retry_assistant_call(
        _produce,
        policy=retry_policy,
        signal=signal,
        callbacks=RetryCallbacks(
            on_retry_scheduled=_on_retry_scheduled,
            on_retry_finished=_on_retry_finished,
        ),
    )

    # message_end 只对最终结果发射一次（重试失败尝试不提交）。
    await emit({"type": "message_end", "message": result})

    return result


# ============================================================================
# 工具执行管道（四阶段）
# ============================================================================


async def _execute_tool_calls(
    tool_calls: list[ToolCall],
    assistant_msg: AssistantMessage,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
) -> dict:
    """执行 LLM 返回的所有工具调用（1.4：支持并行）。

    executeToolCalls()：

    - config.tool_execution == "sequential" → 整批顺序执行
    - 批次内任一工具的 execution_mode == "sequential" → 整批回退顺序执行
    - 其余情况 → 顺序准备 + 并发执行 + 按 assistant 原始顺序输出消息

    返回:
        {"messages": [ToolResultMessage], "terminate": bool}
    """
    has_sequential_tool = any(
        (tool := _find_tool(context.tools or [], tc["name"])) is not None
        and tool.execution_mode == "sequential"
        for tc in tool_calls
    )

    if config.tool_execution == "sequential" or has_sequential_tool:
        return await _execute_tool_calls_sequential(
            tool_calls, assistant_msg, context, config, emit, signal
        )
    return await _execute_tool_calls_parallel(
        tool_calls, assistant_msg, context, config, emit, signal
    )


async def _execute_tool_calls_sequential(
    tool_calls: list[ToolCall],
    assistant_msg: AssistantMessage,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
) -> dict:
    """顺序执行：每个工具调用完成全部三阶段后才轮到下一个。"""
    all_terminate = True
    tool_result_messages: list[ToolResultMessage] = []

    for tc in tool_calls:
        _check_signal(signal)
        await _emit_tool_start(tc, emit)

        prepared = await _prepare_tool_call(tc, assistant_msg, context, config, signal)
        if isinstance(prepared, _ImmediateToolOutcome):
            await _emit_tool_lifecycle(
                emit,
                prepared.tc["id"],
                prepared.tc["name"],
                prepared.args,
                prepared.result,
                prepared.is_error,
            )
            tr_msg = _make_tool_result_message(
                prepared.tc["id"],
                prepared.tc["name"],
                prepared.result,
                prepared.is_error,
            )
            tool_result_messages.append(tr_msg)
            all_terminate = all_terminate and prepared.result.terminate
            continue

        executed = await _execute_tool_call(prepared, emit, signal)
        finalized = await _finalize_tool_call(prepared, executed, config, signal)
        await _emit_tool_lifecycle(
            emit,
            finalized.tc["id"],
            finalized.tc["name"],
            executed.args,
            finalized.result,
            finalized.is_error,
        )
        tr_msg = _make_tool_result_message(
            finalized.tc["id"],
            finalized.tc["name"],
            finalized.result,
            finalized.is_error,
        )
        tool_result_messages.append(tr_msg)
        all_terminate = all_terminate and finalized.result.terminate

    return {"messages": tool_result_messages, "terminate": all_terminate}


async def _execute_tool_calls_parallel(
    tool_calls: list[ToolCall],
    assistant_msg: AssistantMessage,
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
) -> dict:
    """并行执行：顺序准备 → 并发执行 → 按原始顺序输出消息。

    executeToolCallsParallel()：
    - 立即结果（未找到/校验失败/block）在准备循环中当场发 tool_execution_end
    - 已通过的工具并发执行，每个完成后立即发 tool_execution_end（完成顺序）
    - ToolResultMessage 消息在所有工具结束后按 assistant 原始顺序发出
    """
    entries: list[tuple[int, _ImmediateToolOutcome | asyncio.Task]] = []
    try:
        for index, tc in enumerate(tool_calls):
            _check_signal(signal)
            await _emit_tool_start(tc, emit)

            prepared = await _prepare_tool_call(tc, assistant_msg, context, config, signal)
            if isinstance(prepared, _ImmediateToolOutcome):
                await _emit_tool_lifecycle(
                    emit,
                    prepared.tc["id"],
                    prepared.tc["name"],
                    prepared.args,
                    prepared.result,
                    prepared.is_error,
                )
                entries.append((index, prepared))
            else:
                entries.append(
                    (
                        index,
                        asyncio.create_task(_execute_and_finalize(prepared, config, emit, signal)),
                    )
                )
    except BaseException:
        # prepare 阶段异常/中止：取消已启动的并发工具任务，
        # 避免工具继续执行产生副作用（bash 子进程、文件写入）。
        pending = [t for _, t in entries if isinstance(t, asyncio.Task) and not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        raise

    ordered_results: list[_FinalizedToolOutcome | _ImmediateToolOutcome | None] = [None] * len(
        tool_calls
    )
    for index, entry in entries:
        if not isinstance(entry, asyncio.Task):
            ordered_results[index] = entry

    exec_tasks = [(i, t) for i, t in entries if isinstance(t, asyncio.Task)]
    if exec_tasks:
        try:
            outcomes = await asyncio.gather(
                *(t for _, t in exec_tasks),
                return_exceptions=True,
            )
        except BaseException:
            for _, task in exec_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*(t for _, t in exec_tasks), return_exceptions=True)
            raise
        for (index, _), outcome in zip(exec_tasks, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                for _, task in exec_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*(t for _, t in exec_tasks), return_exceptions=True)
                raise outcome
            ordered_results[index] = cast(_FinalizedToolOutcome, outcome)

    tool_result_messages: list[ToolResultMessage] = []
    all_terminate = True
    for outcome in ordered_results:
        finalized = cast(_FinalizedToolOutcome | _ImmediateToolOutcome, outcome)
        tr_msg = _make_tool_result_message(
            finalized.tc["id"],
            finalized.tc["name"],
            finalized.result,
            finalized.is_error,
        )
        tool_result_messages.append(tr_msg)
        all_terminate = all_terminate and finalized.result.terminate

    return {
        "messages": tool_result_messages,
        "terminate": all_terminate and len(tool_result_messages) > 0,
    }


async def _prepare_tool_call(
    tc: ToolCall,
    assistant_msg: AssistantMessage,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> _PreparedToolCall | _ImmediateToolOutcome:
    """准备阶段：查找工具 → 校验参数 → beforeToolCall（prepareToolCall）。

    立即失败（工具未找到 / 参数校验失败 / beforeToolCall block）时返回
    _ImmediateToolOutcome；不发 tool_execution_start（保持 Python 现状）。
    中止（signal 已设置）由 _check_signal 抛 CancelledError 向上传播。
    """
    tc_id: str = tc["id"]
    tc_name: str = tc["name"]
    # 参数已由事件协议解析为对象（ToolCall.arguments）；
    # 但可能为 None（解析失败/尚未解析）——按空参数处理。
    args: dict = tc["arguments"] if tc["arguments"] is not None else {}
    tools = context.tools or []

    tool_def = _find_tool(tools, tc_name)
    if tool_def is None:
        # 工具未找到 → 立即错误
        error_result = AgentToolResult(
            content=[TextContent(type="text", text=f"Tool {tc_name} not found")],
            details={"error": "tool_not_found"},
        )
        return _ImmediateToolOutcome(tc, args, error_result, is_error=True)

    # prepareArguments（对齐 TS agent-loop.ts prepareToolCallArguments）：
    # schema 校验前归一化参数（如 edit 工具的 legacy oldText/newText 载荷）。
    if tool_def.prepare_arguments is not None:
        prepared_args = tool_def.prepare_arguments(args)
        if prepared_args is not None:
            args = prepared_args

    # 参数校验：按 input_schema 校验并返回转换后的参数
    # （validateToolCall：失败返回错误 ToolResult 让 LLM 自纠）。
    try:
        args = validate_arguments(tc_name, tool_def.input_schema, args)
    except ValidationError as exc:
        error_result = AgentToolResult(
            content=[TextContent(type="text", text=str(exc))],
            details={
                "error": "invalid_arguments",
                "message": str(exc),
                "tool_call_id": tc_id,
            },
        )
        return _ImmediateToolOutcome(tc, args, error_result, is_error=True)

    # beforeToolCall 钩子（1.5：接收专用 context 对象）
    if config.before_tool_call is not None:
        before_ctx = BeforeToolCallContext(
            assistant_message=assistant_msg,
            tool_call=tc,
            args=args,
            context=context,
        )
        raw_before = _call_with_signal(config.before_tool_call, before_ctx, signal)
        before_result: BeforeToolCallResult | None
        if asyncio.iscoroutine(raw_before):
            before_result = cast(BeforeToolCallResult | None, await raw_before)
        else:
            before_result = cast(BeforeToolCallResult | None, raw_before)
        if before_result is not None and before_result.block:
            block_msg = before_result.reason or "Tool execution was blocked"
            blocked_result = AgentToolResult(
                content=[TextContent(type="text", text=block_msg)],
                details={"blocked": True, "reason": before_result.reason},
            )
            return _ImmediateToolOutcome(tc, args, blocked_result, is_error=True)

    # beforeToolCall 之后检查中止
    _check_signal(signal)

    return _PreparedToolCall(tc, tool_def, args, assistant_msg, context)


async def _execute_tool_call(
    prepared: _PreparedToolCall,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
) -> _ExecutedToolOutcome:
    """执行阶段：before_execute 替换参数 → tool_execution_start → execute → after_execute。"""
    tc_id: str = prepared.tc["id"]
    tc_name: str = prepared.tc["name"]
    tool_def = prepared.tool
    args: dict = prepared.args

    # Tool 生命周期：before_execute（可选，可替换参数）
    if tool_def.before_execute is not None:
        raw_before_tool = tool_def.before_execute(args, prepared.context)
        if asyncio.iscoroutine(raw_before_tool):
            replaced = await raw_before_tool
        else:
            replaced = raw_before_tool
        if replaced is not None:
            args = replaced

    pending_updates: list[asyncio.Task] = []

    def _on_update(partial: AgentToolResult) -> None:
        # 同步回调：调度 tool_execution_update 事件，执行后统一等待，
        # 保证 update 事件先于 tool_execution_end 发出。
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _emit_update() -> None:
            await emit(
                cast(
                    AgentEvent,
                    {
                        "type": "tool_execution_update",
                        "tool_call_id": tc_id,
                        "tool_name": tc_name,
                        "args": args,
                        "partial_result": partial,
                    },
                )
            )

        pending_updates.append(asyncio.create_task(_emit_update()))

    try:
        result = await tool_def.execute(tc_id, args, signal, _on_update)
        if pending_updates:
            await asyncio.gather(*pending_updates)

        # Tool 生命周期：after_execute（可选，可替换结果）
        if tool_def.after_execute is not None:
            raw_after_tool = tool_def.after_execute(result)
            if asyncio.iscoroutine(raw_after_tool):
                after_val = await raw_after_tool
            else:
                after_val = raw_after_tool
            if after_val is not None:
                result = after_val
    except Exception as exc:
        # 对齐 TS executePreparedToolCall：execute 抛错时也要等待已调度的
        # update 事件任务，避免泄漏/乱序；update 自身异常不影响返回 execute 错误。
        if pending_updates:
            await asyncio.gather(*pending_updates, return_exceptions=True)
        return _ExecutedToolOutcome(
            prepared.tc,
            args,
            AgentToolResult(
                content=[TextContent(type="text", text=str(exc))],
                details={"error": str(exc), "exception_type": type(exc).__name__},
            ),
            is_error=True,
        )

    return _ExecutedToolOutcome(prepared.tc, args, result, is_error=False)


async def _execute_and_finalize(
    prepared: _PreparedToolCall,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
) -> _FinalizedToolOutcome:
    """并行路径的单个工具：执行 → finalize → 发 tool_execution_end（完成顺序）。"""
    executed = await _execute_tool_call(prepared, emit, signal)
    finalized = await _finalize_tool_call(prepared, executed, config, signal)
    await _emit_tool_lifecycle(
        emit,
        finalized.tc["id"],
        finalized.tc["name"],
        executed.args,
        finalized.result,
        finalized.is_error,
    )
    return finalized


async def _finalize_tool_call(
    prepared: _PreparedToolCall,
    executed: _ExecutedToolOutcome,
    config: AgentLoopConfig,
    signal: asyncio.Event | None,
) -> _FinalizedToolOutcome:
    """完成阶段：afterToolCall 字段级覆盖（finalizeExecutedToolCall）。"""
    result = executed.result
    is_error = executed.is_error

    if config.after_tool_call is not None:
        after_ctx = AfterToolCallContext(
            assistant_message=prepared.assistant_message,
            tool_call=prepared.tc,
            args=executed.args,
            result=result,
            is_error=is_error,
            context=prepared.context,
        )
        raw_after = _call_with_signal(config.after_tool_call, after_ctx, signal)
        after_result: AfterToolCallResult | None
        if asyncio.iscoroutine(raw_after):
            after_result = cast(AfterToolCallResult | None, await raw_after)
        else:
            after_result = cast(AfterToolCallResult | None, raw_after)
        if after_result is not None:
            if after_result.content is not None:
                result.content = after_result.content
            if after_result.details is not None:
                result.details = after_result.details
            if after_result.usage is not None:
                result.usage = after_result.usage
            if after_result.is_error is not None:
                is_error = after_result.is_error
            if after_result.terminate is not None:
                result.terminate = after_result.terminate

    return _FinalizedToolOutcome(prepared.tc, result, is_error)


@dataclass(slots=True)
class _PreparedToolCall:
    """准备阶段通过：携带执行与 finalize 所需的全部上下文。"""

    tc: ToolCall
    tool: AgentTool
    args: dict
    assistant_message: AssistantMessage
    context: AgentContext


@dataclass(slots=True)
class _ImmediateToolOutcome:
    """准备阶段直接失败（未找到/校验失败/block/中止）。"""

    tc: ToolCall
    args: dict
    result: AgentToolResult
    is_error: bool


@dataclass(slots=True)
class _ExecutedToolOutcome:
    """执行阶段产物（afterToolCall 覆盖之前）。"""

    tc: ToolCall
    args: dict
    result: AgentToolResult
    is_error: bool


@dataclass(slots=True)
class _FinalizedToolOutcome:
    """afterToolCall 已应用后的最终结果。"""

    tc: ToolCall
    result: AgentToolResult
    is_error: bool


# ============================================================================
# 辅助函数
# ============================================================================


def _check_signal(signal: asyncio.Event | None) -> None:
    """检查取消信号，已设置则抛出 CancelledError。"""
    if signal is not None and signal.is_set():
        raise asyncio.CancelledError()


def _find_tool(tools: list, name: str):
    """在工具列表中按名称查找。"""
    for t in tools:
        if t.name == name:
            return t
    return None


async def _fail_tool_calls_from_truncated(
    tool_calls: list[ToolCall],
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    """截断保护：stop_reason="length" 时将所有工具标记为错误。

    对齐 TS failToolCallsFromTruncatedMessage：每个调用都发 start/end，
    并返回 terminate=false，让循环回填错误结果后再次请求模型。
    """
    results: list[ToolResultMessage] = []
    for tc in tool_calls:
        await _emit_tool_start(tc, emit)
        error_text = (
            f'Tool call "{tc["name"]}" was not executed: the response hit the output '
            "token limit, so its arguments may be truncated. Re-issue the tool call "
            "with complete arguments."
        )
        error_result = AgentToolResult(
            content=[TextContent(type="text", text=error_text)],
            details={},
        )
        await _emit_tool_lifecycle(
            emit,
            tc["id"],
            tc["name"],
            tc.get("arguments") or {},
            error_result,
            True,
        )
        results.append(_make_tool_result_message(tc["id"], tc["name"], error_result, True))
    return results


def _make_tool_result_message(
    tc_id: str,
    tc_name: str,
    result: AgentToolResult,
    is_error: bool,
) -> ToolResultMessage:
    """构造 ToolResultMessage。"""
    msg: ToolResultMessage = {
        "role": "toolResult",
        "tool_call_id": tc_id,
        "tool_name": tc_name,
        "content": list(result.content),
        "is_error": is_error,
        "timestamp": now_ms(),
    }
    if result.details is not None:
        msg["details"] = result.details
    if result.usage is not None:
        msg["usage"] = result.usage
    if result.added_tool_names:
        msg["added_tool_names"] = list(result.added_tool_names)
    return msg


async def _emit_tool_start(tc: ToolCall, emit: AgentEventSink) -> None:
    """发出 tool_execution_start 事件（prepare 之前，对齐 TS）。"""
    await emit(
        {
            "type": "tool_execution_start",
            "tool_call_id": tc["id"],
            "tool_name": tc["name"],
            "args": tc.get("arguments"),
        }
    )


def _call_with_signal(
    fn: Any,
    arg: Any,
    signal: asyncio.Event | None,
) -> Any:
    """按 hook 是否声明第二参数决定是否传入 AbortSignal。"""
    if signal is None:
        return fn(arg)
    try:
        signature = inspect.signature(fn)
        signature.bind(arg, signal)
    except (TypeError, ValueError):
        return fn(arg)
    return fn(arg, signal)


async def _emit_tool_lifecycle(
    emit: AgentEventSink,
    tc_id: str,
    tc_name: str,
    args: dict,
    result: AgentToolResult,
    is_error: bool,
) -> None:
    """发出 tool_execution_end 事件。"""
    await emit(
        {
            "type": "tool_execution_end",
            "tool_call_id": tc_id,
            "tool_name": tc_name,
            "result": result,
            "is_error": is_error,
        }
    )


def _tools_to_pi_ai(tools: list) -> list:
    """将 AgentTool 列表转换为 pi_ai.Tool 列表。"""
    from pi_ai.types import Tool as PiAiTool

    result: list[PiAiTool] = []
    for t in tools:
        result.append(
            PiAiTool(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
                before_execute=t.before_execute,
                after_execute=t.after_execute,
            )
        )
    return result
