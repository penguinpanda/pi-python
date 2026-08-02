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

    context
    config
    emit
    signal

因此：

同样的输入 + 同样的 LLM 返回
=
相同的 Agent 行为


====================================================
Agent Loop 工作流程
====================================================


用户输入:

    UserMessage
          |
          v

run_agent_loop()

          |
          v

    agent_start
          |
          v

    +----------------+
    |   Turn Loop    |
    +----------------+

          |
          |
          v

  stream_assistant_response()

          |
          |
          +----------------+
          |                |
          v                v

     普通回答          Tool Call


                          |
                          v

                 execute_tool_calls()

                          |
                          v

                 ToolResultMessage

                          |
                          v

                 下一轮 LLM


直到：

1. 模型没有 toolCall
2. stopReason == error
3. stopReason == aborted
4. should_stop_after_turn 返回 True


          |
          v

      agent_end



====================================================
核心函数
====================================================

run_agent_loop()

    新用户消息进入 Agent。

    例如：

    用户:
        "帮我搜索天气"


    会:

    context.messages
          +
    UserMessage

    然后启动循环。



run_agent_loop_continue()

    从已有 messages 继续。

    常用于：

    - retry
    - resume
    - interrupted recovery



====================================================
核心设计思想
====================================================

1. Message 是唯一状态

Agent 的状态：

不是 class

不是对象属性

而是:

    messages: list[AgentMessage]


每轮:

messages
    |
    + assistant message
    |
    + tool result


形成完整轨迹。



2. Event 驱动


Agent 不直接操作 UI。

而是:

emit(event)


例如:

{
    "type": "message_update",
    "message": ...
}


外部可以:

- CLI 显示
- WebSocket 推送
- 日志记录
- 数据库存储


3. Tool 是副作用边界


LLM:

纯推理


Tool:

真实世界操作


例如:

    LLM:
        "我要调用 search"

    Tool:
        HTTP 请求


因此 Tool 执行被单独隔离。


"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from pi_ai._types import (
    AssistantMessage,
    DoneEvent,
    ErrorEvent,
    Message,
    StartEvent,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    now_ms,
)

from ._types import (
    AfterToolCallResult,
    AgentContext,
    AgentEventSink,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentToolResult,
    BeforeToolCallResult,
    StreamFn,
)

if TYPE_CHECKING:
    pass


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
            toolCall="weather"
        ),

        ToolResultMessage(
            result="晴天"
        ),

        AssistantMessage(
            "今天晴天"
        )
    ]

    """

    # 不可变：复制上下文
    messages: list[AgentMessage] = list(context.messages)
    tools = list(context.tools) if context.tools else []

    
    """
    将 解释：

    用户消息加入 Agent 历史：

    例如：

    之前:
    [
    assistant:
    "你好"
    ]

    加入：

    user:
    "帮我查天气"

    变成：

    [
    assistant:
    "你好",

    user:
    "帮我查天气"
    ]

    然后通知外部：

    await emit({
        "type":"message_start",
        "message":p
    })

    作用：

    UI 可以显示：

    User:
    帮我查天气
    """
    # 注入新提示
    for p in prompts:
        messages.append(p)
        await emit({"type": "message_start", "message": p})
        await emit({"type": "message_end", "message": p})

    inner_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=messages,
        tools=tools,
    )

    await emit({"type": "agent_start"})

    try:
        result = await _run_loop(inner_context, config, emit, signal, stream_fn)
    except asyncio.CancelledError:
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

    await emit({"type": "agent_start"})

    try:
        result = await _run_loop(inner_context, config, emit, signal, stream_fn)
    except asyncio.CancelledError:
        await emit({"type": "agent_end", "messages": messages})
        raise

    return result


# ============================================================================
# 内部循环
# ============================================================================


async def _run_loop(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
    stream_fn: StreamFn,
) -> list[AgentMessage]:
    """
    真正执行 Agent 推理循环。


    每一次循环代表一次:

    LLM Turn


    一次 Turn:

    1.
    发送 messages 给 LLM


    2.
    接收 assistant


    3.
    检查是否调用工具


    4.
    执行工具


    5.
    把工具结果加入 messages


    6.
    下一轮


    """



    # 不直接使用 context.messages

    # 原因:
    #
    # Agent Loop 是纯函数思想。
    #
    # 输入 context:
    #
    #       context
    #          |
    #          v
    #       新 context
    #
    # 而不是:
    #
    #       context.messages.append(...)
    #
    #
    # 避免外部引用被修改。
    messages: list[AgentMessage] = list(context.messages)
    tools = list(context.tools) if context.tools else []
    current_model = config.model

    while True:
        _check_signal(signal)

        await emit({"type": "turn_start"})

        # -- 构建本轮上下文 --
        turn_context = AgentContext(
            system_prompt=context.system_prompt,
            messages=messages,
            tools=tools,
        )

        # -- 流式获取助手回复 --
        assistant_msg = await _stream_assistant_response(
            turn_context, config, emit, signal, stream_fn
        )
        messages.append(assistant_msg)

        # 检查错误/中止
        stop_reason = assistant_msg.get("stopReason", "stop")
        if stop_reason in ("error", "aborted"):
            await emit({
                "type": "turn_end",
                "message": assistant_msg,
                "tool_results": [],
            })
            await emit({"type": "agent_end", "messages": messages})
            return messages

        # -- 提取工具调用 --
        tool_calls: list[ToolCall] = cast(
            list[ToolCall],
            [
                c
                for c in assistant_msg.get("content", [])
                if c.get("type") == "toolCall"
            ],
        )

        tool_results: list[ToolResultMessage] = []
        has_more_tool_calls = False

        if tool_calls:
            # 截断保护：stopReason="length" 时参数可能不完整
            if stop_reason == "length":
                tool_results = _fail_tool_calls_from_truncated(tool_calls)
            else:
                result = await _execute_tool_calls(
                    tool_calls,
                    tools,
                    config,
                    emit,
                    signal,
                )
                tool_results = result["messages"]
                has_more_tool_calls = not result["terminate"]

            # 追加工具结果消息到上下文
            for tr in tool_results:
                messages.append(tr)
                await emit({"type": "message_start", "message": tr})
                await emit({"type": "message_end", "message": tr})

        # -- turn_end --
        await emit({
            "type": "turn_end",
            "message": assistant_msg,
            "tool_results": tool_results,
        })

        # -- prepare_next_turn --
        if config.prepare_next_turn is not None:
            raw_update = config.prepare_next_turn(turn_context)
            update: object = None
            if asyncio.iscoroutine(raw_update):
                update = await raw_update
            else:
                update = raw_update
            if update is not None:
                _update = cast(AgentLoopTurnUpdate, update)
                if _update.context is not None:
                    context = _update.context
                    messages = list(context.messages)
                    tools = list(context.tools) if context.tools else []
                if _update.model is not None:
                    current_model = _update.model

        # -- should_stop_after_turn --
        if config.should_stop_after_turn is not None:
            raw_stop = config.should_stop_after_turn(turn_context)
            if asyncio.iscoroutine(raw_stop):
                should_stop = await raw_stop
            else:
                should_stop = raw_stop
            if should_stop:
                await emit({"type": "agent_end", "messages": messages})
                return messages

        # -- 无更多工具调用 → 结束 --
        if not has_more_tool_calls:
            await emit({"type": "agent_end", "messages": messages})
            return messages

        # 继续下一轮


# ============================================================================
# 流式助手回复
# ============================================================================


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
    stream_fn: StreamFn,
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

    # 1. transformContext（可选）
    agent_messages = list(context.messages)
    if config.transform_context is not None:
        agent_messages = await config.transform_context(agent_messages)

    # 2. convertToLlm（必须）
    llm_messages = config.convert_to_llm(agent_messages)

    # 3. getApiKey（可选）
    api_key: str | None = None
    if config.get_api_key is not None:
        api_key = config.get_api_key(config.model.provider)

    # 4. 构建 LLM context
    from pi_ai import Context as LlmContext

    llm_context = LlmContext(
        messages=llm_messages,
        tools=_tools_to_pi_ai(context.tools or []),
        systemPrompt=context.system_prompt,
    )

    # 5. 调用 LLM
    options = StreamOptions()
    if api_key is not None:
        options["apiKey"] = api_key

    response = await stream_fn(config.model, llm_context, options)

    # 6. 迭代事件流。
    #
    # 12 事件协议下，每个增量事件都携带 partial 快照，
    # 因此不需要在此自行拼接内容块。
    final_stop_reason = "stop"
    final_error_message: str | None = None
    _final_msg: AssistantMessage | None = None  # DoneEvent/ErrorEvent 的完整消息

    # 当前 partial 消息（随增量事件更新）。
    temp_msg: AssistantMessage = {
        "role": "assistant",
        "content": [],
        "api": config.model.api,
        "provider": config.model.provider,
        "model": config.model.id,
    }
    await emit({"type": "message_start", "message": temp_msg})

    try:
        async for event in response:
            _check_signal(signal)

            event_type = event.get("type")

            if event_type == "start":
                temp_msg = cast(StartEvent, event)["partial"]

            elif event_type in (
                "text_start", "text_delta", "text_end",
                "thinking_start", "thinking_delta", "thinking_end",
                "toolcall_start", "toolcall_delta", "toolcall_end",
            ):
                partial = event["partial"]
                temp_msg = partial
                await emit({
                    "type": "message_update",
                    "message": temp_msg,
                    "assistant_message_event": event,
                })

            elif event_type == "done":
                done_event = cast(DoneEvent, event)
                _final_msg = done_event["message"]
                final_stop_reason = _final_msg.get("stopReason", "stop")
                final_error_message = _final_msg.get("errorMessage")
                break

            elif event_type == "error":
                err_event = cast(ErrorEvent, event)
                _final_msg = err_event["error"]
                final_stop_reason = "error"
                final_error_message = _final_msg.get(
                    "errorMessage", err_event.get("reason", "Unknown error")
                )
                break

    except asyncio.CancelledError:
        final_stop_reason = "aborted"
        final_error_message = "Aborted"
        raise
    else:
        # 如果没有收到 done/error 事件，从 stream.result() 获取最终消息
        if _final_msg is None:
            _final_msg = await response.result()
            if _final_msg is not None:
                final_stop_reason = _final_msg.get("stopReason", "stop")
                final_error_message = _final_msg.get("errorMessage")
    finally:
        # 构建最终消息：优先使用 DoneEvent/ErrorEvent 的完整消息
        result: AssistantMessage
        if _final_msg is not None:
            result = _final_msg
            # 确保 stopReason 被正确设置
            if "stopReason" not in result:
                result["stopReason"] = final_stop_reason
        else:
            # 没有 done/error 时，回退到最后一次 partial 快照
            result = temp_msg
            result["stopReason"] = final_stop_reason
        if final_error_message and "errorMessage" not in result:
            result["errorMessage"] = final_error_message

        await emit({"type": "message_end", "message": result})

    return result


# ============================================================================
# 工具执行管道（四阶段）
# ============================================================================


async def _execute_tool_calls(
    tool_calls: list[ToolCall],
    tools: list,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: asyncio.Event | None,
) -> dict:
    """
    执行 LLM 返回的所有工具调用。


    输入:

    tool_calls:

        AssistantMessage 中的 toolCall


    例如:

    [
    {
    type:"toolCall",
    id:"call_1",
    name:"search",
    arguments:{"query":"python"}
    }
    ]


    tools:

        Agent 注册的工具列表


    config:

        Agent 生命周期 hook


    emit:

        事件输出


    返回:

    {
        messages:
            [
            ToolResultMessage
            ],

        terminate:
            是否终止 Agent Loop
    }


    核心流程:


    ToolCall

    |
    v

    1. 准备阶段

    查找工具
    解析参数
    before hook


    |
    v

    2. 执行阶段

    tool.execute()


    |
    v

    3. 完成阶段

    after hook


    |
    v

    4. 输出阶段

    event
    ToolResultMessage


    """
    all_terminate = True
    tool_result_messages: list[ToolResultMessage] = []

    for tc in tool_calls:
        _check_signal(signal)

        tc_id: str = tc["id"]
        tc_name: str = tc["name"]

        # 参数已由事件协议解析为对象（ToolCall.arguments）。
        args: dict = tc["arguments"]

        # === 阶段 1: 准备 ===
        tool_def = _find_tool(tools, tc_name)
        if tool_def is None:
            # 工具未找到 → 立即错误
            error_result = AgentToolResult(
                content=[TextContent(type="text", text=f"Tool '{tc_name}' not found.")],
                details={"error": "tool_not_found"},
            )
            await _emit_tool_lifecycle(
                emit, tc_id, tc_name, args, error_result, is_error=True
            )
            tr_msg = _make_tool_result_message(tc_id, tc_name, error_result, is_error=True)
            tool_result_messages.append(tr_msg)
            all_terminate = all_terminate and error_result.terminate
            continue

        # beforeToolCall 钩子
        if config.before_tool_call is not None:
            raw_before = config.before_tool_call(
                tc_id, tc_name, args,
                AgentContext(
                    system_prompt="",
                    messages=[],
                    tools=tools,
                ),
            )
            before_result: BeforeToolCallResult | None
            if asyncio.iscoroutine(raw_before):
                before_result = cast(BeforeToolCallResult | None, await raw_before)
            else:
                before_result = cast(BeforeToolCallResult | None, raw_before)
            if before_result is not None and before_result.block:
                block_msg = f"Tool '{tc_name}' blocked: {before_result.reason}"
                blocked_result = AgentToolResult(
                    content=[TextContent(type="text", text=block_msg)],
                    details={"blocked": True, "reason": before_result.reason},
                )
                await _emit_tool_lifecycle(
                    emit, tc_id, tc_name, args, blocked_result, is_error=True
                )
                tr_msg = _make_tool_result_message(
                    tc_id, tc_name, blocked_result, is_error=True
                )
                tool_result_messages.append(tr_msg)
                all_terminate = all_terminate and blocked_result.terminate
                continue

        # === 阶段 2: 执行 ===
        is_error = False
        try:
            await emit({
                "type": "tool_execution_start",
                "tool_call_id": tc_id,
                "tool_name": tc_name,
                "args": args,
            })

            def _on_update(partial: AgentToolResult) -> None:
                # 注意：这是同步回调，不能 await emit
                pass  # 简化：最小核心不做流式 tool update

            result = await tool_def.execute(tc_id, args, signal, _on_update)
        except Exception as exc:
            is_error = True
            result = AgentToolResult(
                content=[TextContent(type="text", text=str(exc))],
                details={"error": str(exc), "exception_type": type(exc).__name__},
            )

        # === 阶段 3: afterToolCall ===
        if config.after_tool_call is not None:
            raw_after = config.after_tool_call(
                tc_id, tc_name, result, is_error,
                AgentContext(
                    system_prompt="",
                    messages=[],
                    tools=tools,
                ),
            )
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
                if after_result.is_error is not None:
                    is_error = after_result.is_error
                if after_result.terminate is not None:
                    result.terminate = after_result.terminate

        # === 阶段 4: 发出事件 + 构造 ToolResultMessage ===
        await _emit_tool_lifecycle(emit, tc_id, tc_name, args, result, is_error)
        tr_msg = _make_tool_result_message(tc_id, tc_name, result, is_error)
        tool_result_messages.append(tr_msg)
        all_terminate = all_terminate and result.terminate

    return {"messages": tool_result_messages, "terminate": all_terminate}


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


def _fail_tool_calls_from_truncated(
    tool_calls: list[ToolCall],
) -> list[ToolResultMessage]:
    """截断保护：stopReason="length" 时将所有工具标记为错误。"""
    results: list[ToolResultMessage] = []
    for tc in tool_calls:
        tc_id = tc["id"]
        tc_name = tc["name"]
        error_text = (
            f"Tool call arguments may be truncated because the model response "
            f"reached its max output length. Tool '{tc_name}' was not executed."
        )
        results.append(ToolResultMessage(
            role="toolResult",
            toolCallId=tc_id,
            toolName=tc_name,
            content=[TextContent(type="text", text=error_text)],
            isError=True,
            timestamp=now_ms(),
        ))
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
        "toolCallId": tc_id,
        "toolName": tc_name,
        "content": list(result.content),
        "isError": is_error,
        "timestamp": now_ms(),
    }
    if result.added_tool_names:
        msg["addedToolNames"] = list(result.added_tool_names)
    return msg


async def _emit_tool_lifecycle(
    emit: AgentEventSink,
    tc_id: str,
    tc_name: str,
    args: dict,
    result: AgentToolResult,
    is_error: bool,
) -> None:
    """发出 tool_execution_end 事件。"""
    await emit({
        "type": "tool_execution_end",
        "tool_call_id": tc_id,
        "tool_name": tc_name,
        "result": result,
        "is_error": is_error,
    })


def _tools_to_pi_ai(tools: list) -> list:
    """将 AgentTool 列表转换为 pi_ai.Tool 列表。"""
    from pi_ai._types import Tool as PiAiTool

    result: list[PiAiTool] = []
    for t in tools:
        result.append(PiAiTool(
            name=t.name,
            description=t.description,
            inputSchema=t.input_schema,
        ))
    return result
