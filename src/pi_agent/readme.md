# pi-agent — 最小核心 LLM Agent 循环

基于 [pi-mono/packages/agent](https://github.com/earendil-works/pi-mono) 的 Python 复刻。

**纯函数引擎 + 有状态包装**：核心循环零类、零可变状态；外层 `Agent` 类提供面向用户的 OOP API。

---

## 架构概览

```
用户 prompt("Hello")
    │
    ▼
Agent (有状态包装)
    │ 互斥锁、abort signal、事件订阅
    ▼
run_agent_loop()  ◄── 纯函数引擎
    │
    ├─ emit agent_start
    ├─ 注入 prompts → emit message_start / message_end
    │
    ▼
_run_loop()  ◄── 核心 while 循环
    │
    ┌──────────────────────────────────────────────┐
    │ 每轮 (turn):                                   │
    │  1. check_signal (取消检查)                     │
    │  2. emit turn_start                            │
    │  3. _stream_assistant_response() → LLM 调用     │
    │     ├─ transformContext (可选)                   │
    │     ├─ convertToLlm (消息转换)                   │
    │     ├─ stream_fn (LLM 流)                       │
    │     └─ 迭代事件流 → emit message_start/update/end│
    │  4. 提取 toolCalls → _execute_tool_calls()      │
    │  5. emit turn_end                              │
    │  6. prepare_next_turn / should_stop_after_turn  │
    │  7. 无 toolCall 或 terminate → break            │
    └──────────────────────────────────────────────┘
    │
    ▼
emit agent_end → 返回 messages
```

**关键设计决策**：

- **事件驱动**：通过 `emit(AgentEvent)` 回调发出 10 种事件，调用方自行归约
- **不可变性**：loop 内部复制 context，不修改调用方传入的值
- **依赖注入**：`StreamFn` 是 Callable 协议，不直接依赖任何具体 provider SDK
- **单轮 tool 循环**：最小核心仅支持顺序单轮 tool call（无 steering/follow-up 队列）

---

## 快速开始

### 安装

```bash
uv sync
```

### Hello World

```python
import asyncio
from pi_agent import Agent, AgentOptions, set_default_stream_fn
from pi_ai import create_default_models


async def main():
    # 1. 创建模型管理器 + 注册全局流函数
    models = create_default_models()
    set_default_stream_fn(models.stream)

    # 2. 创建 Agent
    agent = Agent(AgentOptions(
        system_prompt="You are a concise assistant.",
        model=models.get_model("deepseek", "deepseek-chat"),
    ))

    # 3. 订阅事件
    agent.subscribe(lambda e: print(f"[{e['type']}]"))

    # 4. 发送消息
    await agent.prompt("What is 2+2?")
    await agent.wait_for_idle()

    # 5. 查看结果
    for msg in agent.state.messages:
        print(msg.get("content", ""))


asyncio.run(main())
```

---

## Agent API

### `Agent(options: AgentOptions | None = None)`

| 方法 | 说明 |
|------|------|
| `await prompt(input, images=None)` | 发送用户消息，运行完整 agent loop，阻塞直到完成 |
| `await continue_()` | 从当前 transcript 继续（最后一条消息必须非 assistant） |
| `abort()` | 中止当前运行（设置 abort signal） |
| `subscribe(listener) → unsubscribe` | 订阅生命周期事件，返回取消订阅函数 |
| `await wait_for_idle()` | 等待当前运行结束（含所有事件监听器完成） |
| `reset()` | 清空 transcript 和运行时状态 |
| `.state` → `AgentState` | 当前只读状态快照（tools, messages, is_streaming 等） |

### `AgentOptions`

```python
AgentOptions(
    system_prompt: str = "You are a helpful assistant.",
    model: Model | None = None,
    thinking_level: ThinkingLevel = "off",
    tools: list[AgentTool] | None = None,
    messages: list[AgentMessage] | None = None,
    stream_fn: StreamFn | None = None,           # 显式传入优先于全局默认
    # 钩子（见下方"循环钩子"）
    convert_to_llm: Callable | None = None,
    transform_context: Callable | None = None,
    get_api_key: Callable | None = None,
    before_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    prepare_next_turn: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    tool_execution: ToolExecutionMode = "sequential",
)
```

### `AgentState`

```python
@dataclass
class AgentState:
    system_prompt: str
    model: Model | None
    thinking_level: ThinkingLevel
    tools: list[AgentTool]         # 赋值时自动防御性复制
    messages: list[AgentMessage]   # 赋值时自动防御性复制
    streaming_message: AgentMessage | None
    error_message: AgentMessage | None
    is_streaming: bool
    pending_tool_calls: set[str]
```

---

## 工具系统

### 工具定义 (`AgentTool`)

```python
@dataclass(slots=True)
class AgentTool:
    name: str                          # 工具名 (LLM 可见)
    description: str                   # 工具描述 (给 LLM 看)
    input_schema: dict[str, Any]       # JSON Schema (给 LLM 选择参数)
    label: str                         # UI 显示标签
    execute: Callable[..., Awaitable[AgentToolResult]]
        # 签名: (toolCallId, params, signal?, onUpdate?) → AgentToolResult
    execution_mode: ToolExecutionMode = "sequential"
```

### 工具结果 (`AgentToolResult`)

```python
@dataclass(slots=True)
class AgentToolResult:
    content: list[TextContent | ImageContent]  # 返回给 LLM 的内容
    details: Any = None                         # 附加详情
    usage: Usage | None = None                  # token 用量
    added_tool_names: list[str] | None = None    # 动态添加工具
    terminate: bool = False                      # True → 停止后续 loop
```

### 四阶段执行管道

每轮 LLM 返回 toolCall 后：

```
阶段 1: 准备
  ├─ 按 name 查找工具
  ├─ JSON.parse args
  └─ beforeToolCall 钩子 (可 block 执行)

阶段 2: 执行
  ├─ emit tool_execution_start
  ├─ tool.execute(toolCallId, params, signal, onUpdate)
  └─ try/except → 异常转 is_error=True

阶段 3: 完成
  └─ afterToolCall 钩子 (可覆盖 content/terminate 等)

阶段 4: 发出
  ├─ emit tool_execution_end
  └─ 构造 ToolResultMessage → 追加到上下文
```

**截断保护**：`stopReason="length"` 时，LLM 返回的工具调用参数可能不完整 → 不实际执行工具，直接生成错误 `ToolResultMessage`。

---

## 事件系统

10 种 `AgentEvent`（TypedDict 判别联合）：

| 事件 | 关键字段 | 含义 |
|------|---------|------|
| `agent_start` | — | Agent 循环开始 |
| `agent_end` | `messages` | Agent 循环结束 |
| `turn_start` | — | 单轮开始 |
| `turn_end` | `message`, `tool_results` | 单轮结束 |
| `message_start` | `message` | 消息进入上下文 |
| `message_update` | `message`, `assistant_message_event` | LLM 流式增量 |
| `message_end` | `message` | 消息完成 (追加到 transcript) |
| `tool_execution_start` | `tool_call_id`, `tool_name`, `args` | 工具开始执行 |
| `tool_execution_update` | `tool_call_id`, `partial_result` | 工具流式结果 |
| `tool_execution_end` | `tool_call_id`, `result`, `is_error` | 工具执行完成 |

---

## 循环钩子

通过 `AgentOptions` 或 `AgentLoopConfig` 注入的 8 个可选钩子：

| 钩子 | 签名 | 说明 |
|------|------|------|
| `transform_context` | `(messages) → messages` | 预处理消息列表（如压缩、摘要） |
| `convert_to_llm` | `(messages) → LLM Message[]` | 消息格式转换（唯一转换点） |
| `get_api_key` | `(provider_id) → str \| None` | 动态获取 API Key |
| `before_tool_call` | `(toolCallId, toolName, args, context) → BeforeToolCallResult \| None` | 工具执行前检查（可 block） |
| `after_tool_call` | `(toolCallId, toolName, result, isError, context) → AfterToolCallResult \| None` | 工具执行后处理（可覆盖） |
| `prepare_next_turn` | `(context) → AgentLoopTurnUpdate \| None` | 准备下一轮（可动态添加工具/消息） |
| `should_stop_after_turn` | `(context) → bool` | 判断是否提前终止循环 |
| `tool_execution` | `"sequential"` | 工具执行模式（当前仅 sequential） |

同步/异步钩子自动适配（通过 `asyncio.iscoroutine()` 检测）。

---

## 与 pi_ai 集成

### StreamFn — 依赖注入抽象

```python
# 全局注册（推荐）
from pi_agent import set_default_stream_fn
from pi_ai import create_default_models

models = create_default_models()
set_default_stream_fn(models.stream)

# 或显式传入
agent = Agent(AgentOptions(stream_fn=models.stream))
```

解析优先级：**显式传入 > 全局默认**。未设置时抛 `RuntimeError`。

### 消息转换管道

```
AgentMessage → transformContext (可选预处理)
             → convertToLlm (必须，默认过滤非标准 role)
             → LLM Message
             → stream_fn
             → 事件流
```

### 类型复用

`pi_agent` 直接复用 `pi_ai` 的核心类型：`Model`、`Message`、`AssistantMessageEventStream`、`StreamOptions`、`Usage`、`TextContent`、`ImageContent` 等。

---

## 纯函数入口

除了 `Agent` 类，也可直接使用纯函数引擎：

```python
from pi_agent import run_agent_loop, run_agent_loop_continue

# 独立调用，不依赖 Agent 实例
messages = await run_agent_loop(
    prompts=[user_message],
    context=context,
    config=config,
    emit=my_event_handler,
    signal=None,
    stream_fn=my_stream_fn,
)
```

---

## 开发

```bash
# 运行 pi_agent 测试（48 个测试）
uv run pytest src/pi_agent/tests/ -v

# 运行全部测试（pi_ai + pi_agent）
uv run pytest
```

## 许可

MIT
