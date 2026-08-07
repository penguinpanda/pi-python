# pi-agent — LLM Agent 循环与运行设施

基于 [pi-mono/packages/agent](https://github.com/earendil-works/pi-mono) 的 Python 复刻。

**纯函数引擎 + 有状态包装**：核心循环零类、零可变状态；外层 `Agent` 类提供面向用户的 OOP API。
在 Agent 循环之上还移植了 Session 会话树、技能/提示模板、上下文压缩、分支摘要、AgentHarness 与 Proxy 流函数等运行设施。

---

## 架构概览

```
用户 prompt("Hello")
    │
    ▼
Agent (有状态包装)
    │ 互斥锁、abort signal、双消息队列（steer / follow_up）、事件订阅
    ▼
run_agent_loop()  ◄── 纯函数引擎
    │
    ├─ emit agent_start
    ├─ emit turn_start（首轮，先于 prompts 注入）
    ├─ 注入 prompts → emit message_start / message_end
    │
    ▼
_run_loop()  ◄── 双重嵌套循环
    │
    │  外层 (Follow-up)：内层结束后检查 follow-up 队列，有消息则继续
    │
    │  内层 (Tool + Steering)：每轮 = 注入 pending 消息 → LLM → 工具 → 轮询 steering 队列
    │
    ┌──────────────────────────────────────────────┐
    │ 每轮 (turn):                                   │
    │  1. check_signal (取消检查)                     │
    │  2. emit turn_start（仅第 2 轮起；首轮已由外层发射）│
    │  3. _stream_assistant_response() → LLM 调用     │
    │     ├─ transformContext (可选)                   │
    │     ├─ convertToLlm (消息转换)                   │
    │     ├─ retry_assistant_call (应用层重试)          │
    │     ├─ stream_fn (LLM 流)                       │
    │     └─ 迭代事件流 → emit message_start/update/end│
    │  4. 提取 toolCalls → _execute_tool_calls()      │
    │     ├─ sequential / parallel 两种模式            │
    │     └─ 参数校验 + before/afterToolCall 钩子      │
    │  5. emit turn_end                              │
    │  6. prepare_next_turn / should_stop_after_turn  │
    │  7. 轮询 steering 队列 → 外层 follow-up 队列      │
    └──────────────────────────────────────────────┘
    │
    ▼
emit agent_end → 返回 messages
```

**关键设计决策**：

- **事件驱动**：通过 `emit(AgentEvent)` 回调发出 13 种事件，调用方自行归约
- **不可变性**：loop 内部复制 context，不修改调用方传入的值
- **依赖注入**：`StreamFn` 是 Callable 协议，不直接依赖任何具体 provider SDK
- **双消息队列**：`steer()` 在 turn 边界注入引导消息，`follow_up()` 在 Agent 即将停止时注入后续消息
- **自动重试**：LLM 调用失败按 `RetryPolicy` 指数退避重试（默认启用，max_retries=3），重试过程发 `auto_retry_start` / `auto_retry_end` 事件

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
    agent = Agent(
        AgentOptions(
            system_prompt="You are a concise assistant.",
            model=models.get_model("deepseek", "deepseek-v4-flash"),
        )
    )

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
| `await prompt(input, images=None)` | 发送用户消息（str / AgentMessage / 列表），运行完整 agent loop，阻塞直到完成 |
| `await continue_()` | 从当前 transcript 继续；末条为 assistant 时先消费 steering / follow-up 队列 |
| `steer(message)` / `follow_up(message)` | 入队引导消息 / 后续消息（turn 边界或 Agent 即将停止时注入） |
| `clear_steering_queue()` / `clear_follow_up_queue()` / `clear_all_queues()` | 清空消息队列 |
| `has_queued_messages()` | 任一队列仍有待处理消息时返回 True |
| `abort()` | 中止当前运行（设置 abort signal） |
| `subscribe(listener) → unsubscribe` | 订阅生命周期事件，返回取消订阅函数 |
| `await wait_for_idle()` | 等待当前运行结束（含所有事件监听器完成） |
| `reset()` | 清空 transcript、运行时状态和双消息队列 |
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
    prepare_next_turn_with_context: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    tool_execution: ToolExecutionMode = "parallel",
    # 双消息队列消费策略（"all" / "one-at-a-time"）
    steering_mode: QueueMode = "one-at-a-time",
    follow_up_mode: QueueMode = "one-at-a-time",
    # 提示缓存与会话标识（透传给 StreamOptions）
    session_id: str | None = None,
    cache_retention: CacheRetention | None = None,
    # 推理 token 预算与传输协议（透传给 StreamOptions / SimpleStreamOptions）
    thinking_budgets: ThinkingBudgets | None = None,
    transport: Transport | None = None,
    # 重试策略。None = 默认启用（max_retries=3）；RetryPolicy(enabled=False) 关闭
    retry_policy: RetryPolicy | None = None,
)
```

### `AgentState`

```python
@dataclass
class AgentState:
    system_prompt: str
    model: Model  # 当前模型（必填）
    thinking_level: ThinkingLevel = "off"
    tools: list[AgentTool]  # 赋值时自动防御性复制
    messages: list[AgentMessage]  # 赋值时自动防御性复制
    is_streaming: bool
    streaming_message: AgentMessage | None
    pending_tool_calls: set[str]
    error_message: str | None  # 最近一轮 error 的文本
```

---

## 工具系统

### 工具定义 (`AgentTool`)

```python
@dataclass(slots=True)
class AgentTool:
    name: str  # 工具名 (LLM 可见)
    description: str  # 工具描述 (给 LLM 看)
    input_schema: dict[str, Any]  # JSON Schema (给 LLM 选择参数 + 本地校验)
    label: str  # UI 显示标签
    execute: Callable[..., Awaitable[AgentToolResult]]
    # 签名: (tool_call_id, params, signal?, on_update?) → AgentToolResult
    execution_mode: ToolExecutionMode = "parallel"
    # 单工具执行模式；批次内任一工具声明 "sequential" → 整批回退顺序执行
    # 生命周期钩子（可选）
    before_execute: Callable[[dict, Any], Awaitable[Any]] | None
    after_execute: Callable[[Any], Awaitable[Any]] | None
```

### 工具结果 (`AgentToolResult`)

```python
@dataclass(slots=True)
class AgentToolResult:
    content: list[TextContent | ImageContent]  # 返回给 LLM 的内容
    details: Any = None  # 附加详情
    usage: Usage | None = None  # token 用量
    added_tool_names: list[str] | None = None  # 动态添加工具
    terminate: bool = False  # True → 停止后续 loop
```

### 四阶段执行管道

每轮 LLM 返回 toolCall 后：

```
阶段 1: 准备
  ├─ 按 name 查找工具
  ├─ validate_arguments (按 input_schema 校验/转换)
  ├─ beforeToolCall 钩子 (可 block 执行)
  └─ 中止检查

阶段 2: 执行
  ├─ Tool.before_execute (可选，可替换参数)
  ├─ emit tool_execution_start
  ├─ tool.execute(tool_call_id, params, signal, on_update)
  └─ try/except → 异常转 is_error=True

阶段 3: 完成
  └─ afterToolCall 钩子 (字段级覆盖 content/details/usage/terminate/is_error)

阶段 4: 发出
  ├─ emit tool_execution_end
  └─ 构造 ToolResultMessage → 追加到上下文
```

执行模式：

- `config.tool_execution == "sequential"` → 整批顺序执行；
- 否则 → 顺序准备 + 并发执行 + 按 assistant 原始顺序输出消息（`tool_execution_end` 按完成顺序发出）。

**截断保护**：`stop_reason="length"` 时，LLM 返回的工具调用参数可能不完整 → 不实际执行工具，直接生成错误 `ToolResultMessage`。

---

## 事件系统

13 种 `AgentEvent`（TypedDict 判别联合）：

| 事件 | 关键字段 | 含义 |
|------|---------|------|
| `agent_start` | — | Agent 循环开始 |
| `agent_end` | `messages` | Agent 循环结束 |
| `agent_settled` | — | 状态机进入 settled（一次 run/continue 完全结束） |
| `turn_start` | — | 单轮开始 |
| `turn_end` | `message`, `tool_results` | 单轮结束 |
| `message_start` | `message` | 消息进入上下文 |
| `message_update` | `message`, `assistant_message_event` | LLM 流式增量 |
| `message_end` | `message` | 消息完成 (追加到 transcript) |
| `tool_execution_start` | `tool_call_id`, `tool_name`, `args` | 工具开始执行 |
| `tool_execution_update` | `tool_call_id`, `partial_result` | 工具流式结果 |
| `tool_execution_end` | `tool_call_id`, `result`, `is_error` | 工具执行完成 |
| `auto_retry_start` | `attempt`, `max_attempts`, `delay_ms`, `error_message` | 重试已计划（退避等待开始前） |
| `auto_retry_end` | `success`, `attempt`, `final_error` | 重试循环结束（成功/放弃） |

监听器签名：`async (event, signal) → None`（同步监听器返回 None 也支持），`signal` 为当前运行的取消信号。

---

## 循环钩子

通过 `AgentOptions` 或 `AgentLoopConfig` 注入的钩子：

| 钩子 | 签名 | 说明 |
|------|------|------|
| `transform_context` | `(messages) → messages` | 预处理消息列表（如压缩、摘要） |
| `convert_to_llm` | `(messages) → LLM Message[]` | 消息格式转换（唯一转换点；默认只透传 user/assistant/toolResult，完整转换见 `pi_agent._messages.convert_to_llm`） |
| `get_api_key` | `(provider_id) → str \| None` | 动态获取 API Key |
| `before_tool_call` | `(BeforeToolCallContext) → BeforeToolCallResult \| None` | 工具执行前检查（可 block） |
| `after_tool_call` | `(AfterToolCallContext) → AfterToolCallResult \| None` | 工具执行后处理（字段级覆盖） |
| `prepare_next_turn` | `(context) → AgentLoopTurnUpdate \| None` | 准备下一轮（接收 `AgentContext`；可替换 context / model / thinking_level） |
| `prepare_next_turn_with_context` | `(PrepareNextTurnContext) → AgentLoopTurnUpdate \| None` | 带完整轮次上下文（message / tool_results / context / new_messages）的变体，优先于 `prepare_next_turn` |
| `should_stop_after_turn` | `(context) → bool` | 判断是否提前终止循环 |
| `get_steering_messages` | `() → list[AgentMessage]` | 轮间注入引导消息（steering 队列） |
| `get_follow_up_messages` | `() → list[AgentMessage]` | Agent 即将停止时注入后续消息（follow-up 队列） |

配置项：

- `tool_execution`: `"sequential"` / `"parallel"`（默认 `parallel`）；
- `session_id` / `cache_retention`: 透传给 `StreamOptions`（提示缓存）；
- `thinking_budgets` / `transport`: 透传给 `StreamOptions` / `SimpleStreamOptions`；
- `retry_policy`: 重试策略（默认启用，max_retries=3；传入 `RetryPolicy(enabled=False)` 关闭）。

同步/异步钩子自动适配（通过 `asyncio.iscoroutine()` 检测）。

---

## 内置工具与 ExecutionEnv

`pi_agent.tools` 提供四个可独立创建的工具（对齐 TS harness 工具）：

| 工具 | 创建函数 | 说明 |
|------|----------|------|
| read | `create_read_tool()` | 文本 + 图片（jpg/png/gif/webp/bmp），支持 offset/limit，输出截断（2000 行或 50KB），图片可自动转 Base64 附件 |
| write | `create_write_tool()` | 创建/覆盖文件，自动创建父目录 |
| edit | `create_edit_tool()` | 精确文本替换（`edits[].oldText/newText`），兼容 legacy 参数，返回 diff/patch |
| bash | `create_bash_tool()` | 执行 shell 命令，可设 timeout，输出截断并保存完整输出到临时文件 |

**图片管线**（`pi_agent.tools.image_pipeline`）：read 工具默认接入，提供
`exif_orientation`（EXIF 方向校正）、`resize_image`（最大 2000×2000 等比缩放）、
`convert_image` / `process_image`（多格式转 PNG，BMP/JPEG/GIF/WebP 统一归一化），
剪贴板图片（`pi_tui.ClipboardImage`）复用同一实现。

**工具范围约束**：read/bash 描述声明只操作工作目录内文件、禁止全盘搜索
（`find /`、`grep -r /`、`locate`）；read 对未找到文件返回"不要扩大搜索范围"指引。

所有工具通过 `ExecutionEnv` 进行 I/O（平台无关）：

- `FileSystem` / `Shell` 协议：方法一律返回 `Result[T, FileError]`（不抛异常）；
- `PythonExecutionEnv`：pathlib + asyncio subprocess 实现（Windows 下自动探测 Git bash）；
- `ShellExecOptions`：cwd / env / timeout / abort_signal / on_stdout / on_stderr。

输出截断工具见 `pi_agent.truncate`（`truncate_head` / `truncate_tail` / `format_size`，
默认 `DEFAULT_MAX_LINES=2000`、`DEFAULT_MAX_BYTES=50KB`），bash 输出捕获见 `pi_agent.shell_output`。

---

## Session 系统

`pi_agent.session` 提供 DAG 会话树模型 + 持久化存储 + 会话搜索（对齐 TS harness session）：

- **DAG 会话树**：仅追加不变式，`SessionTreeEntry` 支持
  `message` / `thinking_level_change` / `model_change` / `active_tools_change` /
  `compaction` / `branch_summary` / `custom` / `custom_message` / `label` / `session_info` / `leaf` 条目；
- **`Session`**：`get_branch(fromId)`（根到节点路径）、`build_context()`（重建 LLM 上下文，自动跳过被压缩条目）、
  `move_to(entryId)`（分支切换，可附带 branch_summary）、追加各类条目；
- **存储**：`JsonlSessionStore`（JSONL 文件）、`InMemorySessionStore`，以及 `create_jsonl_session_repo` /
  `create_in_memory_session_repo` / `create_session_repo` 等工厂；
- **搜索**：`ScanningSessionSearch` / `rebuild_session_search_index`，按条目内容建索引。

```python
from pi_agent import create_jsonl_session_repo

repo = create_jsonl_session_repo("~/.pi/agent/sessions")
session = await repo.create({"cwd": "."})
await session.append_message({"role": "user", "content": "hi", "timestamp": 0})
context = await session.build_context()
```

---

## 技能与提示模板

`pi_agent.skills` 对齐 TS `harness/skills.ts`：

- 从目录递归发现 `SKILL.md`，遵循 `.gitignore` / `.ignore` / `.fdignore`；
- YAML frontmatter 元数据（`description`、`disable-model-invocation`）；
- 名称约束：64 字符、`[a-z0-9-]+`，描述上限 1024 字符；
- `format_skill_invocation()` 把技能格式化为 `<skill>` XML 块（XML 转义防注入）。

`pi_agent.prompt_templates` 对齐 TS `harness/prompt-templates.ts`：

- 从 `.md` 文件加载模板（frontmatter 提供 `description`）；
- 参数替换：`$1..$N`、`$@` / `$ARGUMENTS`、`${@:N}` / `${@:N:L}`；
- `load_prompt_templates` / `load_sourced_prompt_templates` / `substitute_args` / `format_prompt_template_invocation`。

---

## 压缩与分支摘要

`pi_agent.compaction` 对齐 TS `harness/compaction/compaction.ts`：

- `CompactionSettings`：`enabled` / `reserve_tokens`（默认 16384）/ `keep_recent_tokens`（默认 20000）；
- `prepare_compaction(entries, messages, settings)`：基于 Session 条目定位切割点（`find_cut_point`）；
- `compact(preparation, model, stream_fn, thinking_level)`：用独立 LLM 调用生成摘要（缓存隔离）；
- 支持迭代压缩（previous_summary 合并）。

`pi_agent.branch_summarization` 对齐 TS `branch-summarization.ts`：

- `collect_entries_for_branch_summary(session, old_leaf_id, target_id)`：收集旧 leaf → 共同祖先的差异条目；
- `generate_branch_summary(...)`：生成结构化分支摘要。

---

## AgentHarness

`pi_agent.AgentHarness` 是更高层的协调器（对齐 TS `harness/agent-harness.ts`）：

- 阶段状态机：`idle → turn / compaction / branch_summary → idle`；
- 双事件系统：`subscribe()` 通配符订阅 + `on(event_type, handler)` 类型化 hook（顺序归约器）；
- Save-point 安全模型：运行期间的配置变更记录为 pending mutation，在安全点 flush；
- 显式调用：`prompt()` / `skill(name, args)` / `prompt_from_template(name, args)` /
  `steer()` / `follow_up()` / `next_turn()` / `compact()` / `navigate_tree()` / `abort()` / `shutdown()`；
- 运行时配置：`get/set_model`、`get/set_thinking_level`、`get/set_tools`、`get/set_active_tools`、
  `get/set_resources`（skills / prompt_templates）、`get/set_stream_options`。

---

## Proxy 流函数

`pi_agent.proxy` 对齐 TS `packages/agent/src/proxy.ts`：把 LLM 调用路由经过服务器中转，
服务器负责认证与 provider 代理，客户端通过 SSE 接收事件。

```python
from pi_agent.proxy import stream_proxy

agent = Agent(
    AgentOptions(
        model=model,
        stream_fn=lambda model, context, options: stream_proxy(
            model,
            context,
            {
                **(options or {}),
                "authToken": token,
                "proxyUrl": "https://genai.example.com",
            },
        ),
    )
)
```

带宽优化：服务器剥离 `partial` 字段，客户端基于事件流本地重建 partial 消息
（文本拼接、thinking 拼接、toolCall 参数流式解析）。

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
from pi_agent import run_agent_loop, run_agent_loop_continue, agent_loop

# 独立调用，不依赖 Agent 实例
messages = await run_agent_loop(
    prompts=[user_message],
    context=context,
    config=config,
    emit=my_event_handler,
    signal=None,
    stream_fn=my_stream_fn,
)

# 从已有上下文继续（重试/恢复场景）
messages = await run_agent_loop_continue(
    context=context,
    config=config,
    emit=my_event_handler,
    signal=None,
    stream_fn=my_stream_fn,
)

# EventStream 包装：agent_end 即流结束事件
stream = agent_loop(prompts, context, config, signal, stream_fn)
messages = await stream.result()
```

---

## 开发

```bash
# 运行 pi_agent 测试
uv run pytest src/pi_agent/tests/ -v

# 运行全部测试（pi_ai + pi_agent + pi_coding_agent）
uv run pytest
```

## 许可

MIT
