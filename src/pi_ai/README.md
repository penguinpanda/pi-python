# pi-ai-python

统一的 LLM API，Provider 抽象模式。

基于 [pi-mono/packages/ai](https://github.com/earendil-works/pi-mono) 的 TypeScript 版本复刻，默认内置 **OpenAI**、**DeepSeek**、**Qwen**、**Ollama** 与 **Faux** 五个 provider。

---

## 支持的模型（默认注册表）

`create_default_models()` 预加载以下静态模型列表：

| Provider | 模型 ID | API 类型 | Thinking | Tool Calling | 图片输入 | max_tokens |
|----------|---------|----------|:--------:|:------------:|:--------:|:---------:|
| OpenAI | `gpt-5-chat-latest` | Responses | ✗ | ✓ | ✓ | 16,384 |
| OpenAI | `gpt-5.6-luna` / `gpt-5.6-sol` / `gpt-5.6-terra` | Responses | ✓ | ✓ | ✓ | 128,000 |
| DeepSeek | `deepseek-v4-flash` | Responses | ✓ | ✓ | ✗ | 384,000 |
| DeepSeek | `deepseek-v4-pro` | Completions | ✓ | ✓ | ✗ | 384,000 |
| Qwen | `qwen-turbo` | Completions | ✗ | ✓ | ✗ | 8,192 |
| Qwen | `qwen-plus` | Completions | ✗ | ✓ | ✗ | 8,192 |
| Qwen | `qwen-max` | Completions | ✗ | ✓ | ✗ | 8,192 |
| Qwen | `qwen3-235b-a22b` | Completions | ✓ | ✓ | ✗ | 131,072 |
| Qwen | `qwen3-30b-a3b` | Completions | ✓ | ✓ | ✗ | 32,768 |
| Qwen | `qwen3-vl-flash` | Completions | ✗ | ✓ | ✓ | 8,192 |
| Qwen | `qwen-vl-plus` | Completions | ✗ | ✓ | ✓ | 4,096 |
| Qwen | `qwen-vl-max` | Completions | ✗ | ✓ | ✓ | 8,192 |
| Ollama | `qwen3:30b` | Completions | ✓ | ✓ | ✗ | 8,192 |
| Ollama | `qwen3:30b-a3b` | Completions | ✓ | ✓ | ✗ | 8,192 |
| Ollama | `richardyoung/qwen3-14b-abliterated:Q5_K_M` | Completions | ✓ | ✓ | ✗ | 8,192 |
| Ollama | `gpt-oss:20b` | Completions | ✓ | ✓ | ✗ | 32,768 |
| Ollama | `llama3.2-vision:latest` | Completions | ✗ | ✓ | ✓ | 4,096 |
| Ollama | `qwen2.5:7b-instruct-q8_0` | Completions | ✗ | ✓ | ✗ | 8,192 |
| Ollama | `deepseek-r1:14b` | Completions | ✓ | ✗ | ✗ | 8,192 |
| Faux | `faux-1` | Completions | ✗ | ✓ | ✗ | 16,384 |

### Ollama 动态发现

静态目录对应 `ollama list` 的输出。`ollama_provider()` 支持运行时动态发现：

- `discover_ollama_models()` 请求 `GET http://127.0.0.1:11434/api/tags`，把实际安装的模型与静态元数据合并；
- 未知模型（新 pull 的）合成默认元数据；请求失败返回 `None` 并回退静态列表；
- `Models.refresh()` 可统一触发各 provider 的刷新（`ollama_provider` 已内置 `fetch_models`），
  结果可持久化到 `ModelsStore`（`InMemoryModelsStore` / `FileModelsStore`），失败时 best-effort 恢复缓存。

### 自动生成的模型目录

`models/generated/providers/*.json` 由 [scripts/generate_models.py](scripts/generate_models.py) 生成，包含
`openai`、`openai-codex`、`azure-openai-responses`、`deepseek`、`mistral`、`ant-ling`、`openrouter`、
`vercel-ai-gateway` 的远程目录；`create_default_models()` 默认即加载并合并这些元数据
（同 id 覆盖已注册 provider 的模型、新 id 追加；无 Python 实现的 provider 保持不可用），
`load_generated_models()` 可单独调用以获取原始目录数据。

---

## 快速开始

### 安装

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest
```

### 认证

API Key 解析优先级：**请求级 `api_key`（StreamOptions） > CredentialStore > 环境变量**。

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# Qwen（DashScope 按量计费）
export DASHSCOPE_API_KEY="sk-..."

# Qwen Token Plan（阿里云百炼 Token Plan 套餐，sk-sp- 前缀）
export QWEN_TOKEN_PLAN_API_KEY="sk-sp-..."      # 国际站
export QWEN_TOKEN_PLAN_CN_API_KEY="sk-sp-..."   # 中国站
```

Ollama 是本地服务，默认不需要 API Key：

```python
from pi_ai import Models
from pi_ai.providers import ollama_provider

models = Models()
models.add_provider(ollama_provider())
```

Ollama 默认地址为 `http://127.0.0.1:11434/v1`
（不使用 `localhost`，避免 httpx 走 Windows 系统代理导致 503）。

也可以在代码中动态设置：

```python
from pi_ai import create_default_models

models = create_default_models()
await models.set_api_key("openai", "sk-...")
await models.set_api_key("deepseek", "sk-...")
```

### OAuth 登录（CLI）

内置三个 OAuth provider（device code / PKCE 流程），凭证保存到当前目录 `auth.json`：

```bash
pi-ai list                # 列出可用 OAuth provider
pi-ai login openai-codex  # 或 github-copilot / openrouter；不带参数时交互选择
```

也可直接运行 `python -m pi_ai login openai-codex`。

### Hello World

```python
import asyncio
from pi_ai import create_default_models, Context


async def main():
    # 创建模型管理器
    # 预加载 OpenAI、DeepSeek、Qwen、Ollama 和 Faux 五个 provider
    models = create_default_models()

    # 获取一个模型（不会发送网络请求）
    model = models.get_model("deepseek", "deepseek-v4-flash")

    # 构造上下文
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": "Hello!"},
        ],
    )

    # 非流式调用：等待完整结果
    msg = await models.complete(model, context)
    print("".join(block["text"] for block in msg["content"] if block["type"] == "text"))

    # 流式调用：逐 Token 输出
    async for event in await models.stream(model, context):
        if event["type"] == "text_delta":
            print(event["delta"], end="", flush=True)


asyncio.run(main())
```

---

## 核心功能

### Models 注册表

`Models` 是整个 SDK 的统一入口（Facade），本身不实现任何模型接口，所有请求转发给对应 Provider：

| 方法 | 说明 |
|------|------|
| `add_provider(provider)` / `remove_provider(id)` | 注册 / 移除 Provider（同 ID 覆盖） |
| `get_provider(id)` / `get_providers()` | 查询 Provider |
| `get_models(provider_id=None)` | 模型列表（可按 provider 过滤） |
| `get_model(provider_id, model_id)` | 精确查找模型 |
| `get_model_by_id(model_id)` | 跨所有 provider 按 ID 查找 |
| `await refresh(options)` | 并发刷新支持动态发现的 provider，单点失败不阻断（`ModelsRefreshResult` 收集错误） |
| `await stream(model, context, options)` | 流式调用，返回 `AssistantMessageEventStream` |
| `await complete(model, context, options)` | 非流式调用，内部仍走 stream 并等待最终消息 |
| `await set_api_key(provider_id, api_key)` | 写入 CredentialStore（优先于环境变量） |

### 非流式调用（`complete`）

```python
msg = await models.complete(model, context)

# msg 结构：
# {
#     "role": "assistant",
#     "content": [
#         {"type": "text", "text": "..."},
#         {"type": "thinking", "thinking": "..."},       # 推理模型
#         {"type": "toolCall", "id": "...", "name": "...", "raw_arguments": "...", "arguments": {...}},  # 工具调用
#     ],
#     "usage": {"input": 100, "output": 50, ...},
#     "stop_reason": "stop",
# }
```

### 流式调用（`stream`）

```python
async for event in await models.stream(model, context):
    match event["type"]:
        case "text_delta":
            print(event["delta"], end="", flush=True)
        case "toolcall_delta":
            pass  # 工具参数增量（原始 JSON 片段）
        case "thinking_delta":
            pass  # 推理过程（可选输出）
        case "done":
            print(f"\ndone ({event['message']['usage']})")
```

事件类型一览（12 种）：

| 事件类型 | 说明 | 关键字段 |
|----------|------|----------|
| `start` | 流开始 | `partial` |
| `text_start` / `text_delta` / `text_end` | 文本块生命周期 | `content_index`, `delta`, `partial` |
| `thinking_start` / `thinking_delta` / `thinking_end` | 推理块生命周期 | `content_index`, `delta`, `partial` |
| `toolcall_start` / `toolcall_delta` / `toolcall_end` | 工具调用块生命周期 | `content_index`, `delta` / `tool_call`, `partial` |
| `done` | 流正常结束 | `message`（完整 `AssistantMessage`） |
| `error` | 流异常结束 | `reason`, `error` |

每个增量事件（`start` / `*_start` / `*_delta` / `*_end`）都携带 `partial` 字段——
当前累积状态的 `AssistantMessage` 快照，消费方无需自行拼接。
`toolcall_end` 额外携带已解析的 `ToolCall`（`arguments` 为对象）。

也可以通过 `await result()` 等待最终结果：

```python
stream = await models.stream(model, context)
# ... 可以同时消费事件 ...
msg = await stream.result()  # 等到 done/error
```

### 流选项（StreamOptions）

`stream()` 和 `complete()` 都支持可选的第三个参数 `StreamOptions`：

| 参数 | 类型 | 说明 |
|------|------|------|
| `temperature` | `float` | 采样温度 |
| `max_tokens` | `int` | 最大输出 Token 数 |
| `thinking_budget` | `int \| None` | 推理 Token 预算（仅推理模型） |
| `thinking_enabled` | `bool \| None` | 是否启用推理 |
| `api_key` | `str` | 本次请求使用的 API Key（覆盖默认） |
| `base_url` | `str` | Provider 层解析后的 Base URL（回退 `model.base_url`） |
| `headers` | `dict` | 额外的 HTTP 请求头 |
| `max_retries` | `int` | Provider 层客户端重试上限 |
| `max_retry_delay_ms` | `int` | 最大重试延迟（毫秒） |
| `signal` | `asyncio.Event` | 流式中止信号 |
| `http_client` | `AsyncHTTPClient` | 可注入的异步 HTTP 客户端 |
| `transport` | `Transport` | 首选传输协议（`sse` / `websocket` / `auto` 等） |
| `cache_retention` | `CacheRetention` | 提示缓存保留策略（默认 `short`） |
| `session_id` | `str` | 会话标识（用于提示缓存 key） |
| `timeout_ms` | `int` | HTTP 请求超时 |
| `on_payload` / `on_response` | `Callable` | 请求发送前 / 响应接收后回调 |
| `metadata` / `env` | `dict` | 附加元数据 / Provider 作用域环境变量 |

```python
options = {
    "temperature": 0.7,
    "max_tokens": 2000,
    "headers": {"X-Custom-Header": "value"},
}
msg = await models.complete(model, context, options)
```

### 重试

应用层重试工具（`pi_ai.utils.retry`）：

- `RetryPolicy`：`enabled` / `max_retries`（默认 3）/ `base_delay_ms`（默认 2000）/ `max_delay_ms` / `jitter`；
- `retry_assistant_call(produce, policy, signal, callbacks)` 包装单次 LLM 调用，退避期间支持中止；
- `is_retryable_error()` 判断错误是否可重试；
- `compute_backoff_delay()` 计算指数退避延迟。

### 提示缓存

`StreamOptions.session_id` + `cache_retention` 会解析为请求体参数：

- `prompt_cache_key`（OpenAI 限制 64 字符，超长自动截断）；
- `prompt_cache_retention`（`long` 时发送 `24h`）。

相关实现见 `pi_ai.utils.prompt_cache`（`clamp_openai_prompt_cache_key` / `resolve_cache_retention`）。

### Token 估算与上下文溢出

`pi_ai.utils.estimate` / `overflow` 提供：

- `estimate_context_tokens` / `estimate_message_tokens` / `estimate_tools_tokens` / `calculate_context_tokens`；
- `is_context_overflow(message, context_window)` 与 `get_overflow_patterns()`（`OVERFLOW_PATTERNS` / `NON_OVERFLOW_PATTERNS`）。

---

## 工具调用（Tool Calling）

```python
import asyncio
from pi_ai import create_default_models, Context, Tool


async def main():
    models = create_default_models()
    model = models.get_model("deepseek", "deepseek-v4-flash")

    # 定义工具
    tools = [
        Tool(
            name="get_weather",
            description="获取指定城市的天气",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名称"}},
                "required": ["city"],
            },
        ),
    ]

    context = Context(
        messages=[
            {"role": "user", "content": "北京今天天气怎么样？"},
        ],
        tools=tools,
    )

    # 第一轮：模型返回 toolCall
    msg = await models.complete(model, context)
    for block in msg["content"]:
        if block["type"] == "toolCall":
            print(f"模型要调用工具: {block['name']}")
            print(f"参数: {block['arguments']}")

    # 第二轮：将工具执行结果返回给模型
    context.messages.append(msg)  # assistant 消息
    context.messages.append(
        {
            "role": "toolResult",
            "tool_call_id": msg["content"][0]["id"],
            "tool_name": "get_weather",
            "content": [{"type": "text", "text": "北京今天晴，25°C"}],
        }
    )

    final_msg = await models.complete(model, context)
    print("".join(block["text"] for block in final_msg["content"] if block["type"] == "text"))


asyncio.run(main())
```

`Tool` 支持生命周期钩子与可选实现：

- `handler`：可选的 Python 执行函数（None 表示仅定义、无实现）；
- `before_execute(args, context)`：执行前钩子，返回 dict 替换传给 handler 的参数；
- `after_execute(result)`：执行后钩子，返回新值替换最终结果；
- `constrained_sampling`：`json_schema` / `grammar` 受约束采样配置。

---

## 推理模式（Thinking）

DeepSeek V4 系列（`deepseek-v4-flash` / `deepseek-v4-pro`）与 OpenAI `gpt-5.6-*` 支持推理过程（Thinking），模型会先生成内部思维链再输出最终回复：

```python
from pi_ai import create_default_models, Context

models = create_default_models()
model = models.get_model("deepseek", "deepseek-v4-flash")

context = Context(
    messages=[
        {"role": "user", "content": "证明 √2 是无理数"},
    ],
)

async for event in await models.stream(model, context):
    if event["type"] == "thinking_delta":
        # 推理过程
        print(f"[思考] {event['delta']}", end="", flush=True)
    elif event["type"] == "text_delta":
        # 最终回复
        print(event["delta"], end="", flush=True)
    elif event["type"] == "done":
        msg = event["message"]
        # content 中同时包含 thinking 和 text
        for block in msg["content"]:
            if block["type"] == "thinking":
                print(f"\n\n推理 Token: {len(block['text'])}")
```

通过 `StreamOptions` 控制 thinking 行为：

```python
# 关闭或限制 thinking
options = {
    "thinking_enabled": False,
    # "thinking_budget": 2048,  # 限制 thinking token 数量
}
async for event in await models.stream(model, context, options):
    ...
```

---

## 错误处理

当模型返回错误或网络异常时，事件流会以 `error` 事件结束：

```python
async for event in await models.stream(model, context):
    if event["type"] == "error":
        print(f"请求失败: {event.get('reason', 'unknown')}")
        # event["error"] 包含一个 fallback AssistantMessage
        break
    elif event["type"] == "text_delta":
        print(event["delta"], end="")
```

---

## API 注册表（自定义 API 协议）

API 协议实现通过注册表按 `model.api` 分发（对齐 TS `apiProviderRegistry`），新增协议无需改动调度代码：

```python
from pi_ai import ApiProvider, register_api_provider

register_api_provider(
    ApiProvider(
        api="my-api",
        stream=my_stream,  # (model, context, options) -> EventStream
        streamSimple=my_stream_simple,
    )
)
```

内置注册：`openai-completions`、`openai-responses`、`pi-messages`。
`pi-messages` 是 pi 自有线协议（`POST {baseUrl}/messages`，SSE 事件流）。

---

## 图片生成（Images）

聊天 Provider 之外有平行的图片生成设施：

```python
from pi_ai.images import generate_images
from pi_ai.types import ImagesModel, ImagesContext

model = ImagesModel(id="some-image-model", api="openrouter-images", provider="openrouter")
result = await generate_images(
    model,
    ImagesContext(input=[{"type": "text", "text": "a cat"}]),
)
```

- 内置 `openrouter-images` API（走 Chat Completions 的 `modalities: ["image"]` 扩展）；
- `generate_images` 永不 reject：失败返回 `stop_reason="error"` 的 `AssistantImages`；
- 顶层入口 `pi_ai.images.generate_images(model, context, options)` 按 `model.api`
  从图片 API 注册表分发；
- `pi_ai.images_models` 提供与聊天侧平行的 Provider 集合
  （`ImagesModels` / `create_images_provider` / `create_images_models`）。

---

## 可观测性（Trace）

`pi_ai.trace` 提供运行时追踪器（类型定义在 `pi_ai.types.trace`）：

```python
from pi_ai.trace import TraceTracer, run_with_trace

tracer = TraceTracer("my-agent")
await run_with_trace(tracer, some_async_work())
# tracer.trace 收集完整 Trace（含嵌套 span）
```

`trace_id` 可贯穿 `Context` → 请求 → 事件。

---

## 自定义 Provider

通过 `create_provider()` 可以接入任意兼容 OpenAI Chat Completions API 的服务：

```python
from pi_ai import Models, Model, create_provider
from pi_ai.auth import env_api_key_auth

# 接入 OpenRouter
models = Models()
models.add_provider(
    create_provider(
        id="openrouter",
        name="OpenRouter",
        api_kind="completions",
        base_url="https://openrouter.ai/api/v1",
        auth=env_api_key_auth("OpenRouter API key", ["OPENROUTER_API_KEY"]),
        models=[
            Model(
                id="anthropic/claude-sonnet-4",
                provider="openrouter",
                api="openai-completions",
                name="Claude Sonnet 4",
                input=["text"],
                output=["text"],
                max_tokens=16384,
            ),
        ],
    )
)

model = models.get_model("openrouter", "anthropic/claude-sonnet-4")
```

`create_provider` 还支持：

- `stream_fn`：自定义流函数（跳过认证与 API 分发，用于测试）；
- `fetch_models`：动态模型刷新实现（配合 `Models.refresh()`）；
- `auth=None`：本地服务（如 Ollama）不需要 API Key。

---

## 架构概览

```
                         Models  ◄── 统一入口（Facade）
                           │
                  ┌────────┼────────┐
                  ▼                 ▼
           Provider(OpenAI)   Provider(DeepSeek)
           api=responses      api=completions
                  │                 │
           resolve_api_key()  resolve_api_key()
                  │                 │
                  ▼                 ▼
      API 注册表分发（按 model.api）
          responses_stream()  chat_completions_stream()
                  │                 │
                  ▼                 ▼
           AssistantMessageEventStream
```

- **`Models`** — 注册表，管理所有 Provider，调度请求，管理凭证与模型目录
- **`Provider`** — 封装 provider 配置（base URL、认证、模型列表、API 类型）
- **`api/`** — 按 API 协议实现（Completions / Responses / pi-messages），经注册表按 `model.api` 分发
- **`EventStream`** — 基于 `asyncio.Queue` 的生产者-消费者异步事件流
- **`auth/`** — API Key 凭证 + OAuth（openai-codex / github-copilot / openrouter）
- **`models/`** — 模型注册表、持久化 ModelsStore、自动生成的模型目录

### 类型组织（`pi_ai.types`）

所有公共类型定义在 `pi_ai/types/` 包（原 `pi_ai/_types.py`，现为兼容 re-export）：

```
pi_ai/types/
├── common.py     # 基础枚举 / 协议（StopReason、AsyncHTTPClient ...）
├── content.py    # ContentBlock（Text / Image / ToolCall / Thinking / Code）
├── message.py    # Message（System / User / Assistant / ToolResult / Agent）
├── tool.py       # Tool（含 before_execute / after_execute 生命周期钩子）
├── model.py      # Model / ModelCost
├── context.py    # Context（含 state / memory / trace_id）+ MemoryStore
├── stream.py     # 流事件（BaseEvent + 12 种事件）+ StreamOptions
├── image.py      # 图片生成类型
├── compat.py     # Provider 兼容配置（Compat）
└── trace.py      # 可观测性（Trace / TraceSpan）
```

扩展要点：

- `Message` 联合含 `AgentMessage`（planner / observation / memory 等通用 role），
  Agent 层可携带任意 Agent role；转换函数对未知 role 安全跳过。
- `ContentBlock` 继承 `BaseContent`，新增类型只需继承并把 `type` 收窄为唯一字面量即可插件化扩展。
- `ToolCall` 含 `raw_arguments`（流式原始 JSON）与 `arguments`（解析后 dict / None）。
- `Tool` 支持 `before_execute` / `after_execute` 生命周期钩子（默认 None）。
- `Context` 可注入 `state` / `memory`（`MemoryStore`）/ `trace_id`。

### 工具函数（`pi_ai.utils`）

- **重试**：`RetryPolicy` / `retry_assistant_call` / `is_retryable_error`
- **Token / 上下文**：`estimate_*` / `calculate_context_tokens` / `is_context_overflow`
- **校验**：`validate_tool_arguments` / `validate_tool_call` / `coerce_with_json_schema`
- **JSON**：`repair_json` / `parse_streaming_json` / `partial_json`
- **诊断**：`create_assistant_message_diagnostic` / `append_assistant_message_diagnostic`
- **其它**：`prompt_cache`（提示缓存参数）、`provider_env`、`uuid`（uuidv7）、`http_cache`

---

## 开发

```bash
# 安装依赖
uv sync

# 运行全部测试
uv run pytest

# 运行指定测试文件
uv run pytest src/pi_ai/tests/test_models.py -v

# 运行集成测试（需要 API Key）
OPENAI_API_KEY="sk-..." uv run pytest src/pi_ai/tests/test_stream.py -v
```

## 许可

MIT
