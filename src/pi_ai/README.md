# pi-ai-python

统一的 LLM API，Provider 抽象模式。

基于 [pi-mono/packages/ai](https://github.com/earendil-works/pi-mono) 的 TypeScript 版本复刻，目前支持 **OpenAI**、**DeepSeek** 和 **Ollama** 三个 provider。

---

## 支持的模型

| Provider | 模型 ID | API 类型 | Thinking | Tool Calling | 图片输入 | maxTokens |
|----------|---------|----------|:--------:|:------------:|:--------:|:---------:|
| OpenAI | `gpt-4o` | Responses | ✗ | ✓ | ✓ | 16,384 |
| OpenAI | `gpt-4o-mini` | Responses | ✗ | ✓ | ✓ | 16,384 |
| OpenAI | `o4-mini` | Responses | ✓ | ✓ | ✗ | 100,000 |
| DeepSeek | `deepseek-chat` | Completions | ✗ | ✓ | ✗ | 65,536 |
| DeepSeek | `deepseek-reasoner` | Completions | ✓ | ✗ | ✗ | 65,536 |
| DeepSeek | `deepseek-v4-flash` | Completions | ✓ | ✓ | ✗ | 384,000 |
| Ollama | `qwen3:30b` | Completions | ✓ | ✓ | ✗ | 8,192 |
| Ollama | `qwen3:30b-a3b` | Completions | ✓ | ✓ | ✗ | 8,192 |
| Ollama | `richardyoung/qwen3-14b-abliterated:Q5_K_M` | Completions | ✓ | ✓ | ✗ | 8,192 |
| Ollama | `gpt-oss:20b` | Completions | ✓ | ✓ | ✗ | 32,768 |
| Ollama | `llama3.2-vision:latest` | Completions | ✗ | ✓ | ✓ | 4,096 |
| Ollama | `qwen2.5:7b-instruct-q8_0` | Completions | ✗ | ✓ | ✗ | 8,192 |
| Ollama | `deepseek-r1:14b` | Completions | ✓ | ✗ | ✗ | 8,192 |

Ollama 模型列表对应本地 `ollama list` 的输出；新增或卸载模型后需要同步更新
`OLLAMA_MODELS`（或改为运行时调用 `/api/tags` 动态生成）。

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

通过环境变量设置 API Key：

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."
```

Ollama 是本地服务，默认不需要 API Key：

```python
from pi_ai import Models
from pi_ai.providers import ollama_provider

models = Models()
models.add_provider(ollama_provider())
```

Ollama 默认地址为 `http://127.0.0.1:11434/v1`
（不使用 `localhost`，避免 httpx 走 Windows 系统代理导致 503），
如果修改了 `OLLAMA_HOST`，需要同步调整 Provider 的 `base_url`。

也可以在代码中动态设置：

```python
from pi_ai import create_default_models

models = create_default_models()
await models.set_api_key("openai", "sk-...")
await models.set_api_key("deepseek", "sk-...")
```

### Hello World

```python
import asyncio
from pi_ai import create_default_models, Context


async def main():
    # 创建模型管理器
    # 预加载 OpenAI 和 DeepSeek 两个 provider
    models = create_default_models()

    # 获取一个模型（不会发送网络请求）
    model = models.get_model("deepseek", "deepseek-v4-flash")

    # 构造上下文
    context = Context(
        system="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": "Hello!"},
        ],
    )

    # 非流式调用：等待完整结果
    msg = await models.complete(model, context)
    print(
        "".join(
            block["text"]
            for block in msg["content"]
            if block["type"] == "text"
        )
    )

    # 流式调用：逐 Token 输出
    async for event in await models.stream(model, context):
        if event["type"] == "delta":
            print(event["text"], end="", flush=True)


asyncio.run(main())
```

---

## 核心功能

### 非流式调用（`complete`）

`complete()` 等待整个模型回复完成后一次性返回 `AssistantMessage`：

```python
msg = await models.complete(model, context)

# msg 结构：
# {
#     "role": "assistant",
#     "content": [
#         {"type": "text", "text": "..."},
#         {"type": "thinking", "text": "..."},      # 推理模型
#         {"type": "toolCall", "tool": {...}},       # 工具调用
#     ],
#     "usage": {"input": 100, "output": 50, ...},
#     "stopReason": "end",
# }
```

### 流式调用（`stream`）

`stream()` 返回 `AssistantMessageEventStream`，支持 `async for` 逐事件消费：

```python
async for event in await models.stream(model, context):
    match event["type"]:
        case "delta":
            print(event["text"], end="", flush=True)
        case "toolCallDelta":
            print(f"\n🔧 calling {event['tool']}...")
        case "thinkingDelta":
            pass  # 推理过程（可选输出）
        case "done":
            print(f"\n✅ done ({event['message']['usage']})")
```

事件类型一览：

| 事件类型 | 说明 | 关键字段 |
|----------|------|----------|
| `delta` | 文本增量 | `text` |
| `toolCallDelta` | 工具调用增量 | `tool` |
| `thinkingDelta` | 推理过程增量 | `thinking` |
| `done` | 流正常结束 | `message`（完整 `AssistantMessage`） |
| `error` | 流异常结束 | `reason`, `error` |

也可以通过 `await result()` 等待最终结果：

```python
stream = await models.stream(model, context)
# ... 可以同时消费事件 ...
msg = await stream.result()  # 等到 done/error
```

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
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
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
            print(f"模型要调用工具: {block['toolName']}")
            print(f"参数: {block['args']}")

    # 第二轮：将工具执行结果返回给模型
    context.messages.append(msg)  # assistant 消息
    context.messages.append({
        "role": "toolResult",
        "toolCallId": msg["content"][0]["toolCallId"],
        "toolName": "get_weather",
        "content": [{"type": "text", "text": "北京今天晴，25°C"}],
    })

    final_msg = await models.complete(model, context)
    print(
        "".join(
            block["text"]
            for block in final_msg["content"]
            if block["type"] == "text"
        )
    )

asyncio.run(main())
```

---

## 推理模式（Thinking）

DeepSeek Reasoner 和 o4-mini 支持推理过程（Thinking），模型会先生成内部思维链再输出最终回复：

```python
from pi_ai import create_default_models, Context

models = create_default_models()
model = models.get_model("deepseek", "deepseek-reasoner")

context = Context(
    messages=[
        {"role": "user", "content": "证明 √2 是无理数"},
    ],
)

async for event in await models.stream(model, context):
    if event["type"] == "thinkingDelta":
        # 推理过程
        print(f"[思考] {event['thinking']}", end="", flush=True)
    elif event["type"] == "delta":
        # 最终回复
        print(event["text"], end="", flush=True)
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
    "thinkingEnabled": False,
    # "thinkingBudget": 2048,  # 限制 thinking token 数量
}
async for event in await models.stream(model, context, options):
    ...
```

---

## 流选项（StreamOptions）

`stream()` 和 `complete()` 都支持可选的第三个参数 `StreamOptions`：

| 参数 | 类型 | 说明 |
|------|------|------|
| `temperature` | `float` | 采样温度 |
| `maxTokens` | `int` | 最大输出 Token 数 |
| `thinkingBudget` | `int` | 推理 Token 预算（仅推理模型） |
| `thinkingEnabled` | `bool` | 是否启用推理 |
| `apiKey` | `str` | 本次请求使用的 API Key（覆盖默认） |
| `headers` | `dict` | 额外的 HTTP 请求头 |

```python
options = {
    "temperature": 0.7,
    "maxTokens": 2000,
    "headers": {"X-Custom-Header": "value"},
}
msg = await models.complete(model, context, options)
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
    elif event["type"] == "delta":
        print(event["text"], end="")
```

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
        api_kind="openai-completions",
        base_url="https://openrouter.ai/api/v1",
        auth=env_api_key_auth("OPENROUTER_API_KEY"),
        models=[
            Model(
                id="anthropic/claude-sonnet-4",
                provider="openrouter",
                api="openai-completions",
                name="Claude Sonnet 4",
                input=["text"],
                output=["text"],
                maxTokens=16384,
                thinking=False,
                supportsToolCalling=True,
                supportsImages=False,
            ),
        ],
    )
)

model = models.get_model("openrouter", "anthropic/claude-sonnet-4")
```

---

## 架构概览

```
                         Models  ◄── 统一入口（Facade）
                           │
                  ┌────────┼────────┐
                  ▼                 ▼
           Provider(OpenAI)   Provider(DeepSeek)
           api_kind=          api_kind=
           responses          completions
                  │                 │
           resolve_api_key()  resolve_api_key()
                  │                 │
                  ▼                 ▼
          responses_stream()  chat_completions_stream()
                  │                 │
                  ▼                 ▼
           AssistantMessageEventStream
```

- **`Models`** — 注册表，管理所有 Provider，调度请求
- **`Provider`** — 封装 provider 配置（base URL、认证、模型列表、API 类型）
- **`EventStream`** — 基于 `asyncio.Queue` 的生产者-消费者异步事件流
- **`api/`** — 将不同 API 协议（Completions / Responses）统一转为 SDK 事件

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
