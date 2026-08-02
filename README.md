# pi-python

基于 [pi-mono](https://github.com/earendil-works/pi-mono) 的 Python 复刻，三层架构的 AI 编码代理工具链。

---

## 包结构

```
pi-python/
├── src/pi_ai/              # LLM API 抽象层（Provider 模式）
├── src/pi_agent/           # Agent 循环引擎（纯函数核心 + 有状态包装）
└── src/pi_coding_agent/    # CLI 编码代理（工具 + 会话 + 配置）
```

| 包 | 文档 | 说明 | 打包 |
|---|------|------|:---:|
| `pi_ai` | [README](src/pi_ai/README.md) | 统一 LLM API，Provider 抽象模式。支持 OpenAI (Responses API) 和 DeepSeek (Completions API) | ✓ |
| `pi_agent` | [README](src/pi_agent/README.md) | 最小核心 Agent 循环。事件驱动、工具调用、循环钩子 | ✓ |
| `pi_coding_agent` | [README](src/pi_coding_agent/README.md) | CLI 编码代理。7 个编码工具、会话持久化、双层配置 | ✓ |

### 架构

```
pi_coding_agent (CLI + Tools + Sessions)
    └─ pi_agent (Agent Loop + Events + Hooks)
        └─ pi_ai (Models + Providers + Streams)
```

- **pi_ai** — 底层 LLM 调用：`Models` 注册表管理多个 Provider，`complete()` / `stream()` 统一非流式/流式调用，`EventStream` 生产者-消费者异步事件流
- **pi_agent** — 中间层 Agent 循环：纯函数引擎 `run_agent_loop()` + 有状态 `Agent` 包装类，事件驱动、工具调用、取消机制、循环钩子
- **pi_coding_agent** — 顶层 CLI：`pi -p "..."` 单次编码查询，7 个编码工具（read/write/edit/bash/grep/find/ls），JSONL 会话持久化，双层 settings.json 配置

---

## 快速开始

### 安装

```bash
git clone https://github.com/penguinpanda/pi-python.git
cd pi-python
uv sync
```

### 认证

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:DEEPSEEK_API_KEY="sk-..."

# Linux/macOS
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
```

### pi_ai — 直接调用 LLM

```python
import asyncio
from pi_ai import create_default_models, Context

async def main():
    models = create_default_models()
    model = models.get_model("deepseek", "deepseek-chat")

    async for event in await models.stream(model, Context(
        messages=[{"role": "user", "content": "Hello!"}],
    )):
        if event["type"] == "text_delta":
            print(event["delta"], end="", flush=True)

asyncio.run(main())
```

详见 [src/pi_ai/README.md](src/pi_ai/README.md)。

### pi_agent — Agent 循环

```python
import asyncio
from pi_agent import Agent, AgentOptions, set_default_stream_fn
from pi_ai import create_default_models

async def main():
    models = create_default_models()
    set_default_stream_fn(models.stream)

    agent = Agent(AgentOptions(
        model=models.get_model("deepseek", "deepseek-chat"),
    ))
    agent.subscribe(lambda e: print(f"[{e['type']}]"))
    await agent.prompt("What is 2+2?")
    await agent.wait_for_idle()

asyncio.run(main())
```

详见 [src/pi_agent/README.md](src/pi_agent/README.md)。

### pi_coding_agent — CLI 编码代理

```bash
# 单次 print 模式
uv run python -m pi_coding_agent -p "read README.md and summarize it"

# 指定模型
uv run python -m pi_coding_agent --model deepseek-v4-flash -p "explain this code"

# 不持久化会话
uv run python -m pi_coding_agent --no-session -p "what is 2+2?"
```

详见 [src/pi_coding_agent/README.md](src/pi_coding_agent/README.md)。

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

---

## 开发

```bash
# 安装依赖
uv sync

# 运行全部测试（pi_ai + pi_agent + pi_coding_agent）
uv run pytest

# 按包运行测试
uv run pytest src/pi_ai/tests/ -v
uv run pytest src/pi_agent/tests/ -v
uv run pytest src/pi_coding_agent/tests/ -v

# 运行集成测试（需要 API Key）
$env:OPENAI_API_KEY="sk-..."; uv run pytest src/pi_ai/tests/test_stream.py -v
```

## 许可

MIT
