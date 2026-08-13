# pi-python
[English](README.md) | [中文](README.zh.md)

Python implementation of the [pi agent framework](https://github.com/earendil-works/pi-mono).
Based on the original pi project (MIT License, Copyright (c) 2025 Mario Zechner); see [NOTICE](NOTICE).

pi-python 是基于 [pi-mono](https://github.com/earendil-works/pi-mono) 的 Python 复刻，三层架构的 AI 编码代理工具链。

---

## 包结构

```
pi-python/
├── src/pi_ai/              # LLM API 抽象层（Provider 模式）
├── src/pi_agent/           # Agent 循环引擎（纯函数核心 + 有状态包装）
├── src/pi_coding_agent/    # CLI 编码代理（工具 + 会话 + 配置）
├── src/pi_tui/             # 内置引擎 TUI（无 Textual；主题/快捷键/选择器）
├── src/pi_protocol/        # protocol v2 线协议（pydantic schema + JSONL framing）
├── src/pi_storage/         # PostgreSQL 会话存储（asyncpg + 迁移 + 搜索）
├── src/pi_server/          # 常驻 pi 服务（stdio JSONL，attach/detach + 快照推送）
└── src/pi_evals/           # TS packages/evals 完整移植（harness + runner + 对比评测）
```

| 包 | 文档 | 说明 | 打包 |
|---|------|------|:---:|
| `pi_ai` | [README](src/pi_ai/README.md) | 统一 LLM API，Provider 抽象模式。支持 OpenAI (Responses API)、DeepSeek/Qwen (Completions API)、Ollama 本地、Radius 网关（动态目录）与 Faux 测试 Provider；OAuth 浏览器/设备码登录（Codex/OpenRouter/xAI/Radius） | ✓ |
| `pi_agent` | [README](src/pi_agent/README.md) | 最小核心 Agent 循环。事件驱动、工具调用、循环钩子、harness/session v4 | ✓ |
| `pi_coding_agent` | [README](src/pi_coding_agent/README.md) | CLI 编码代理。编码工具、DAG 会话持久化、双层配置、扩展/技能/信任/压缩、包管理子命令（install/remove/update/list/config）、远程模型目录 overlay（ETag 4h） | ✓ |
| `pi_tui` | [README](src/pi_tui/README.md) | 内置引擎 TUI：主题、快捷键、选择器、剪贴板图片、mermaid 终端图渲染、suspend（Ctrl+Z） | ✓ |
| `pi_protocol` | [README](src/pi_protocol/README.md) | protocol v2：Command/Result/Snapshot/Progress/Error + JSONL framing | ✓ |
| `pi_storage` | [README](src/pi_storage/README.md) | PostgreSQL SessionStore/SessionSearch（`docker compose up -d pg`） | ✓ |
| `pi_server` | [README](src/pi_server/README.md) | 常驻服务：`python -m pi_server`（stdio JSONL） | ✓ |
| `pi_evals` | [README](src/pi_evals/README.md) | TS `packages/evals` 完整移植：pi-coding-agent harness、judge/harness table/artifacts/summary、`pi-evals` CLI runner | ✓ |

### 架构

```
pi_coding_agent (CLI + Tools + Sessions)
    └─ pi_agent (Agent Loop + Events + Hooks)
        └─ pi_ai (Models + Providers + Streams)
```

- **pi_ai** — 底层 LLM 调用：`Models` 注册表管理多个 Provider，`complete()` / `stream()` 统一非流式/流式调用，`EventStream` 生产者-消费者异步事件流
- **pi_agent** — 中间层 Agent 循环：纯函数引擎 `run_agent_loop()` + 有状态 `Agent` 包装类，事件驱动、工具调用、取消机制、循环钩子
- **pi_coding_agent** — 顶层 CLI：`pi-python -p "..."` 单次编码查询，编码工具（read/write/edit/bash/grep/find/ls），JSONL 会话持久化，双层 settings.json 配置，Slash 命令，项目信任，系统提示构建器（AGENTS.md/CLAUDE.md），turn timings / cache stats，包管理子命令（`pi install/remove/update/list/config`），远程模型目录 overlay
- **pi_tui / pi_protocol / pi_storage / pi_server / pi_evals** — TUI 引擎层（含 mermaid 终端图渲染与 Ctrl+Z suspend）、protocol v2、PostgreSQL 存储、常驻服务与评测 harness（见下表）

---

## 快速开始

### 安装

```bash
git clone https://github.com/penguinpanda/pi-python.git
cd pi-python
uv sync
```

### 本地检查

一键运行与 GitHub Actions 相同的检查（ruff lint / ruff format / mypy / pytest 带覆盖率）：

```bash
python scripts/check.py
```

可选：提交前自动检查

```bash
uv tool install pre-commit
pre-commit install
```

### 认证

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:DEEPSEEK_API_KEY="sk-..."
$env:DASHSCOPE_API_KEY="sk-..."
$env:QWEN_TOKEN_PLAN_API_KEY="sk-sp-..."
$env:QWEN_TOKEN_PLAN_CN_API_KEY="sk-sp-..."

# Linux/macOS
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export DASHSCOPE_API_KEY="sk-..."
export QWEN_TOKEN_PLAN_API_KEY="sk-sp-..."
export QWEN_TOKEN_PLAN_CN_API_KEY="sk-sp-..."
```

### pi_ai — 直接调用 LLM

```python
import asyncio
from pi_ai import create_default_models, Context


async def main():
    models = create_default_models()
    model = models.get_model("deepseek", "deepseek-v4-flash")

    async for event in await models.stream(
        model,
        Context(
            messages=[{"role": "user", "content": "Hello!"}],
        ),
    ):
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

    agent = Agent(
        AgentOptions(
            model=models.get_model("deepseek", "deepseek-v4-flash"),
        )
    )
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

# TUI 交互模式（无参数且 stdin 为 TTY 时默认进入）
uv run python -m pi_coding_agent --mode tui

# 包管理
uv run python -m pi_coding_agent install npm:pi-extension-example
uv run python -m pi_coding_agent list
```

详见 [src/pi_coding_agent/README.md](src/pi_coding_agent/README.md)。

---

## 支持的模型

| Provider | 模型 ID | API 类型 | Thinking | Tool Calling | 图片输入 | max_tokens |
|----------|---------|----------|:--------:|:------------:|:--------:|:---------:|
| OpenAI | `gpt-5-chat-latest` | Responses | ✗ | ✓ | ✓ | 16,384 |
| OpenAI | `gpt-5.6-luna` / `gpt-5.6-sol` / `gpt-5.6-terra` | Responses | ✓ | ✓ | ✓ | 128,000 |
| DeepSeek | `deepseek-v4-flash` | Responses | ✓ | ✓ | ✗ | 384,000 |
| DeepSeek | `deepseek-v4-pro` | Completions | ✓ | ✓ | ✗ | 384,000 |
| Qwen | `qwen-turbo` / `qwen-plus` / `qwen-max` | Completions | ✗ | ✓ | ✗ | 8,192 |
| Qwen | `qwen3-235b-a22b` | Completions | ✓ | ✓ | ✗ | 131,072 |
| Qwen | `qwen3-30b-a3b` | Completions | ✓ | ✓ | ✗ | 32,768 |
| Qwen | `qwen3-vl-flash` / `qwen-vl-max` | Completions | ✗ | ✓ | ✓ | 8,192 |
| Qwen | `qwen-vl-plus` | Completions | ✗ | ✓ | ✓ | 4,096 |
| Ollama | `qwen3:30b` / `gpt-oss:20b` / `deepseek-r1:14b` 等 6 个静态模型 | Completions | 按模型 | ✓ | 按模型 | 本地 |

> 更多模型来自 `src/pi_ai/models/generated/providers/`（OpenRouter 273、Vercel AI Gateway 196、OpenAI Codex 7 等），`--list-models` 可查看全部。

---

## 开发

```bash
# 安装依赖
uv sync

# 运行全部测试（pi_ai + pi_agent + pi_coding_agent + pi_tui + 新包）
uv run pytest

# 静态检查（ruff lint + format + mypy；CI 中全部阻塞）
uv run ruff check .
uv run ruff format .
uv run mypy src/pi_ai src/pi_agent src/pi_coding_agent src/pi_tui src/pi_protocol src/pi_storage src/pi_server src/pi_evals

# 按包运行测试
uv run pytest src/pi_ai/tests/ -v
uv run pytest src/pi_agent/tests/ -v
uv run pytest src/pi_coding_agent/tests/ -v
uv run pytest src/pi_protocol/tests/ src/pi_server/tests/ src/pi_evals/ -v

# 运行评测（CLI 模型选择或 PI_PROVIDER/PI_MODEL 环境变量；默认 faux）
uv run pi-evals

# PostgreSQL 存储测试（先启动 compose 的 pg 服务）
docker compose -f docker/compose.yaml up -d pg
$env:PI_PG_DSN="postgresql://pi:pi@127.0.0.1:5432/pi"
uv run pytest src/pi_storage/tests/ -v

# 运行集成测试（需要 API Key）
$env:OPENAI_API_KEY="sk-..."; uv run pytest src/pi_ai/tests/test_stream.py -v
```

## 许可

MIT
