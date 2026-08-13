# pi-python

[English](README.md) | [中文](README.zh.md)

Python implementation of the [pi agent framework](https://github.com/earendil-works/pi-mono).
Based on the original pi project (MIT License, Copyright (c) 2025 Mario Zechner); see [NOTICE](NOTICE).

A three-layer AI coding agent toolchain: LLM SDK → Agent loop → Coding agent CLI/TUI.

---

## Packages

```
pi-python/
├── src/pi_ai/              # Unified LLM API (provider abstraction)
├── src/pi_agent/           # Agent loop engine (pure-function core + stateful wrapper)
├── src/pi_coding_agent/    # Coding agent CLI (tools + sessions + config)
├── src/pi_tui/             # Built-in engine TUI (no Textual; themes/keybindings/selectors)
├── src/pi_protocol/        # protocol v2 wire protocol (pydantic schema + JSONL framing)
├── src/pi_storage/         # PostgreSQL session storage (asyncpg + migrations + search)
├── src/pi_server/          # Persistent pi service (stdio JSONL, attach/detach + snapshot push)
└── src/pi_evals/           # Full port of TS packages/evals (harness + runner + comparative evals)
```

| Package | Docs | Description | Packaged |
|---|------|------|:---:|
| `pi_ai` | [README](src/pi_ai/README.md) | Unified LLM API with provider abstraction. OpenAI (Responses), DeepSeek/Qwen (Completions), Ollama, Radius gateway (dynamic catalog), Faux test provider; OAuth browser/device-code login (Codex/OpenRouter/xAI/Radius) | ✓ |
| `pi_agent` | [README](src/pi_agent/README.md) | Minimal core Agent loop: event-driven, tool calling, loop hooks, harness/session v4 | ✓ |
| `pi_coding_agent` | [README](src/pi_coding_agent/README.md) | Coding agent CLI: tools, session persistence, two-level config, extensions/skills/trust/compaction, package management subcommands (install/remove/update/list/config), remote model catalog overlay (ETag 4h) | ✓ |
| `pi_tui` | [README](src/pi_tui/README.md) | Built-in engine TUI: themes, keybindings, selectors, clipboard images, mermaid terminal diagrams, suspend (Ctrl+Z) | ✓ |
| `pi_protocol` | [README](src/pi_protocol/README.md) | protocol v2: Command/Result/Snapshot/Progress/Error + JSONL framing | ✓ |
| `pi_storage` | [README](src/pi_storage/README.md) | PostgreSQL SessionStore/SessionSearch (`docker compose up -d pg`) | ✓ |
| `pi_server` | [README](src/pi_server/README.md) | Persistent service: `python -m pi_server` (stdio JSONL) | ✓ |
| `pi_evals` | [README](src/pi_evals/README.md) | Full port of TS `packages/evals`: pi-coding-agent harness, judge/harness table/artifacts/summary, `pi-evals` CLI runner | ✓ |

### Architecture

```
pi_coding_agent (CLI + Tools + Sessions)
    └─ pi_agent (Agent Loop + Events + Hooks)
        └─ pi_ai (Models + Providers + Streams)
```

- **pi_ai** — LLM calls: `Models` registry managing multiple providers, unified `complete()` / `stream()`, `EventStream` producer-consumer async events
- **pi_agent** — Agent loop: pure-function engine `run_agent_loop()` + stateful `Agent` wrapper; event-driven, tool calling, cancellation, loop hooks
- **pi_coding_agent** — top-level CLI: `pi-python -p "..."` one-shot queries, coding tools (read/write/edit/bash/grep/find/ls), JSONL session persistence, two-level settings.json, slash commands, project trust, system prompt builder (AGENTS.md/CLAUDE.md), turn timings / cache stats, package management subcommands, remote model catalog overlay
- **pi_tui / pi_protocol / pi_storage / pi_server / pi_evals** — TUI engine (with mermaid diagrams and Ctrl+Z suspend), protocol v2, PostgreSQL storage, persistent service, evals harness (see table above)

---

## Quick Start

### Installation

```bash
git clone https://github.com/penguinpanda/pi-python.git
cd pi-python
uv sync
```

### Local checks

One-command checks identical to GitHub Actions (ruff lint / ruff format / mypy / pytest with coverage):

```bash
python scripts/check.py
```

Optional: pre-commit

```bash
uv tool install pre-commit
pre-commit install
```

### Authentication

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

### pi_ai — direct LLM calls

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

See [src/pi_ai/README.md](src/pi_ai/README.md).

### pi_agent — Agent loop

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

See [src/pi_agent/README.md](src/pi_agent/README.md).

### pi_coding_agent — coding agent CLI

```bash
# one-shot print mode
uv run python -m pi_coding_agent -p "read README.md and summarize it"

# specify a model
uv run python -m pi_coding_agent --model deepseek-v4-flash -p "explain this code"

# don't persist the session
uv run python -m pi_coding_agent --no-session -p "what is 2+2?"

# TUI interactive mode (default when stdin is a TTY and no message is given)
uv run python -m pi_coding_agent --mode tui

# package management
uv run python -m pi_coding_agent install npm:pi-extension-example
uv run python -m pi_coding_agent list
```

See [src/pi_coding_agent/README.md](src/pi_coding_agent/README.md).

---

## Supported models

| Provider | Model ID | API | Thinking | Tool Calling | Image Input | max_tokens |
|----------|---------|-----|:--------:|:------------:|:-----------:|:---------:|
| OpenAI | `gpt-5-chat-latest` | Responses | ✗ | ✓ | ✓ | 16,384 |
| OpenAI | `gpt-5.6-luna` / `gpt-5.6-sol` / `gpt-5.6-terra` | Responses | ✓ | ✓ | ✓ | 128,000 |
| DeepSeek | `deepseek-v4-flash` | Responses | ✓ | ✓ | ✗ | 384,000 |
| DeepSeek | `deepseek-v4-pro` | Completions | ✓ | ✓ | ✗ | 384,000 |
| Qwen | `qwen-turbo` / `qwen-plus` / `qwen-max` | Completions | ✗ | ✓ | ✗ | 8,192 |
| Qwen | `qwen3-235b-a22b` | Completions | ✓ | ✓ | ✗ | 131,072 |
| Qwen | `qwen3-30b-a3b` | Completions | ✓ | ✓ | ✗ | 32,768 |
| Qwen | `qwen3-vl-flash` / `qwen-vl-max` | Completions | ✗ | ✓ | ✓ | 8,192 |
| Qwen | `qwen-vl-plus` | Completions | ✗ | ✓ | ✓ | 4,096 |
| Ollama | `qwen3:30b` / `gpt-oss:20b` / `deepseek-r1:14b` and 3 more static models | Completions | per model | ✓ | per model | local |

> More models come from `src/pi_ai/models/generated/providers/` (OpenRouter 273, Vercel AI Gateway 196, OpenAI Codex 7, ...); use `--list-models` to see them all.

---

## Development

```bash
# install dependencies
uv sync

# run all tests
uv run pytest

# static checks (ruff lint + format + mypy; all blocking in CI)
uv run ruff check .
uv run ruff format .
uv run mypy src/pi_ai src/pi_agent src/pi_coding_agent src/pi_tui src/pi_protocol src/pi_storage src/pi_server src/pi_evals

# run tests per package
uv run pytest src/pi_ai/tests/ -v
uv run pytest src/pi_agent/tests/ -v
uv run pytest src/pi_coding_agent/tests/ -v
uv run pytest src/pi_protocol/tests/ src/pi_server/tests/ src/pi_evals/ -v

# run evals (CLI model selection or PI_PROVIDER/PI_MODEL env vars; faux by default)
uv run pi-evals

# PostgreSQL storage tests (start the compose pg service first)
docker compose -f docker/compose.yaml up -d pg
$env:PI_PG_DSN="postgresql://pi:pi@127.0.0.1:5432/pi"
uv run pytest src/pi_storage/tests/ -v

# integration tests (require API keys)
$env:OPENAI_API_KEY="sk-..."; uv run pytest src/pi_ai/tests/test_stream.py -v
```

## License

MIT
