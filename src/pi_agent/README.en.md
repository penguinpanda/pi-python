# pi-agent — LLM Agent Loop & Runtime Facilities

[English](README.en.md) | [中文](README.md)

A Python port of [pi-mono/packages/agent](https://github.com/earendil-works/pi-mono).

**Pure-function engine + stateful wrapper**: the core loop is class-free and mutation-free; an outer `Agent` class provides the user-facing OOP API. On top of the loop: Session tree, skills/prompt templates, context compaction, branch summarization, AgentHarness, and Proxy stream functions.

## Architecture

```
Agent (stateful wrapper)
  │ mutex, abort signal, dual message queues (steer / follow_up), event subscription
  ▼
run_agent_loop()  ◄── pure-function engine
  ├─ agent_start / turn_start / message_start / message_end events
  ▼
_run_loop()  ◄── nested loops
  ├─ outer (follow-up): continue while follow-up queue has messages
  └─ inner (tool + steering): pending messages → LLM → tools → steering poll
```

## Quick Start

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

## Key facilities

| Module | Description |
|---|---|
| `run_agent_loop()` | pure-function loop engine |
| `Agent` | stateful wrapper (mutex, abort, steer/follow-up queues, events) |
| `tools/` | read / write / edit / bash (with 100ms-throttled live progress), grep, find, ls |
| `env.py` | `ExecutionEnv` (filesystem + subprocess), abort-aware process killing |
| `compaction.py` | context compaction policies (overflow recovery + threshold triggers) |
| `branch_summarization.py` | branch summaries on tree navigation |
| `skills.py` / `prompt_templates.py` | skill discovery, frontmatter parsing, gitignore matching |
| `session/v4/` | JSONL v4 sessions: lanes, facts, usage records, PostgreSQL backend |
| `harness` (`_harness.py`) | AgentHarness: run/compaction/navigation operations, telemetry hooks, shutdown |
| `proxy.py` | stream function proxy (provider remapping) |

## Events

`agent_start`, `turn_start`, `message_start`, `message_delta`, `message_end`, `tool_execution_start/update/end`, `compaction_start/end`, `agent_end`, `auto_retry_start/end`, `summarization_retry_scheduled`, and more.

## Tests

```bash
uv run pytest src/pi_agent/tests/ -v
```

## License

MIT
