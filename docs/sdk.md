# SDK（Python 移植）

pi-python 分几个包，对应 TS 的 `pi-ai` / `pi-agent-core` / `pi-coding-agent` / `pi-tui`：

| 包 | 职责 | 主要入口 |
| --- | --- | --- |
| `pi_ai` | 模型注册表、Provider、API 流、类型、认证 | `Models`、`Provider`、`Context`、`stream` |
| `pi_agent` | 最小 agent 循环、有状态 Agent、harness、会话树、压缩/分支摘要 | `Agent`、`AgentHarness`、`Session` |
| `pi_coding_agent` | CLI 编码代理：会话、资源、扩展、ModelRuntime、RPC、TUI | `AgentSession`、`ResourceLoader`、`run_print_mode` / `run_tui_mode` / `run_rpc_mode` |
| `pi_tui` | Textual 组件、选择器、主题、快捷键 | `PiTuiApp`（在 `pi_coding_agent.modes.interactive`） |
| `pi_protocol` / `pi_storage` / `pi_server` | RPC 协议 / 存储 / 服务端（移植中） | 见各自 `__init__.py` |

## 快速开始

```python
from pi_ai import create_default_models
from pi_agent import Agent, set_default_stream_fn

models = create_default_models()
set_default_stream_fn(models.stream)

agent = Agent()
await agent.prompt("Hello!")
```

`pi_coding_agent` 的完整入口：

```python
from pi_coding_agent import AgentSession, SessionManager, create_all_tools, run_print_mode

sm = SessionManager.in_memory(cwd=".")
model = ...  # 来自 ModelRuntime / Models
agent = Agent(model=model, tools=create_all_tools(cwd="."))
session = AgentSession(agent, sm, cwd=".", model=model)
await run_print_mode(session, "read README.md")
```

## 核心概念

### Models / Provider / Context / stream

- `Models` 是统一入口：`add_provider` / `get_model` / `get_models` / `stream` / `complete` / `set_api_key` / `refresh`
- `Provider` 保存配置、模型列表与认证，`stream()` 解析凭证后按 `model.api` 调 API 注册表
- `Context(system_prompt=..., messages=..., tools=...)` 是一次请求的输入；流函数 `stream_fn(model, context, options)` 返回 `AssistantMessageEventStream`
- options：`api_key` / `base_url` / `headers` / `env` / `max_tokens` / `reasoning` / `signal` / `cache_retention` / `session_id` / `max_retries` 等

### Agent / AgentState

`pi_agent.Agent(model=..., system_prompt=..., tools=..., session_id=..., cache_retention=..., retry_policy=..., thinking_level=..., get_api_key=...)`。`agent.prompt(text)` 运行完整 loop 并返回最后一条 assistant 消息；`agent.continue_()` 从当前上下文继续（最后一条不能是 assistant）；`agent.state.messages` / `agent.state.model` / `agent.state.thinking_level`。

事件流（`pi_agent._agent_loop`）：`agent_start/end`、`turn_start/end`、`message_start/update/end`、`tool_execution_start/end`、`auto_retry_start/end`。

### AgentSession（应用层）

`pi_coding_agent.AgentSession` 把 Agent + SessionManager + ModelRuntime + 扩展 runner 组合起来：

- `prompt(text)` / `continue_()` / `steer()` / `follow_up()` / `next_turn()`
- `compact()` 手动压缩；`_check_compaction()` 自动压缩（溢出/阈值，溢出恢复会移除错误消息并 `continue_()` 重试）
- `navigate_to(entry_id, summarize=True)` 分支导航（可选 branch summary）
- `set_model(model)` / `set_thinking_level(level)` / `set_session_name(name)`
- 事件：`compaction_start/end`、`navigated`、`skill_invocation` 等（`_emit` 通知监听器）

### 会话与压缩

- `SessionManager`：JSONL 会话树（`SessionTreeNode`），`build_context()` 重建上下文
- 压缩：`prepare_compaction` → `compact`（摘要 LLM，`cache_retention=none` + 随机 `session_id`）→ 追加 `compaction` 条目
- 分支：`collect_entries_for_branch_summary` / `generate_branch_summary`

### 资源与信任

- `ResourceLoader`：扩展 / 技能 / 提示模板 / 主题 / 上下文文件聚合
- `TrustManager` / `resolve_project_trusted`：项目 `.pi/` 资源信任门控
- `load_project_context_files`：全局 agent 目录 + cwd 祖先链上的 `AGENTS.md` / `CLAUDE.md`
- `build_system_prompt`：默认系统提示（工具、指南、pi 文档段、项目上下文、技能）

## 运行模式

- **Print**：`run_print_mode(session, prompt)`，无 UI，扩展 UI 走 `NoopUIContext`
- **TUI**：`run_tui_mode(...)`，Textual 应用 `PiTuiApp`
- **RPC**：`run_rpc_mode` + `RpcMessageHandler` / `RpcClient`（stdin/stdout JSONL 协议，`pi_protocol`）
- **CLI**：`python -m pi_coding_agent`（`main`），支持 `--model` / `--provider` / `--models` / `--list-models` / `--system-prompt` / `--append-system-prompt` / `--session` / `--no-session` / `--tools` / `--exclude-tools` / `--no-tools` / `--extension`（`-e`）/ `--skill` / `--prompt-template`（均可重复）/ `--no-skills` / `--no-prompt-templates` / `--no-context-files`（`-nc`）/ `--preset`

## 完整示例

```python
import asyncio
from pi_ai import create_default_models
from pi_agent import Agent, set_default_stream_fn
from pi_coding_agent import (
    AgentSession,
    SessionManager,
    create_coding_tools,
    run_print_mode,
)


async def main() -> None:
    models = create_default_models()
    set_default_stream_fn(models.stream)
    model = models.get_models()[0]
    sm = SessionManager.in_memory(cwd=".")
    agent = Agent(model=model, tools=create_coding_tools(cwd="."))
    session = AgentSession(agent, sm, cwd=".", model=model)
    await run_print_mode(session, "List the files and summarize the project")


asyncio.run(main())
```

## 未移植（TS SDK 独有）

- `createAgentSessionRuntime` / `AgentSessionRuntime` 的完整选项集
- 包安装（`pi install` 等）与 OAuth 之外的完整 CLI 子命令
