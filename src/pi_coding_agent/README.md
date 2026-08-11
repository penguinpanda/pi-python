# pi-coding-agent — CLI 编码代理

基于 [pi-mono/packages/coding-agent](https://github.com/earendil-works/pi-mono) 的 Python 全功能复刻。

在 `pi_agent` + `pi_ai` 之上构建的编码代理，提供 7 个编码工具、DAG 会话持久化、双层配置、
项目信任、系统提示构建器、扩展/技能/模板、自动上下文压缩、turn 级重试、turn timings / cache stats，
以及 print / RPC / TUI 三种运行模式。

---

## 架构概览

```
CLI (_cli.py)
  ├─ SettingsManager.create(cwd) ── 双层 settings.json + 项目信任感知
  ├─ resolve_project_trusted(cwd, trust_manager, settings) ── trust.json + 资源门控
  ├─ pi_ai.create_default_models() → Models registry
  ├─ pi_agent.set_default_stream_fn(models.stream)
  ├─ build_system_prompt() ── 工具说明 + 指南 + AGENTS.md/CLAUDE.md 上下文 + 技能
  ├─ pi_agent.Agent(AgentOptions(system_prompt, model, session_id))
  │   └─ Agent.state.tools ← create_all_tools(cwd)
  └─ pi_coding_agent.AgentSession(agent, session_manager, cwd, model)
      ├─ subscribe(agent events) → persistence + forwarding
      ├─ 自动压缩（溢出恢复 / 阈值）+ turn 级重试
      └─ 模式分发：print_mode / rpc_mode / tui_mode
```

三层依赖关系：`pi_coding_agent` → `pi_agent` → `pi_ai`；TUI 组件层在 `pi_tui`。

---

## 快速开始

### 安装

```bash
uv sync
```

### 认证

API Key 方式（环境变量）：

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
$env:DEEPSEEK_API_KEY="sk-..."
$env:DASHSCOPE_API_KEY="sk-..."

# Linux/macOS
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export DASHSCOPE_API_KEY="sk-..."
```

OAuth 方式（凭证保存到 `~/.pi/agent/auth.json`）：

```bash
pi login                # 交互选择 provider（openai-codex / github-copilot / openrouter）
pi login openai-codex   # 指定 provider
pi logout openai-codex  # 注销
pi list                 # 查看登录状态
```

### CLI 用法

```bash
# 单次 print 模式
uv run python -m pi_coding_agent -p "read README.md and summarize it"

# 指定模型 / provider
uv run python -m pi_coding_agent --model deepseek-v4-flash -p "explain this code"
uv run python -m pi_coding_agent --provider openai --model gpt-5-chat-latest -p "what is Python?"

# 本地 Ollama（需 Ollama 已运行且模型已 pull；启动时自动 /api/tags 动态发现）
uv run python -m pi_coding_agent --provider ollama --model qwen3:30b -p "what is Python?"

# Faux Provider（离线验证管道）
uv run python -m pi_coding_agent --provider faux --model faux-1 -p "hi"

# 列出所有可用模型（可配合 --provider 过滤）
uv run python -m pi_coding_agent --list-models
uv run python -m pi_coding_agent --list-models --provider ollama

# JSON Lines 输出（print 模式）
uv run python -m pi_coding_agent --json -p "hi"

# RPC 模式（stdin/stdout JSONL，32 命令）
uv run python -m pi_coding_agent --mode rpc

# TUI 模式（内置引擎交互界面）
uv run python -m pi_coding_agent --mode tui

# 自定义系统提示 / 追加
uv run python -m pi_coding_agent --system-prompt "You are a Python expert." -p "..."
uv run python -m pi_coding_agent --append-system-prompt "只回答中文" -p "..."

# 会话控制
uv run python -m pi_coding_agent --no-session -p "..."
uv run python -m pi_coding_agent --session ~/.pi/agent/sessions/abc123.jsonl -p "continue"
uv run python -m pi_coding_agent -c -p "continue"

# 工具控制
uv run python -m pi_coding_agent --tools read,bash -p "..."
uv run python -m pi_coding_agent --exclude-tools bash -p "..."
uv run python -m pi_coding_agent --no-tools -p "what is 2+2?"

# 通过 stdin 输入
echo "read README.md" | uv run python -m pi_coding_agent -p
```

### TUI Slash 命令（30 条）

`/model` `/thinking` `/oauth` `/extensions` `/name` `/compact` `/new` `/quit` `/help`
`/hotkeys` `/session` `/reload` `/trust` `/changelog` `/copy` `/export` `/tree` `/fork` `/input`
`/clone` `/settings` `/scoped-models` `/login` `/logout` `/share` `/import` `/resume`
`/debug` `/arminsayshi` `/dementedelves`

其中 `/settings`、`/trust`、`/thinking`、`/oauth`、`/scoped-models`、`/extensions` 在
TUI 中打开对应选择器；`/reload` 会重建系统提示并纳入 context files（AGENTS.md/CLAUDE.md）。
启动时聊天区会显示已加载资源汇总（`[Context]` / `[Skills]` / `[Prompts]` /
`[Extensions]` / `[Themes]`，空段省略；`--no-context-files` 只禁用 Context 段）。

---

## CLI 参考

```
pi [-p] [--mode print|rpc|tui] [--model MODEL] [--provider PROVIDER] [--models LIST]
   [--list-models] [--system-prompt PROMPT] [--append-system-prompt PROMPT]
   [--session PATH] [-c|--continue] [--no-session] [--setup] [--json]
   [--tools WHITELIST] [--exclude-tools BLACKLIST] [--no-tools]
   [--version]
   [message]

pi login [provider] | pi logout <provider> | pi list
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `-p, --print` | flag | 单次 print 模式（有 message 时默认） |
| `--mode` | print/rpc/tui | 运行模式：print（默认）/ rpc（JSONL）/ tui（内置引擎） |
| `--json` | flag | print 模式输出 JSON Lines 事件 |
| `--model` | str | 模型 ID（如 `deepseek-v4-flash`、`gpt-5-chat-latest`） |
| `--provider` | str | Provider ID（如 `deepseek`、`openai`、`qwen`、`ollama`、`faux`） |
| `--models` | str | 逗号分隔的 Ctrl+P 循环模型范围 |
| `--list-models` | flag | 列出所有可用模型并退出（可配合 `--provider`） |
| `--system-prompt` / `--append-system-prompt` | str | 覆盖 / 追加系统提示（settings 的 `systemPrompt`/`appendSystemPrompt` 也生效） |
| `--session` / `-c` / `--no-session` | str/flag | 打开指定会话 / 继续最近会话 / 内存会话 |
| `--setup` | flag | 首次启动向导 |
| `--tools` / `--exclude-tools` / `--no-tools` | str/flag | 工具白名单 / 黑名单 / 全禁用 |
| `--no-context-files` / `-nc` | flag | 禁用 AGENTS.md / CLAUDE.md 发现与加载 |
| `--version` | flag | 打印版本号 |
| `login` / `logout` / `list` | 子命令 | OAuth 登录 / 注销 / 查看登录状态 |
| `message` | positional | 用户消息（可选，可通过 stdin pipe） |

**模型解析优先级**：CLI 参数 > `settings.json`（`defaultProvider`/`defaultModel`）> 第一个可用模型

**启动流程**：

1. 解析参数；`login` / `logout` / `list` 子命令在 argparse 之前拦截
2. `SettingsManager.create(cwd)` — 双层 settings.json（项目信任前先加载全局）
3. `resolve_project_trusted()` — 未信任项目不加载 `.pi` 资源并提示
4. `create_default_models()` + `set_agent_stream_fn()` — 模型注册表 + 全局流函数
5. 系统提示构建：工具说明 + 指南 + AGENTS.md/CLAUDE.md 上下文 + 技能清单
6. 会话管理（新建/继续/打开/内存）→ Agent + AgentSession → 模式分发

---

## 程序化用法

```python
import asyncio
from pi_agent import Agent, AgentOptions, set_default_stream_fn
from pi_ai import create_default_models
from pi_coding_agent import AgentSession, run_print_mode
from pi_coding_agent._session_manager_v4 import create_session_manager


async def main():
    models = create_default_models()
    set_default_stream_fn(models.stream)

    model = models.get_model("deepseek", "deepseek-v4-flash")
    agent = Agent(
        AgentOptions(
            system_prompt="You are a helpful coding assistant.",
            model=model,
        )
    )

    session_manager = await create_session_manager(cwd=".")
    session = AgentSession(
        agent=agent,
        session_manager=session_manager,
        cwd=".",
        model=model,
    )
    session.subscribe(lambda e: print(f"[{e['type']}]"))

    exit_code = await run_print_mode(session, "read README.md and summarize")
    await session.dispose()
    print(f"Exit: {exit_code}")


asyncio.run(main())
```

### 会话工厂

```python
from pi_coding_agent._session_manager_v4 import (
    create_session_manager,
    in_memory_session_manager,
    open_session_manager,
)

manager = await create_session_manager(cwd=".")       # 新建持久化会话（v4 默认）
manager = await in_memory_session_manager(cwd=".")    # 内存会话
manager = await open_session_manager(
    "~/.pi/agent/sessions/.../session.jsonl",
    cwd_override=".",
)
```

---

## 工具参考

7 个编码工具 + 3 种组合器：

| # | 工具 | LLM 名称 | 关键参数 | 特性 |
|---|------|----------|----------|------|
| 1 | read | `read` | `path`(必填), `offset`, `limit` | 行号标头，路径遍历防护，图片自动归一化（BMP/JPEG/GIF/WebP → PNG） |
| 2 | write | `write` | `path`(必填), `content`(必填) | 自动创建父目录 |
| 3 | edit | `edit` | `path`(必填), `edits[].oldText/newText` | 精确文本替换（兼容 legacy oldText/newText），返回 unified diff/patch |
| 4 | bash | `bash` | `command`(必填), `timeout` | 平台感知 shell，输出截断；禁止全盘搜索约束 |
| 5 | grep | `grep` | `pattern`(必填), `path`, `include`, `max_results` | `file:lineno:line` 输出，忽略 `.git/`、`node_modules/` 等 |
| 6 | find | `find` | `pattern`(必填), `path`, `max_results` | 支持 `**` 递归，仅文件 |
| 7 | ls | `ls` | `path` | 目录优先排序，显示大小 |

组合器：`create_all_tools(cwd)`（7 个）、`create_coding_tools(cwd)`（read/bash/edit/write）、
`create_readonly_tools(cwd)`（read/grep/find/ls）。工具在系统提示
“Available tools”段的单行说明优先级：`prompt_snippet` → description 第一行 → 工具名。

---

## 会话管理

会话默认使用 JSONL v4（`V4SessionManager`，见 `_session_manager_v4.py`）：
首行 header（`kind: header` / `version: 4` / `id` / `createdAt` / `cwd`），后续为带
全局 seq 的 mutation 行（message / compaction / fact / operation records 等）。
打开 v3 文件时惰性转换（`.bak` 备份），`PI_SESSION_FORMAT=v3` 可回退旧实现。
文件布局详见 [agent-directory.md](../../docs/agent-directory.md)。

统一入口 `create_session_manager` / `open_session_manager` /
`in_memory_session_manager` / `fork_session_manager`
（`pi_coding_agent._session_manager_v4`）返回 `SessionManagerLike`，支持：
`append_message` / `append_compaction` / `append_model_change` /
`append_thinking_level_change` / `move_to`（分支导航）/ `fork` / `get_tree` /
`get_branch` / `get_entries` / `get_leaf_id` / `list_sessions` / `build_context`
（沿 parentId 链重建，遇 compaction 停止回溯），以及 v4 records：
`start_operation` / `finish_operation` / `record_usage` / `find_records` /
`open_operations` / `recovery_state` / `edit_session_message`。

`AgentSession` 额外提供：`prompt` / `steer` / `follow_up` / `abort` / `compact` /
`navigate_to` / `set_model` / `cycle_model` / `set_thinking_level` / `cycle_thinking_level` /
`get_session_stats`（含 `turnTimings` + `cacheStats`）/ `rebuild_system_prompt`（/reload 用）。

构造参数（关键字）：

```python
AgentSession(
    agent, session_manager, cwd, model,
    *,
    tools_override=None,          # 默认 create_all_tools(cwd)
    turn_retry_policy=None,       # 默认启用 max_retries=3
    compaction_settings=None,     # 默认启用
    model_runtime=None,           # setModel/cycleModel/缓存统计
    scoped_models=None,           # Ctrl+P 循环范围
    skill_loader=None,            # /skill:name 展开
    template_loader=None,         # /templateName 展开
    extension_runner=None,        # 扩展事件/命令
    system_prompt_builder=None,   # /reload 重建系统提示
)
```

---

## 配置

`SettingsManager` 实现双层 settings.json（项目覆盖全局，深度合并）：

```
~/.pi/agent/settings.json     ← 全局配置
<cwd>/.pi/settings.json       ← 项目配置（未信任项目不加载/不写入）
```

常用设置项：

```json
{
    "defaultProvider": "deepseek",
    "defaultModel": "deepseek-v4-flash",
    "defaultProjectTrust": "ask",
    "systemPrompt": "You are a Python expert.",
    "appendSystemPrompt": ["只回答中文"],
    "keybindings": {"app.model.select": "ctrl+m"},
    "compaction": {"enabled": true, "reserveTokens": 16384, "keepRecentTokens": 20000},
    "restrictUntrustedTools": false,
    "retry": {"enabled": true, "maxRetries": 3, "baseDelayMs": 2000},
    "httpIdleTimeoutMs": 300000,
    "enableSkillCommands": true
}
```

路径常量：`get_agent_dir()` → `~/.pi/agent/`、`get_sessions_dir()` →
`~/.pi/agent/sessions/`、`get_settings_path()` → `~/.pi/agent/settings.json`、
`get_project_settings_path(cwd)` → `<cwd>/.pi/settings.json`；`themes/` /
`tools/` / `bin/` / `pi-debug.log` 仅定义路径约定（占位），见
[agent-directory.md](../../docs/agent-directory.md)。

---

## 运行模式

### Print 模式（默认）

`run_print_mode(session, message)` / `run_print_mode_json(session, message)`：
prompt → 等 agent 完成 → 输出最后一条 assistant 纯文本；`stop_reason` 为
`error`/`aborted` 返回 1。

### RPC 模式（`--mode rpc`）

stdin/stdout JSONL 无头协议，32 个命令（`prompt`/`abort`/`steer`/`follow_up`/`set_model`/
`cycle_model`/`compact`/`get_tree`/`fork`/`switch_session`/`get_state`/`get_messages`/
`get_entries`/`get_session_stats` 等），`pi_coding_agent.rpc.rpc_client` 提供客户端封装。

### TUI 模式（`--mode tui`）

内置引擎交互界面：主题（dark/light + 自定义 JSON）、快捷键（settings 覆盖）、
30 个 Slash 命令、模型/会话/设置/信任/思考/OAuth/作用域模型/扩展选择器、
剪贴板图片、外部编辑器（Ctrl+G）、树导航与 fork。

---

## 开发

```bash
# 运行 pi_coding_agent 测试
uv run pytest src/pi_coding_agent/tests/ -v

# 运行全部测试
uv run pytest
```

## 许可

MIT
