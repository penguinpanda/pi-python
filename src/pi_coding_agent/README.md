# pi-coding-agent — CLI 编码代理

基于 [pi-mono/packages/coding-agent](https://github.com/earendil-works/pi-mono) 的 Python 最小核心复刻。

在 `pi_agent` + `pi_ai` 之上构建的编码代理 CLI，提供文件读写、搜索、Shell 执行等编码工具。

---

## 架构概览

```
CLI (_cli.py)
  ├─ pi_ai.create_default_models() → Models registry
  ├─ pi_agent.set_default_stream_fn(models.stream)
  ├─ pi_agent.Agent(AgentOptions(system_prompt, model))
  │   └─ Agent.state.tools ← create_all_tools(cwd)
  └─ pi_coding_agent.AgentSession(agent, session_manager, cwd, model)
      ├─ subscribe(agent events) → persistence + forwarding
      └─ print_mode.run_print_mode(session, message)
          └─ subscribe → agent_end → extract final assistant text → stdout
```

三层依赖关系：`pi_coding_agent` → `pi_agent` → `pi_ai`

---

## 快速开始

### 安装

```bash
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

### CLI 用法

```bash
# 单次 print 模式
uv run python -m pi_coding_agent -p "read README.md and summarize it"

# 指定模型
uv run python -m pi_coding_agent --model deepseek-v4-flash -p "explain this code"

# 指定 provider
uv run python -m pi_coding_agent --provider openai --model gpt-4o -p "what is Python?"

# 本地 Ollama（需 Ollama 已运行且模型已 pull）
uv run python -m pi_coding_agent --provider ollama --model qwen3:30b -p "what is Python?"

# Faux Provider（离线验证管道，无脚本化响应时返回错误消息）
uv run python -m pi_coding_agent --provider faux --model faux-1 -p "hi"

# 列出所有可用模型（可配合 --provider 过滤）
uv run python -m pi_coding_agent --list-models
uv run python -m pi_coding_agent --list-models --provider ollama

# 自定义系统提示
uv run python -m pi_coding_agent --system-prompt "You are a Python expert." -p "..."

# 不持久化会话（纯内存模式）
uv run python -m pi_coding_agent --no-session -p "..."

# 从已有会话继续
uv run python -m pi_coding_agent --session ~/.pi/agent/sessions/abc123.jsonl -p "continue"

# 禁用所有工具
uv run python -m pi_coding_agent --no-tools -p "what is 2+2?"

# 通过 stdin 输入
echo "read README.md" | uv run python -m pi_coding_agent -p

# 查看版本
uv run python -m pi_coding_agent --version
```

---

## CLI 参考

```
pi [-p] [--model MODEL] [--provider PROVIDER] [--list-models]
   [--system-prompt PROMPT] [--append-system-prompt PROMPT]
   [--session PATH] [--no-session]
   [--tools WHITELIST] [--exclude-tools BLACKLIST] [--no-tools]
   [--version]
   [message]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `-p, --print` | flag | 单次 print 模式（非交互） |
| `--model` | str | 模型 ID（如 `deepseek-chat`、`gpt-4o`） |
| `--provider` | str | Provider ID（如 `deepseek`、`openai`、`ollama`、`faux`） |
| `--list-models` | flag | 列出所有可用模型并退出（可配合 `--provider` 过滤） |
| `--system-prompt` | str | 覆盖系统提示 |
| `--append-system-prompt` | str | 追加到系统提示 |
| `--session` | str | 已有会话 JSONL 文件路径 |
| `--no-session` | flag | 不持久化会话到磁盘 |
| `--tools` | str | 工具白名单（未实现） |
| `--exclude-tools` | str | 工具黑名单（未实现） |
| `--no-tools` | flag | 禁用所有工具 |
| `--version` | flag | 打印版本号 `pi 0.1.0 (minimal core)` |
| `message` | positional | 用户消息（可选，可通过 stdin pipe） |

**模型解析优先级**：CLI 参数 > `settings.json` > 第一个可用模型

**工作流程**：

1. 解析参数 → 确定 `cwd`
2. `load_settings(cwd)` — 双层合并配置
3. `create_default_models()` — 创建模型注册表
4. `set_agent_stream_fn(models.stream)` — 注册全局流函数
5. 模型解析 → 会话管理（新建/打开/内存） → 创建 Agent + AgentSession
6. `run_print_mode(session, message)` → 输出最终 assistant 文本 → 返回退出码

---

## 程序化用法

```python
import asyncio
from pi_agent import Agent, AgentOptions, set_default_stream_fn
from pi_ai import create_default_models
from pi_coding_agent import AgentSession, SessionManager, run_print_mode


async def main():
    # 初始化
    models = create_default_models()
    set_default_stream_fn(models.stream)

    model = models.get_model("deepseek", "deepseek-chat")
    agent = Agent(AgentOptions(
        system_prompt="You are a helpful coding assistant.",
        model=model,
    ))

    # 新建持久化会话
    session_manager = SessionManager.create(cwd=".")

    # 创建 AgentSession（自动注入所有编码工具）
    session = AgentSession(
        agent=agent,
        session_manager=session_manager,
        cwd=".",
        model=model,
    )

    # 订阅事件（可选）
    session.subscribe(lambda e: print(f"[{e['type']}]"))

    # 运行 print 模式
    exit_code = await run_print_mode(session, "read README.md and summarize")
    print(f"Exit: {exit_code}")

    # 清理
    await session.dispose()


asyncio.run(main())
```

### 内存模式（不持久化）

```python
session_manager = SessionManager.in_memory(cwd=".")
```

### 打开已有会话

```python
session_manager = SessionManager.open(
    "~/.pi/agent/sessions/abc123.jsonl",
    cwd_override=".",
)
```

### 自定义工具集

```python
from pi_coding_agent import create_readonly_tools

# 仅使用只读工具
session = AgentSession(
    agent=agent,
    session_manager=session_manager,
    cwd=".",
    model=model,
    tools_override=create_readonly_tools("."),
)
```

---

## 工具参考

7 个编码工具 + 3 种组合器。所有工具签名一致：

```python
async def execute(tool_call_id: str, params: dict, **_kwargs) -> AgentToolResult
```

### 工具清单

| # | 工具 | LLM 名称 | 关键参数 | 特性 |
|---|------|----------|----------|------|
| 1 | **read** | `read` | `path`(必填), `offset`(1-based), `limit`(默认2000) | 行号标头 `[Lines X-Y of Z]`，路径遍历防护 |
| 2 | **write** | `write` | `path`(必填), `content`(必填) | 自动创建父目录，返回字节数 |
| 3 | **edit** | `edit` | `path`(必填), `diff`(unified diff, 必填) | 从后往前应用 hunks，手动 difflib 解析 |
| 4 | **bash** | `bash` | `command`(必填), `timeout`(默认120s) | 平台感知 shell（win: `cmd /c`，其他: `bash -c`），输出截断 50KB |
| 5 | **grep** | `grep` | `pattern`(必填, regex), `path`(默认`.`), `include`(glob), `max_results`(默认100) | 输出格式 `file:lineno:line`，自动忽略 `.git/`、`node_modules/` 等 |
| 6 | **find** | `find` | `pattern`(必填, glob), `path`(默认`.`), `max_results`(默认200) | 支持 `**` 递归，仅文件 |
| 7 | **ls** | `ls` | `path`(默认`.`) | 目录优先排序，显示大小 (B/K/M)，icon: `📁`/`📄` |

### 被忽略的目录

`grep` 和 `find` 自动忽略：`.git`、`__pycache__`、`node_modules`、`.venv`、`venv`、`.tox`、`.mypy_cache`、`.pytest_cache`

### 工具组合器

| 函数 | 包含的工具 |
|------|-----------|
| `create_all_tools(cwd)` | read + write + edit + bash + grep + find + ls |
| `create_coding_tools(cwd)` | read + bash + edit + write |
| `create_readonly_tools(cwd)` | read + grep + find + ls |

---

## 会话管理

### JSONL 存储格式

会话持久化为 JSONL 文件（`~/.pi/agent/sessions/{session_id}.jsonl`）：

```
{"type":"session","version":3,"id":"abc123...","timestamp":"...","cwd":"/path"}
{"type":"message","id":"msg1","parentId":null,"timestamp":"...","message":{...}}
{"type":"message","id":"msg2","parentId":"msg1","timestamp":"...","message":{...}}
```

首行 `SessionHeader` + 后续 `SessionMessageEntry`。

### 单链表上下文重建

`parentId` 从 root → leaf 形成单链。`build_context()` 从 leaf 回溯再反转，按时间顺序重建消息列表。

### SessionManager

| 工厂方法 | 说明 |
|----------|------|
| `SessionManager.create(cwd, sessions_dir?, session_id?)` | 新建 JSONL 文件 + 写入 header |
| `SessionManager.open(filepath, cwd_override?)` | 读取已有 JSONL 文件 |
| `SessionManager.in_memory(cwd, session_id?)` | 纯内存模式，不写磁盘 |

| 方法/属性 | 说明 |
|-----------|------|
| `.session_id` → `str` | 会话 ID |
| `.cwd` → `str` | 工作目录 |
| `.is_persisted()` → `bool` | 是否持久化到磁盘 |
| `await .append_message(msg) → str` | 追加消息 + 写入 JSONL，返回 entryId |
| `.build_context()` → `list[AgentMessage]` | 沿 parentId 链重建消息 |
| `.get_entries()` → `list[SessionMessageEntry]` | 返回所有条目 |

### AgentSession

中枢编排类，连接 Agent + 工具 + 持久化 + 事件转发。

```python
class AgentSession:
    def __init__(
        self,
        agent: Agent,
        session_manager: SessionManager,
        cwd: str,
        model: Model,
        *,
        tools_override: list[AgentTool] | None = None,  # 默认 create_all_tools(cwd)
    )

    async def prompt(self, text: str) -> None
    async def abort(self) -> None
    async def wait_for_idle(self) -> None
    def subscribe(self, listener) -> Callable[[], None]
    def get_messages(self) -> list[AgentMessage]
    async def dispose(self) -> None
```

**事件桥接**：`AgentSession` 监听 agent 的 `message_end` 事件 → 自动 `session_manager.append_message(msg)`，同时转发给外部监听器。

---

## 配置

### 双层 settings.json

`load_settings(cwd)` 实现双层合并（项目覆盖全局，深度字典合并）：

```
~/.pi/agent/settings.json     ← 全局配置
<cwd>/.pi/settings.json       ← 项目配置（覆盖全局）
```

### 支持的设置项（最小核心）

```json
{
    "defaultProvider": "deepseek",
    "defaultModel": "deepseek-chat",
    "tools": {
        "exclude": ["bash"]
    },
    "sessionDir": "/custom/sessions/path"
}
```

### 路径常量

| 函数 | 返回值 |
|------|--------|
| `get_agent_dir()` | `~/.pi/agent/` |
| `get_sessions_dir()` | `~/.pi/agent/sessions/` |
| `get_settings_path()` | `~/.pi/agent/settings.json` |
| `get_project_settings_path(cwd)` | `<cwd>/.pi/settings.json` |

---

## Print 模式

当前唯一运行模式。`run_print_mode(session, message) → int`：

1. `session.subscribe(on_event)` — 注册监听器
2. `await session.prompt(message)` — 发送消息
3. `await session.wait_for_idle()` — 等待完成
4. 提取：`agent_end` 事件中从后往前找最后一条 `assistant` 消息 → 提取纯文本 → `print(final_text)`
5. 错误处理：`stop_reason` 为 `error`/`aborted` 返回 1；异常返回 1
6. `finally` 中取消订阅 + `await session.dispose()`

返回码：`0` = 成功，`1` = 错误/中止。

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
