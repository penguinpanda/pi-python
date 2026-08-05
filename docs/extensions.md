# 扩展（Python 移植）

扩展是 `.py` 模块，通过工厂函数拿到 `ExtensionAPI`，注册事件、工具、命令、快捷键、provider 等。实现：`src/pi_coding_agent/extensions/`（`loader.py` 发现/加载、`types.py` 类型与 API、`runner.py` 生命周期/事件分发）。

## 快速开始

`hello.py`：

```python
from pi_coding_agent import ExtensionAPI, ToolDefinition

def create_extension(pi: ExtensionAPI):
    # 1. 订阅事件
    @pi.on("message_end")
    def on_message_end(event, ctx):
        pass

    # 2. 注册自定义工具
    pi.register_tool(ToolDefinition(
        name="greet",
        label="Greeting",
        description="Generate a greeting",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Name to greet"}},
            "required": ["name"],
        },
        execute=lambda tool_call_id, params, signal, on_update, ctx: {
            "content": [{"type": "text", "text": f"Hello, {params['name']}!"}],
            "details": {},
        },
    ))

    # 3. 注册 /command
    pi.register_command("hello", {
        "description": "Say hello",
        "handler": lambda args, ctx: ctx.ui.notify("Hello!", "info"),
    })
```

工厂函数也可以是 async：`async def create_extension(pi)`。

## 位置与发现

- 全局：`~/.pi/agent/extensions/*.py`
- 项目：`.pi/extensions/*.py`（项目信任后）
- 显式路径：`ExtensionLoader.discover_all(explicit_paths=[...])`（文件或目录）
- 子目录入口：`index.py` 或 `pi_extension.py`
- 优先级：项目 > 全局 > 显式，按解析路径去重

CLI 暂未提供 TS 的 `--extension` / `-e` 标志；扩展由 `ResourceLoader` 在启动 / `/reload` 时加载。

## ExtensionAPI 方法

已实现（`src/pi_coding_agent/extensions/types.py`）：

| 方法 | 作用 |
| --- | --- |
| `on(event_type, handler)` | 订阅事件（handler 为 async 或 sync） |
| `register_tool(ToolDefinition \| dict)` | 注册 LLM 工具 |
| `register_command(name, options)` | 注册 `/command` |
| `register_shortcut(shortcut, options)` | 注册快捷键 |
| `register_flag(name, options)` / `get_flag(name)` | 注册/读取 CLI 标志值 |
| `register_message_renderer(custom_type, fn)` | 自定义消息渲染 |
| `register_entry_renderer(custom_type, fn)` | 自定义会话条目渲染 |
| `register_markdown_transformer(fn)` | Markdown 变换 |
| `register_autocomplete(provider)` | 自动补全 provider |
| `register_provider(name, config)` / `unregister_provider(name)` | 注册/注销 provider |
| `set_model` / `get_thinking_level` / `set_thinking_level` | 模型与思考级别 |
| `set_session_name` / `get_session_name` | 会话名 |
| `send_user_message(content, options=None)` | 注入用户消息 |
| `get_active_tools` / `set_active_tools` / `get_all_tools` | 工具开关 |
| `get_commands` | 已注册命令列表 |
| `exec(command, args=None, options=None)` | 执行 shell（`PythonExecutionEnv`） |
| `events` | `EventBus`（扩展间通信：`events.on(event, fn)` / `events.emit(event, data)`） |

工具定义字段：`name`、`label`、`description`、`parameters`（JSON Schema dict）、`execute`。执行返回 `{"content": [{"type": "text", "text": ...}], "details": {...}}`，`details` 会随会话持久化（分支/压缩后可恢复状态）。

## 事件

已接线的子集：

- **Agent 循环事件**：`agent_start` / `agent_end`、`turn_start` / `turn_end`、`message_start` / `message_update` / `message_end`、`tool_execution_start` / `tool_execution_end`、`auto_retry_start` / `auto_retry_end`（`_session._handle_agent_event` 全部转发给扩展 runner）
- **`input`**：输入变换链，`(text, action)` 返回；`handled` 短路（`ExtensionRunner.emit_input`）
- **`project_trust`**：返回 `"yes"` / `"no"` / `"undecided"` 决定信任
- **`session_shutdown`**：卸载时通知（`ExtensionRunner.unload`）

未移植的 TS 事件：`resources_discover`、`session_info_changed`、`session_before_*`、`before_agent_start` 的完整语义、`before_provider_headers` / `before_provider_request` / `after_provider_response`、`tool_call` 阻断与 `tool_result`、`user_bash`、`model_select` / `thinking_level_select` 等。

## ExtensionContext / ExtensionCommandContext

`ExtensionContext` 属性与方法：`ui`、`mode`、`cwd`、`session`、`model`、`thinking_level`、`is_idle()`、`has_pending_messages()`、`abort()`、`shutdown()`、`compact()`（async）、`get_system_prompt()`。

`ctx.ui` 协议（`UIContext`）：`select(title, options, timeout=None)`、`confirm(title, message, timeout=None)`、`input(title, placeholder=None, timeout=None)`（async）；`notify(message, type=None)`、`set_status(key, text)`、`set_title(title)`、`set_editor_text(text)`。Print 模式用 `NoopUIContext`（全部 no-op），TUI / RPC 各有实现。

`ExtensionCommandContext` 额外提供：`wait_for_idle()`、`new_session(options=None)`、`fork(entry_id, options=None)`、`navigate_tree(target_id, options=None)`、`switch_session(session_path, options=None)`、`reload()`。

## 生命周期与错误

- `ExtensionLoader.load_extension(path, runtime, event_bus)`：importlib 动态加载，执行 `create_extension(api)` / `factory(api)`；失败返回 `ExtensionError(extension_path, event, error)`
- `ExtensionRunner`：`bind(...)` / `bind_session(session)` 绑定运行时动作；`emit_event` / `emit_input` / `emit_project_trust`；`on_error` 监听
- 动作在绑定前调用会抛 “not initialized”（`_not_initialized`）

## 示例

Python 化示例见 `examples/extensions/` 下的 `.py` 文件；TS 原示例在 TS 仓库 `packages/coding-agent/examples/extensions/`。
