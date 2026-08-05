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
    pi.register_tool(
        ToolDefinition(
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
        )
    )

    # 3. 注册 /command
    pi.register_command(
        "hello",
        {
            "description": "Say hello",
            "handler": lambda ctx, args: ctx.ui.notify("Hello!", "info"),
        },
    )
```

工厂函数也可以是 async：`async def create_extension(pi)`。

## 位置与发现

- 全局：`~/.pi/agent/extensions/*.py`
- 项目：`.pi/extensions/*.py`（项目信任后）
- 显式路径：`ExtensionLoader.discover_all(explicit_paths=[...])`（文件或目录）
- 子目录入口：`index.py` 或 `pi_extension.py`
- 优先级：项目 > 全局 > 显式，按解析路径去重

CLI 支持 `--extension` / `-e`（可重复，文件或目录）；扩展默认由 `ResourceLoader` 在启动 / `/reload` 时从全局与项目目录发现。

## ExtensionAPI 方法

已实现（`src/pi_coding_agent/extensions/types.py`）：

| 方法 | 作用 |
| --- | --- |
| `on(event_type, handler)` | 订阅事件（handler 为 async 或 sync） |
| `register_tool(ToolDefinition \| dict)` | 注册 LLM 工具 |
| `register_command(name, options)` | 注册 `/command` |
| `register_shortcut(shortcut, options)` | 注册快捷键 |
| `register_flag(name, options)` / `get_flag(name)` | 注册/读取 CLI 标志值 |
| `register_message_renderer(custom_type, fn)` | 自定义消息渲染（TUI 已接线；custom 角色消息也进入 LLM 上下文） |
| `register_tool_renderer(tool_name, fn)` | 内置/自定义工具结果的紧凑渲染（TUI 已接线，返回字符串） |
| `register_entry_renderer(custom_type, fn)` | 自定义会话条目渲染（/tree 已接线） |
| `register_markdown_transformer(fn)` | Markdown 变换链（TUI 已接线，失败跳过保留上一步） |
| `register_autocomplete(provider)` | 自动补全 provider（TUI 已接线：编辑器 Tab → provider(text) → 选择器插入） |
| `register_provider(name, config)` / `unregister_provider(name)` | 注册/注销 provider |
| `set_model` / `get_thinking_level` / `set_thinking_level` | 模型与思考级别 |
| `set_session_name` / `get_session_name` | 会话名 |
| `send_user_message(content, options=None)` | 注入用户消息 |
| `send_message(content, options=None)` | 注入自定义消息（role=custom，写会话树） |
| `append_entry(custom_type, data=None)` | 追加自定义会话条目（custom 类型） |
| `set_label(entry_id, label)` | 给会话条目设置 label（/tree 用） |
| `get_active_tools` / `set_active_tools` / `get_all_tools` | 工具开关 |
| `get_commands` | 已注册命令列表 |
| `exec(command, args=None, options=None)` | 执行 shell（`PythonExecutionEnv`） |
| `events` | `EventBus`（扩展间通信：`events.on(event, fn)` / `events.emit(event, data)`） |

工具定义字段：`name`、`label`、`description`、`parameters`（JSON Schema dict）、`execute`。执行返回 `{"content": [{"type": "text", "text": ...}], "details": {...}}`，`details` 会随会话持久化（分支/压缩后可恢复状态）。

**工具合并**：`register_tool` 的工具会在会话初始化与 `/reload` 时合并进 agent 工具集；同名工具**覆盖**内置工具（对齐 TS tool-override）。`execute` 返回 dict（`content` / `details` / `terminate` / `usage`）会自动归一化为 `AgentToolResult`。

**命令 handler 签名**：`(ctx, args)`（与 TS 的 `(args, ctx)` 相反；这是 pi-python 的既有约定，registry 与测试均按此执行）。

**自定义消息**：`pi.send_message(content, {"customType": ...})` 写入 `custom_message` 条目；`build_context` 以 `role="custom"` 进入上下文，`convert_to_llm` 包装为 user 消息；TUI 用 `register_message_renderer(customType)` 渲染，未注册时回退文本。`register_entry_renderer` 供 `/tree` 渲染 `append_entry` 产生的 custom 条目（返回字符串）。

## 事件

已接线的子集：

- **Agent 循环事件**：`agent_start` / `agent_end`、`turn_start` / `turn_end`、`message_start` / `message_update` / `message_end`、`tool_execution_start` / `tool_execution_end`、`auto_retry_start` / `auto_retry_end`（`_session._handle_agent_event` 全部转发给扩展 runner）
- **`session_start`**：会话创建时（含恢复会话）
- **`session_info_changed`**：会话名变化时（`name` / `previousName`）
- **`session_before_switch` / `session_before_fork`**：新会话 / 切换 / fork 前；返回 `{"cancel": True}` 阻止（switch 的 `position` 为 `"at"`，new/fork 为 `"before"`）
- **`session_before_compact`**：压缩前；返回 `{"cancel": True}` 取消，或 `{"compaction": {summary, firstKeptEntryId, tokensBefore, usage?}}` 提供自定义摘要（手动 `/compact` 与自动压缩都走）
- **`session_compact`**：压缩完成后（事件含 `compactionEntry` / `fromExtension` / `reason` / `willRetry`）
- **`session_before_tree` / `session_tree`**：`/tree` 导航前（返回 `{"cancel": True}` 或 `{"summary": {...}}` 自定义摘要）与导航后（`newLeafId` / `oldLeafId` / `summaryEntry` / `fromExtension`）
- **`before_agent_start`**：每轮 prompt 前；handler 可返回 `{"system_prompt": ...}`（当轮生效、结束后恢复）或 `{"prompt": ...}` 替换本轮输入
- **`model_select`**：`set_model` / 模型循环切换时（事件含 `model` / `previousModel` / `provider` / `modelId`）
- **`thinking_level_select`**：思考级别变化时
- **`tool_call`**：工具执行前；返回 `{"block": True, "reason": ...}` 阻断，或 `{"input": ...}` 改写参数（通过包装 agent 的 `before_tool_call`）
- **`tool_result`**：工具执行后；返回 `{"content" / "details" / "is_error" / "usage" / "terminate"}` 覆盖结果（包装 `after_tool_call`）
- **`user_bash`**：`!` / `!!` 命令执行前；返回 `{"result": {output, exitCode, cancelled, truncated}}` 直接替换，或 `{"operations": {"exec": fn}}` 自定义执行后端
- **`resources_discover`**：动态资源发现；返回 `{"skills"/"prompts"/"themes": [...]}` 对象形式或 `{"skillPaths"/"promptPaths"/"themePaths": [...]}` 路径形式（动态技能/模板参与 `/skill:` 与 `/模板名` 展开，并进入系统提示）
- **`before_provider_request`**：每次 provider 请求前；返回 `{"stream_options": {...}}` 覆盖请求选项
- **`before_provider_headers`**：请求头合并（返回 `{"headers": {...}}`）
- **`after_provider_response`**：每次 provider 请求完成后（`response` 为最终 assistant 消息）
- **`context`**：每次 LLM 请求前；返回 `{"messages": [...]}` 非破坏性过滤/修改消息
- **`tool_execution_update`**：工具执行期间 on_update 进度（与 `tool_execution_start` / `tool_execution_end` 配套）
- **`input`**：输入变换链，`(text, action)` 返回；`handled` 短路；事件带 `streamingBehavior`（`"steer"` / `"followUp"` / None，扩展投递消息时透传，见 `ExtensionRunner.emit_input`）
- **`project_trust`**：返回 `"yes"` / `"no"` / `"undecided"` 决定信任
- **`session_shutdown`**：卸载时通知（`ExtensionRunner.unload`）

未移植的 TS 事件：无（替换完成后的 `session_shutdown` / `session_start` 由宿主流程负责）。

## bash 工具

`create_all_tools` / `create_bash_tool` 支持 `session_env_provider`（注入 `PI_SESSION_ID` 等）与 `spawn_hook`（对齐 TS `spawnHook`），见 [environment-variables.md](environment-variables.md)。

## ExtensionContext / ExtensionCommandContext

`ExtensionContext` 属性与方法：`ui`、`mode`、`cwd`、`session`、`session_manager`（会话树管理器）、`model`、`thinking_level`、`scoped_models`（`--models` 循环列表）、`has_ui`（print 模式为 False）、`is_project_trusted()`（项目信任状态，CLI/TUI 启动时写入 `session.project_trusted`）、`model_registry`（`find(provider, id)` / `complete(model, context, options)`）、`signal`（当前 turn 中止信号，无运行 turn 为 None）、`is_idle()`、`has_pending_messages()`、`abort()`、`shutdown()`、`compact()`（async）、`get_system_prompt()`、`get_system_prompt_options()`、`get_context_usage()`（按最后一条带 usage 的 assistant 消息估算 tokens；无 usage 数据返回 None）。

`ctx.ui` 协议（`UIContext`）：`select(title, options, timeout=None)`、`confirm(title, message, timeout=None)`、`input(title, placeholder=None, timeout=None)`（async）；`notify(message, type=None)`、`set_status(key, text)`、`set_title(title)`、`set_editor_text(text)`、`set_footer(text)`、`set_header(text)`、`set_editor_component(component)`（组件必须是 `PiEditor` 子类；替换后聚焦新编辑器，旧编辑器隐藏）、`set_widget(key, lines, options)`（`options.placement` 为 `"aboveEditor"`（默认）或 `"belowEditor"`，空列表移除）、`set_overlay(key, lines, options)`（浮层：锚点 / margin / 动画 / 边框标题 / `position: absolute`）、`set_hidden_thinking_label(label=None)`（折叠 thinking 块标签）、`set_working_message(text=None)`（流式工作提示文案）、`set_theme(theme=None)`（切换主题，None 恢复配置主题）。Print 模式用 `NoopUIContext`（全部 no-op）；TUI 模式由 `TuiUIContext`（`modes/interactive/ui_context.py`）实现，并提供 `ctx.ui.theme`（`fg(name, text)` / `bg(name, text)` ANSI 颜色）；RPC 用 `RpcUiContext`（`setFooter` / `setHeader` / `setEditorComponent` / `setWidget` / `setOverlay` / `setHiddenThinkingLabel` / `setWorkingMessage` / `setTheme` 走 `extension_ui_request`）。

`ExtensionCommandContext` 额外提供：`wait_for_idle()`、`new_session(options=None)`、`fork(entry_id, options=None)`、`navigate_tree(target_id, options=None)`、`switch_session(session_path, options=None)`、`reload()`。

## 生命周期与错误

- `ExtensionLoader.load_extension(path, runtime, event_bus)`：importlib 动态加载，执行 `create_extension(api)` / `factory(api)`；失败返回 `ExtensionError(extension_path, event, error)`
- `ExtensionRunner`：`bind(...)` / `bind_session(session)` 绑定运行时动作；`emit_event` / `emit_input` / `emit_project_trust`；`on_error` 监听
- 动作在绑定前调用会抛 “not initialized”（`_not_initialized`）

## 示例

Python 化示例见 `examples/extensions/` 下的 `.py` 文件；TS 原示例在 TS 仓库 `packages/coding-agent/examples/extensions/`。
