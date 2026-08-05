# TUI（Python 移植）

pi-python 的 TUI 基于 **Textual**，应用在 `src/pi_coding_agent/modes/interactive/app.py`（`PiTuiApp`），组件/选择器/主题/快捷键在 `src/pi_tui/`。

## 架构

```text
PiTuiApp (Textual App)
├── PiChatContainer (VerticalScroll)   # 消息区
│   ├── MessageEntry (Static)          # 常规消息（markdown 渲染）
│   └── BashExecutionEntry (Static)    # bash 执行块
├── PiEditor (TextArea)                # 输入区（/ 命令、! bash、alt+enter 排队）
├── PiStatusBar (Label)                # 状态栏
├── PiFooter (Label)                   # 页脚
├── PiToolbar (Input)                  # 工具栏
├── PiHeader (Static)
└── ModalScreen 选择器
    ├── ModelSelector / ThinkingSelector / ScopedModelsSelector
    ├── SessionPicker / TreeSelector
    ├── TextInputDialog / ChoiceSelector / SettingsSelector
    ├── OAuthSelector / ExtensionSelector / TrustSelector
```

组件列表见 `src/pi_tui/components.py`，选择器见 `src/pi_tui/selectors.py`。

## 输入与命令

- `/name`：提示模板展开；`/skill:name`：技能展开（`_session.expand_prompt`）
- `/command`：`SlashCommandRegistry` + 内置命令（`slash_commands.py` 的 `register_builtin_commands`）：`/new`、`/resume`、`/tree`、`/fork`、`/model`、`/thinking`、`/tools`、`/settings`、`/export`、`/reload` 等
- `!cmd`：直接执行 shell；`alt+enter`：排队 follow-up
- 外部编辑器：`ctrl+g`（`$VISUAL` / `$EDITOR` 回退）

## 主题与快捷键

- 主题：`src/pi_tui/theme.py`（内置 dark/light + JSON 主题 + `auto` 终端背景检测），详见 [themes.md](themes.md)
- 快捷键：`src/pi_tui/keybindings.py`（settings `keybindings` 节覆盖，`/reload` 生效），详见 [keybindings.md](keybindings.md)
- CSS：`_build_css(colors)` 把主题颜色注入 Textual CSS（`--pi-*` 变量）

## 扩展可用的 UI

`ExtensionContext.ui`（`UIContext` 协议）：

```python
async def handler(event, ctx):
    choice = await ctx.ui.select("Pick", ["a", "b"])
    ok = await ctx.ui.confirm("Sure?", "Continue?")
    text = await ctx.ui.input("Name", placeholder="...")
    ctx.ui.notify("done", "success")
    ctx.ui.set_status("turn", "working")
    ctx.ui.set_title("pi")
    ctx.ui.set_editor_text("prefilled")
```

Print 模式降级为 `NoopUIContext`；RPC 模式有 `RpcUiContext`。自定义 Textual 组件/渲染器注册（TS 的 `registerMessageRenderer` / `registerEntryRenderer` / overlay）只有底层注册表，UI 接线未完整移植。

## 剪贴板图片

`src/pi_tui/clipboard_image.py`（`ClipboardImage`）：Windows `alt+v` / 其他平台 `ctrl+v` 从剪贴板粘贴图片（Windows 走 PowerShell/clipboard API，macOS 走 `osascript`，其他平台走 `xclip`）。

## 未移植（TS TUI 独有）

- Overlay 合成系统（anchors / margins / stacking / 动画）
- 自定义编辑器组件替换（`setEditorComponent`）、vim 模式编辑器
- 全屏 alt-screen 滚动、OSC 8 链接点击、鼠标拖选
- 树过滤模式（default / no-tools / user-only / labeled-only / all）、label 时间戳
- `registerMessageRenderer` / `registerEntryRenderer` 的完整渲染管线
- 自动补全 provider 栈（如 GitHub issue autocomplete）

## 测试

`src/pi_coding_agent/tests/`：`test_tui_app.py`、`test_tui_theme.py`、`test_tui_keybindings.py`、`test_tui_slash_commands.py`、`test_tui_message_rendering.py`、`test_tui_clipboard_image.py`。
