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
- `/command`：`SlashCommandRegistry` + 内置命令（`slash_commands.py` 的 `register_builtin_commands`）：`/new`、`/resume`、`/tree`、`/fork`、`/input`、`/model`、`/thinking`、`/tools`、`/settings`、`/export`、`/reload` 等
- `/input [text]`：挂起当前任务 → 选择一条历史 user 消息 → 把输入合并进该消息
  （旧内容 + 空行 + 新内容）→ 会话回卷到该消息并重建 → `continue_()` 继续任务；
  旧分支保留在 JSONL 文件里，树仍可见（`SessionManager.edit_message`）
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

Print 模式降级为 `NoopUIContext`；RPC 模式有 `RpcUiContext`。自定义 Textual 组件/渲染器注册（TS 的 `registerMessageRenderer` / `registerEntryRenderer`）只有底层注册表，UI 接线未完整移植。

**已接线的渲染管线**：`register_message_renderer`（custom 角色消息）、`register_markdown_transformer`（user/assistant/thinking 文本）、`register_entry_renderer`（/tree 的 custom 条目）、`ctx.ui.set_footer` / `set_header`（替换底部栏 / 顶部栏文本）、`ctx.ui.set_editor_component`（用 `PiEditor` 子类替换编辑器）、`ctx.ui.set_widget`（编辑器上方 / 下方多行组件，`aboveEditor` / `belowEditor`）。渲染器返回字符串，失败/缺失时回退内置渲染。

## Overlay 与焦点管理

`src/pi_tui/overlay/` 是独立的 overlay 运行时（核心不依赖 Textual，可单测）：

- `model.py`：`OverlayLayout` / `OverlayStyle` / `OverlayBehavior` 三类职责分离，`OverlayEntry` / `OverlayHandle`（hide / setHidden / focus / unfocus / isFocused）。
- `layout.py`：`resolve_layout()` 纯函数——锚点、margin、百分比/绝对 row/col、offset、maxHeight、minWidth、边界 clamp。
- `focus.py`：`OverlayFocusController` 三态状态机（inactive / active / blocked），只管理“哪个 overlay 拥有焦点权”与 preFocus / blocked / resume 恢复关系，不直接操作 Textual widget。
- `manager.py`：`OverlayManager` 管理 show / update / remove / setHidden / focusOrder 置顶、可见性回调、resize 重排、事件路由（topmost capturing overlay 优先，未处理则冒泡到基座）。
- `widgets.py`：`OverlayLayer`（保留类，空渲染，不挂载）与 `OverlayWidget`（Static 子类，can_focus）；overlay 直接以 `layer: overlay` 挂到 Screen。

`ctx.ui.set_overlay(key, lines, options)` 支持：锚点 / margin / offset / 百分比 / maxHeight / 边框标题 / `nonCapturing` / `visible(w,h)` 回调 / 动画；capturing overlay 会接管键盘焦点，关闭或 unfocus 后恢复到 preFocus（编辑器或上层 overlay）。ModalScreen 选择器暂不接入这套焦点协议。

`ctx.ui.set_overlay_component(key, widget, options)`：组件树 overlay——`OverlayWidget`
双模（行文本 / 组件），组件模式复用根节点、焦点自动落到子树内第一个可聚焦组件，
同一 key 可在行文本与组件间切换。

**所有选择器均已 overlay 化**：`ChoiceSelector` / `TextInputDialog` /
`ThinkingSelector` / `SettingsSelector` / `ModelSelector` / `SessionPicker` /
`TreeSelector` / `OAuthSelector` / `ScopedModelsSelector` / `ExtensionSelector` /
`TrustSelector` 都继承 `OverlayDialog`（Widget），`push_screen` 自动桥接到
overlay 层（居中、80% 宽、maxHeight 60%），dismiss 时移除 overlay 并把焦点
恢复到打开前位置（编辑器或上层 overlay）。pi 自身不再使用 ModalScreen。

## 通用列表组件

`src/pi_tui/lists.py`：`SelectList`（模糊筛选：前缀/子串/子序列，上下键导航，
Enter 选择 / Escape 取消）与 `SettingsList`（label + 当前值两列，Enter 循环取值）。
`ChoiceSelector` / `ThinkingSelector` 已改为复用 `SelectList`，保留旧
`action_select` / `action_cancel` 兼容入口。

`src/pi_tui/components.py`：`PiEditorVim(PiEditor)`——vim 风格编辑器（Esc 切换
normal/insert，normal 模式支持 h/j/k/l、0/$、i/a/o、dd、x、u），可经
`set_editor_component` 替换默认编辑器。

`src/pi_tui/markdown.py`：消息正文用 rich.markdown 渲染（标题/列表/代码块/表格/
链接），`MessageEntry` = label（粗体）+ Markdown 正文；扩展 markdown transformer
产出 Rich markup（含 `[/` 闭合标签）时保留原样，避免二次转义。

`src/pi_tui/autocomplete.py`：`CombinedAutocompleteProvider`——扩展自动补全
provider 栈（多 provider 并发收集、按 value 去重、保持注册顺序、支持同步/异步、
单 provider 异常跳过）；Tab 触发后异步收集并弹 overlay 选择器插入选中值。

终端能力：`src/pi_tui/terminal.py` 的 OSC 11 背景色查询在启动时探测终端背景色，
用于 `auto` 主题的深/浅选择（Textual 自身不提供该查询）；`src/pi_tui/terminal_image.py`
提供 kitty/iTerm2 图像序列生成（独立模块，暂未接入聊天渲染管线）。

流式渲染：`message_start` 挂一个 Assistant 占位条目，`message_update` 用 partial
快照增量更新（text / thinking / toolCall 统一文本），`message_end` 移除占位并追加
最终消息；每次增量自动滚动到底；`agent_settled` 兜底清理。`MessageEntry.set_text()`
支持未挂载/已挂载两种状态更新；带 `error_message` 的 assistant 消息标签显示为
`Assistant (error)`。

详细的 TS pi-tui 差距分析、剩余差距与下一步开发路线见 [tui-gap.md](tui-gap.md)。

## 剪贴板图片

`src/pi_tui/clipboard_image.py`（`ClipboardImage`）：Windows `alt+v` / 其他平台 `ctrl+v` 从剪贴板粘贴图片（Windows 走 PowerShell/clipboard API，macOS 走 `osascript`，其他平台走 `xclip`）。

## 未移植（TS TUI 独有）

- 渲染回调 API（`ctx.ui.set_overlay_renderer(key, fn(width, height) -> list[str])`）已实现；
  `SelectList` / `SettingsList` / `PiEditorVim` / Markdown 渲染均已实现
- ModalScreen 与 overlay 焦点协议的统一（一期共存）
- 全屏 alt-screen 滚动、OSC 8 链接点击、鼠标拖选
- 树过滤模式与 label 时间戳已实现：TreeSelector 按 `f` 循环 default / no-tools /
  user-only / labeled-only / all（default 隐藏 label/custom/model_change/
  thinking_level_change/session_info 等记账条目）；按 `t` 开关 `[+label time]`
  （label 时间戳显示为本地 HH:MM:SS）
- 自动补全 provider 栈（如 GitHub issue autocomplete）

## 测试

`src/pi_coding_agent/tests/`：`test_tui_app.py`、`test_tui_theme.py`、`test_tui_keybindings.py`、`test_tui_slash_commands.py`、`test_tui_message_rendering.py`、`test_tui_clipboard_image.py`。
