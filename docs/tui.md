# TUI（Python 移植）

pi-python 的 TUI 基于内置引擎（无 Textual），应用在
`src/pi_coding_agent/modes/interactive/app.py`（`PiTuiApp`），
引擎与组件/选择器/主题/快捷键在 `src/pi_tui/`。

## 架构

```text
PiTuiApp (pi_tui.engine.App)
├── screen (Container, vertical)
│   ├── PiHeader                       # logo + 快捷键提示
│   ├── PiChatContainer (ScrollView)   # 消息区（滚动条 + 鼠标滚轮）
│   │   ├── MessageEntry               # 常规消息（markdown 渲染）
│   │   └── BashExecutionEntry         # bash 执行块
│   ├── Static (pi-widgets-above)      # slash 补全 / setWidget
│   ├── PiStatusBar                    # 状态栏
│   ├── PiEditor (Editor)              # 输入区（/ 命令、! bash、alt+enter 排队）
│   ├── Static (pi-widgets-below)
│   └── PiFooter                       # 页脚
└── overlay 层（OverlayManager + OverlayWidget）
    ├── ModelSelector / ThinkingSelector / ScopedModelsSelector
    ├── SessionPicker / TreeSelector
    ├── TextInputDialog / ChoiceSelector / SettingsSelector
    ├── OAuthSelector / ExtensionSelector / TrustSelector
```

引擎位于 `src/pi_tui/engine/`：

- `cells.py`：`Cell` / `Line` 单元格行模型，组件渲染与屏幕差分的基础。
- `text.py`：Rich renderable → 定宽 `Line` 列表（markup / markdown / OSC8 链接）。
- `keys.py`：终端输入解析——UTF-8、CSI、SS3、kitty 协议、SGR 鼠标、bracketed paste、OSC。
- `terminal.py`：raw 模式、alt-screen、尺寸查询、差分写入（只重画变化行）、
  OSC 133 / 2026 / 52、硬件光标、Windows VT 模式；`FakeTerminal` 供无 TTY 测试。
- `widgets.py`：`Widget` / `Container`（vertical/horizontal 布局）、
  `Static` / `Input` / `Editor`（vim 模式、undo、word navigation）、
  `ScrollView`（滚动条）、`SelectList` / `SettingsList`、`Loader`、`Markdown`。
- `app.py`：事件循环、焦点、overlay 合成、快捷键分发、剪贴板、生命周期。

组件列表见 `src/pi_tui/components.py`，选择器见 `src/pi_tui/selectors.py`。

## 输入与命令

- `/name`：提示模板展开；`/skill:name`：技能展开（`_session.expand_prompt`）
- `/command`：`SlashCommandRegistry` + 内置命令（`slash_commands.py` 的 `register_builtin_commands`）：`/new`、`/resume`、`/tree`、`/fork`、`/input`、`/model`、`/thinking`、`/tools`、`/settings`、`/export`、`/reload` 等
- `/input [text]`：挂起当前任务 → 选择一条历史 user 消息 → 把输入合并进该消息
  （旧内容 + 空行 + 新内容）→ 会话回卷到该消息并重建 → `continue_()` 继续任务；
  旧分支保留在 JSONL 文件里，树仍可见（`SessionManager.edit_message`）
- `!cmd`：直接执行 shell；`alt+enter`：排队 follow-up
- 列表弹层（TreeSelector / ChoiceSelector / SessionPicker）：选中即复制——
  Enter 选择时把完整内容直接写入剪贴板（树选择器复制完整消息文本，
  会话选择器复制 session 路径）
- 输入框：`ctrl+c` 清空（对齐 TS）；多行编辑支持 vim 模式（`PiEditorVim`，
  Esc 切换 normal/insert，normal 模式 h/j/k/l、0/$、i/a/o、dd、x、u、undo；shift/词选区、ctrl+k kill、ctrl+y yank、选区复制）
- 输出框：点击一条消息复制其完整文本（bash 条目复制 `$ 命令` + 输出）；
  `ctrl+x` 仍可复制最后一条 assistant 消息
- 外部编辑器：`ctrl+g`（`$VISUAL` / `$EDITOR` 回退）

## 主题与快捷键

- 主题：`src/pi_tui/theme.py`（内置 dark/light + JSON 主题 + `auto` 终端背景检测），详见 [themes.md](themes.md)
- 快捷键：`src/pi_tui/keybindings.py`（settings `keybindings` 节覆盖，`/reload` 生效），详见 [keybindings.md](keybindings.md)
- 主题颜色直接作为 `Rich Style` 应用到各组件（无 CSS）；`/reload` 会重建键位表与主题样式

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

Print 模式降级为 `NoopUIContext`；RPC 模式有 `RpcUiContext`。

**已接线的渲染管线**：`register_message_renderer`（custom 角色消息）、
`register_markdown_transformer`（user/assistant/thinking 文本）、
`register_entry_renderer`（/tree 的 custom 条目）、`ctx.ui.set_footer` /
`set_header`（替换底部栏 / 顶部栏文本）、`ctx.ui.set_editor_component`
（用 `PiEditor` 子类替换编辑器）、`ctx.ui.set_widget`（编辑器上方 / 下方多行组件，
`aboveEditor` / `belowEditor`）。渲染器返回字符串，失败/缺失时回退内置渲染。

## Overlay 与焦点管理

`src/pi_tui/overlay/` 是独立的 overlay 运行时（不依赖任何 UI 框架，可单测）：

- `model.py`：`OverlayLayout` / `OverlayStyle` / `OverlayBehavior` 三类职责分离，`OverlayEntry` / `OverlayHandle`（hide / setHidden / focus / unfocus / isFocused）。
- `layout.py`：`resolve_layout()` 纯函数——锚点、margin、百分比/绝对 row/col、offset、maxHeight、minWidth、边界 clamp。
- `focus.py`：`OverlayFocusController` 三态状态机（inactive / active / blocked），管理焦点权与 preFocus / blocked / resume 恢复关系。
- `manager.py`：`OverlayManager` 管理 show / update / remove / setHidden / focusOrder 置顶、可见性回调、resize 重排、事件路由（topmost capturing overlay 优先，未处理则冒泡到基座）。
- `widgets.py`：`OverlayWidget`（引擎版，行文本 / 组件树双模）；`OverlayLayer` 为兼容占位。

`ctx.ui.set_overlay(key, lines, options)` 支持：锚点 / margin / offset / 百分比 /
maxHeight / 边框标题 / `nonCapturing` / `visible(w,h)` 回调；capturing overlay
会接管键盘焦点，关闭或 unfocus 后恢复到 preFocus（编辑器或上层 overlay）。

`ctx.ui.set_overlay_component(key, widget, options)`：组件树 overlay——组件模式
复用根节点、焦点自动落到子树内第一个可聚焦组件，同一 key 可在行文本与组件间切换。

**所有选择器均为 overlay**：`ChoiceSelector` / `TextInputDialog` /
`ThinkingSelector` / `SettingsSelector` / `ModelSelector` / `SessionPicker` /
`TreeSelector` / `OAuthSelector` / `ScopedModelsSelector` / `ExtensionSelector` /
`TrustSelector` 都继承 `OverlayDialog`，`push_screen` 自动桥接到 overlay 层
（居中、80% 宽、maxHeight 60%），dismiss 时移除 overlay 并把焦点恢复到打开前位置。

## 通用列表组件

`src/pi_tui/lists.py`：`SelectList`（模糊筛选：前缀/子串/子序列，上下键导航，
Enter 选择 / Escape 取消）与 `SettingsList`（label + 当前值两列，Enter 循环取值）。
`ChoiceSelector` / `ThinkingSelector` 复用 `SelectList`。

`src/pi_tui/markdown.py`：消息正文用 rich.markdown 渲染（标题/列表/代码块/表格/
链接），`MessageEntry` = label（粗体）+ Markdown 正文；扩展 markdown transformer
产出 Rich markup（含 `[/` 闭合标签）时保留原样，避免二次转义。

`src/pi_tui/autocomplete.py`：`CombinedAutocompleteProvider`——扩展自动补全
provider 栈（多 provider 并发收集、按 value 去重、保持注册顺序、支持同步/异步、
单 provider 异常跳过）；Tab 触发后异步收集并弹 overlay 选择器插入选中值。

终端能力：`src/pi_tui/terminal.py` 的 OSC 11 背景色查询在启动时探测终端背景色，
用于 `auto` 主题的深/浅选择；`src/pi_tui/terminal_image.py` 提供 kitty/iTerm2
图像序列生成（kitty placement / iTerm2 inline 已接入消息渲染）；`src/pi_tui/links.py` 把工具结果 /
工具调用 / markdown 正文中的绝对路径自动转成 OSC 8 可点击链接；`TerminalImage` 可作 overlay 图片组件且移除时清理 kitty 图片；引擎支持硬件光标
（`PI_HARDWARE_CURSOR=1`）、同步输出（OSC 2026）、OSC 133 prompt 标记、
OSC 52 剪贴板、SGR 鼠标滚轮、滚动条拖拽、鼠标拖选复制（高亮 + 双击选词）、overlay 动画、`App.flash` 闪烁提示。

流式渲染：`message_start` 挂一个 Assistant 占位条目，`message_update` 用 partial
快照增量更新（text / thinking / toolCall 统一文本），`message_end` 移除占位并追加
最终消息；每次增量自动滚动到底；`agent_settled` 兜底清理。带 `error_message` 的
assistant 消息标签显示为 `Assistant (error)`。


## 剪贴板图片

`src/pi_tui/clipboard_image.py`（`ClipboardImage`）：Windows `alt+v` / 其他平台 `ctrl+v` 从剪贴板粘贴图片（Windows 走 PowerShell/clipboard API，macOS 走 `osascript`，其他平台走 `xclip`）。

## 测试

引擎测试：`src/pi_tui/tests/`（布局 / 焦点 / manager 纯单测 + 引擎 keys /
cells / editor / lists / markdown / links / tree-selector / terminal-image）。
应用集成测试：`src/pi_coding_agent/tests/test_tui_app.py` 用 `FakeTerminal`
无头驱动（输入注入、输出捕获），覆盖启动渲染、提交、overlay 选择器、
流式消息、slash 通知、bash、主题切换与 ctrl+d 退出。
