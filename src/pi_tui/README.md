# pi-tui — 终端 UI 框架（内置引擎）

基于 [pi-mono/packages/tui](https://github.com/earendil-works/pi-mono) 的 Python 复刻，
与 `pi_ai` / `pi_agent` 平级的可复用终端 UI 框架包（**无 Textual**）：
内置引擎、主题、快捷键、基础组件、选择器与剪贴板图片处理。
应用层（AgentSession 绑定、slash 命令、会话切换）位于 `pi_coding_agent.modes.interactive`。

---

## 架构概览

```text
pi_tui
├── engine/                        # 内置终端引擎
│   ├── cells.py                   # Cell / Line 单元格行模型 + ANSI/OSC8 输出
│   ├── text.py                    # Rich renderable → 定宽 Line
│   ├── keys.py                    # UTF-8 / CSI / SS3 / kitty / SGR 鼠标 / paste / OSC
│   ├── terminal.py                # raw / alt-screen / 尺寸 / 行差分 / OSC 133/2026/52
│   ├── widgets.py                 # Widget / Container / Text / Input / Editor / ScrollView / 列表
│   ├── overlay_widget.py          # OverlayWidget（行文本 / 组件树双模）
│   └── app.py                     # App 基类（事件循环、焦点、overlay 合成、快捷键）
├── components.py                  # PiHeader / PiChatContainer / PiEditor / PiStatusBar / PiFooter
├── keybindings.py                 # KeybindingsManager + DEFAULT_APP_KEYBINDINGS
├── selectors.py                   # ModelSelector / SessionPicker / TreeSelector 等
├── theme.py                       # ThemeLoader + DARK_THEME / LIGHT_THEME / 自定义 JSON
├── overlay/                       # overlay 模型 / 布局 / 焦点状态机 / 管理器
├── markdown.py / autocomplete.py / terminal_image.py / clipboard_image.py
└── lists.py                       # SelectList / SettingsList
```

依赖：`rich`（文本样式 / markdown 解析）+ `pillow`（剪贴板图片处理）。
引擎本身不依赖 `pi_ai`；仅 `clipboard_image` 复用 `pi_agent.tools.image_pipeline`
（见下文剪贴板图片一节），其余部分可独立复用。

---

## 快速开始

### 安装

```bash
uv sync
```

### 最小示例：组合内置组件

```python
import asyncio

from pi_tui.engine import App, FakeTerminal
from pi_tui.components import PiChatContainer, PiEditor, PiFooter, PiHeader
from pi_tui.keybindings import KeybindingsManager
from pi_tui.theme import ThemeLoader


class MyApp(App):
    def __init__(self, terminal=None) -> None:
        self._keybindings = KeybindingsManager()
        self._theme = ThemeLoader().resolve("dark")
        super().__init__(terminal=terminal or FakeTerminal(size=(100, 30)))

    def on_mount(self) -> None:
        self.screen.mount(PiHeader(self._keybindings, height=1, id="pi-header"))
        self._chat = PiChatContainer(height="1fr", id="pi-chat")
        self.screen.mount(self._chat)
        self.screen.mount(PiEditor(height=6, id="pi-editor"))
        self.screen.mount(PiFooter("", height=1, id="pi-footer"))

    def on_pi_editor_submitted(self, message: PiEditor.Submitted) -> None:
        self._chat.add_message_agent({"role": "user", "content": message.text})


asyncio.run(MyApp().run_async())
```

### 主题注入

主题颜色直接作为 `Rich Style` 应用到组件（`bgcolor` / `color`）：

```python
theme = ThemeLoader().load("dark")
header.base_style = Style(bgcolor=theme.colors["bgToolbar"], color=theme.colors["textAlt"])
```

`Theme.css_variables()` 保留为兼容 API（旧 CSS 模板不再使用）。

---

## 引擎（engine/）

- `cells.py`：`Cell`（字符 + Rich Style + OSC8 link）与 `Line`（定宽单元格行）；
  `line_to_ansi` 输出 SGR + OSC8 链接 + 尾部重置。
- `text.py`：`render_markup` / `render_markdown` / `render_renderable` → `list[Line]`。
- `keys.py`：`KeyParser` 增量解析，`Key` 规范名（`ctrl+p` / `shift+tab` / `f1`），
  `MouseEvent`（SGR press/motion/wheel）。
- `terminal.py`：POSIX raw（termios/tty）与 Windows VT（SetConsoleMode）、alt-screen、
  `ScreenBuffer.diff` 行差分、OSC 133 prompt / OSC 2026 同步输出 / OSC 52 剪贴板、
  `PI_HARDWARE_CURSOR` 硬件光标；`FakeTerminal` 供无 TTY 测试。
- `widgets.py`：`Widget` / `Container`（vertical/horizontal、fixed/auto/fr）、
  `Static`（Rich markup）、`Input`、`Editor`（多行、undo/redo、vim 模式、
  word navigation、补全导航）、`ScrollView`（滚动条、滚轮）、
  `SelectList` / `SettingsList`、`Loader` / `CancellableLoader`、`Markdown`。
- `app.py`：`request_render` / `_compose`（base + overlay 合成）/ `focus` /
  `dispatch_message`（`on_<namespace>_<snake>` 处理器）/ 快捷键分发 / 剪贴板 / 生命周期。

---

## 组件（components.py）

### 消息渲染：`message_to_entries(message, *, show_tools, show_thinking)`

把 `AgentMessage` 转成 `[(label, text)]`，按 role 分组：

| role | label | 说明 |
|------|-------|------|
| `user` | `User` | 字符串或内容块文本 |
| `toolResult` | `Tool: <tool_name>` / `Tool: <tool_name> (error)` | 工具执行结果（错误带标记） |
| `assistant` | `Thinking` / `Assistant` / `Tool call` | 按 `show_thinking` / `show_tools` 过滤；toolCall 逐条渲染 |
| `compactionSummary` | `Compaction summary` | 压缩摘要 |
| `branchSummary` | `Branch summary` | 分支摘要 |
| `skillInvocation` | `Skill` | 技能调用提示 |
| `system` | `System` | 系统消息 |
| 其它（custom） | `Agent` | 降级为文本 |

内容块支持 `text` / `thinking` / `toolCall` / `image`（`[image]` 占位）。

### 组件清单

| 组件 | 基类 | 职责 |
|------|------|------|
| `PiHeader` | `Static` | Logo + 快捷键提示（读取 KeybindingsManager 生成） |
| `PiChatContainer` | `ScrollView` | 消息列表；`add_message_agent()` 追加并滚动到底部，`clear_messages()` 清空，`set_visibility()` 控制工具/思考块显隐 |
| `ToolExecutionEntry` / `BashExecutionEntry` | `Widget` | 工具执行与 bash 交互条目：名称/参数 + 流式输出 + 状态色，可展开/折叠 |
| `PiEditor` | `Editor` | 多行输入；Enter 提交并发出 `Submitted` 事件（空文本忽略）；vim 模式 `PiEditorVim` |
| `PiStatusBar` | `Label` | 状态栏：Working / Compacting / Idle 等 |
| `PiFooter` | `Label` | 底部栏：`update_info(model, thinking, message_count, session_name)` |
| `PiToolbar` | `Input` | 工具条输入（占位，后续用于搜索等） |

---

## 快捷键（keybindings.py）

快捷键以 action id（如 `app.model.cycleForward`）为键，默认绑定如下：

| action id | 默认键 | 动作 | 说明 |
|-----------|--------|------|------|
| `app.interrupt` | `escape` | `interrupt` | 取消 / 中止 |
| `app.clear` | `ctrl+c` | `clear` | 清空编辑器 |
| `app.exit` | `ctrl+d` | `exit` | 编辑器为空时退出 |
| `tui.altScreen.previousPrompt` | `ctrl+shift+up` | `previous_prompt` | 跳到上一条语义 prompt |
| `tui.altScreen.nextPrompt` | `ctrl+shift+down` | `next_prompt` | 跳到下一条语义 prompt |
| `app.thinking.cycle` | `shift+tab` | `cycle_thinking` | 切换思考级别 |
| `app.model.cycleForward` | `ctrl+p` | `cycle_model_forward` | 下一个模型 |
| `app.model.cycleBackward` | `shift+ctrl+p` | `cycle_model_backward` | 上一个模型 |
| `app.model.select` | `ctrl+l` | `select_model` | 打开模型选择器 |
| `app.tools.expand` | `ctrl+o` | `toggle_tools` | 切换工具输出 |
| `app.thinking.toggle` | `ctrl+t` | `toggle_thinking` | 切换思考块显示 |
| `app.message.followUp` | `alt+enter` | `follow_up` | 排队后续消息 |
| `app.message.dequeue` | `alt+up` | `dequeue` | 恢复已排队消息 |
| `app.clipboard.pasteImage` | `alt+v`（Windows）/ `ctrl+v`（其它平台） | `paste_image` | 粘贴剪贴板图片 |
| `app.session.new` | `ctrl+n` | `new_session` | 新建会话 |
| `app.session.resume` | `ctrl+r` | `resume_session` | 恢复会话 |
| `app.message.copy` | `ctrl+x` | `copy_last_message` | 复制最后一条 assistant 消息 |
| `app.editor.external` | `ctrl+g` | `external_editor` | 打开外部编辑器 |

### settings.json 覆盖

`KeybindingsManager.load_from_settings(settings)` 读取 `settings["keybindings"]`，
值为 `action_id → key`，支持三种形式：

```json
{
    "keybindings": {
        "app.model.cycleForward": "ctrl+k",
        "app.model.select": ["ctrl+l", "ctrl+m"],
        "app.message.copy": []
    }
}
```

- 字符串 → 绑定单键；数组 → 第一个为主键，其余为备用键；
- 空数组 → 禁用该 action（`is_enabled()` 返回 False，不出现在 `all_bindings()`）；
- `None` / 未知 action → 忽略，保持默认。

### KeybindingsManager API

| 方法 | 说明 |
|------|------|
| `register(binding)` | 注册 / 覆盖一个绑定 |
| `set_user_bindings(user_bindings)` | 应用 settings 覆盖 |
| `load_from_settings(settings)` | 从 settings.json 加载 `keybindings` 节 |
| `resolve(key)` → `action_id` | 按键名解析动作 |
| `get_action_key(action_id)` | 查询默认主键 |
| `is_enabled(action_id)` | 动作是否已启用 |
| `all_bindings()` | 全部启用的绑定（供 App 快捷键分发） |

---

## 选择器（selectors.py）

基于 overlay 的选择器（统一走 `OverlayManager`，不再有 ModalScreen）：

| 组件 | 触发方式 | 功能 |
|------|----------|------|
| `ModelSelector` | `Ctrl+L`（`app.model.select`） | 模型选择：按 provider/id/name 分组显示 + 实时搜索 + 键盘导航，`>` 标记当前模型 |
| `SessionPicker` | `--resume` / `app.session.resume` | 会话恢复选择：按修改时间倒序，显示 session id + 时间 |
| `TreeSelector` | `/tree` / `/fork` | 会话树导航 / fork 目标选择（ASCII 树 + 键盘导航，`f` 过滤模式，`t` label 时间戳） |
| `TextInputDialog` | OAuth 回调等 | 通用文本输入弹层（Enter 提交，Esc 取消） |
| `ChoiceSelector` | settings 子项 | 通用选项列表弹层 |
| `SettingsSelector` | `/settings` | 设置菜单（bool/choice/string 三类设置项，落盘项目 `.pi/settings.json`） |
| `TrustSelector` | `/trust` / 启动未信任项目 | 项目信任决策（Trust / Trust parent / Do not trust 等） |
| `ThinkingSelector` | `/thinking` | 思考级别选择 |
| `OAuthSelector` | `/oauth [login|logout]` | OAuth provider 选择（显示登录状态） |
| `ScopedModelsSelector` | `/scoped-models` | 模型范围多选（Enter 切换，Esc 保存） |
| `ExtensionSelector` | `/extensions` | 扩展列表（显示命令/工具数量） |

选择器即 overlay：`push_screen` 桥接到 overlay 层（居中、80% 宽、maxHeight 60%），
dismiss 时移除 overlay 并把焦点恢复到打开前位置。

---

## 主题（theme.py）

56 种命名颜色，覆盖背景、边框、状态、文本、基础色板、Markdown/diff 与 thinking/工具执行语义，内置 dark / light 两套主题：

| 分类 | 键 |
|------|-----|
| 背景 | `bg` `bgAlt` `bgBase` `bgHover` `bgInactive` `bgLoading` `bgPanel` `bgPanelAlt` `bgPrompt` `bgToolbar` `bgUserInput` |
| 边框 | `border` `borderActive` `borderInactive` |
| 状态 | `error` `info` `success` `warning` `accent` `accentMuted` |
| 文本 | `text` `textAlt` `textDim` `textDisabled` `textLight` `textSelected` `textSystem` `textWarning` `dim` |
| 色板 | `black` `red` `green` `yellow` `blue` `magenta` `cyan` `white` |
| Markdown / diff | `markdownHeading` `markdownLink` `diffAdd` `diffRemove` `diffChange` |
| thinking / 工具执行 / bash | `thinkingText` `thinkingOff` `thinkingMinimal` `thinkingLow` `thinkingMedium` `thinkingHigh` `thinkingXhigh` `thinkingMax` `toolPendingBg` `toolSuccessBg` `toolErrorBg` `toolTitle` `toolOutput` `bashMode` |

内置主题为 Catppuccin 风格：`dark`（Mocha）、`light`（Latte）。

### ThemeLoader

| 方法 | 说明 |
|------|------|
| `available()` | 内置 dark / light + `theme_dir/*.json` 自定义主题名 |
| `load(name)` | 加载主题；未知名称抛 `ThemeError` |
| `detect_terminal_background()` | 先 OSC 11 查询，失败回退 `COLORFGBG`（值 `>= 7` 判为 light），未知默认 dark |
| `resolve(name)` | `None` / `"auto"` 时按终端背景自动选择 |

### 自定义 JSON 主题

主题文件须为 JSON 对象且包含全部 56 个颜色键，值为十六进制字符串（`validate_theme_colors` 校验）：

```json
{
    "bg": "#000000",
    "text": "#ffffff"
}
```

缺少键或非 hex 值会抛 `ThemeError`。加载方式：

```python
loader = ThemeLoader(theme_dir="~/.pi/themes")
theme = loader.load("my-theme")
```

---

## 剪贴板图片（clipboard_image.py）

跨平台剪贴板图片读取 + Pillow 处理，对齐 TS `clipboard-image.ts` + `image-process.ts`：

- **读取**：Windows PowerShell（`System.Windows.Forms.Clipboard`）→ macOS `osascript` → Linux `wl-paste`，失败回退 `xclip`；
- **处理**：复用 `pi_agent.tools.image_pipeline`（EXIF 方向校正 → 缩放到 `MAX_IMAGE_DIMENSION=2000` 以内 → 转 PNG，保留 alpha）；
- 超时 10 秒，失败 / 无图片返回 `None`，不抛异常。

```python
import asyncio
from pi_tui.clipboard_image import ClipboardImage

data = await ClipboardImage.read()  # PNG bytes | None
png = ClipboardImage.process(data)  # 规范化 → PNG bytes
```

---

## 与 pi_coding_agent 集成

`pi_tui` 本身不绑定 Agent；`pi_coding_agent.modes.interactive.app` 中的 `PiTuiApp` 负责组装：

- `on_mount()` 挂载 Header / ChatContainer / StatusBar / Editor / Footer；
- `KeybindingsManager` 默认表 + settings 覆盖直接驱动 `App._dispatch_binding`；
- `ThemeLoader.resolve(theme_name)` 生成主题色 → 应用为各组件的 `base_style`；
- 会话事件（`message_end` / `agent_settled` / `compaction_start` / `model_changed` 等）驱动 UI 更新；
- slash 命令经 `SlashCommandRegistry` 执行，模型 / 会话选择器经 `push_screen` 弹出。

```bash
# 通过 CLI 启动 TUI（交互模式）
uv run python -m pi_coding_agent --mode tui
```

---

## 开发

```bash
# 运行全部测试
uv run pytest

# 引擎与 TUI 测试（无头 FakeTerminal，不需要真实 TTY）
uv run pytest src/pi_tui/tests/ -v
uv run pytest src/pi_coding_agent/tests/test_tui_app.py -v
uv run pytest src/pi_coding_agent/tests/test_tui_keybindings.py -v
uv run pytest src/pi_coding_agent/tests/test_tui_theme.py -v
uv run pytest src/pi_coding_agent/tests/test_tui_clipboard_image.py -v
```

## 许可

MIT
