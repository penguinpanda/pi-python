# TS pi-tui 与 Python pi_tui 差距分析（下一步开发路线）

> 状态：路线图（Phase 1-3）已全部完成；剩余条目均为可选或明确建议维持 Textual
> 边界的低优先级项（见各节“建议”）。

> 对比基准：TS 主仓库 `packages/tui`（HEAD `4b85cd978`）与 pi-python `src/pi_tui`
> （Textual 8.2.8 之上）。本文面向“下一步开发”，先讲现状，再列差距，最后给阶段路线。

## 1. 一句话结论

TS pi-tui 是**自研终端渲染器 + 组件树 + overlay 焦点协议**；Python pi_tui 是
**Textual 应用 + 自研 overlay 运行时**。2026-08-05 的 overlay 改造后，两者在
“overlay 生命周期 / 布局 / 焦点归属”三个核心语义上已经对齐大部分；
剩余差距集中在四块：**组件树 API、焦点边界语义、通用组件库、终端级能力**。

## 2. 架构对照

| 维度 | TS pi-tui | Python pi_tui（现在） |
| --- | --- | --- |
| 渲染引擎 | 自研：`Component.render(width)` 生成文本行，差分写入终端，支持主屏/alt-screen | Textual compositor（CSS 布局、鼠标、resize 全由 Textual 负责） |
| 组件模型 | `Component` / `Container` 树，每个组件可渲染、可聚焦（`Focusable`） | Textual `Widget` 树 |
| Overlay | `showOverlay(component, options)` + `OverlayStackEntry` + `OverlayHandle` | `OverlayManager` + `OverlayLayer` + `OverlayHandle` |
| Overlay 布局 | `resolveOverlayLayout`：anchor / margin / 百分比 / 绝对 row/col / offset / maxHeight | `resolve_layout()` 纯函数，语义一致 |
| 焦点 | `TuiBase.setFocus` + eligible/blocked/inactive + preFocus 链 + `isOverlayFocusAncestor` + `CURSOR_MARKER` | `OverlayFocusController` 三态（active/blocked/inactive）+ `DescendantFocus` 同步，无 eligible 专有状态 |
| 输入 | 自解析终端 stdin（kitty 协议、key release/repeat、input listener） | Textual 键盘事件分发 |
| 终端能力 | 硬件光标、OSC11 背景色查询、颜色方案通知、kitty/iTerm2 图像、OSC8 链接、同步输出、OSC133、鼠标选择、滚动条 | 无这些查询/协议；由 Textual 提供基础鼠标/滚动/alt-screen |
| 模态 | 所有选择器都是 overlay（同一栈） | `ModalScreen` push/pop（独立于 overlay 焦点协议） |
| 布局系统 | 自研 `layout.ts`（LayoutFrame、ScrollView、scrollbar 几何） | Textual CSS layout |
| 编辑器 | `Editor`：vim 模式、undo stack、kill ring、word navigation、自定义 `EditorComponent` | `PiEditor(TextArea)`：Enter 提交 / Shift+Enter 换行 / Tab 补全 / ctrl+d / ctrl+x / shift+tab，无 vim 模式 |
| 通用组件 | Box / HStack / VStack / Spacer / Text / TruncatedText / Input / SelectList / SettingsList / ScrollView / Loader / CancellableLoader / Markdown / Image / AltScreenFlash | MessageEntry / BashExecutionEntry / PiHeader / PiChatContainer / PiEditor / PiStatusBar / PiFooter / PiToolbar（+ Textual 内置组件） |
| 测试 | tui 单元测试 + overlay 回归测试 | `src/pi_tui/tests/`（布局 / 焦点 / manager 纯单测）+ `test_tui_app.py`（Textual pilot 集成） |

## 3. 已对齐 / 已实现（现状基线）

### 3.1 Overlay 生命周期与布局

- `OverlayLayout / OverlayStyle / OverlayBehavior` 三类职责分离（对应 TS 的 options 拆分）。
- `resolve_layout()` 纯函数：anchor 九宫格、margin（数字/四边）、百分比与绝对 row/col、
  offset、maxHeight、minWidth、边界 clamp；含随机不变量测试。
- `OverlayHandle`：hide / setHidden / focus / unfocus(target) / isFocused / isHidden。
- z-order：`focusOrder` 单调递增，focus()/setHidden(false) 会置顶（Textual 无 `z` 样式，
  通过异步重挂载到 `OverlayLayer` 末尾实现）。
- `visible(w,h)` 回调 + resize 重排 + 焦点重定向。
- `nonCapturing`：不抢焦点、不参与事件路由。
- 事件路由：`manager.handle_event` 从最上层可见 capturing overlay 向下冒泡到基座。

### 3.2 焦点

- `OverlayFocusController` 三态状态机：inactive / active / blocked。
- `active` 等价 TS 的 `eligible`：焦点暂时离开 overlay 时，下一次输入自动恢复；
  组件模式判断“焦点仍在 overlay 子树内”时不做无谓恢复。
- preFocus 记录与链式回退（`retarget_pre_focus`），移除上层 overlay 时焦点回到
  topmost visible 或 preFocus。
- blocked + `RestoreMode.OVERLAY / TARGET`：显式 unfocus(target) 后，blocked_by 失焦时
  跳到目标；焦点在基座间移动时 blocked_by 跟随更新。
- 与 Textual 真实焦点同步：`DescendantFocus` → `manager.on_widget_focused`；
  输入前 `route_input()` 做恢复检查。
- ModalScreen：Textual 屏幕栈原生保存每屏焦点；`pop_screen` 后追加
  `route_input()`，让选择器关闭后立即与 overlay 焦点状态机对齐。
- 轻量选择器已 overlay 化：`ChoiceSelector` / `TextInputDialog` / `ThinkingSelector` /
  `SettingsSelector` 继承 `OverlayDialog`（Widget），`PiTuiApp.push_screen` 对这四个
  类自动桥接到 `OverlayManager`（居中、80% 宽、maxHeight 60%），dismiss 移除 overlay
  并回调结果。
- 重型选择器（Model / Session / Tree / OAuth / Scoped / Extension / Trust）也已
  迁移到 `OverlayDialog`：全部选择器统一走 overlay 栈，pi 自身不再使用 ModalScreen。
- TreeSelector 树过滤模式（对齐 TS）：`f` 循环 default / no-tools / user-only /
  labeled-only / all；default 隐藏 label/custom/model_change/thinking_level_change/
  session_info 记账条目，标题显示当前模式。
- TreeSelector label 时间戳（对齐 TS `[+label time]`）：`t` 开关，标签行显示本地
  HH:MM:SS；`SessionTreeNode.label_timestamp` 由 `get_tree` 从 label 条目填充。

### 3.3 接入面

- `ctx.ui.set_overlay(key, lines, options)`：行文本 overlay 全集（anchor/margin/offset/
  百分比/maxHeight/border/title/nonCapturing/visible/animate）。
- `ctx.ui.set_overlay_component(key, widget, options)`：组件树 overlay——`OverlayWidget`
  双模（行文本 / 组件），组件模式复用根节点、焦点落到子树内第一个可聚焦组件，
  同一 key 可在行文本与组件间切换。
- `set_widget`（editor 上/下）、`set_footer/set_header`、`set_editor_component`、
  message/markdown/entry/tool 渲染器接线。
- RPC/Print 上下文：`RpcUiContext` 已补 setFooter/setHeader/setWidget/setOverlay 等请求；
  Print 走 Noop。

## 4. 差距清单

### A. Overlay 组件树 API（已基本完成）

**TS 做法**：`showOverlay(component)` 接受任意 Component 子树，overlay 内容可以是
Box/VStack/HStack/Text/Input/SelectList 的组合，渲染成文本行后由
`compositeOverlays` 合成。

**Python 现状**：`OverlayWidget(Static)` 双模（行文本 / 组件）；扩展侧已有
`set_overlay_component(key, widget, options)`，组件模式复用根节点、焦点落到子树。

**剩余建议**：
1. 渲染回调 API（`set_overlay_renderer(key, fn(width, height) -> list[str])`）可选。
2. `handle_event` 已保留钩子，等通用组件（SelectList 等）落地后自然生效。
3. 补 Tab 循环、set_component 换子树的集成测试。

### B. 焦点边界语义（已补齐核心，剩余低优先级）

**TS 做法**：`OverlayFocusRestoreState` 有 `eligible`（overlay 曾持焦、焦点暂时离开）
和 `blocked`（焦点被基座组件卡住）两种恢复状态；输入时如果焦点不在 blocked_by 上会
自动夺回 overlay；支持 `wantsKeyRelease`；preFocus 沿 `isOverlayFocusAncestor` 链判断
嵌套 overlay 的恢复归属。

**Python 现状**：`active` 即 TS eligible（输入时焦点不在 overlay 子树内则自动恢复）；
`blocked_by` 跟随当前基座焦点，blocked_by 卸载/焦点清空时按 RestoreMode 恢复；
组件模式支持“焦点在子树内”判断。无 `wantsKeyRelease`（低优先级，Textual 键盘
语义下意义不大）。

**剩余建议**：补 `wantsKeyRelease`（仅当需要按键释放语义的组件时）。

### C. ModalScreen 与 overlay 统一（已完成：轻量选择器已迁移）

**TS 做法**：模型/会话/设置等选择器全部走 `showOverlay`，天然共享 overlay 栈、
焦点恢复和 z-order。

**Python 现状**：全部选择器（Choice/TextInput/Thinking/Settings/Model/Session/
Tree/OAuth/Scoped/Extension/Trust）都已改为 `OverlayDialog`（Widget）并通过
`push_screen` 桥接挂进 `OverlayLayer`——选择器即 overlay，焦点 / z-order /
关闭恢复全部走同一套管理器；不再有 ModalScreen。

**建议**：无（已全部迁移）。

### D. 通用组件库（进行中：SelectList/SettingsList 已完成）

**TS 有而 Python 之前没有**：

- `SelectList` / `SettingsList`：已实现（`src/pi_tui/lists.py`）——SelectList 支持
  前缀/子串/子序列模糊筛选、上下键导航、Enter 选择 / Escape 取消；SettingsList
  支持 label + 当前值两列、Enter 循环取值；ChoiceSelector / ThinkingSelector 已
  改为复用 SelectList（保留 ModalScreen 与 action_select/action_cancel 兼容）。
- `ScrollView`（自研滚动 + 滚动条）、`Loader` / `CancellableLoader`：未移植。
- `Markdown`：已实现（`src/pi_tui/markdown.py`，基于 rich.markdown）——标题 /
  列表 / 代码块（语法高亮）/ 表格 / 链接；`MessageEntry` 渲染 label + Markdown 正文，
  扩展 transformer 输出 Rich markup 时保留原样避免二次转义。`Image`（终端图像协议）未移植。
- `Box` / `HStack` / `VStack` / `Spacer` / `Text` / `TruncatedText`：Textual 内置
  Vertical/Horizontal/Static 覆盖，未单独移植。
- `Editor` 的 vim 模式：已实现 `PiEditorVim(PiEditor)`（Esc 切换 normal/insert、
  h/j/k/l、0/$、i/a/o、dd、x、u，Enter 提交语义与 PiEditor 一致，可作为
  `set_editor_component` 的替换编辑器）；undo stack / kill ring / word navigation
  未单独移植（TextArea 自带 undo）。

**剩余建议**：
1. `ScrollView` 行为如果 Textual 的 `VerticalScroll` 不满足再移植；
2. `Image`（终端图像协议）未移植（建议维持 Textual 边界）。

### E. 终端级能力（低优先级，建议维持 Textual 边界）

**TS 独有**：`PI_HARDWARE_CURSOR`、`queryTerminalBackgroundColor`（OSC11）、
颜色方案通知、kitty/iTerm2 图像传输、OSC8 链接点击、alt-screen 合成、
同步输出（OSC 2026）、OSC133 prompt、鼠标拖选 + 滚动条拖拽。

**Python 现状**：全部 N/A（Textual 内部实现鼠标/alt-screen/OSC8 链接的部分能力，
但没有硬件光标、OSC 查询和图像协议）。

**建议**：不追求对齐。若未来需要：先做 `PI_HARDWARE_CURSOR`（Textual 无 API，
需在渲染层拦截），图像协议等 Textual 生态支持后再评估。已在
`examples/extensions/STATUS.md` 标注 N/A。

### F. 输入 / 按键系统（自动补全 provider 栈已完成）

TS 有 `keys.ts`（kitty 协议解析、key release/repeat 过滤）、`autocomplete.ts`
（provider 栈）、`fuzzy.ts`、`kill-ring.ts`、`undo-stack.ts`、`word-navigation.ts`。
Python 有 `KeybindingsManager` + Textual 键盘事件；自动补全 provider 栈已实现
（`src/pi_tui/autocomplete.py` 的 `CombinedAutocompleteProvider`）：多 provider
并发收集、按 value 去重、保持注册顺序、支持同步/异步、单个异常跳过；Tab →
`PiEditor.AutocompleteRequested` → 异步收集 → overlay 选择器 → 插入。

**剩余**：kitty 协议解析 / key release / kill-ring / word-navigation 依赖 Textual
键盘栈，不必移植。

### G. 渲染 / 文本工具（低优先级）

TS `utils.ts` 有 `visibleWidth` / `sliceByColumn` / `wrapTextWithAnsi` /
`stripTerminalSequences` / `getOsc8LinkAtColumn` 等。Python 依赖 Textual/Rich 的
文本测量与 ANSI 处理，overlay 合成由 Textual 完成，不需要自研。

**建议**：仅当实现组件树 API 时需要“按列截断”工具时再补。

## 5. 下一步开发路线

### Phase 1：Overlay 组件树 API（已完成）

- `OverlayWidget` 双模（行文本 / 组件），组件模式复用根节点（避免 Textual 异步
  remove 撞同 id）。
- 扩展 API：`ctx.ui.set_overlay_component(key, widget, options)`；TUI 返回
  `OverlayHandle`，RPC 发送 `setOverlayComponent` 意图请求，Print 走 Noop。
- 焦点：组件模式聚焦子树内第一个可聚焦组件；`manager.ensure_focus` 处理子组件
  异步挂载后的焦点落位。
- 测试：manager 单测 + Textual pilot 集成（挂载 / 聚焦 / 切换行文本 / 移除）。

### Phase 2：焦点语义与 ModalScreen 统一（已完成）

- `active` 覆盖 TS eligible 语义：输入时焦点不在 overlay 子树内则自动恢复；
  组件子树焦点不触发无谓恢复（`_is_same_or_descendant`）。
- `pop_screen` 后 `route_input()`：ModalScreen 关闭后立即与 overlay 焦点状态机对齐
  （Textual 屏幕栈原生保存每屏焦点）。
- 边界测试：active 恢复 / 子树焦点 / blocked_by 卸载（焦点清空）/ ModalScreen
  打开-关闭后焦点回到 overlay 或编辑器。

### Phase 3：通用组件移植 + 选择器迁移（已完成）

- ✅ `SelectList`（fuzzy + 键盘导航）→ `SettingsList`（循环取值）。
- ✅ 轻量选择器（Choice/TextInput/Thinking/Settings）迁移到 `OverlayLayer`
  （`OverlayDialog` + `push_screen` 桥接），关闭后焦点回到打开前位置。
- 可选：`Markdown` 渲染对齐、`PiEditorVim`。

### Phase 4：终端能力（可选，按需）

- 硬件光标 / OSC 查询 / 图像协议，或明确长期保持 Textual 边界并在文档中记录。

## 6. 涉及文件

TS 参考：

- `packages/tui/src/tui.ts`（TUI 接口、OverlayOptions、焦点协议、compositeOverlays）
- `packages/tui/src/tui-alt-screen.ts` / `tui-main-screen.ts`（渲染器、鼠标、图像）
- `packages/tui/src/components/`（通用组件）
- `packages/tui/src/keys.ts` / `autocomplete.ts` / `layout.ts` / `terminal-image.ts`

Python 现状：

- `src/pi_tui/overlay/`（model / layout / focus / manager / widgets）
- `src/pi_coding_agent/modes/interactive/app.py`（PiTuiApp 接线）
- `src/pi_coding_agent/modes/interactive/ui_context.py`（扩展 API）
- `src/pi_tui/components.py` / `selectors.py` / `keybindings.py` / `theme.py`
- `docs/tui.md`（用户文档）、`examples/extensions/STATUS.md`（示例状态）
