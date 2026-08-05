# 快捷键（Python 移植）

实现：`src/pi_tui/keybindings.py`（`KeybindingsManager` + `DEFAULT_APP_KEYBINDINGS`）。pi-python 只实现了**应用级动作**的默认键位表；编辑器光标移动、选择列表、树导航等按键由 Textual 原生处理，未做成 TS 那样的大表。

## 配置方式

在 settings.json 的 `keybindings` 节覆盖（不是 TS 的 `~/.pi/agent/keybindings.json` 文件）：

```json
{
  "keybindings": {
    "app.model.select": "ctrl+m",
    "app.thinking.cycle": ["shift+tab", "ctrl+t"]
  }
}
```

- 单键字符串 / 多键数组都支持。
- `null`（或不设置）→ 保持默认。
- 空列表 `[]` → 禁用该动作。
- 编辑后 `/reload` 生效（`modes/interactive/slash_commands.py` 的 `_reload`）。

## 默认动作表

| action id | 默认键 | 说明 |
| --- | --- | --- |
| `app.interrupt` | `escape` | 取消 / 中止 |
| `app.clear` | `ctrl+c` | 清空编辑器 |
| `app.exit` | `ctrl+d` | 编辑器为空时退出 |
| `app.thinking.cycle` | `shift+tab` | 循环思考级别 |
| `app.thinking.toggle` | `ctrl+t` | 展开 / 折叠 thinking 块 |
| `app.model.select` | `ctrl+l` | 打开模型选择器 |
| `app.model.cycleForward` | `ctrl+p` | 切换到下一个模型 |
| `app.model.cycleBackward` | `shift+ctrl+p` | 切换到上一个模型 |
| `app.tools.expand` | `ctrl+o` | 展开 / 折叠工具输出 |
| `app.message.followUp` | `alt+enter` | 排队 follow-up 消息 |
| `app.message.dequeue` | `alt+up` | 把排队的消息恢复到编辑器 |
| `app.message.copy` | `ctrl+x` | 复制最后一条 assistant 消息 |
| `app.session.new` | `ctrl+n` | 新会话 |
| `app.session.resume` | `ctrl+r` | 恢复会话选择器 |
| `app.editor.external` | `ctrl+g` | 用外部编辑器编辑当前输入 |
| `app.clipboard.pasteImage` | Windows `alt+v`，其他 `ctrl+v` | 粘贴剪贴板图片 |

## 按键格式

`modifier+key`：`ctrl` / `shift` / `alt` 可组合（如 `ctrl+shift+x`、`alt+ctrl+x`）；key 支持 `a-z`、`0-9`、方向键 / 功能键（`f1`-`f12`）、`escape`、`enter`、`tab`、`space`、`backspace`、`delete`、`home`、`end`、`pageUp` / `pageDown` 以及符号键。

## 未移植（TS 独有）

- `~/.pi/agent/keybindings.json` 独立文件与旧 id 自动迁移。
- 编辑器 / 选择器 / 树导航 / 全屏 alt-screen 的整套键位表（`tui.editor.*`、`tui.select.*`、`app.tree.*`、`app.altScreen.*`、`app.models.*` 等）。
