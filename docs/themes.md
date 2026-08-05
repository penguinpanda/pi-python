# 主题（Python 移植）

主题是定义 TUI 命名颜色的 JSON 文件。实现：`src/pi_tui/theme.py`（`ThemeLoader`、`Theme`、`validate_theme_colors`），发现逻辑在 `src/pi_coding_agent/resource_loader.py` 的 `_load_themes`。

## 位置

- 内置：`dark`（Catppuccin Mocha 系）、`light`（Latte 系）。
- 全局：`~/.pi/agent/themes/*.json`
- 项目：`.pi/themes/*.json`（项目信任后）
- settings 的 `themes` 数组：文件或目录（`ThemeLoader(directory).load(name)`）

## 选择主题

settings.json：

```json
{
  "theme": "my-theme"
}
```

`theme` 缺省或为 `"auto"` 时自动选择：根据 `COLORFGBG` 环境变量判断终端背景（Windows 终端默认深色，未知默认 `dark`）。

## 主题格式

JSON 对象，必须包含 `src/pi_tui/theme.py` `COLOR_KEYS` 里的**全部**颜色键，且每个值都是 `#rrggbb` 十六进制字符串：

```json
{
  "bg": "#1e1e2e",
  "bgAlt": "#181825",
  "accent": "#89b4fa",
  "text": "#cdd6f4",
  "diffAdd": "#a6e3a1",
  "diffRemove": "#f38ba8"
}
```

缺少键或非十六进制值会抛 `ThemeError`，主题不加载。

## 颜色键

约 43 个语义键，分几类（完整列表见 `COLOR_KEYS`）：

- 背景：`bg`、`bgAlt`、`bgBase`、`bgHover`、`bgInactive`、`bgLoading`、`bgPanel`、`bgPanelAlt`、`bgPrompt`、`bgToolbar`、`bgUserInput`
- 边框：`border`、`borderActive`、`borderInactive`
- 状态：`error`、`info`、`success`、`warning`、`accent`、`accentMuted`
- 文本：`text`、`textAlt`、`textDim`、`textDisabled`、`textLight`、`textSelected`、`textSystem`、`textWarning`、`dim`
- 基础色板：`black` / `red` / `green` / `yellow` / `blue` / `magenta` / `cyan` / `white`
- Markdown / diff：`markdownHeading`、`markdownLink`、`diffAdd`、`diffRemove`、`diffChange`

`Theme.css_variables(prefix="pi")` 把它们转成 `--pi-<key>` CSS 变量供 Textual CSS 模板使用。

## 未移植（TS 独有）

- `vars` 变量引用、`$schema`、`export`（HTML 导出配色）段、51 个 token 结构。
- 编辑活动主题文件时的自动热重载。
