# ~/.pi/agent 目录约定

全局 agent 目录：`~/.pi/agent/`（`PI_CODING_AGENT_DIR` 可覆盖，见
[environment-variables.md](environment-variables.md)）。下表列出 Python
当前已实现 / 已消费的条目，以及仅定义路径约定的占位条目。

| 条目 | 用途 | Python 状态 |
| --- | --- | --- |
| `auth.json` | 各 provider 的 api_key / oauth 凭证 | 已实现（`AuthStorage`） |
| `models.json` | 模型元数据缓存（在线刷新可关闭） | 已实现（模型解析/缓存） |
| `models-store.json` | 动态模型目录缓存（models + etag / lastModified / checkedAt） | 已实现（`FileModelsStore`，CLI/server 默认路径） |
| `settings.json` | 全局配置（双层合并：项目 `.pi/settings.json` 覆盖） | 已实现（`load_settings`） |
| `trust.json` | 项目信任决策记录 | 已实现（`TrustManager` + `resolve_project_trusted`） |
| `sessions/` | 会话存储 | 已实现，per-cwd 布局（见下） |
| `prompts/` | 提示模板（slash command 模板） | 已实现（`get_prompts_dir`） |
| `skills/` | 技能目录 | 已实现（`get_skills_dir`） |
| `themes/` | 用户自定义主题 JSON | 启动时自动创建目录；`ThemeLoader` 尚未消费（内容按需放置） |
| `tools/` | 自定义工具 / 旧版二进制 | 启动时自动创建目录；工具全部内置，未消费 |
| `bin/` | 托管二进制（TS 的 fd/rg 自动解压位置） | 启动时自动创建目录；Python 未自动提取 |
| `pi-debug.log` | 调试日志 | 占位：`get_debug_log_path()` 已定义，Python 尚无日志文件 |

路径函数见 `src/pi_coding_agent/_config.py`：

- `get_agent_dir()` → `~/.pi/agent/`
- `get_sessions_dir()` → `~/.pi/agent/sessions/`
- `get_skills_dir()` / `get_prompts_dir()` / `get_themes_dir()` /
  `get_tools_dir()` / `get_bin_dir()`
- `get_debug_log_path()` → `~/.pi/agent/pi-debug.log`

`ensure_agent_dirs()` 在 CLI/TUI 启动时补齐 `sessions/` `prompts/` `skills/`
`extensions/` `themes/` `tools/` `bin/` 七个约定目录；`auth.json` /
`models.json` / `settings.json` 等文件仍按需懒创建（首次读/写时才生成）。

## sessions/ 布局（与 TS JsonlSessionStore 对齐）

```
~/.pi/agent/sessions/
  └── --<encoded-cwd>--/
        └── <timestamp>_<sessionId>.jsonl
```

- `encoded-cwd`：把 cwd 前导 `/` 或 `\` 去掉，再把 `/` `\` `:` 替换为 `-`，
  最后包上 `--`（例如 `/tmp/proj` → `--tmp-proj--`）。
- 文件名：`{timestamp}_{sessionId}.jsonl`，timestamp 中的 `:` 和 `.`
  替换为 `-`。
- 旧版本平铺文件（`sessions/*.jsonl`）启动/列出时自动迁移到对应
  per-cwd 子目录；读不出 cwd 的文件放入 `--legacy--/`，不丢失。
- 迁移逻辑：`SessionManager.migrate_flat_sessions()`；列出/创建/恢复均
  走 `SessionManager.list_sessions()` / `create()`，调用方无需感知子目录。

## 展示层工具（render_utils）

`src/pi_coding_agent/tools/render_utils.py` 对齐 TS
`core/tools/render-utils.ts`：路径缩短（`shorten_path`）、OSC 8 文件超链接
（`link_path` / `hyperlink`）、文本规范化（`str_value` / `replace_tabs` /
`normalize_display_text` / `strip_ansi`）、图片回退（`get_text_output` /
`get_image_dimensions` / `image_fallback`）与工具路径渲染
（`invalid_arg_text` / `render_tool_path`）。终端能力探测为保守子集：
识别 kitty / Ghostty / WezTerm / Warp / iTerm2 / Windows Terminal / VSCode /
Alacritty；tmux / screen / 未知终端默认关闭 OSC 8。
