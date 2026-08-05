# Changelog

## [Unreleased]

### Added

- 交互模式支持 `!cmd` / `!!cmd` 本地 shell 命令（对齐 TS）：`!` 执行并进入 LLM 上下文，`!!` 执行但不进上下文，输出流式渲染
- 引入 ruff（lint + format）与 mypy 配置及依赖
- GitHub Actions CI：uv sync → ruff lint/format → pytest（含 PostgreSQL 存储测试）
- `/trust` slash command and CLI startup project-trust resolution (`trust.json` persistence)
- `/changelog` slash command backed by `CHANGELOG.md` parsing
- TUI trust selector and settings selector entry points
- Structured system-prompt builder with project context files (AGENTS.md/CLAUDE.md)
- Turn-level timings and prompt-cache waste statistics in session stats
- Typed SettingsManager with file/in-memory storage, migration, and project-trust gating
- Unified resource loader aggregating skills/prompts/extensions/themes/context files
- Tool-scope constraints against whole-disk searches (read/bash/find/grep guidance)
- Image pipeline (EXIF orientation, resize, multi-format to PNG) wired into read/clipboard
- `pi-protocol` v2 package: commands/results/snapshots/events with JSONL framing
- `pi-storage` PostgreSQL session store (asyncpg) with migrations and tsvector/pg_trgm search
- `pi-server` persistent stdio service with attach/detach and snapshot push
- `pi-evals` harness with smoke and extensions evals (faux provider)
- TUI tool-execution, skill-invocation, compaction/branch-summary message entries
- TUI thinking/oauth/scoped-models selectors and extension selector (`/thinking`, `/oauth`, `/extensions`)

### Changed

- mypy 全仓清零（原 664 个错误）：TypedDict 判别字段基类重构（NotRequired/Literal 收窄）、`NotRequired` 改用 `typing_extensions`、运行时凭证/事件/配置对象的类型收窄；CI 中 mypy 由非阻塞报告改为阻塞检查
- 修复一批运行时隐患：`skills.py`/`prompt_templates.py` 访问不存在的 `.message` 属性（改为 `str(error)`）、TUI `scroll_visible(entry)` 应为 `scroll_to_widget(entry)`、`_CliAuthInteraction` 读取 camelCase 事件键、`openrouter_images` 输出补齐 `url` 键等
- `!command` 配置值改为 `shlex.split` 参数数组执行（不再经 shell，消除命令注入边界）
- 提交 `uv.lock` 锁定依赖；`docs/` 保持不入库；运行时 `.pi/` 目录忽略
- 修复 `app.py` 中 `/reload` 未导入 `Path` 的运行时错误、`env.py` 回调异常被静默吞掉的问题
- 清理 `_agent_loop.py` 游离 docstring 与各类 lint 问题（未使用变量/导入、异常链、zip strict 等）
- 默认 pytest 不再强制开启覆盖率（CI 中显式开启）

## [0.1.0]

### Added

- Unified LLM API layer (`pi-ai`) with providers, auth, models, and streaming
- Agent core loop (`pi-agent`) with tools, retry, compaction, and branching
- Coding agent (`pi-coding-agent`) with print/RPC/TUI modes and slash commands
- Textual-based TUI (`pi-tui`) with selectors, keybindings, and themes
- Docker development environment and session JSONL persistence
