# Changelog

## [Unreleased]

### Added

- DeepSeek 前缀缓存支持（P0–P3）：严格 usage 解析（`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` 原始字段与计费）、DeepSeek 禁用长缓存参数、`compaction.cacheFirst` 稳定截断低价值工具输出、上下文指纹与 `hitRate` 统计、`showCacheMissNotices` TUI 提示
- cache-first 增强：预算驱动尾部剪枝（未超阈值不截）、warm/cold 缓存状态策略（冷缓存跳过剪枝）、DeepSeek 模型自动开启 `compaction.cacheFirst`（显式配置优先）、指纹归因（compaction/system/tools/append）并入 miss 提示、`cacheStats` 增加 `hitTokens`/`hitRate`
- cache-first 剪枝增强：head+tail 截断（只读工具长头短尾、bash 均衡）、最近 16K token 保护尾不剪、截断前归档原始输出、warm/cold TTL 校准为 10 分钟
- pi-evals harness 选项支持 `compaction_settings` / `cache_first` / `show_cache_miss_notices`，使 eval 测试可走真实缓存路径
- extensions_eval 增加 `default-system-prompt-cache-first` 对照（`cache_first=True`），量化缓存策略收益
- 新增 `long_session_cache_eval.py`：单会话 5 轮长会话 eval（大文件读入触发剪枝阈值），对比 cache-first 开关的 tokens / latency / cost
- 新增 `scripts/check.py` 与 pre-commit 配置，本地一键复现 CI 的 ruff/mypy/pytest 检查
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
- `pi-evals` 完整移植 TS `packages/evals`：`createPiCodingAgentHarness`
  （隔离工作区 / transform / output / session 快照）、vitest-evals 等价物
  （judge / harness table / artifacts / summary）、`pi-evals` CLI runner，
  附 smoke / extensions evals（默认 faux provider）
- `pi-evals` harness 新增 `thinking_level` 选项（对齐 TS `thinkingLevel`，
  显式值 > `PI_REASONING_LEVEL` 环境变量 > 默认 `off`），支持 `max` 等推理
  强度；非法值报错，实际级别按模型支持范围 clamp
- TUI tool-execution, skill-invocation, compaction/branch-summary message entries
- TUI thinking/oauth/scoped-models selectors and extension selector (`/thinking`, `/oauth`, `/extensions`)
- 内置 TUI 引擎 `src/pi_tui/engine/`：单元格渲染与行差分、终端输入解析
  （UTF-8/CSI/SS3/kitty/SGR 鼠标/paste/OSC）、raw/alt-screen 终端驱动、
  组件树（Editor vim/undo、ScrollView、SelectList/SettingsList、Loader、Markdown）、
  App 基类（事件循环、焦点、overlay 合成、快捷键分发）；`FakeTerminal` 无头测试
- TUI 终端协议：OSC 11 背景色、OSC 52 剪贴板、OSC 133 prompt、OSC 2026 同步输出、
  `PI_HARDWARE_CURSOR` 硬件光标、SGR 鼠标滚轮与滚动条
- 新增 `docs/tui-ts-feature-gap.md`：面向界面对齐 TS 的逐项功能差距与实施建议（布局/聊天/编辑器/终端协议/鼠标/设置接线，含里程碑与验证方式）
- `AgentOptions` 新增 `prepare_next_turn_with_context`（接收 `PrepareNextTurnContext`：message / tool_results / context / new_messages）以及 `thinking_budgets` / `transport` 透传字段
- 新增 `pi_agent._messages.convert_to_llm` / `pi_coding_agent.messages`：应用层完整消息转换（bashExecution / compactionSummary / branchSummary / custom 包装为 user 消息）
- CLI / `pi_server` 默认用 `~/.pi/agent/models-store.json` 持久化动态模型目录缓存
  （`FileModelsStore`：models + etag / lastModified / checkedAt，跨进程复用条件刷新）

### Changed

- **TUI 彻底移除 Textual**：`PiTuiApp` 与全部组件/选择器改到内置引擎之上，
  依赖改为显式 `rich>=13.0`；`uv.lock` 移除 textual 及其传递依赖
- mypy 全仓清零（原 664 个错误）：TypedDict 判别字段基类重构（NotRequired/Literal 收窄）、`NotRequired` 改用 `typing_extensions`、运行时凭证/事件/配置对象的类型收窄；CI 中 mypy 由非阻塞报告改为阻塞检查
- 修复一批运行时隐患：`skills.py`/`prompt_templates.py` 访问不存在的 `.message` 属性（改为 `str(error)`）、TUI `scroll_visible(entry)` 应为 `scroll_to_widget(entry)`、`_CliAuthInteraction` 读取 camelCase 事件键、`openrouter_images` 输出补齐 `url` 键等
- `!command` 配置值改为 `shlex.split` 参数数组执行（不再经 shell，消除命令注入边界）
- 提交 `uv.lock` 锁定依赖；`docs/` 保持不入库；运行时 `.pi/` 目录忽略
- 修复 `app.py` 中 `/reload` 未导入 `Path` 的运行时错误、`env.py` 回调异常被静默吞掉的问题
- 清理 `_agent_loop.py` 游离 docstring 与各类 lint 问题（未使用变量/导入、异常链、zip strict 等）
- 默认 pytest 不再强制开启覆盖率（CI 中显式开启）
- `AgentHarness.compact()` / `navigate_tree()` 接入 DAG Session 与 `pi_agent.compaction` / `branch_summarization`，不再抛 `not_implemented`
- TUI 界面按差距文档补齐：启动资源独立容器、多行可展开 header、队列消息显示区、
  状态 spinner 动画、编辑器边框与 padding、工具执行条目（可展开）、补全下拉 overlay、
  OSC133 prompt 滚动（ctrl+shift+up/down）、kitty 键盘协议协商、焦点事件 `?1004h`、
  OSC 9;4 终端进度、终端标题更新、OSC8 链接点击打开、拖选自动滚动、
  图片宽度设置接线（showImages/imageWidthCells）、clearOnShrink 选项、
  regular 主屏模式（uiMode 设置）、滚动条 hover 高亮、颜色方案通知解析、
  autocompleteMaxVisible/quietStartup 接线、Markdown 标题/代码主题着色、
  /debug /arminsayshi /dementedelves 彩蛋命令、ScrollView.scroll_by overscroll 语义、
  drop files 提示（与 TS 一致）
- 修复退出挂起：POSIX 输入读取改用 select 超时，避免退出时 `asyncio.run` 等待阻塞的读线程，
  shell 提示符在退出后立即出现（真实 tmux 冒烟验证 fullscreen/regular 两种模式）
- 编辑器补齐 `ctrl+shift+Home/End` 选区到文档首/尾（对齐 TS）
- 输入解析补齐 `wantsKeyRelease` 接口：kitty release 事件带 `Key.release` 标记，
  仅分发给声明 `wants_key_release` 的组件（对齐 TS isKeyRelease 过滤）
- 扩展 UI API 补齐 `setWorkingVisible` / `setWorkingIndicator` / `pasteToEditor` /
  `getEditorText` / `editor`（抽象接口 + Noop + TuiUIContext 实现）
- 文档统一移除“自研”表述
- `turn_start` 改为由 `run_agent_loop` / `run_agent_loop_continue` 外层发射（先于 prompts 注入，对齐 TS agent-loop.ts），`_run_loop` 首轮不再重复发射
- `pi_agent` 默认 `convert_to_llm` 改为最小过滤（只透传 user/assistant/toolResult，对齐 TS `defaultConvertToLlm`）；压缩/分支摘要等丰富转换移至应用层转换器，`AgentHarness` 与 `AgentSession` 已接线
- TUI 大内容量渲染优化：`MessageEntry` 跨帧缓存（natural_size / render 按内容版本失效）、
  布局合成整行复用 + 共享行写时复制、regular 模式按行对象同一性增量 diff 且只对变化行
  转 ANSI；1000 条消息的逐帧渲染从数百毫秒降至 ~10-20ms，输入/退出不再被渲染阻塞

### Fixed

- TUI 布局对齐 TS：容器分配保持挂载顺序，聊天区固定在编辑器上方（1fr），
  编辑器/状态栏/页脚固定在底部，提交消息后输入框不再被挤出可视区；
  启动资源提示随聊天区显示在输入框上方
- TUI 退出行为：退出 alt-screen 后清空主屏视口，只保留 shell 提示符，
  不再残留 TUI 的 header / status / footer 文档
- 修复滚动视口子组件 `app` 未传播导致流式占位（Speaking）无法移除的问题；
  `message_start` 仅对 assistant 创建流式占位，user/custom 消息由 `message_end` 追加
- 输入解析：分片 CSI/OSC 序列等待补齐而非被 final flush 丢弃；kitty release
  与未知 CSI 序列被增量消费
- 编辑器新增输入历史（对齐 TS）：提交后记录，上下键召回，上限 100 条
- 编辑器补齐 TS emacs/alt 绑定：`ctrl+b/f` 左右移、`alt+b/f` 词导航、
  `alt+backspace` / `alt+d` 删词、`alt+y` yank pop、`ctrl+-` undo、
  `ctrl+]` / `ctrl+alt+]` 字符跳转；`\x1f` 解析为 `ctrl+-`（对齐 TS keys.ts）
- TUI 注册 SIGINT/SIGTERM/SIGHUP 优雅退出，进程被杀时恢复终端
- Windows：`GetConsoleScreenBufferInfo` 尺寸读取字段修正（srWindow Right/Bottom），
  控制台输入模式关闭行缓冲与回显（对齐 raw 模式）
- TUI 默认把硬件光标定位到输入光标处并显示（对齐 TS CURSOR_MARKER），
  Windows IME 候选窗口跟随输入位置而不是停在最右侧；regular 模式同样生效，
  `PI_HARDWARE_CURSOR=0` 可关闭；编辑器光标列按可见宽度计算（CJK/emoji 不偏移）
- TUI 退出后清空主屏，不再回写最后一帧文档（原先对齐 TS 的退出行为会残留
  header / status / footer，导致 shell 提示符上方堆满 TUI 内容）

## [0.1.0]

### Added

- Unified LLM API layer (`pi-ai`) with providers, auth, models, and streaming
- Agent core loop (`pi-agent`) with tools, retry, compaction, and branching
- Coding agent (`pi-coding-agent`) with print/RPC/TUI modes and slash commands
- Textual-based TUI (`pi-tui`) with selectors, keybindings, and themes
- Docker development environment and session JSONL persistence
