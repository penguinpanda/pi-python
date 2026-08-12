# Changelog

## [Unreleased]

### Breaking Changes

- CLI 命令 `pi` 更名为 `pi-python`（`pi-ai` / `pi-evals` 保持不变）

### Added

- Agent 内联 `AI_TELEMETRY_SCHEMA` / `HARNESS_TELEMETRY_SCHEMA`
  （`pi_agent/telemetry_schema.py`，Schema 不再依赖 `pi_telemetry` 定义）
- `TaggedError` 基类与 TS harness 对应具体错误子类，支持 `match/case` 与
  `to_json()`（`pi_agent._harness_types`）
- Session v4 新增 `reducer.py`：record log 12 类 corruption 校验与
  `reduce_lane_state()` lane 状态归约（对齐 TS `harness/reducer.ts`）
- `pi_agent` 新增独立 `format_skills_for_system_prompt()`，对齐 TS
  `harness/system-prompt.ts` 的 `<available_skills>` XML 格式化
- 新增 Google Generative AI / Google Vertex / Mistral / Azure OpenAI / OpenAI Codex / AWS Bedrock 核心 API 与 provider
- AWS Bedrock 支持 SigV4 与 bearer token；Cloudflare Workers AI / AI Gateway 自定义认证；OpenAI Codex deferred fetch/cancel
- OpenAI 兼容 completions provider 支持动态 `/models` 发现（Groq/Together/Cerebras/Fireworks/xAI/NVIDIA/HuggingFace/Baseten/Moonshot/Xiaomi/Z.ai/OpenCode/Xiaomi Token Plan 等）
- `Models` 新增 `get_auth` / `check_auth` / `get_available` / `login` / `logout`；OpenRouter 与 GitHub Copilot 支持 API key + OAuth 双认证
- 新增 `pi_telemetry` 包、`TerminalProtocol`/`ProcessTerminal`、TUI 子系统独立模块、`TruncatedText`、`EditorComponent`
- 新增 parity/golden 测试目录与 `create_agent_session` SDK 入口
- `create_agent_session` 补齐 TS `CreateAgentSessionOptions` 选项集：`agent_dir` /
  `settings_manager` / `resource_loader` / 工具 allowlist-denylist / `no_tools` /
  `custom_tools` / 会话模型恢复 / `session_start_event`，返回
  `CreateAgentSessionResult`
- Mistral Conversations 补齐 TS 一致的 9 位字母数字 tool call ID 规范化
  （含冲突避让），经 `chat_completions_stream` 可选 normalizer 接入
- Google Generative AI 补齐 strict tool sampling（Gemini 3+ `VALIDATED`）与
  Gemini 3 / Gemma 4 advanced thinking 配置（thinkingLevel / thinkingBudget）
- Google Vertex 支持 Application Default Credentials 回退（`google-auth` +
  `requests`，未显式传 token 时自动刷新）
- OpenAI Codex SSE 对齐 `/codex/responses` URL，并支持 zstd 请求体压缩
  （`zstandard` level 3，transport 层按 `content-encoding: zstd` 压缩）
- OpenAI Codex websocket transport 接入 `responses_stream` 处理管线
  （`responses_websockets=2026-02-06`，transport=websocket/websocket-cached/auto）
- AWS Bedrock 支持 shared credentials file 回退
  （`~/.aws/credentials` / `AWS_SHARED_CREDENTIALS_FILE`，含 profile/session token）
- Azure OpenAI Responses 测试补齐真实 Responses 流事件解析（text/usage/completed）
- 移植 TS 内置 `llama.cpp` 扩展：注册 OpenAI 兼容 provider 与 `/llama` 命令，
  CLI 启动时自动加载（完整下载/加载 UI 保持 TS 独有）
- `scripts/check.py` 覆盖率扩展到全部 9 个 src 包（含 `pi_telemetry`）
- 新增 `sdk-options.md`：`create_agent_session` 完整选项、作用与选择流程说明
- 新增 `scripts/check.py` 与 pre-commit 配置，本地一键复现 CI 的 ruff/mypy/pytest 检查
- 交互模式支持 `!cmd` / `!!cmd` 本地 shell 命令：`!` 执行并进入 LLM 上下文，`!!` 执行但不进上下文，输出流式渲染
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
- `pi-evals` harness 新增 `thinking_level` 选项（ `thinkingLevel`，
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
- 新增 `docs/tui-ts-feature-gap.md`：面向界面 的逐项功能差距与实施建议（布局/聊天/编辑器/终端协议/鼠标/设置接线，含里程碑与验证方式）
- `AgentOptions` 新增 `prepare_next_turn_with_context`（接收 `PrepareNextTurnContext`：message / tool_results / context / new_messages）以及 `thinking_budgets` / `transport` 透传字段
- 新增 `pi_agent._messages.convert_to_llm` / `pi_coding_agent.messages`：应用层完整消息转换（bashExecution / compactionSummary / branchSummary / custom 包装为 user 消息）
- CLI / `pi_server` 默认用 `~/.pi/agent/models-store.json` 持久化动态模型目录缓存
  （`FileModelsStore`：models + etag / lastModified / checkedAt，跨进程复用条件刷新）
- 工具 `promptGuidelines` 支持：内置 read/edit/write/bash 携带与 TS v0.84.0 一致的
  指南，扩展 `ToolDefinition` 支持 `prompt_guidelines` / `source_info`，
  `get_all_tools` 返回新字段
- 扩展 API 补齐：`ctx.ui.custom`（自定义交互组件）、`ToolDefinition.execution_mode`
  与 `render_call`/`render_result`、CLI 扩展 flags 两段解析、
  `before_agent_start` message 注入、`send_message` 的 `deliverAs`/`triggerTurn`
- 新增 `scripts/verify_web_search.py`：离线验证 DeepSeek Responses 服务端
  web_search 的请求构造 / web_search_call 捕获 / stateless 回放，`--live`
  可选真实调用（需 `DEEPSEEK_API_KEY` 或 `--api-key`）
- 补齐 TS 文档：从 pi TS 仓库复制 `packages/coding-agent/docs` 中 Python
  缺失的文档到 `src/pi_coding_agent/docs/`（原样保留，含 `docs.json` 与
  `images/`），已有中文移植的同名文档不覆盖
- Responses 对齐 TS：请求显式 `store:false`、启用 reasoning 时请求
  `reasoning.encrypted_content`（终态回填）、assistant 消息写入
  `text_signature`、支持 `reasoning_summary` / `refusal.delta` /
  `include_system_prompt`，并接入 deferred tools（tool_search 协议）
- 新增 Qwen Token Plan provider（阿里云百炼 Token Plan，OpenAI 兼容模式）：
  国际站 `qwen-token-plan`（`QWEN_TOKEN_PLAN_API_KEY`）与中国站
  `qwen-token-plan-cn`（`QWEN_TOKEN_PLAN_CN_API_KEY`），模型目录经
  `scripts/generate_models.py` 从 TS 上游数据生成（16 个 tool-capable 模型/站，
  含 qwen3.x / deepseek / kimi / glm / minimax），并注册默认模型 `qwen3.7-max`
- Session v4 新增后端无关 conformance 工厂
  `create_session_backend_conformance()`（`pi_agent.session.v4.testing`），
  InMemory / JSONL / PostgreSQL 后端复用同一组一致性用例
- JSONL v4 仓库支持注入 `JsonlSessionRepoFileSystem`（默认
  `LocalFileSystem`）与 `JsonlSessionRepoOptions` 构造，`JsonlSessionStorage`
  新增 `drain()`；公开导出 `JsonlSessionCreateOptions` /
  `JsonlSessionListOptions` / `JsonlSessionMetadata` / `JsonlV4Header`
- `/resume` Session Selector 对齐 TS：recent / threaded / fuzzy 排序、
  named / all 过滤、current / all scope、path display、rename、
  delete confirmation（`trash` 优先，`unlink` 回退）
- Interactive Mode 统一 autocomplete provider：内置命令、prompt templates、
  extension commands、skills、`/model`、`/login`、fd/readdir 路径补全统一走
  编辑器内联渲染，扩展 autocomplete 不再弹模态选择器
- 扩展命令支持 `getArgumentCompletions`（兼容 `getArgumentCompletions`
  camelCase 注册键），参数补全透传到统一 autocomplete provider
- Session Picker 快捷键并入 `KeybindingsManager` 独立 action 命名空间，
  支持 settings `keybindings` 覆盖与禁用
- 新增托管工具下载/缓存（`tools/_ensure_tool.py`）：fd/rg 缺失时按平台
  从 GitHub 下载并缓存到 `~/.pi/agent/bin`，带重试；`PI_OFFLINE` 下跳过
- `/model` 补全对齐 TS：按 id / provider / name 模糊搜索，不再使用显式
  aliases 或自动派生去日期后缀 id
- 差距审查（`docs/nd_upload/0812/code_review/pi-python-vs-ts-gap-report.md`）
  批次闭合：TUI 启动 changelog 通知（`lastChangelogVersion` 记录 +
  `collapseChangelog` 设置）、Azure base-URL 归一化（Azure 主机强制
  `/openai/v1`）、`/hotkeys` 分组表格输出、对话框可见倒计时
  （`auto-cancel in Xs`）、`/session` usage 分组明细
  （`get_usage_cost_breakdown` 按 provider/model 归组）
- 设置选择器键族补全：transport / steeringMode / followUpMode / theme /
  retryEnabled / httpIdleTimeoutMs / hideThinkingBlock / showCacheMissNotices /
  quietStartup / collapseChangelog / defaultThinkingLevel，新增
  `set_global_setting`（globalSettings 分层写入）
- Codex WebSocket 连接失败自动回退 SSE（`_CodexWsConnectError` + 会话级
  `websocketSseFallbackSessions` 记忆）；流开始后失败抛
  `_CodexWsStreamError`（对齐 TS 不回退语义）
- AWS Bedrock：`additionalModelRequestFields.thinking`（budget 型 +
  `thinking_display`）、toolResult 连续消息合并为单条 user 消息与 image
  内容转换、`cachePoint` prompt caching 注入（`_supports_prompt_caching`
  + `AWS_BEDROCK_FORCE_CACHE`）
- 会话树选择器折叠/展开（ctrl+left/right + `⊞`/`⊟` 标记）与 shift+l
  label 编辑（`set_label`）
- ScopedModels 选择器：ctrl+a 全选 / ctrl+x 全清 / ctrl+p 切换 provider /
  alt+up/down 重排 / ctrl+s 持久化到 `settings.scopedModels`（选中顺序即
  模型循环顺序）
- find/ls 工具响应取消信号（`_aborted` → `Operation aborted` 结果）
- 扩展 UI context 补齐 `getTheme` / `getAllThemes` / `onTerminalInput` /
  `getContextUsage`（输入 hook 可消费事件）
- legacy auth 一次性迁移（`migrate_auth_to_auth_json`：oauth.json +
  settings.json apiKeys → auth.json 0600，CLI 启动执行）与 TUI CSI 16t
  cell 尺寸查询
- 包管理子命令：`pi install/remove/update`（npm:/git:/local 源，
  user/project scope，settings.packages 持久化）
- PI_EXPERIMENTAL 门控与 first-time setup 自动触发（官方发行 + 默认 agent
  目录 + settings.json 不存在）+ enableAnalytics 同意询问
- OAuth 浏览器流程：OpenAI Codex PKCE loopback（端口 1455 + 手动粘贴回退 +
  device-code 404 自动切换）、OpenRouter loopback（ephemeral 端口 + 回调内
  exchange + 手动粘贴竞争）、xAI device-code（SuperGrok/X Premium 订阅）
- Radius 网关 provider（pi-messages + `GET /v1/config` 动态 catalog）与
  订阅 OAuth（网关发现 + loopback + `/v1/oauth/token`）
- 远程模型目录 overlay（`with_remote_catalog`：ETag/Last-Modified 条件请求、
  4h 新鲜度窗口、304/404/501 语义），CLI 创建 runtime 时叠加到内置 provider
- mermaid 代码块终端图渲染：内置 flowchart TD/LR + sequenceDiagram 渲染器，
  `off|final|streaming` 三模式 transformer（宽度检查、警告、thinking 跳过），
  `mermaidRenderingMode` 设置
- SDK 导出 `AgentSessionRuntime`（session/cwd/diagnostics 访问器 +
  `set_rebind_session`）

### Changed

- 对齐 TS 工具管理：移除 fd/rg 下载的版本标记文件与
  `PI_FD_PATH` / `PI_RG_PATH` 外部路径覆盖；`@` 路径补全在 fd 缺失或
  失败时返回空，不再回退 readdir
- `list_sessions` 移除本地 JSON search index，始终扫描会话文件（对齐 TS）
- `/model` 补全移除显式 aliases 与去日期后缀 id 自动派生，搜索字段与 TS
  `getModelSelectorSearchText` 一致
- 会话运行时工厂固定走 v4：移除 `PI_SESSION_FORMAT=v3` 运行时回退，旧 v3
  文件由 v4 仓库始终惰性迁移；`pi_evals` harness 与 `pi_server` 会话工厂
  迁移到 v4
- 移除旧 v3 会话实现（`session.py` / `memory.py` / `repo.py` / `search.py` /
  `types.py` / `jsonl.py`）；v3 读取器收拢为 `pi_agent.session.v4.v3_reader`，
  仅用于惰性迁移
- Azure OpenAI Responses 默认 `apiVersion` 对齐 TS `v1`（`AZURE_OPENAI_API_VERSION` 可覆盖）
- `AgentHarness` 公开 API  legacy（0.84.0 之前）：`models: Models` + `session: Session`
  必填并移除 `stream_fn`；新增 `request_shutdown()` / `wait_for_shutdown()`、
  compaction/branch-summary `retry` 与 retry 事件、`before_provider_payload` /
  `after_provider_response` hooks、`NavigateOptions` 完整导航选项、泛型工具/资源类型与
  `Result` 辅助；`shutdown()` 保留为兼容别名
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
- 编辑器补齐 `ctrl+shift+Home/End` 选区到文档首/尾
- 输入解析补齐 `wantsKeyRelease` 接口：kitty release 事件带 `Key.release` 标记，
  仅分发给声明 `wants_key_release` 的组件（ isKeyRelease 过滤）
- 扩展 UI API 补齐 `setWorkingVisible` / `setWorkingIndicator` / `pasteToEditor` /
  `getEditorText` / `editor`（抽象接口 + Noop + TuiUIContext 实现）
- 文档统一移除“自研”表述
- `question.py` / `questionnaire.py` / `plan_mode.py` 示例完整对齐 TS：
  编号选项与描述、自定义输入 Esc 返回、tab bar 问卷、bash 白名单、
  Plan 提取与 `[DONE:n]` 进度、会话恢复
- `turn_start` 改为由 `run_agent_loop` / `run_agent_loop_continue` 外层发射（先于 prompts 注入， agent-loop.ts），`_run_loop` 首轮不再重复发射
- `pi_agent` 默认 `convert_to_llm` 改为最小过滤（只透传 user/assistant/toolResult， `defaultConvertToLlm`）；压缩/分支摘要等丰富转换移至应用层转换器，`AgentHarness` 与 `AgentSession` 已接线
- TUI 大内容量渲染优化：`MessageEntry` 跨帧缓存（natural_size / render 按内容版本失效）、
  布局合成整行复用 + 共享行写时复制、regular 模式按行对象同一性增量 diff 且只对变化行
  转 ANSI；1000 条消息的逐帧渲染从数百毫秒降至 ~10-20ms，输入/退出不再被渲染阻塞
- DeepSeek 模型元数据统一从 `models/generated` 生成目录加载（移除手写
  `DEEPSEEK_MODELS`）；`openai-completions` 的 DeepSeek thinking 参数：
  未指定 effort 时显式发送 `thinking.type=disabled`，map 缺失或为 `null` 的级别
  按原值透传 `reasoning_effort`
- `deepseek-v4-flash` 切换到 OpenAI Responses API：`responses.py` 补齐
  `instructions` / `max_output_tokens` / `reasoning` / `incomplete` / `failed` /
  usage 缓存与推理明细；DeepSeek 官方 Responses 默认启用服务端 `web_search`
  （可显式关闭），并支持 `web_search_call` stateless 回放
- bash 子进程环境：先删除 PI_SESSION_ID / PI_SESSION_FILE / PI_PROVIDER /
  PI_MODEL / PI_REASONING_LEVEL 再按需注入，并把 pi bin 目录前置到 PATH；
  激活工具变化（扩展注册 / `set_active_tools`）会重建系统提示
- 技能系统对齐 TS v0.84.0：frontmatter 改用完整 YAML 解析（PyYAML），
  gitignore 匹配改用 pathspec（完整 gitignore 语义），harness 层补齐 symlink
  解析与 TS 路径语义；显式技能路径补齐非 .md 警告与来源归属
- `examples/` 与 `docs/` 迁入 `src/pi_coding_agent/`（对齐 TS
  `packages/coding-agent/{examples,docs}`）：`get_package_dir()` 默认指向
  包目录，系统提示中的 docs/examples/README 路径随包分发；新增
  `examples/README.md` 索引；`docs/` 中 TS 没有的 `agent-directory.md` 与
  `docs/nd_upload/` 保留在仓库根原位置
- 会话接入 JSONL v4（`pi_agent.session.v4`）：多 lane / 全局 seq / facts /
  自包含 compaction；新会话默认写 v4，旧 v3 文件打开时惰性转换并保留 `.bak`，
  `PI_SESSION_FORMAT=v3` 可回退；`V4SessionManager` 与 `SessionManager` API
  对齐并接入 CLI / RPC / TUI / server
- v4 会话写入 operation records（run / compaction / navigation 的开始与结束 +
  usage），`findOpenOperations` 支持挂起恢复检测；`/input` 编辑改为 v4 原生
  追加改写（合并后的 user 消息 + 移动 lane，旧条目保留）
- `AgentSession` 新增挂起恢复入口（`recovery_state` / `open_operations` /
  `resume_suspended_operation`：重放挂起 run 的原始 prompt）；v3
  `SessionManager` 标记为 legacy，默认会话统一走 v4（`PI_SESSION_FORMAT=v3`
  仅作调试回退）
- `AgentSession` 把 assistant / compaction / branch_summary 的真实 usage 写入
  v4 usage records，`get_session_stats`（v4）反映实际 token 与成本
- `V4SessionManager` 写入改为增量缓存更新（不再每次全量重读会话），
  大会话下避免 O(n²) 开销
- deferred responses 基础层：`DeferredHandle` 类型、`split_deferred_tools`、
  Provider/Models 的 `fetch_deferred` / `cancel_deferred`（faux 参考实现）、
  `AgentSession.fetch_deferred` / `cancel_deferred` 与 `write_deferred` 记录
- `pi_storage` 新增 PostgreSQL v4 会话后端（对齐 TS sqlite-node）：
  `PostgresV4SessionRepo` / `PostgresV4SessionStorage` / `PgSessionSearch`，
  含 lanes / records / lane_moves / facts / branch cache / session_stats /
  writer lease（TTL 30s + 心跳 + fence）；`V4SessionManager.from_repo` /
  `open_with_repo` 与工厂 `repo` 参数接入，`AgentSession.dispose` 释放连接

### Fixed

- Markdown 表格换行单元格保留补位空格，换行后的 `│` 与首行/边框保持对齐
- Markdown 表格在终端宽度过窄时不再变成空行：按 token 行号从源文本回退
  到原始 Markdown 表格（对齐 TS `token.raw` 回退）
- `_CloudflareAuth` 补 `resolve`，修复默认 ModelRuntime/CLI/evals 启动时
  `AttributeError`（Cloudflare provider 只有 `resolve_auth`）
- `model_to_dict` 处理 `cost=None`，修复 `get_available_models` 在包含
  无成本模型时序列化失败
- 修复 DeepSeek Responses 多轮 400 “reasoning_text must be passed back”：
  流式下 reasoning item 的 `content` 为空时，用已累积的 reasoning delta
  文本补全回放项，确保后续轮次把 reasoning_text 原样传回
- 修复 `test_cli.py` 对 `_cli.SessionManager` 的引用：改为 patch
  `create_session_manager` 工厂（配合 CLI 默认 v4 会话接线）
- `Agent._run_prompt` / `_run_continue` 将 context 快照构造移入 try/finally，
  构造异常时也能复位运行状态并置位 `_settled`（ `runWithLifecycle`）；
  运行中 `prompt()` / `continue_()` 的 `RuntimeError` 消息补充
  `steer()` / `follow_up()` / 等待完成提示
- TUI 布局：容器分配保持挂载顺序，聊天区固定在编辑器上方（1fr），
  编辑器/状态栏/页脚固定在底部，提交消息后输入框不再被挤出可视区；
  启动资源提示随聊天区显示在输入框上方
- TUI 退出行为：退出 alt-screen 后清空主屏视口，只保留 shell 提示符，
  不再残留 TUI 的 header / status / footer 文档
- 修复滚动视口子组件 `app` 未传播导致流式占位（Speaking）无法移除的问题；
  `message_start` 仅对 assistant 创建流式占位，user/custom 消息由 `message_end` 追加
- 输入解析：分片 CSI/OSC 序列等待补齐而非被 final flush 丢弃；kitty release
  与未知 CSI 序列被增量消费
- 编辑器新增输入历史：提交后记录，上下键召回，上限 100 条
- 编辑器补齐 TS emacs/alt 绑定：`ctrl+b/f` 左右移、`alt+b/f` 词导航、
  `alt+backspace` / `alt+d` 删词、`alt+y` yank pop、`ctrl+-` undo、
  `ctrl+]` / `ctrl+alt+]` 字符跳转；`\x1f` 解析为 `ctrl+-`（ keys.ts）
- TUI 注册 SIGINT/SIGTERM/SIGHUP 优雅退出，进程被杀时恢复终端
- Windows：`GetConsoleScreenBufferInfo` 尺寸读取字段修正（srWindow Right/Bottom），
  控制台输入模式关闭行缓冲与回显（对齐 raw 模式）
- TUI 默认把硬件光标定位到输入光标处并显示（ CURSOR_MARKER），
  Windows IME 候选窗口跟随输入位置而不是停在最右侧；regular 模式同样生效，
  `PI_HARDWARE_CURSOR=0` 可关闭；编辑器光标列按可见宽度计算（CJK/emoji 不偏移）
- TUI 退出后清空主屏，不再回写最后一帧文档（原先 的退出行为会残留
  header / status / footer，导致 shell 提示符上方堆满 TUI 内容）
- 修复 TUI 编辑器软件光标（反色块）在 CJK/emoji 下按可见列索引单元格导致
  每输入一个宽字符光标多空一列的问题，改为按字符格定位；硬件光标位置不变
- 修复 Responses 流在 `response.completed` 时把 `output_text` 在 toolCall 后
  重复追加的问题：回放时 function_call 与 function_call_output 之间会插入
  多余 message item，DeepSeek 报 `No tool output found for tool call` 并导致
  后续所有轮次卡死；`transform_messages` 同时对旧会话中重复的 text 块去重
- 修复 TUI 消息正文含绝对路径时整体退化为纯文本的问题：现在始终走 markdown
  渲染，路径仍通过 `linkify_lines` 生成 OSC 8 链接
- 修复 PostgreSQL v4 会话后端 conformance 问题：`get_log` 保留 `entry_seq`
  列、fork 初始化 `session_sequences` 并补齐默认 `parentSessionId`，
  `session_sequences` 写入改为 upsert

## [0.1.0]

### Added

- Unified LLM API layer (`pi-ai`) with providers, auth, models, and streaming
- Agent core loop (`pi-agent`) with tools, retry, compaction, and branching
- Coding agent (`pi-coding-agent`) with print/RPC/TUI modes and slash commands
- Textual-based TUI (`pi-tui`) with selectors, keybindings, and themes
- Docker development environment and session JSONL persistence
