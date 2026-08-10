# TS 示例移植状态

本目录同步自 TS 仓库 `packages/coding-agent/examples/extensions/`。`*.ts` 保留原文；已 Python 化的以 `*.py` 提供。

## 已移植

| TS 文件 | Python 文件 | 说明 |
| --- | --- | --- |
| `hello.ts` | `hello.py` | 最小自定义工具 |
| `notify.ts` | `notify.py` | agent_end 桌面通知 |
| `project-trust.ts` | `project_trust.py` | project_trust 事件 |
| `send-user-message.ts` | `send_user_message.py` | send_user_message |
| `shutdown-command.ts` | `shutdown_command.py` | /quit + 退出工具 |
| `input-transform.ts` | `input_transform.py` | input 事件变换 |
| `status-line.ts` | `status_line.py` | set_status（无主题色） |
| `session-name.ts` | `session_name.py` | 会话名 |
| `event-bus.ts` | `event_bus.py` | 扩展间事件 |
| `timed-confirm.ts` | `timed_confirm.py` | 超时对话框 |
| `trigger-compact.ts` | `trigger_compact.py` | 手动压缩（无自动阈值） |
| `permission-gate.ts` | `permission_gate.py` | tool_call 阻断 |
| `prompt-customizer.ts` | `prompt_customizer.py` | before_agent_start 系统提示覆盖 |
| `model-status.ts` | `model_status.py` | model_select 状态栏 |
| `custom-compaction.ts` | `custom_compaction.py` | session_before_compact 自定义摘要 |
| `dynamic-resources/` | `dynamic_resources.py` | resources_discover 路径形式 |
| `provider-payload.ts` | `provider_payload.py` | provider 请求记录/覆盖 |
| `custom-footer.ts` | `custom_footer.py` | set_footer 自定义页脚 |
| `custom-header.ts` | `custom_header.py` | set_header 自定义页头 |
| `widget-placement.ts` | `widget_placement.py` | set_widget 上下组件 |
| `todo.ts` | `todo.py` | todo 工具 + details 状态恢复 |
| `tools.ts` | `tools.py` | /tools 工具开关 + 持久化 |
| `qna.ts` | `qna.py` | model_registry 问题抽取 |
| `message-renderer.ts` | `message_renderer.py` | 自定义消息渲染 |
| `entry-renderer.ts` | `entry_renderer.py` | 自定义条目渲染 |
| `bookmark.ts` | `bookmark.py` | set_label 书签 |
| `pirate.ts` | `pirate.py` | 动态系统提示 |
| `system-prompt-header.ts` | `system_prompt_header.py` | 系统提示长度状态 |
| `claude-rules.ts` | `claude_rules.py` | 项目规则注入 |
| `protected-paths.ts` | `protected_paths.py` | 敏感路径阻断 |
| `confirm-destructive.ts` | `confirm_destructive.py` | 会话替换确认 |
| `dirty-repo-guard.ts` | `dirty_repo_guard.py` | git 脏仓库保护 |
| `reload-runtime.ts` | `reload_runtime.py` | reload 命令+工具 |
| `auto-commit-on-exit.ts` | `auto_commit_on_exit.py` | 退出自动提交 |
| `git-checkpoint.ts` | `git_checkpoint.py` | stash 检查点 |
| `summarize.ts` | `summarize.py` | 对话摘要 |
| `question.ts` | `question.py` | 用户提问工具（完整对齐） |
| `commands.ts` | `commands.py` | 列出命令 |
| `inline-bash.ts` | `inline_bash.py` | 内联 bash 展开 |
| `input-transform-streaming.ts` | `input_transform_streaming.py` | streaming 感知输入门 |
| `structured-output.ts` | `structured_output.py` | 终止工具结果 |
| `dynamic-tools.ts` | `dynamic_tools.py` | 动态注册工具 |
| `kimi-deferred-tools.ts` | `kimi_deferred_tools.py` | 延迟工具加载 |
| `file-trigger.ts` | `file_trigger.py` | 触发文件注入 |
| `tool-override.ts` | `tool_override.py` | 覆盖内置 read |
| `questionnaire.ts` | `questionnaire.py` | 多问题问卷（完整对齐） |
| `working-indicator.ts` | `working_indicator.py` | 流式指示器 |
| `titlebar-spinner.ts` | `titlebar_spinner.py` | 标题动画 |
| `modal-editor.ts` | `modal_editor.py` | vim 模式编辑器 |
| `rainbow-editor.ts` | `rainbow_editor.py` | 彩虹编辑器 |
| `overlay-test.ts` | `overlay_test.py` | 浮层测试 |
| `hidden-thinking-label.ts` | `hidden_thinking_label.py` | 折叠标签 |
| `interactive-shell.ts` | `interactive_shell.py` | user_bash 覆盖 |
| `bash-spawn-hook.ts` | `bash_spawn_hook.py` | spawn hook |
| `truncated-tool.ts` | `truncated_tool.py` | 输出截断 |
| `working-message-test.ts` | `working_message_test.py` | 工作提示持久化 |
| `mac-system-theme.ts` | `mac_system_theme.py` | 系统主题同步 |
| `preset.ts` | `preset.py` | 命名预设 |
| `handoff.ts` | `handoff.py` | 上下文移交 |
| `git-merge-and-resolve.ts` | `git_merge_and_resolve.py` | 自动 merge/冲突 |
| `built-in-tool-renderer.ts` | `built_in_tool_renderer.py` | 内置工具渲染 |
| `minimal-mode.ts` | `minimal_mode.py` | 最小化工具显示 |
| `ssh.ts` | `ssh.py` | 远程 SSH 执行 |
| `rpc-demo.ts` | `rpc_demo.py` | RPC UI 演示 |
| `border-status-editor.ts` | `border_status_editor.py` | 编辑器状态边框 |
| `with-deps/` | `with_deps.py` + `with_deps_lib/` | 本地依赖解析 |
| `tic-tac-toe.ts` | `tic_tac_toe.py` | 井字棋工具 |
| `subagent/` | `subagent.py` | 独立子代理（简化） |
| `plan-mode/` | `plan_mode.py` + `plan_mode_utils.py` | 只读规划模式（完整对齐） |
| `github-issue-autocomplete.ts` | `github_issue_autocomplete.py` | Tab 自动补全 |
| `snake.ts` | `snake.py` | 覆盖层贪吃蛇（简化） |
| `space-invaders.ts` | `space_invaders.py` | 覆盖层太空侵略者（简化） |
| `custom-provider-anthropic/`、`custom-provider-gitlab-duo/` | `custom_provider.py` | OpenAI 兼容 provider（简化） |
| `overlay-qa-tests.ts` | `overlay_qa_tests.py` | 覆盖层 QA 检查 |

## 待移植（依赖尚未实现的 API）

单文件示例：无（全部已移植或已标注简化移植）。

子目录：`doom-overlay/`、`gondolin/`、`sandbox/`。

## 依赖的未实现能力（按示例分类）

- UI：`set_overlay_component`（组件实例）已实现；`SelectList` / `SettingsList` 已提供；全部选择器（Choice/TextInput/Thinking/Settings/Model/Session/Tree/OAuth/Scoped/Extension/Trust）已迁移到 overlay 层

## 已知限制（非 pi-python 扩展 API 缺口）

- `doom-overlay/`、`gondolin/`、`sandbox/`：依赖外部运行时（WASM 游戏、微 VM、OS 沙箱），需要引入对应外部子系统后才能移植。
- `PI_SKIP_VERSION_CHECK` / `PI_TELEMETRY` / `PI_SHARE_VIEWER_URL` / `PI_HARDWARE_CURSOR`：pi-python 没有版本检查 / 遥测 / share 上传 / 硬件光标对应的功能消费者；当前 Textual 版本也无硬件光标 API，标注 N/A。
- overlay 运行时：`src/pi_tui/overlay/` 已实现 `OverlayManager` + `OverlayFocusController` + `resolve_layout` + `OverlayHandle`（hide / setHidden / focus / unfocus），支持锚点 / margin / 百分比 / maxHeight / nonCapturing / visible / 动画 / z-order / 事件路由；`OverlayWidget` 双模支持行文本与组件子树（`set_overlay_component`），overlay 直接挂 Screen overlay 层；全部选择器已 overlay 化。
