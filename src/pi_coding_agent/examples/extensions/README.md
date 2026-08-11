# Extension 示例（Python 移植）

pi-python 的扩展是 `.py` 模块，导出 `create_extension(api)`（或 `factory(api)`）工厂函数，同步 / 异步均可。本目录同时保留 TS 原示例（`*.ts`）作为对照。

## 用法

```bash
# 复制到全局或项目扩展目录，启动 / /reload 时自动加载
cp hello.py ~/.pi/agent/extensions/
cp hello.py .pi/extensions/          # 项目需先被信任
```

CLI 暂未提供 TS 的 `--extension` / `-e` 标志；扩展由 `ResourceLoader` 发现（子目录入口为 `index.py` / `pi_extension.py`）。

## 已移植示例

| 文件 | 说明 |
| --- | --- |
| `hello.py` | 最小自定义工具（`register_tool`） |
| `notify.py` | `agent_end` 时桌面通知（OSC 777 / OSC 99 / Windows toast） |
| `project_trust.py` | `project_trust` 事件：自定义信任选择 |
| `send_user_message.py` | `/ask` `/steer` `/followup`：`send_user_message` |
| `shutdown_command.py` | `/quit` + 退出工具：`ctx.shutdown()` |
| `input_transform.py` | `input` 事件：`?quick` 变换、`ping`/`time` 即时处理 |
| `status_line.py` | `turn_start` / `turn_end` 时 `ctx.ui.set_status` |
| `session_name.py` | `/session-name`：设置 / 读取会话名 |
| `event_bus.py` | 扩展间通信：`pi.events` |
| `timed_confirm.py` | 带超时的 `confirm` / `select` 对话框 |
| `trigger_compact.py` | `/trigger-compact`：手动触发压缩 |
| `permission_gate.py` | `tool_call` 事件：危险 bash 命令确认/阻断 |
| `prompt_customizer.py` | `before_agent_start` 事件：动态补充系统提示 |
| `model_status.py` | `model_select` 事件：状态栏显示模型变化 |
| `custom_compaction.py` | `session_before_compact`：自定义压缩摘要 |
| `dynamic_resources.py` | `resources_discover`：动态提供技能/模板 |
| `provider_payload.py` | `before/after_provider_request`：记录/覆盖请求 |
| `custom_footer.py` | `set_footer`：切换自定义页脚（token/模型统计） |
| `custom_header.py` | `set_header`：替换页头为 pi 吉祥物文本 |
| `widget_placement.py` | `set_widget`：编辑器上/下方组件 |
| `overlay_demo.py` | `set_overlay`：锚点/边框/动画浮层演示 |
| `todo.py` | `todo` 工具 + `/todos`：状态经工具 details 随分支恢复 |
| `tools.py` | `/tools`：开关工具并持久化到会话条目 |
| `qna.py` | `/qna`：`model_registry` 抽取问题写入编辑器 |
| `message_renderer.py` | `register_message_renderer` + `send_message` |
| `entry_renderer.py` | `register_entry_renderer` + `append_entry` |
| `bookmark.py` | `/bookmark` / `/unbookmark`：`set_label` |
| `pirate.py` | `/pirate`：before_agent_start 动态系统提示 |
| `system_prompt_header.py` | agent_start 状态栏显示系统提示长度 |
| `claude_rules.py` | 扫描 `.claude/rules/` 并注入系统提示 |
| `protected_paths.py` | `tool_call` 阻断敏感路径写入 |
| `confirm_destructive.py` | session_before_switch/fork 确认 |
| `dirty_repo_guard.py` | git 脏仓库时阻止切换/分支 |
| `reload_runtime.py` | `/reload-runtime` + 工具排队 follow-up |
| `auto_commit_on_exit.py` | session_shutdown 自动 git 提交 |
| `git_checkpoint.py` | turn_start stash + fork 恢复 |
| `summarize.py` | `model_registry` 摘要写入编辑器 |
| `question.py` | `question` 工具：select/input 询问用户 |
| `commands.py` | `/commands`：`get_commands` 列出命令 |
| `inline_bash.py` | `input` 事件展开 `!{command}` |
| `input_transform_streaming.py` | `streamingBehavior=steer` 跳过预处理 |
| `structured_output.py` | `terminate: true` 工具结束回合 |
| `dynamic_tools.py` | 运行时动态注册工具 |
| `kimi_deferred_tools.py` | 搜索并按需激活工具 |
| `file_trigger.py` | 轮询触发文件并注入消息 |
| `tool_override.py` | 同名覆盖内置 `read`（审计+阻断） |
| `questionnaire.py` | 多问题问卷工具 |
| `working_indicator.py` | 状态栏流式指示器 |
| `titlebar_spinner.py` | 终端标题 braille 动画 |
| `modal_editor.py` | `set_editor_component` vim 模式编辑器 |
| `rainbow_editor.py` | `set_editor_component` 彩虹动画编辑器 |
| `overlay_test.py` | `set_overlay` 边缘用例演示 |
| `hidden_thinking_label.py` | `set_hidden_thinking_label` 折叠标签 |
| `interactive_shell.py` | `user_bash` operations 覆盖 |
| `bash_spawn_hook.py` | `create_bash_tool` spawn_hook + 覆盖 bash |
| `truncated_tool.py` | `rg` 工具输出截断 + 临时文件 |
| `working_message_test.py` | `set_working_message` 持久化测试 |
| `mac_system_theme.py` | `set_theme` 跟随 macOS 深色/浅色 |
| `preset.py` | `/preset`：加载 presets.json 并应用 |
| `handoff.py` | `/handoff`：生成新会话聚焦提示 |
| `git_merge_and_resolve.py` | agent_end 自动 merge + 冲突上报 |
| `minimal_mode.py` | `register_tool_renderer` 最小化工具显示 |
| `built_in_tool_renderer.py` | read/bash/edit/write 紧凑渲染 |
| `ssh.py` | `user_bash` operations 走远程 SSH |
| `rpc_demo.py` | RPC UI 方法演示（select/confirm/input/...） |
| `border_status_editor.py` | 编辑器边框显示模型/思考状态 |
| `with_deps.py` | 扩展目录加入 sys.path，import 本地依赖 |
| `tic_tac_toe.py` | 无共享状态的井字棋工具 |
| `subagent.py` | 进程内独立子代理（/subagent） |
| `plan_mode.py` | `/plan` 只读规划模式切换 |
| `github_issue_autocomplete.py` | `register_autocomplete`：Tab 补全 gh issues |
| `snake.py` | set_overlay 贪吃蛇（命令方向控制） |
| `space_invaders.py` | set_overlay 太空侵略者（命令控制） |
| `custom_provider.py` | 环境变量配置 OpenAI 兼容 provider |
| `overlay_qa_tests.py` | set_overlay 锚点/边框/动画 QA 检查 |

全部 TS 示例的移植状态见 [STATUS.md](STATUS.md)。

## 扩展 API 速查

```python
from pi_coding_agent import ExtensionAPI, ToolDefinition


def create_extension(pi: ExtensionAPI):
    @pi.on("message_end")
    async def on_message_end(event, ctx):
        pass

    pi.register_tool(
        ToolDefinition(
            name="greet",
            label="Greeting",
            description="Generate a greeting",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            execute=lambda tool_call_id, params, signal, on_update, ctx: {
                "content": [{"type": "text", "text": f"Hello, {params['name']}!"}],
                "details": {},
            },
        )
    )

    pi.register_command(
        "hello",
        {
            "description": "Say hello",
            "handler": lambda ctx, args: ctx.ui.notify("Hello!", "info"),
        },
    )
```

完整文档见 [docs/extensions.md](../../docs/extensions.md)。

注意：pi-python 的扩展命令 handler 签名为 `(ctx, args)`（与 TS 的 `(args, ctx)` 相反）。
