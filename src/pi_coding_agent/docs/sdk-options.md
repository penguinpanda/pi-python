# create_agent_session 选项说明

`create_agent_session(options)` 是 `pi-coding-agent` 的 SDK 入口，对齐 TS
`packages/coding-agent/src/core/sdk.ts` 的 `CreateAgentSessionOptions`。
它把模型运行时、资源加载器、会话管理器、工具集、扩展 runner 和
`AgentSession` 组装成一个可直接使用的会话对象。

## 返回值

`create_agent_session` 返回 `CreateAgentSessionResult`：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `session` | `AgentSession` | 组装完成的会话，可直接 `prompt()` / `continue_()` |
| `extensions_result` | `ResourceLoadResult` | 已加载的技能、提示模板、扩展、主题、诊断信息 |
| `model_fallback_message` | `str \| None` | 会话模型恢复失败时的回退说明 |

## 选项表

### 目录与配置

| 选项 | 类型 | 作用 |
| --- | --- | --- |
| `cwd` | `str` | 项目工作目录；工具和项目资源都基于它解析。默认 `"."` |
| `agent_dir` | `str \| None` | 全局配置目录；用于读取 `settings.json` / `auth.json` / `models.json` 和全局资源。默认使用 `~/.pi/agent` |
| `settings_manager` | `SettingsManager \| None` | 双层设置管理器；未传时自动按 `cwd` + `agent_dir` 创建 |
| `resource_loader` | `DefaultResourceLoader \| None` | 统一资源加载器；未传时自动创建并 `reload()` |
| `model_runtime` | `ModelRuntime \| None` | 模型/认证运行时；未传时用内置 provider 创建 |
| `session_manager` | `SessionManagerLike \| None` | 会话持久化管理器；未传时使用内存 v4 会话 |

### 模型与思考

| 选项 | 类型 | 作用 |
| --- | --- | --- |
| `model` | `Model \| None` | 显式指定模型；未传时优先恢复已有会话模型，否则按设置默认值或第一个可用模型选择 |
| `thinking_level` | `ModelThinkingLevel \| None` | 思考级别；未传时恢复会话级别或设置默认值，最终按模型能力 clamp |
| `scoped_models` | `list[ScopedModel] \| None` | 模型循环列表；传入后 `get_available_models()` 优先使用该列表 |

### 工具

| 选项 | 类型 | 作用 |
| --- | --- | --- |
| `tools` | `list[str] \| None` | 工具 allowlist；只启用列出的内置工具名 |
| `exclude_tools` | `list[str] \| None` | 工具 denylist；在 allowlist 之后移除 |
| `no_tools` | `"all" \| "builtin" \| None` | `"all"` 不启用任何内置工具；`"builtin"` 关闭内置工具但保留自定义工具 |
| `custom_tools` | `list[ToolDefinition \| AgentTool] \| None` | 额外注册的自定义工具；`ToolDefinition` 会自动归一化为 `AgentTool` |

### 运行时扩展与持久化

| 选项 | 类型 | 作用 |
| --- | --- | --- |
| `turn_retry_policy` | `RetryPolicy \| None` | turn 级重试策略 |
| `compaction_settings` | `CompactionSettings \| None` | 自动压缩设置 |
| `skill_loader` | `SkillLoader \| None` | 技能加载器；未传时复用资源加载器 |
| `template_loader` | `PromptTemplateLoader \| None` | 提示模板加载器；未传时复用资源加载器 |
| `extension_runner` | `ExtensionRunner \| None` | 扩展 runner；未传时用已加载扩展创建并执行资源发现 |
| `system_prompt_builder` | `Callable[[], str] \| None` | 系统提示重建回调；`/reload` 等场景使用 |
| `extension_state` | `dict \| None` | CLI/TUI 共享的扩展状态；未传时自动创建 `active_tools` |
| `session_start_event` | `dict \| None` | 会话启动事件附加元数据 |
| `system_prompt` | `str \| None` | 显式系统提示；未传时使用资源加载器生成的提示 |

## 选择流程

1. 创建 `SettingsManager` / `ModelRuntime` / `ResourceLoader` / `SessionManager`。
2. 如果已有会话，从 `model_change` 条目恢复模型；恢复失败时记录 `model_fallback_message`。
3. 仍未选到模型时，按设置默认 provider/model 或第一个可用模型选择。
4. 思考级别按 显式选项 > 会话历史 > 设置默认值 解析，再用 `clamp_thinking_level()` 收敛。
5. 工具集按 `tools` / `exclude_tools` / `no_tools` 过滤，追加 `custom_tools`。
6. 组装 `Agent` 与 `AgentSession`，并把模型/思考级别写入新会话。

## 示例

```python
import asyncio

from pi_coding_agent import CreateAgentSessionOptions, create_agent_session
from pi_coding_agent.extensions import ToolDefinition
from pi_agent import AgentToolResult


async def main() -> None:
    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=".",
            tools=["read", "bash"],
            exclude_tools=["write", "edit"],
            custom_tools=[
                ToolDefinition(
                    name="hello",
                    description="Say hello",
                    execute=lambda *args, **kwargs: AgentToolResult(
                        content=[{"type": "text", "text": "hello"}]
                    ),
                )
            ],
            session_start_event={"source": "sdk"},
        )
    )
    session = result.session
    await session.prompt("list the project")


asyncio.run(main())
```

## 与 TS 的差异

- `AgentSessionRuntime` 类本身未移植；其选项集已由 `create_agent_session` 对齐。
- 默认 `session_manager` 使用内存 v4 会话，便于无文件副作用使用；需要持久化时显式传
  `create_session_manager(cwd)` 或已打开的会话管理器。
