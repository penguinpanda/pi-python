# 环境变量（Python 移植状态）

TS 的环境变量分三类：进程配置、bash 工具会话注入、进程标记。pi-python 目前只实现了一部分；本文列出已实现项，并标注 TS 独有、尚未移植的项。

## 已实现

| 变量 | 作用 | 实现位置 |
| --- | --- | --- |
| `PI_CACHE_RETENTION` | 设为 `long` 时对支持的 provider 使用长提示缓存，否则默认 `short` | `src/pi_ai/utils/prompt_cache.py`（`resolve_cache_retention`） |
| `PI_PACKAGE_DIR` | 覆盖 pi 包根目录（Nix/Guix store 路径等）；默认向上找 `pyproject.toml` 所在目录 | `src/pi_coding_agent/_config.py`（`get_package_dir`） |
| `PI_PROVIDER` / `PI_MODEL` | 仅 `pi_evals` harness 用于选择真实模型（默认 faux provider），不是 CLI 通用配置 | `src/pi_evals/harness.py` |
| `PI_CODING_AGENT` | CLI 入口设置 `true`，子进程可据此识别自己在 pi 内 | `src/pi_coding_agent/_cli.py`（`main`） |
| `PI_CODING_AGENT_DIR` | 覆盖全局 agent 目录（默认 `~/.pi/agent`） | `src/pi_coding_agent/_config.py`（`get_agent_dir`） |
| `PI_CODING_AGENT_SESSION_DIR` | 覆盖会话存储目录（`--session-dir` 优先） | `src/pi_coding_agent/_config.py`（`get_sessions_dir`） |
| `PI_OFFLINE` | `1` / `true` / `yes` 时禁止模型目录网络刷新（`allow_model_network=False`） | `src/pi_coding_agent/_cli.py`（`_allow_model_network`） |

底层 HTTP 客户端（httpx/openai）按惯例读取 `HTTP_PROXY` / `HTTPS_PROXY` 等代理变量；pi 本身没有额外的代理逻辑。

## bash 工具会话注入（已实现）

AgentSession 创建工具时给 bash 工具注入会话环境变量（`_session_env_vars`）：

| 变量 | 说明 |
| --- | --- |
| `PI_SESSION_ID` | 当前会话 ID |
| `PI_SESSION_FILE` | 持久化会话文件路径（内存会话不设置） |
| `PI_PROVIDER` / `PI_MODEL` | 当前模型 provider / id |
| `PI_REASONING_LEVEL` | 当前思考级别 |

`pi_coding_agent.tools.create_bash_tool(cwd, session_env_provider=..., expose_session_environment=..., spawn_hook=...)` / `create_all_tools(...)` 可覆盖注入或追加环境变量；`spawn_hook(ctx)` 对齐 TS `createBashTool` 的 spawnHook（[bash.py](C:/coding/AI/Agent/pi-python/src/pi_agent/tools/bash.py)）。`expose_session_environment=False` 关闭注入。

## 未移植（TS 独有）

- 进程配置：`PI_SKIP_VERSION_CHECK`、`PI_TELEMETRY`、`PI_SHARE_VIEWER_URL`、`PI_HARDWARE_CURSOR`、`VISUAL` / `EDITOR` 均未读取。

## 用途

当被问“当前用哪个模型 / provider”时，应查会话状态（`pi_coding_agent._session` 的 `model` / `thinking_level`），而不是依赖环境变量。
