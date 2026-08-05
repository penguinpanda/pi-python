# 环境变量（Python 移植状态）

TS 的环境变量分三类：进程配置、bash 工具会话注入、进程标记。pi-python 目前只实现了一部分；本文列出已实现项，并标注 TS 独有、尚未移植的项。

## 已实现

| 变量 | 作用 | 实现位置 |
| --- | --- | --- |
| `PI_CACHE_RETENTION` | 设为 `long` 时对支持的 provider 使用长提示缓存，否则默认 `short` | `src/pi_ai/utils/prompt_cache.py`（`resolve_cache_retention`） |
| `PI_PACKAGE_DIR` | 覆盖 pi 包根目录（Nix/Guix store 路径等）；默认向上找 `pyproject.toml` 所在目录 | `src/pi_coding_agent/_config.py`（`get_package_dir`） |
| `PI_PROVIDER` / `PI_MODEL` | 仅 `pi_evals` harness 用于选择真实模型（默认 faux provider），不是 CLI 通用配置 | `src/pi_evals/harness.py` |

底层 HTTP 客户端（httpx/openai）按惯例读取 `HTTP_PROXY` / `HTTPS_PROXY` 等代理变量；pi 本身没有额外的代理逻辑。

## 未移植（TS 独有）

- `PI_CODING_AGENT=true` 进程标记：CLI/RPC 入口未设置，子进程无法据此识别自己在 pi 内。
- bash 工具会话注入：`PI_SESSION_ID` / `PI_SESSION_FILE` / `PI_PROVIDER` / `PI_MODEL` / `PI_REASONING_LEVEL` 不会注入 LLM 调用的 bash 工具环境；`createBashTool(exposeSessionEnvironment=...)` / `spawnHook` 也未移植（见 `src/pi_agent/tools/bash.py`）。
- 进程配置：`PI_CODING_AGENT_DIR`、`PI_CODING_AGENT_SESSION_DIR`、`PI_OFFLINE`、`PI_SKIP_VERSION_CHECK`、`PI_TELEMETRY`、`PI_SHARE_VIEWER_URL`、`PI_HARDWARE_CURSOR`、`VISUAL` / `EDITOR` 均未读取。

## 用途

当被问“当前用哪个模型 / provider”时，应查会话状态（`pi_coding_agent._session` 的 `model` / `thinking_level`），而不是依赖环境变量。
