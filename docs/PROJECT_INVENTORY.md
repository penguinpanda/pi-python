# PROJECT_INVENTORY

> 依据 `docs/nd_upload/0812/code_review/review.md` 第一阶段生成的资产盘点。
> 统计口径：`src/<pkg>` 下非 `__pycache__` 的 Python 文件；测试为 `tests/test_*.py`。

## 规模总览（2026-08）

| 包 | 文件数 | 代码行数 | 测试文件数 | 职责 |
| --- | --- | --- | --- | --- |
| `pi_ai` | 170 | 33,124 | 68 | 统一 LLM API 层：models/provider/auth/streaming（responses/completions/pi-messages 等管线） |
| `pi_agent` | 86 | 25,532 | 35 | Agent 核心循环：tools/env/retry/compaction/harness/session v4 |
| `pi_coding_agent` | 202 | 42,445 | 63 | 应用层：CLI（print/RPC/TUI 模式）、sessions、tools、extensions、skills、settings、包管理、远程目录 overlay |
| `pi_tui` | 69 | 17,271 | 32 | 独立 TUI 框架：engine（渲染/输入/布局）+ 组件/选择器/主题（无 Textual 依赖） |
| `pi_protocol` | 5 | 1,108 | 2 | v2 协议：commands/results/snapshots/events + JSONL framing |
| `pi_storage` | 6 | 2,976 | 2 | PostgreSQL 会话存储（asyncpg + 迁移 + 搜索） |
| `pi_server` | 5 | 937 | 1 | stdio 持久服务（attach/detach + snapshot push） |
| `pi_evals` | 49 | 4,697 | 3 | 评测框架：harness/judge/artifacts/summary + CLI |
| `pi_telemetry` | 5 | 252 | 1 | telemetry schema 与上报 |

**合计**：~597 文件 / ~128k 行 / 207 个测试文件；全量 pytest ~2397 passed。

## 入口

- CLI：`python -m pi_coding_agent`（print / RPC / TUI 三模式 + login/logout/list/auth/install/remove/update 子命令）
- SDK：`pi_coding_agent.sdk.create_agent_session` / `AgentSessionRuntime`
- Server：`pi_server`（`python -m pi_server`）
- Evals：`pi_evals` CLI runner
- Harness：`pi_agent.AgentHarness`

## 模块边界与依赖方向

```
pi_ai          ← pi_agent ← pi_coding_agent（+ pi_tui 独立框架被 pi_coding_agent 使用）
pi_protocol / pi_storage / pi_telemetry 被上层按需引用
pi_server → pi_protocol / pi_storage / pi_coding_agent
pi_evals → pi_coding_agent / pi_agent
```

- 公共 API：`pi_ai`（types/models/provider/auth）、`pi_agent`（Agent/Harness/tools/env）、`pi_coding_agent`（sdk/_cli）、`pi_tui`（engine/组件/主题）
- 内部实现：各包 `_` 前缀模块、`session/v4`、`engine/`
- 适配层：`providers/`（40+ provider 注册）、`auth/oauth/`（device-code + loopback 浏览器流程）、`modes/`（print/RPC/interactive）
- 测试：各包 `tests/`，faux provider 为主（无真实 API key）
