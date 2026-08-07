# src — pi-python 源码目录

本目录包含 pi-python 的八个 Python 包，自底向上依赖：

```
pi_coding_agent (CLI 编码代理)
    └─ pi_agent (Agent 循环 / Session / Harness / 工具)
        └─ pi_ai (LLM API 抽象 / Provider / 认证)
```

其余包（`pi_tui`、`pi_protocol`、`pi_storage`、`pi_server`、`pi_evals`）与核心三层相对独立，分别承担 TUI 引擎、线协议、会话存储、常驻服务与评测。

| 包 | README | 说明 |
|---|--------|------|
| `pi_ai` | [README](pi_ai/README.md) | 统一 LLM API：Models 注册表 + Provider 抽象 + 事件流协议 |
| `pi_agent` | [README](pi_agent/README.md) | Agent 核心循环（纯函数引擎 + Agent 包装）+ Session / 技能 / 压缩 / Harness |
| `pi_coding_agent` | [README](pi_coding_agent/README.md) | CLI 编码代理：7 个编码工具、JSONL 会话、双层配置、自动压缩 |
| `pi_tui` | [README](pi_tui/README.md) | 内置引擎 TUI（无 Textual）：主题 / 快捷键 / 选择器 / 剪贴板图片 |
| `pi_protocol` | [README](pi_protocol/README.md) | protocol v2 线协议：Command / Result / Snapshot / Progress / Error + JSONL framing |
| `pi_storage` | [README](pi_storage/README.md) | PostgreSQL 会话存储：SessionStore / SessionSearch（asyncpg + 迁移） |
| `pi_server` | [README](pi_server/README.md) | 常驻 pi 服务：stdio JSONL，attach/detach + 快照推送 |
| `pi_evals` | [README](pi_evals/README.md) | TS packages/evals 移植：harness + runner + 对比评测 |

## 目录结构

```
src/
├── pi_ai/               # LLM 抽象层（provider / api / auth / models / types / utils）
├── pi_agent/            # Agent 循环 + 运行设施（session / tools / skills / compaction ...）
├── pi_coding_agent/     # CLI 编码代理（tools / sessions / compaction）
├── pi_tui/              # 独立可复用 TUI 框架（engine / overlay，无 Textual）
├── pi_protocol/         # protocol v2 schema + JSONL framing
├── pi_storage/          # PostgreSQL 会话存储（asyncpg + 迁移 + 搜索）
├── pi_server/           # 常驻 pi 服务（stdio JSONL）
└── pi_evals/            # 评测 harness + runner（pi-evals CLI）
```

## 依赖方向

- `pi_ai` 不依赖其它包
- `pi_agent` 依赖 `pi_ai`（复用 Model / Message / EventStream 等类型）
- `pi_coding_agent` 依赖 `pi_agent` + `pi_ai` + `pi_tui`
- `pi_tui` 依赖 `pi_agent`（仅 `clipboard_image` 复用其 `tools.image_pipeline`）；自身引擎无第三方依赖
- `pi_protocol` 独立（pydantic schema）
- `pi_storage` 独立
- `pi_server` 依赖 `pi_protocol` + `pi_agent` + `pi_ai` + `pi_coding_agent`
- `pi_evals` 依赖 `pi_agent` + `pi_ai` + `pi_coding_agent`（评测 harness 驱动 Agent 循环）

## 开发

```bash
# 安装依赖
uv sync

# 运行全部测试
uv run pytest

# 按包运行
uv run pytest src/pi_ai/tests/ -v
uv run pytest src/pi_agent/tests/ -v
uv run pytest src/pi_coding_agent/tests/ -v
uv run pytest src/pi_protocol/tests/ src/pi_server/tests/ src/pi_evals/ -v

# PostgreSQL 存储测试（先启动 compose 的 pg 服务）
docker compose -f docker/compose.yaml up -d pg
$env:PI_PG_DSN="postgresql://pi:pi@127.0.0.1:5432/pi"
uv run pytest src/pi_storage/tests/ -v
```

各包详细文档见上表链接。
