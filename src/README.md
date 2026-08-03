# src — pi-python 源码目录

本目录包含 pi-python 的三个 Python 包，自底向上依赖：

```
pi_coding_agent (CLI 编码代理)
    └─ pi_agent (Agent 循环 / Session / Harness / 工具)
        └─ pi_ai (LLM API 抽象 / Provider / 认证)
```

| 包 | README | 说明 |
|---|--------|------|
| `pi_ai` | [README](pi_ai/README.md) | 统一 LLM API：Models 注册表 + Provider 抽象 + 事件流协议 |
| `pi_agent` | [README](pi_agent/readme.md) | Agent 核心循环（纯函数引擎 + Agent 包装）+ Session / 技能 / 压缩 / Harness |
| `pi_coding_agent` | [README](pi_coding_agent/README.md) | CLI 编码代理：7 个编码工具、JSONL 会话、双层配置、自动压缩 |

## 目录结构

```
src/
├── pi_ai/               # LLM 抽象层（provider / api / auth / models / types / utils）
├── pi_agent/            # Agent 循环 + 运行设施（session / tools / skills / compaction ...）
└── pi_coding_agent/     # CLI 编码代理（tools / sessions / compaction）
```

## 依赖方向

- `pi_ai` 不依赖其它两个包
- `pi_agent` 依赖 `pi_ai`（复用 Model / Message / EventStream 等类型）
- `pi_coding_agent` 依赖 `pi_agent` + `pi_ai`

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
```

各包详细文档见上表链接。
