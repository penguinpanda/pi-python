# pi-evals — 评测 harness

对齐 TS [packages/evals](https://github.com/earendil-works/pi-mono/tree/main/packages/evals)：
把 pi-harness 移植为 pytest 用例（默认 faux provider，零网络），
并提供 smoke / extensions 两类 eval 与结果汇总。

## 内容

- `harness.py` — `PiCodingAgentHarness`：
  - `run(input)` 支持字符串或步骤序列（`{"type":"prompt","content":...}` /
    `{"type":"reload"}`）；
  - 结果 `EvalResult`：`output` / `errors` / `usage`（provider/model/tokens/cost）/
    `transcript`（message/tool_call/tool_result 事件）/ `artifacts` / `duration_ms`；
  - 模型选择：显式 `model={"provider":..., "id":...}` > 环境变量
    `PI_PROVIDER` / `PI_MODEL`；未设置时报错。
- `smoke_eval.py` — 基本 prompt 端到端（faux provider 脚本化响应）。
- `extensions_eval.py` — 临时扩展加载 + 命令注册 + prompt 端到端。
- `reporter.py` — `report_results()` 汇总表（ok/FAIL、时长、token、错误列表）。

## 用法

```python
import asyncio
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime
from pi_evals import PiCodingAgentHarness

async def main():
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    core = faux_provider()
    core.set_responses([faux_assistant_message("Paris")])
    models.add_provider(core.provider)
    runtime = ModelRuntime(models, store)

    harness = PiCodingAgentHarness(
        runtime=runtime,
        model={"provider": "faux", "id": "faux-1"},
    )
    result = await harness.run("What's the capital of France?")
    print(result.output, result.errors)

asyncio.run(main())
```

## 测试 / 运行 eval

```bash
uv run pytest src/pi_evals/ -v
```

设置 `PI_PROVIDER` / `PI_MODEL` 可用真实模型运行同一 harness（默认仍建议 faux）。

## 许可

MIT
