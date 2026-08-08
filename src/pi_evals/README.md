# pi-evals — 评测 harness（TS packages/evals 完整移植）

对齐 TS [packages/evals](https://github.com/earendil-works/pi-mono/tree/main/packages/evals)：
把 pi-harness、vitest-evals 等价物（judge / harness table / artifacts /
summary / runner）移植为 Python，并提供 smoke / extensions 两类 eval。
默认运行时与 CLI 一致（真实 providers + `~/.pi/agent` 的 auth.json /
models.json / models-store.json）；测试通过 `runtime` 注入 faux provider
（零网络、可脚本化响应）。

## 内容

- `harness.py` — `create_pi_coding_agent_harness()`（对齐 TS
  `createPiCodingAgentHarness`）：
  - 每次 run 在隔离的临时 workspace / agent / sessions 目录创建
    `AgentSession`，结束后删除并保留 session JSONL 快照 artifact；
  - `run(input)` 支持字符串或步骤序列（`{"type":"prompt","content":...}` /
    `{"type":"reload"}`）；
  - 选项：`name` / `model` / `no_tools` / `transform_system_prompt` /
    `output` / `runtime`（测试注入）；
  - 结果 `HarnessRun`：`output` / `events` / `errors` / `usage` /
    `timings` / `artifacts`；usage 含 input/output/total tokens、toolCalls、
    缓存 token 与（有定价时）estimatedCostUsd；
  - 模型选择：显式 `model={"provider":..., "id":...}` > 环境变量
    `PI_PROVIDER` / `PI_MODEL`；未设置时报错。
- `vitest_evals/` — 通用框架：
  - `harness.py`：`Harness` / `HarnessRun` / `HarnessContext` /
    `create_harness` / 稳定 JSON 规范化；
  - `judge.py`：`create_judge` / `normalize_tool_calls` / 平均分；
  - `harness_table.py`：`eval_harness_table`（baseline + candidates ×
    repetitions）与迭代 artifact；
  - `artifacts.py`：session / source artifact 落盘；
  - `summary.py`：pass-rate lift、token / latency / cost 配对差值报告；
  - `suite.py`：`describe_eval` case 注册与 `run_case` 执行。
- `smoke_eval.py` — 基本 prompt 端到端（走默认 harness runtime，断言 usage 的
  provider/model 匹配 `PI_PROVIDER` / `PI_MODEL` 环境变量）。
- `extensions_eval.py` — 扩展编写 + reload + 工具使用；系统提示词
  baseline/candidate 对比（judge 评分，不设阈值）。
- `runner.py` / `__main__.py` — CLI runner（对齐 TS `run-evals.mjs`）。

## 用法

```bash
# 默认 faux 之外的模型（CLI 优先，需成对指定）
uv run pi-evals --provider openai --model gpt-5

# 等价环境变量
PI_PROVIDER=openai PI_MODEL=gpt-5 uv run pi-evals

# 只跑指定 eval 模块；其余参数透传
uv run pi-evals src/pi_evals/smoke_eval.py
uv run pi-evals --artifact-dir out-evals src/pi_evals/extensions_eval.py

# 对比重复次数（默认 1；等价环境变量 PI_EVAL_REPETITIONS=3）
uv run pi-evals --repetitions 3 --provider deepseek --model deepseek-v4-flash
```

### 推理强度（max）

用最大推理强度跑 eval：harness 选项 `thinking_level`（对齐 TS
`createPiCodingAgentHarness` 的 `thinkingLevel`），或等价环境变量
`PI_REASONING_LEVEL`（与 `pi_coding_agent` 的扩展环境变量同名）。显式选项
优先于环境变量，两者都未设置时默认 `off`。

```bash
# CLI：环境变量方式开 max
PI_REASONING_LEVEL=max uv run pi-evals --provider deepseek --model deepseek-v4-flash

# 等价编程式（harness 选项）
harness = create_pi_coding_agent_harness(
    model={"provider": "deepseek", "id": "deepseek-v4-flash"},
    thinking_level="max",
)
```

合法级别：`off` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max`；
非法值直接报 `ValueError`。注意实际生效级别会被模型支持范围 clamp——
`max`/`xhigh` 必须模型显式映射（`thinking_level_map` 非 null）才可用，
模型不支持时自动降到最高支持级别，`reasoning` 为 `false` 的模型则保持
`off`。

产物写在 `src/pi_evals/.eval/<timestamp>_<uuid>/`（可用
`--artifact-dir` 或 `PI_EVAL_ARTIFACT_DIR` 覆盖）：`runs.jsonl` 索引每次
harness run，`sessions/<sha256(runId)>/session.jsonl` 为会话快照，
`sources/<sha256(runId)>/` 为 eval 生成的源文件。

编程式用法：

```python
import asyncio
from pi_ai import Models
from pi_ai.providers.faux import faux_assistant_message, faux_provider
from pi_coding_agent.auth_storage import AuthStorage
from pi_coding_agent.model_runtime import ModelRuntime
from pi_evals import create_pi_coding_agent_harness
from pi_evals.vitest_evals import HarnessContext


async def main():
    store = AuthStorage.in_memory()
    models = Models(credentials=store)
    core = faux_provider()
    core.set_responses([faux_assistant_message("Paris")])
    models.add_provider(core.provider)
    runtime = ModelRuntime(models, store)

    harness = create_pi_coding_agent_harness(
        model={"provider": "faux", "id": "faux-1"},
        runtime=runtime,
        no_tools=True,
    )
    result = await harness.run(
        "What's the capital of France? Respond with only the city name.",
        HarnessContext(),
    )
    print(result.output, result.errors)


asyncio.run(main())
```

## 对比评测

用 `eval_harness_table` + `describe_eval` 注册 baseline/candidate 对比：

```python
from pi_evals import create_judge, describe_eval, eval_harness_table

baseline = create_pi_coding_agent_harness(name="without-docs", transform_system_prompt=strip_docs)
candidate = create_pi_coding_agent_harness(name="default", transform_system_prompt=strip_cwd)

for row in eval_harness_table("my eval set", baseline=baseline, candidate=candidate):

    @describe_eval(
        f"{row.name} repetition {row.repetition}", harness=row.harness, judge_threshold=None
    )
    async def _case(ctx):
        await ctx.run("Complete the task.")
```

`judge_threshold=None` 时低分只作为观测进入对比报告，不使 runner 失败；
缺省阈值 1.0 时低于阈值直接失败。

## 测试

```bash
uv run pytest src/pi_evals/ -v
```

设置 `PI_PROVIDER` / `PI_MODEL` 可用真实模型运行同一 harness（默认仍建议
faux）。

## 许可

MIT
