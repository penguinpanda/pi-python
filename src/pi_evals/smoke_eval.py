"""smoke eval：基本 prompt 端到端（对齐 TS packages/evals/src/smoke.eval.ts）。"""

from __future__ import annotations

import os

from .harness import create_pi_coding_agent_harness
from .vitest_evals.suite import describe_eval

pi_coding_agent_harness = create_pi_coding_agent_harness(no_tools=True)


@describe_eval("Pi Coding Agent smoke", harness=pi_coding_agent_harness)
async def _smoke_basic_prompt(ctx):
    result = await ctx.run("What's the capital of France? Respond with only the city name.")
    output = result.output
    assert isinstance(output, str)
    assert output.strip() == "Paris"
    assert result.errors == []
    assert result.usage["provider"] == os.environ.get("PI_PROVIDER")
    assert result.usage["model"] == os.environ.get("PI_MODEL")
    assert result.usage["totalTokens"] > 0
