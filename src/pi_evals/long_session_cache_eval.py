"""Long-session cache-first eval：单会话多轮，量化 cacheFirst 剪枝收益。

场景：同一 session 连续 5 个 prompt，其中首轮读取大文件把上下文顶到
剪枝阈值以上；cache-first 开启后后续请求会截断大工具输出，未开启则
每轮都携带完整大输出。对比表输出 tokens / latency / cost。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pi_coding_agent.compaction import CompactionSettings

from .harness import create_pi_coding_agent_harness
from .vitest_evals.harness_table import eval_harness_table
from .vitest_evals.judge import JudgeContext, create_judge
from .vitest_evals.suite import describe_eval

# 剪枝目标阈值（token）：上下文超过后 cache-first 开始截断大工具输出。
CACHE_FIRST_TARGET_THRESHOLD = 10_000
# DeepSeek V4 权威 context window（生成目录值）；用于把 reserve_tokens
# 换算成“达到目标阈值即剪枝”。
DEEPSEEK_CONTEXT_WINDOW = 1_000_000
CACHE_FIRST_RESERVE_TOKENS = DEEPSEEK_CONTEXT_WINDOW - CACHE_FIRST_TARGET_THRESHOLD

BIG_FILE_NAME = "data/big.txt"
BIG_FILE_LINES = 3000


def _big_line(index: int) -> str:
    return f"line-{index:04d}: " + "x" * 50


def _setup_workspace(root: Path) -> None:
    """写入大文件与目标文件：大文件用于把上下文顶过剪枝阈值。"""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    lines = [_big_line(index) for index in range(1, BIG_FILE_LINES + 1)]
    (data_dir / "big.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data_dir / "target.txt").write_text("ANSWER=42\n", encoding="utf-8")


STEPS: list[dict[str, str]] = [
    {
        "type": "prompt",
        "content": "Read data/big.txt without a limit and repeat its first line exactly.",
    },
    {
        "type": "prompt",
        "content": "Use bash to run `wc -l data/big.txt` and reply with only the number.",
    },
    {
        "type": "prompt",
        "content": "Read data/target.txt and reply with only the value after ANSWER=.",
    },
    {
        "type": "prompt",
        "content": "Use bash to compute 7*6 and reply with only the number.",
    },
    {
        "type": "prompt",
        "content": "Use bash to print the first 3 lines of data/big.txt and reply with only those lines.",
    },
]


def _assistant_texts(session: Any) -> list[str]:
    texts: list[str] = []
    for message in session.get_messages():
        if message.get("role") != "assistant":
            continue
        parts = [
            block.get("text", "")
            for block in message.get("content") or []
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "".join(parts)
        if text:
            texts.append(text)
    return texts


def _count_tool_calls(session: Any) -> int:
    count = 0
    for message in session.get_messages():
        if message.get("role") != "assistant":
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                count += 1
    return count


def _long_session_output(response: str, session: Any) -> dict[str, Any]:
    """聚合长会话领域输出：各轮回复、工具调用数、最终回复。"""
    return {
        "response": response,
        "assistantMessages": _assistant_texts(session),
        "toolCalls": _count_tool_calls(session),
    }


def create_long_session_harness(name: str, *, cache_first: bool = False):
    """创建长会话 harness；cache_first 时用显式 compaction_settings 压低剪枝阈值。"""

    def output(args: dict[str, Any]) -> dict[str, Any]:
        return _long_session_output(args["response"], args["session"])

    return create_pi_coding_agent_harness(
        name=name,
        workspace_setup=_setup_workspace,
        output=output,
        compaction_settings=(
            CompactionSettings(cache_first=True, reserve_tokens=CACHE_FIRST_RESERVE_TOKENS)
            if cache_first
            else None
        ),
    )


def _long_session_judge(ctx: JudgeContext) -> dict[str, Any]:
    failures: list[str] = []
    output = ctx.output
    if not isinstance(output, dict):
        failures.append("output is unavailable")
    else:
        assistant = output.get("assistantMessages")
        if not isinstance(assistant, list) or len(assistant) < len(STEPS):
            failures.append("not all steps produced an assistant response")
        tool_calls = output.get("toolCalls", 0)
        if not isinstance(tool_calls, int) or tool_calls < 4:
            failures.append("too few tool calls to be a realistic long session")
        response = str(output.get("response") or "")
        if not any(token in response for token in ("line-0001", "line-0002", "line-0003")):
            failures.append("final response did not include the first lines of big.txt")
    return {
        "score": 1 if not failures else 0,
        "metadata": {
            "rationale": "Long session workflow completed." if not failures else "; ".join(failures)
        },
    }


long_session_judge = create_judge("LongSessionJudge", _long_session_judge)

long_session_harness_table = eval_harness_table(
    "Pi long session cache-first",
    baseline=create_long_session_harness("default-system-prompt"),
    candidates=[
        create_long_session_harness("default-system-prompt-cache-first", cache_first=True),
    ],
)

for _row in long_session_harness_table:

    @describe_eval(
        f"{_row.name} repetition {_row.repetition}",
        harness=_row.harness,
        judges=[long_session_judge],
        judge_threshold=None,
    )
    async def _long_session_case(ctx, _row=_row):
        result = await ctx.run(STEPS)
        assert isinstance(result.output, dict)
