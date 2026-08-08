"""Compaction / branch-summarization 模板常量 parity 测试。

断言 Python 侧运行时常量（pi_coding_agent.compaction / pi_agent.branch_summarization）
与 TS 侧同名常量逐字符一致（golden/compaction_*.txt，由 dump-compaction-prompts.ts
从 TS 源码原样提取生成）。

golden 缺失时测试跳过（不是失败）——需要先在装有 pi mono-repo 的机器上运行
TS dump 脚本生成（见 README.md）。
"""

from __future__ import annotations

import difflib
import importlib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"

# (常量名, Python 模块)。BRANCH_* 仅存在于 agent 层（pi_agent），其余在
# coding-agent 层（pi_coding_agent）；TS 两套模板当前内容一致。
CASES: list[tuple[str, str]] = [
    ("SUMMARIZATION_SYSTEM_PROMPT", "pi_coding_agent.compaction"),
    ("SUMMARIZATION_PROMPT", "pi_coding_agent.compaction"),
    ("UPDATE_SUMMARIZATION_PROMPT", "pi_coding_agent.compaction"),
    ("TURN_PREFIX_SUMMARIZATION_PROMPT", "pi_coding_agent.compaction"),
    ("BRANCH_SUMMARY_PROMPT", "pi_agent.branch_summarization"),
    ("BRANCH_SUMMARY_PREAMBLE", "pi_agent.branch_summarization"),
]

_TS_DUMP_COMMAND = (
    "node --experimental-strip-types src/pi_agent/tests/parity/dump-compaction-prompts.ts"
)


def _normalize_newlines(text: str) -> str:
    # 只归一 CRLF/LF，不做任何 strip——尾换行等真实差异必须保留。
    return text.replace("\r\n", "\n")


def _diff(expected: str, actual: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="TS golden",
            tofile="PY",
            lineterm="",
        )
    )


@pytest.mark.parametrize("name,module", CASES)
def test_compaction_prompt_matches_ts_golden(name: str, module: str) -> None:
    golden = GOLDEN_DIR / f"compaction_{name}.txt"
    if not golden.exists():
        pytest.skip(f"TS golden 缺失，请先运行: {_TS_DUMP_COMMAND}")

    actual = _normalize_newlines(getattr(importlib.import_module(module), name))
    expected = _normalize_newlines(golden.read_text(encoding="utf-8"))

    assert actual == expected, _diff(expected, actual)
