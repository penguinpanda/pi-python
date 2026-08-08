"""Compaction 函数运行时 parity 测试。

对 fixtures/compaction_functions.json 中的固定输入，断言 Python
format_file_operations / serialize_conversation 的输出与 TS 侧真实函数
（formatFileOperations / serializeConversation）的输出逐字符一致
（golden/compaction_formatFileOperations_<i>.txt、compaction_serializeConversation_<i>.txt，
由 dump-compaction-functions.ts 运行真实 TS 代码生成）。

golden 缺失时测试跳过（不是失败）。
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest

from pi_coding_agent.compaction import format_file_operations, serialize_conversation

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"
GOLDEN_DIR = HERE / "golden"

FIXTURE = json.loads((FIXTURES_DIR / "compaction_functions.json").read_text(encoding="utf-8"))

# Python 函数名 → golden 文件前缀
FUNCTIONS: dict[str, str] = {
    "formatFileOperations": "compaction_formatFileOperations",
    "serializeConversation": "compaction_serializeConversation",
}

_TS_DUMP_COMMAND = (
    "node --experimental-strip-types src/pi_agent/tests/parity/dump-compaction-functions.ts"
)


def _cases() -> list[tuple[str, int]]:
    return [(func, index) for func in FUNCTIONS for index in range(len(FIXTURE[func]))]


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


@pytest.mark.parametrize("func,index", _cases())
def test_compaction_function_matches_ts_golden(func: str, index: int) -> None:
    golden = GOLDEN_DIR / f"{FUNCTIONS[func]}_{index}.txt"
    if not golden.exists():
        pytest.skip(f"TS golden 缺失，请先运行: {_TS_DUMP_COMMAND}")

    case = FIXTURE[func][index]
    if func == "formatFileOperations":
        actual = format_file_operations(case["readFiles"], case["modifiedFiles"])
    else:
        actual = serialize_conversation(case)
    expected = golden.read_text(encoding="utf-8")

    assert actual == expected, _diff(expected, actual)
