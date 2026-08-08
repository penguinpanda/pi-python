"""Prompt 模板函数运行时 parity 测试。

对 fixtures/prompt_templates.json 中的固定输入，断言 Python 实现与 TS 侧
真实函数（packages/coding-agent/src/core/prompt-templates.ts）的输出逐字符
一致：

- substitute_args            ↔ substituteArgs
- parse_command_args         ↔ parseCommandArgs（golden 为紧凑 JSON 数组）
- PromptTemplateLoader.expand ↔ expandPromptTemplate

golden 缺失时测试跳过（不是失败）。
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest

from pi_agent.prompt_templates import substitute_args
from pi_coding_agent.prompt_templates import (
    PromptTemplate,
    PromptTemplateLoader,
    parse_command_args,
)

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"
GOLDEN_DIR = HERE / "golden"

FIXTURE = json.loads((FIXTURES_DIR / "prompt_templates.json").read_text(encoding="utf-8"))

_TS_GOLDEN_HINT = "TS golden 缺失：需在 pi TS mono-repo 中生成后拷入 golden/（见 README.md）"


def _cases(func: str) -> list[tuple[str, int]]:
    return [(func, index) for index in range(len(FIXTURE[func]))]


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


@pytest.mark.parametrize("func,index", _cases("substituteArgs"))
def test_substitute_args_matches_ts_golden(func: str, index: int) -> None:
    golden = GOLDEN_DIR / f"prompttemplates_substituteArgs_{index}.txt"
    if not golden.exists():
        pytest.skip(_TS_GOLDEN_HINT)

    case = FIXTURE[func][index]
    actual = substitute_args(case["content"], case["args"])
    expected = golden.read_text(encoding="utf-8")
    assert actual == expected, _diff(expected, actual)


@pytest.mark.parametrize("func,index", _cases("parseCommandArgs"))
def test_parse_command_args_matches_ts_golden(func: str, index: int) -> None:
    golden = GOLDEN_DIR / f"prompttemplates_parseCommandArgs_{index}.txt"
    if not golden.exists():
        pytest.skip(_TS_GOLDEN_HINT)

    case = FIXTURE[func][index]
    # 与 TS dump 相同的序列化方式（紧凑分隔符）比较数组。
    actual = json.dumps(parse_command_args(case), ensure_ascii=False, separators=(",", ":"))
    expected = golden.read_text(encoding="utf-8")
    assert actual == expected, _diff(expected, actual)


@pytest.mark.parametrize("func,index", _cases("expandPromptTemplate"))
def test_expand_prompt_template_matches_ts_golden(func: str, index: int, tmp_path: Path) -> None:
    golden = GOLDEN_DIR / f"prompttemplates_expandPromptTemplate_{index}.txt"
    if not golden.exists():
        pytest.skip(_TS_GOLDEN_HINT)

    case = FIXTURE[func][index]
    loader = PromptTemplateLoader(global_dir=tmp_path)
    # 直接注入模板（测试聚焦 expand 逻辑，绕过文件加载）。
    loader._templates = {
        t["name"]: PromptTemplate(
            name=t["name"],
            description="",
            argument_hint=None,
            content=t["content"],
            file_path="",
            source="path",
        )
        for t in case["templates"]
    }
    actual = loader.expand(case["text"])
    expected = golden.read_text(encoding="utf-8")
    assert actual == expected, _diff(expected, actual)
