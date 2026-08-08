"""System prompt 运行时 parity 测试。

核心断言：对同一组固定 options（fixtures/*.json），Python 的
build_system_prompt 输出必须与 TS 侧 buildSystemPrompt 的 golden 输出
（golden/*.txt，由 dump-system-prompt.ts 生成）逐字符一致。

golden 文件缺失时测试跳过（不是失败）——需要先在装有 pi mono-repo 依赖的
机器上运行 TS dump 脚本生成（见 README.md）。
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import pytest

from pi_coding_agent.skills import Skill
from pi_coding_agent.system_prompt import BuildSystemPromptOptions, build_system_prompt

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"
GOLDEN_DIR = HERE / "golden"

# 与 dump-system-prompt.ts 的 PI_PACKAGE_DIR 保持一致，保证 "Pi documentation"
# 段中的路径两边相同。
PACKAGE_DIR = "C:/pi-pkg"

# 生成 golden 的命令，缺失时在 skip 信息里提示。
_TS_DUMP_COMMAND = (
    "PI_PACKAGE_DIR=C:/pi-pkg node --experimental-strip-types "
    "src/pi_agent/tests/parity/dump-system-prompt.ts"
)


def _build_options(fixture: dict) -> BuildSystemPromptOptions:
    """把 fixture JSON（camelCase，对齐 TS BuildSystemPromptOptions）映射为 Python options。"""
    skills = [
        Skill(
            name=item["name"],
            description=item["description"],
            file_path=item["filePath"],
            base_dir=item["baseDir"],
            source=item.get("source", "local"),
            disable_model_invocation=bool(item.get("disableModelInvocation", False)),
        )
        for item in (fixture.get("skills") or [])
    ]
    return BuildSystemPromptOptions(
        cwd=fixture["cwd"],
        custom_prompt=fixture.get("customPrompt"),
        selected_tools=fixture.get("selectedTools"),
        tool_snippets=fixture.get("toolSnippets"),
        prompt_guidelines=fixture.get("promptGuidelines") or [],
        append_system_prompt=fixture.get("appendSystemPrompt"),
        context_files=fixture.get("contextFiles") or [],
        skills=skills or None,
    )


def _fixture_names() -> list[str]:
    return sorted(
        path.stem
        for path in FIXTURES_DIR.glob("*.json")
        # system prompt fixtures 必有 cwd 字段；compaction/prompt_templates
        # 等其它 parity 输入不含，避免误收集。
        if "cwd" in json.loads(path.read_text(encoding="utf-8"))
    )


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


@pytest.mark.parametrize("fixture_name", _fixture_names())
def test_system_prompt_matches_ts_golden(
    fixture_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = GOLDEN_DIR / f"{fixture_name}.txt"
    if not golden.exists():
        pytest.skip(f"TS golden 缺失，请先运行: {_TS_DUMP_COMMAND}")

    fixture = json.loads((FIXTURES_DIR / f"{fixture_name}.json").read_text(encoding="utf-8"))
    monkeypatch.setenv("PI_PACKAGE_DIR", PACKAGE_DIR)
    actual = build_system_prompt(_build_options(fixture))
    expected = golden.read_text(encoding="utf-8")

    assert actual == expected, _diff(expected, actual)
