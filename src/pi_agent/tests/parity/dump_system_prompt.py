"""Dump build_system_prompt output for each parity fixture (Python side).

只负责生成 Python 侧输出：读取 fixtures/*.json，调用真实的
pi_coding_agent.system_prompt.build_system_prompt，把输出写入
python_out/<name>.txt。比较（TS golden vs python_out）请运行
compare_outputs.py。

运行（仓库根）:

    python src/pi_agent/tests/parity/dump_system_prompt.py

PI_PACKAGE_DIR 固定为 C:/pi-pkg（可用环境变量覆盖）；TS 与 Python 两侧必须
使用同一个值，否则 "Pi documentation" 段的路径不同。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pi_coding_agent.skills import Skill
from pi_coding_agent.system_prompt import BuildSystemPromptOptions, build_system_prompt

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"
OUT_DIR = HERE / "python_out"

DEFAULT_PACKAGE_DIR = "C:/pi-pkg"


def build_options(fixture: dict) -> BuildSystemPromptOptions:
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


def main() -> None:
    os.environ.setdefault("PI_PACKAGE_DIR", DEFAULT_PACKAGE_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        # 仅 system prompt fixtures 有 cwd；compaction/prompt_templates 输入不含。
        if "cwd" not in fixture:
            continue
        prompt = build_system_prompt(build_options(fixture))
        out = OUT_DIR / f"{path.stem}.txt"
        out.write_text(prompt, encoding="utf-8", newline="\n")
        print(f"wrote {out.relative_to(HERE)} ({len(prompt)} chars)")


if __name__ == "__main__":
    main()
