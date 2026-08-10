"""pi_agent 提示模板（prompt_templates.py）测试。"""

from __future__ import annotations

import pytest

from pi_agent.env import PythonExecutionEnv
from pi_agent.prompt_templates import (
    format_prompt_template_invocation,
    load_prompt_templates,
    load_sourced_prompt_templates,
    substitute_args,
)


def test_substitute_args_positional_and_all():
    assert substitute_args("$1 and $2", ["a", "b"]) == "a and b"
    assert substitute_args("$@", ["a", "b"]) == "a b"
    assert substitute_args("$ARGUMENTS", ["a"]) == "a"
    assert substitute_args("$9", ["a"]) == ""


def test_substitute_args_defaults_and_slices():
    assert substitute_args("${1:-fallback}", []) == "fallback"
    assert substitute_args("${@:-none}", []) == "none"
    assert substitute_args("${ARGUMENTS:-none}", []) == "none"
    assert substitute_args("${2:-fb}", ["a"]) == "fb"
    assert substitute_args("${@:2}", ["a", "b", "c"]) == "b c"
    assert substitute_args("${@:2:1}", ["a", "b", "c"]) == "b"
    assert substitute_args("${@:0}", ["a", "b"]) == "a b"
    assert substitute_args("${@:2:0}", ["a", "b", "c"]) == ""


def test_format_invocation_passthrough():
    assert format_prompt_template_invocation("t", "Hello $1", ["world"]) == "Hello world"


def _write_template(root, relative: str, body: str, frontmatter: str | None = None):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}" if frontmatter else body, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_load_template_file_and_dir(tmp_path):
    _write_template(
        tmp_path / "prompts",
        "greet.md",
        "Hi $1",
        "name: greet\ndescription: Greeting",
    )
    (tmp_path / "prompts" / "notes.txt").write_text("ignored", encoding="utf-8")
    env = PythonExecutionEnv(str(tmp_path))
    result = await load_prompt_templates(env, str(tmp_path / "prompts"))
    assert len(result["promptTemplates"]) == 1
    template = result["promptTemplates"][0]
    assert template["name"] == "greet"
    assert template["description"] == "Greeting"
    assert template["content"] == "Hi $1"
    assert result["diagnostics"] == []


@pytest.mark.asyncio
async def test_load_template_default_name_and_parse_failed(tmp_path):
    env = PythonExecutionEnv(str(tmp_path))
    bare = _write_template(tmp_path, "bare.md", "Body")
    result = await load_prompt_templates(env, str(bare))
    assert result["promptTemplates"][0]["name"] == "bare"

    bad = _write_template(tmp_path, "bad.md", "Body", "description: [unclosed")
    result = await load_prompt_templates(env, str(bad))
    assert result["promptTemplates"] == []
    assert any(d.code == "parse_failed" for d in result["diagnostics"])


@pytest.mark.asyncio
async def test_load_prompt_templates_missing_and_non_md(tmp_path):
    env = PythonExecutionEnv(str(tmp_path))
    missing = await load_prompt_templates(env, str(tmp_path / "nope"))
    assert missing["promptTemplates"] == []
    assert missing["diagnostics"] == []
    txt = _write_template(tmp_path, "x.txt", "Body")
    non_md = await load_prompt_templates(env, str(txt))
    assert non_md["promptTemplates"] == []


@pytest.mark.asyncio
async def test_load_sourced_prompt_templates(tmp_path):
    _write_template(tmp_path / "prompts", "g.md", "G", "description: G")
    env = PythonExecutionEnv(str(tmp_path))
    result = await load_sourced_prompt_templates(
        env, [{"path": str(tmp_path / "prompts"), "source": "user"}]
    )
    assert result["promptTemplates"][0]["source"] == "user"
    assert result["promptTemplates"][0]["promptTemplate"]["name"] == "g"
