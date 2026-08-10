"""pi_agent harness skills 对齐 TS v0.84.0 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_agent.env import PythonExecutionEnv
from pi_agent.skills import (
    _dirname_env_path,
    _relative_env_path,
    format_skill_invocation,
    load_sourced_skills,
    load_skills,
)


def _write_skill(root: Path, relative: str, body: str, frontmatter: str | None = None) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if frontmatter is None:
        frontmatter = "description: Test skill"
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_full_yaml_frontmatter(tmp_path):
    _write_skill(
        tmp_path / "skills" / "full",
        "SKILL.md",
        "Body",
        'description: "Multi\\nline description"\nname: full\n'
        "tags:\n  - a\n  - b\ndisable-model-invocation: true\n",
    )
    env = PythonExecutionEnv(str(tmp_path))
    result = await load_skills(env, str(tmp_path / "skills"))
    loaded = result["skills"][0]
    assert loaded["name"] == "full"
    assert loaded["description"] == "Multi\nline description"
    assert loaded["disableModelInvocation"] is True


@pytest.mark.asyncio
async def test_invalid_yaml_reports_parse_failed(tmp_path):
    bad_dir = tmp_path / "skills" / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "SKILL.md").write_text("---\ndescription: [unclosed\n---\n\nBody", encoding="utf-8")
    env = PythonExecutionEnv(str(tmp_path))
    result = await load_skills(env, str(tmp_path / "skills"))
    assert result["skills"] == []
    assert any(d.code == "parse_failed" for d in result["diagnostics"])


@pytest.mark.asyncio
async def test_gitignore_full_semantics(tmp_path):
    root = tmp_path / "skills"
    _write_skill(root / "visible", "SKILL.md", "v")
    _write_skill(root / "x.log" / "app", "SKILL.md", "log")
    _write_skill(root / "nested" / "deep", "SKILL.md", "deep")
    _write_skill(root / "chardir" / "c", "SKILL.md", "char")
    _write_skill(root / "ignored" / "i", "SKILL.md", "ignored")
    _write_skill(root / "keepme" / "k", "SKILL.md", "keep", "name: keepme\ndescription: Keep")
    (root / ".gitignore").write_text(
        "*.log\nnested/\nch?rdir/\nignored/\n!keepme/\n", encoding="utf-8"
    )
    env = PythonExecutionEnv(str(tmp_path))
    result = await load_skills(env, str(root))
    names = {skill["name"] for skill in result["skills"]}
    assert names == {"visible", "keepme"}


@pytest.mark.asyncio
async def test_symlinked_skill_dir_and_file(tmp_path):
    target = tmp_path / "target"
    _write_skill(target / "linked", "SKILL.md", "linked", "name: linked\ndescription: Linked")
    _write_skill(target / "file_skill", "SKILL.md", "file", "name: alias2\ndescription: File")
    try:
        (tmp_path / "skills" / "alias").symlink_to(target / "linked", target_is_directory=True)
        (tmp_path / "skills" / "alias2").mkdir()
        (tmp_path / "skills" / "alias2" / "SKILL.md").symlink_to(target / "file_skill" / "SKILL.md")
    except OSError:
        pytest.skip("symlink not supported")
    env = PythonExecutionEnv(str(tmp_path))
    result = await load_skills(env, str(tmp_path / "skills"))
    names = {skill["name"] for skill in result["skills"]}
    assert names == {"linked", "alias2"}


@pytest.mark.asyncio
async def test_load_sourced_skills_map_skill(tmp_path):
    _write_skill(tmp_path / "s" / "a", "SKILL.md", "A", "name: a\ndescription: A")
    env = PythonExecutionEnv(str(tmp_path))
    result = await load_sourced_skills(
        env,
        [{"path": str(tmp_path / "s"), "source": "user"}],
        map_skill=lambda skill, source: {**skill, "source": source.upper()},
    )
    assert result["skills"][0]["skill"]["source"] == "USER"


def test_dirname_env_path():
    assert _dirname_env_path("C:/pi/pkg/SKILL.md") == "C:/pi/pkg"
    assert _dirname_env_path("C:/SKILL.md") == "C:/"
    assert _dirname_env_path("SKILL.md") == "/"
    assert _dirname_env_path("/a/b/SKILL.md") == "/a/b"


def test_relative_env_path():
    assert _relative_env_path("C:/root", "C:/root/a/b") == "a/b"
    assert _relative_env_path("/root/", "/root") == ""
    assert _relative_env_path("/root", "/outside/x") == "outside/x"


def test_format_skill_invocation_directory():
    block = format_skill_invocation("n", "d", "body", "C:/skills/n/SKILL.md")
    assert "References are relative to C:/skills/n." in block
