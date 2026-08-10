"""SkillLoader 单元测试。"""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.skills import (
    SkillLoader,
    format_skills_for_prompt,
)


def _write_skill(
    directory: Path,
    body: str,
    name: str | None = None,
    frontmatter: str | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    if frontmatter is not None:
        path.write_text(f"---\n{frontmatter}\n---\n\n{body}", encoding="utf-8")
    else:
        lines = []
        if name is not None:
            lines.append(f"name: {name}")
        lines.append("description: Test skill")
        path.write_text(f"---\n{chr(10).join(lines)}\n---\n\n{body}", encoding="utf-8")
    return path


class TestSkillLoaderLoad:
    def test_loads_global_and_project(self, tmp_path):
        global_dir = tmp_path / "agent" / "skills"
        project_dir = tmp_path / "proj" / ".pi" / "skills"
        _write_skill(global_dir / "greet", "Hello skill", name="greet")
        _write_skill(project_dir / "lint", "Lint skill", name="lint")

        loader = SkillLoader(global_dir=global_dir, project_dir=project_dir)
        result = loader.load()
        names = {skill.name for skill in result.skills}
        assert names == {"greet", "lint"}
        assert result.diagnostics == []

        greet = loader.get("greet")
        assert greet is not None
        assert greet.source == "user"
        assert greet.base_dir == str(global_dir / "greet")
        lint = loader.get("lint")
        assert lint.source == "project"

    def test_skill_root_stops_recursion(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(root / "outer", "Outer", name="outer")
        _write_skill(root / "outer" / "nested", "Nested", name="nested")
        loader = SkillLoader(global_dir=root)
        result = loader.load()
        # outer 含 SKILL.md → 不再递归 nested
        assert [skill.name for skill in result.skills] == ["outer"]

    def test_gitignore_filtering(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(root / "visible", "Visible", name="visible")
        _write_skill(root / "ignored", "Ignored", name="ignored")
        (root / ".gitignore").write_text("ignored/\n", encoding="utf-8")
        loader = SkillLoader(global_dir=root)
        result = loader.load()
        assert [skill.name for skill in result.skills] == ["visible"]

    def test_root_markdown_files_loaded(self, tmp_path):
        root = tmp_path / "skills"
        root.mkdir(parents=True)
        (root / "guide.md").write_text(
            "---\ndescription: Root guide\n---\n\nBody", encoding="utf-8"
        )
        loader = SkillLoader(global_dir=root)
        result = loader.load()
        # 根目录 .md：无 frontmatter name 时以父目录名（source 根）命名（对齐 TS）。
        assert [skill.name for skill in result.skills] == ["skills"]
        assert result.skills[0].description == "Root guide"

    def test_missing_description_drops_skill_with_warning(self, tmp_path):
        root = tmp_path / "skills"
        (root / "noskill").mkdir(parents=True)
        (root / "noskill" / "SKILL.md").write_text(
            "---\nname: noskill\n---\n\nBody", encoding="utf-8"
        )
        loader = SkillLoader(global_dir=root)
        result = loader.load()
        assert result.skills == []
        assert any(diag.type == "warning" for diag in result.diagnostics)

    def test_invalid_name_warns_but_loads(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(root / "Bad_Name", "Body", name="Bad_Name")
        loader = SkillLoader(global_dir=root)
        result = loader.load()
        assert [skill.name for skill in result.skills] == ["Bad_Name"]
        assert any("invalid characters" in diag.message for diag in result.diagnostics)

    def test_name_collision_diagnostic(self, tmp_path):
        global_dir = tmp_path / "skills"
        project_dir = tmp_path / "proj" / ".pi" / "skills"
        _write_skill(global_dir / "dup", "Global", name="dup")
        _write_skill(project_dir / "dup", "Project", name="dup")
        loader = SkillLoader(global_dir=global_dir, project_dir=project_dir)
        result = loader.load()
        assert len(result.skills) == 1
        assert any(diag.type == "collision" for diag in result.diagnostics)

    def test_explicit_path(self, tmp_path):
        skill_dir = tmp_path / "extra"
        _write_skill(skill_dir, "Extra", name="extra")
        loader = SkillLoader(global_dir=tmp_path / "empty")
        result = loader.load(explicit_paths=[str(skill_dir)])
        assert [skill.name for skill in result.skills] == ["extra"]
        assert result.skills[0].source == "path"

    def test_missing_explicit_path_diagnostic(self, tmp_path):
        loader = SkillLoader(global_dir=tmp_path / "empty")
        result = loader.load(explicit_paths=[str(tmp_path / "nope")])
        assert result.skills == []
        assert any(diag.code == "path_missing" for diag in result.diagnostics)

    def test_full_yaml_frontmatter(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(
            root / "full",
            "Body",
            frontmatter=(
                'description: "Multi\\nline"\nname: full\ntags:\n  - a\n  - b\n'
                "disable-model-invocation: true"
            ),
        )
        loader = SkillLoader(global_dir=root)
        result = loader.load()
        skill = result.skills[0]
        assert skill.description == "Multi\nline"
        assert skill.disable_model_invocation is True

    def test_invalid_yaml_parse_failed(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(root / "bad", "Body", frontmatter="description: [unclosed")
        loader = SkillLoader(global_dir=root)
        result = loader.load()
        assert result.skills == []
        assert any(diag.code == "parse_failed" for diag in result.diagnostics)

    def test_gitignore_nested_star_pattern(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(root / "ok", "Ok")
        _write_skill(root / "x.log" / "app", "Log")
        (root / ".gitignore").write_text("*.log\n", encoding="utf-8")
        loader = SkillLoader(global_dir=root)
        result = loader.load()
        assert [skill.name for skill in result.skills] == ["ok"]

    def test_explicit_non_markdown_warns(self, tmp_path):
        root = tmp_path / "skills"
        notes = tmp_path / "notes.txt"
        notes.write_text("x", encoding="utf-8")
        loader = SkillLoader(global_dir=root)
        result = loader.load(explicit_paths=[str(notes)])
        assert result.skills == []
        assert any(diag.code == "path_not_markdown" for diag in result.diagnostics)

    def test_explicit_path_source_attribution(self, tmp_path):
        global_dir = tmp_path / "agent" / "skills"
        project_dir = tmp_path / "proj" / ".pi" / "skills"
        outside = tmp_path / "extra"
        _write_skill(global_dir / "g", "G", frontmatter="name: g\ndescription: G")
        _write_skill(project_dir / "p", "P", frontmatter="name: p\ndescription: P")
        _write_skill(outside / "x", "X", frontmatter="name: x\ndescription: X")
        loader = SkillLoader(global_dir=global_dir, project_dir=project_dir)
        result = loader.load(explicit_paths=[str(global_dir), str(project_dir), str(outside)])
        sources = {skill.name: skill.source for skill in result.skills}
        assert sources == {"g": "user", "p": "project", "x": "path"}


class TestSkillFormatting:
    def test_format_for_prompt_excludes_disabled(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(root / "normal", "Normal", name="normal")
        disabled_dir = root / "hidden"
        disabled_dir.mkdir(parents=True)
        (disabled_dir / "SKILL.md").write_text(
            "---\nname: hidden\ndescription: Hidden\ndisable-model-invocation: true\n---\n\nBody",
            encoding="utf-8",
        )
        loader = SkillLoader(global_dir=root)
        loader.load()
        prompt = loader.format_for_prompt()
        assert "<available_skills>" in prompt
        assert "normal" in prompt
        assert "hidden" not in prompt

    def test_format_invocation(self, tmp_path):
        root = tmp_path / "skills"
        _write_skill(root / "greet", "Hello from skill", name="greet")
        loader = SkillLoader(global_dir=root)
        loader.load()
        skill = loader.get("greet")
        assert skill is not None
        block = loader.format_invocation(skill, "Use it now")
        assert '<skill name="greet"' in block
        assert "Hello from skill" in block
        assert "Use it now" in block
        assert "References are relative to" in block

    def test_format_skills_for_prompt_empty(self):
        assert format_skills_for_prompt([]) == ""
