"""PromptTemplateLoader 单元测试。"""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.prompt_templates import (
    PromptTemplateLoader,
    parse_command_args,
)


def _write_template(
    path: Path, body: str, description: str | None = None, hint: str | None = None
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    if description is not None:
        lines.append(f"description: {description}")
    if hint is not None:
        lines.append(f"argument-hint: {hint}")
    lines.append("---")
    path.write_text("\n".join(lines) + "\n\n" + body, encoding="utf-8")
    return path


class TestLoad:
    def test_loads_global_and_project(self, tmp_path):
        global_dir = tmp_path / "agent" / "prompts"
        project_dir = tmp_path / "proj" / ".pi" / "prompts"
        _write_template(global_dir / "greet.md", "Hi $1!", description="Greeting")
        _write_template(project_dir / "review.md", "Review $@", hint="<code>")

        loader = PromptTemplateLoader(global_dir=global_dir, project_dir=project_dir)
        templates = loader.load()
        names = {template.name for template in templates}
        assert names == {"greet", "review"}

        greet = loader.get("greet")
        assert greet.description == "Greeting"
        assert greet.content == "Hi $1!"
        assert greet.source == "user"
        review = loader.get("review")
        assert review.argument_hint == "<code>"
        assert review.source == "project"

    def test_description_falls_back_to_first_line(self, tmp_path):
        root = tmp_path / "prompts"
        _write_template(root / "bare.md", "First line of body\n\nMore")
        loader = PromptTemplateLoader(global_dir=root)
        templates = loader.load()
        assert templates[0].description == "First line of body"

    def test_description_truncated(self, tmp_path):
        root = tmp_path / "prompts"
        long_line = "x" * 100
        _write_template(root / "long.md", long_line)
        loader = PromptTemplateLoader(global_dir=root)
        templates = loader.load()
        assert templates[0].description == "x" * 60 + "..."

    def test_explicit_file_path(self, tmp_path):
        path = _write_template(tmp_path / "direct.md", "Direct $@", description="Direct")
        loader = PromptTemplateLoader(global_dir=tmp_path / "empty")
        templates = loader.load(explicit_paths=[str(path)])
        assert [template.name for template in templates] == ["direct"]
        assert templates[0].source == "path"

    def test_missing_dir_returns_empty(self, tmp_path):
        loader = PromptTemplateLoader(global_dir=tmp_path / "nope")
        assert loader.load() == []


class TestParseCommandArgs:
    def test_simple(self):
        assert parse_command_args("one two three") == ["one", "two", "three"]

    def test_quoted(self):
        assert parse_command_args('one "two words" three') == ["one", "two words", "three"]
        assert parse_command_args("one 'two words'") == ["one", "two words"]

    def test_empty(self):
        assert parse_command_args("") == []


class TestExpand:
    def test_expand_template(self, tmp_path):
        root = tmp_path / "prompts"
        _write_template(
            root / "fmt.md",
            "$1 | $2 | $@ | ${@:2} | ${@:1:2} | ${2:-none} | ${@:-empty}",
            description="Fmt",
        )
        loader = PromptTemplateLoader(global_dir=root)
        loader.load()
        template = loader.get("fmt")
        assert template is not None
        expanded = loader.expand_template(template, "a b c")
        assert expanded == "a | b | a b c | b c | a b | b | a b c"

    def test_expand_default_when_missing(self, tmp_path):
        root = tmp_path / "prompts"
        _write_template(root / "def.md", "${1:-hello}", description="Def")
        loader = PromptTemplateLoader(global_dir=root)
        loader.load()
        template = loader.get("def")
        assert loader.expand_template(template, "") == "hello"
        assert loader.expand_template(template, "world") == "world"

    def test_expand_prompt_text(self, tmp_path):
        root = tmp_path / "prompts"
        _write_template(root / "greet.md", "Hello $1", description="Greet")
        loader = PromptTemplateLoader(global_dir=root)
        loader.load()
        assert loader.expand("/greet world") == "Hello world"
        assert loader.expand("/missing arg") == "/missing arg"
        assert loader.expand("plain text") == "plain text"

    def test_expand_quoted_args(self, tmp_path):
        root = tmp_path / "prompts"
        _write_template(root / "quote.md", "[$1]", description="Quote")
        loader = PromptTemplateLoader(global_dir=root)
        loader.load()
        assert loader.expand('/quote "hello world"') == "[hello world]"
