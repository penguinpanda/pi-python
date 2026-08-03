"""Phase 4 Skills / Templates / Compaction / 分支摘要测试。"""

from __future__ import annotations

import asyncio

import pytest
from pi_ai._types import Model, TextContent
from pi_ai.providers.faux import faux_assistant_message, faux_provider

from pi_agent import (
    PythonExecutionEnv,
    collect_entries_for_branch_summary,
    format_prompt_template_invocation,
    format_skill_invocation,
    generate_branch_summary,
    load_prompt_templates,
    load_skills,
    prepare_compaction,
    substitute_args,
)
from pi_agent.compaction import DEFAULT_COMPACTION_SETTINGS, compact
from pi_agent.session import InMemorySessionStorage, Session


def _make_model() -> Model:
    return Model(
        id="test-model",
        provider="test",
        api="openai-completions",
        name="Test",
        max_tokens=4096,
        context_window=128000,
    )


def _assistant_with_usage(text: str):
    return {
        "role": "assistant",
        "content": [TextContent(type="text", text=text)],
        "api": "test",
        "provider": "test",
        "model": "test",
        "usage": {
            "input": 100,
            "output": 20,
            "cache_read": 0,
            "cache_write": 0,
            "total_tokens": 120,
            "cost": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "total": 0},
        },
    }


def _user_message(text: str):
    return {"role": "user", "content": text}


class TestSkills:
    @pytest.mark.asyncio
    async def test_load_skills_from_skil_md(self, tmp_path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: A test skill\n---\n\nDo the thing.\n",
            encoding="utf-8",
        )
        env = PythonExecutionEnv(str(tmp_path))
        result = await load_skills(env, str(tmp_path))
        assert len(result["skills"]) == 1
        skill = result["skills"][0]
        assert skill["name"] == "my-skill"
        assert skill["description"] == "A test skill"
        assert "Do the thing." in skill["content"]

    @pytest.mark.asyncio
    async def test_skill_name_validation_diagnostics(self, tmp_path):
        skill_dir = tmp_path / "Bad_Name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: test\n---\n\ncontent\n",
            encoding="utf-8",
        )
        env = PythonExecutionEnv(str(tmp_path))
        result = await load_skills(env, str(tmp_path))
        # TS 语义：名称校验失败仍返回 skill，但产出 invalid_metadata 诊断
        assert len(result["skills"]) == 1
        codes = [d.code for d in result["diagnostics"]]
        assert "invalid_metadata" in codes

    @pytest.mark.asyncio
    async def test_format_skill_invocation(self):
        text = format_skill_invocation(
            name="test-skill",
            description="desc",
            content="content",
            file_path="/tmp/skills/test-skill/SKILL.md",
        )
        assert '<skill name="test-skill"' in text
        assert "content" in text
        assert "</skill>" in text


class TestPromptTemplates:
    @pytest.mark.asyncio
    async def test_substitute_args(self):
        content = "$1 and $2 and $ARGUMENTS and ${@:2} and ${@:1:2}"
        result = substitute_args(content, ["a", "b", "c"])
        assert result == "a and b and a b c and b c and a b"

    @pytest.mark.asyncio
    async def test_load_templates_from_dir(self, tmp_path):
        (tmp_path / "greet.md").write_text(
            "---\ndescription: Greeting\n---\n\nHi $1!\n",
            encoding="utf-8",
        )
        env = PythonExecutionEnv(str(tmp_path))
        result = await load_prompt_templates(env, str(tmp_path))
        assert len(result["promptTemplates"]) == 1
        template = result["promptTemplates"][0]
        assert template["name"] == "greet"
        assert template["description"] == "Greeting"
        assert format_prompt_template_invocation(
            template["name"], template["content"], ["World"]
        ) == "Hi World!"


class TestCompaction:
    @pytest.mark.asyncio
    async def test_prepare_compaction_none_when_empty(self):
        ok_flag, preparation = prepare_compaction([], DEFAULT_COMPACTION_SETTINGS)
        assert ok_flag is True
        assert preparation is None

    @pytest.mark.asyncio
    async def test_prepare_compaction_small_history(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        await session.append_message(_user_message("q1"))
        await session.append_message(_assistant_with_usage("a1"))
        entries = await session.get_branch()

        ok_flag, preparation = prepare_compaction(entries, DEFAULT_COMPACTION_SETTINGS)
        assert ok_flag is True
        assert preparation is not None
        # 小历史低于保留预算 → 切割点取第一个合法切割点（TS 语义）
        assert preparation.first_kept_entry_id == entries[0]["id"]
        assert preparation.messages_to_summarize == []

    @pytest.mark.asyncio
    async def test_compact_generates_summary(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        await session.append_message(_user_message("Help me build a calculator"))
        await session.append_message(_assistant_with_usage("I built the calculator."))
        entries = await session.get_branch()
        ok_flag, preparation = prepare_compaction(entries, DEFAULT_COMPACTION_SETTINGS)
        assert preparation is not None

        core = faux_provider()
        core.set_responses([faux_assistant_message("## Goal\nBuild a calculator\n\n## Progress\n- done")])

        ok_flag, result = await compact(preparation, core.stream, _make_model())
        assert ok_flag is True
        assert "## Goal" in result.summary
        assert result.first_kept_entry_id == preparation.first_kept_entry_id
        assert result.tokens_before >= 0
        assert result.usage is not None


class TestBranchSummarization:
    @pytest.mark.asyncio
    async def test_collect_entries_and_generate(self):
        storage = InMemorySessionStorage()
        session = Session(storage)
        a_id = await session.append_message(_user_message("start"))
        await session.append_message(_assistant_with_usage("explore"))
        b_id = await session.append_message(_user_message("branch question"))

        old_leaf_id = b_id
        target_id = a_id
        collected = await collect_entries_for_branch_summary(session, old_leaf_id, target_id)
        assert len(collected["entries"]) == 2

        core = faux_provider()
        core.set_responses([faux_assistant_message("## Goal\nExplored branch")])
        ok_flag, result = await generate_branch_summary(
            collected["entries"],
            stream_fn=core.stream,
            model=_make_model(),
        )
        assert ok_flag is True
        assert "Summary of that exploration" in result["summary"]
        assert "## Goal" in result["summary"]
