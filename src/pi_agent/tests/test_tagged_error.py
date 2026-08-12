"""TaggedError / match-case 错误测试。"""

from __future__ import annotations

from typing import assert_never

import pytest

from pi_agent import AgentHarnessError, TaggedError, UnknownSkillError, UnknownTemplateError
from pi_agent._harness_types import Result, err, ok


def test_tagged_error_attributes_and_json() -> None:
    error = UnknownSkillError("review", "Skill review not found")
    assert error._tag == "UnknownSkill"
    assert error.tag == "UnknownSkill"
    assert error.name == "review"
    payload = error.to_json()
    assert payload["_tag"] == "UnknownSkill"
    assert payload["message"] == "Skill review not found"
    assert payload["name"] == "review"


def test_agent_harness_error_is_tagged() -> None:
    error = AgentHarnessError("busy", "Harness is busy")
    assert isinstance(error, TaggedError)
    assert error.code == "busy"
    assert error._tag == "AgentHarnessError"


def test_match_case_dispatch() -> None:
    def classify(error: TaggedError) -> str:
        match error:
            case UnknownSkillError():
                return "skill"
            case UnknownTemplateError():
                return "template"
            case _ as unreachable:
                assert_never(unreachable)

    assert classify(UnknownSkillError("review", "missing")) == "skill"
    assert classify(UnknownTemplateError("plan", "missing")) == "template"


def test_result_interacts_with_tagged_error() -> None:
    failed: Result[str, UnknownSkillError] = err(UnknownSkillError("x", "missing"))
    assert failed.is_ok() is False
    with pytest.raises(UnknownSkillError):
        failed.get_or_throw()

    success: Result[str, UnknownSkillError] = ok("value")
    assert success.get_or_throw() == "value"
