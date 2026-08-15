"""CLI secret 提示回归测试：secret 类型必须走 getpass 不回显。"""

from __future__ import annotations

import pytest

from pi_ai.cli import _CliAuthInteraction


@pytest.mark.asyncio
async def test_secret_prompt_uses_getpass(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_getpass(prompt: str) -> str:
        calls.append(("getpass", prompt))
        return "sk-secret"

    monkeypatch.setattr("pi_ai.cli.getpass.getpass", fake_getpass)

    async def fake_input(prompt: str) -> str:
        calls.append(("input", prompt))
        return "should-not-be-used"

    monkeypatch.setattr("builtins.input", fake_input)

    interaction = _CliAuthInteraction()
    result = await interaction.prompt({"type": "secret", "message": "Enter API key"})
    assert result == "sk-secret"
    assert [kind for kind, _ in calls] == ["getpass"]


@pytest.mark.asyncio
async def test_text_prompt_still_uses_input(monkeypatch) -> None:
    calls: list[str] = []

    def fake_input(prompt: str) -> str:
        calls.append(prompt)
        return "value"

    monkeypatch.setattr("builtins.input", fake_input)
    interaction = _CliAuthInteraction()
    result = await interaction.prompt({"type": "text", "message": "Name"})
    assert result == "value"
    assert calls == ["Name: "]
