"""pi_ai.cli（对齐 TS packages/ai/src/cli.ts）测试。"""

import json

import pytest

from pi_ai import cli


class _FakeFlow:
    name = "Fake OAuth"

    async def login(self, interaction):
        return {
            "type": "oauth",
            "access": "sk-fake",
            "refresh": "rf-fake",
            "expires": 9999999999999,
        }


def _fake_providers():
    return [("fake", "Fake OAuth", _FakeFlow())]


@pytest.fixture
def fake_providers(monkeypatch):
    monkeypatch.setattr("pi_ai.cli.builtin_oauth_providers", _fake_providers)


class TestUsage:
    def test_help_shows_usage(self, fake_providers, capsys):
        assert cli.main(["help"]) == 0
        out = capsys.readouterr().out
        assert "Usage: pi-ai <command> [provider]" in out
        assert "login [provider]" in out
        assert "list" in out
        assert "Fake OAuth" in out

    def test_no_args_shows_usage(self, fake_providers, capsys):
        assert cli.main([]) == 0
        assert "Usage:" in capsys.readouterr().out


class TestList:
    def test_list_providers(self, fake_providers, capsys):
        assert cli.main(["list"]) == 0
        out = capsys.readouterr().out
        assert "fake" in out
        assert "Fake OAuth" in out


class TestLogin:
    def test_login_with_provider_persists(self, fake_providers, monkeypatch, tmp_path, capsys):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr(cli, "AUTH_FILE", str(auth_file))
        assert cli.main(["login", "fake"]) == 0
        assert "Credentials saved to" in capsys.readouterr().out

        raw = json.loads(auth_file.read_text(encoding="utf-8"))
        assert raw["fake"]["access"] == "sk-fake"
        assert raw["fake"]["refresh"] == "rf-fake"

    def test_login_select_provider(self, fake_providers, monkeypatch, tmp_path, capsys):
        auth_file = tmp_path / "auth.json"
        monkeypatch.setattr(cli, "AUTH_FILE", str(auth_file))
        monkeypatch.setattr("builtins.input", lambda _prompt: "1")
        assert cli.main(["login"]) == 0
        out = capsys.readouterr().out
        assert "Select a provider:" in out
        assert "Credentials saved to" in out
        assert auth_file.exists()

    def test_login_unknown_provider(self, fake_providers, capsys):
        assert cli.main(["login", "nope"]) == 1
        assert "Unknown provider: nope" in capsys.readouterr().err


def test_unknown_command(fake_providers, capsys):
    assert cli.main(["bogus"]) == 1
    assert "Unknown command: bogus" in capsys.readouterr().err


def _run(coro):
    import asyncio

    return asyncio.run(coro)


class TestCliAuthInteraction:
    def test_prompt_select_invalid_then_valid(self, capsys, monkeypatch):
        answers = iter(["99", "1"])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
        interaction = cli._CliAuthInteraction()
        result = _run(
            interaction.prompt(
                {
                    "type": "select",
                    "message": "Pick",
                    "options": [{"id": "a", "label": "A"}],
                }
            )
        )
        assert result == "a"
        assert "Invalid selection." in capsys.readouterr().out

    def test_prompt_input_with_placeholder(self, capsys, monkeypatch):
        prompts: list[str] = []
        monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "typed")
        interaction = cli._CliAuthInteraction()
        result = _run(interaction.prompt({"message": "Name", "placeholder": "hint"}))
        assert result == "typed"
        assert prompts == ["Name (hint): "]

    def test_notify_events(self, capsys):
        interaction = cli._CliAuthInteraction()
        interaction.notify({"type": "auth_url", "url": "https://x", "instructions": "Open it"})
        interaction.notify(
            {"type": "device_code", "verification_uri": "https://y", "user_code": "ABC"}
        )
        interaction.notify({"type": "info", "message": "working"})
        interaction.notify({"type": "progress", "message": "still working"})
        out = capsys.readouterr().out
        assert "https://x" in out and "Open it" in out
        assert "https://y" in out and "Enter code: ABC" in out
        assert "working" in out and "still working" in out
