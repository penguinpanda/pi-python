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
    def test_login_with_provider_persists(
        self, fake_providers, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.chdir(tmp_path)
        assert cli.main(["login", "fake"]) == 0
        assert "Credentials saved to auth.json" in capsys.readouterr().out

        raw = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
        assert raw["fake"]["access"] == "sk-fake"
        assert raw["fake"]["refresh"] == "rf-fake"

    def test_login_select_provider(
        self, fake_providers, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("builtins.input", lambda _prompt: "1")
        assert cli.main(["login"]) == 0
        out = capsys.readouterr().out
        assert "Select a provider:" in out
        assert "Credentials saved to auth.json" in out

    def test_login_unknown_provider(self, fake_providers, capsys):
        assert cli.main(["login", "nope"]) == 1
        assert "Unknown provider: nope" in capsys.readouterr().err


def test_unknown_command(fake_providers, capsys):
    assert cli.main(["bogus"]) == 1
    assert "Unknown command: bogus" in capsys.readouterr().err
