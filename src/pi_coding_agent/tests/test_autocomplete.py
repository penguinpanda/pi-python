"""slash 命令自动补全 provider 测试。"""

from __future__ import annotations

from types import SimpleNamespace

from pi_coding_agent.modes.interactive.autocomplete import create_slash_command_provider


class _Command:
    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description


class _Registry:
    def __init__(self, commands) -> None:
        self._commands = commands

    def list(self):
        return list(self._commands)


class _Templates:
    def __init__(self, names: list[str]) -> None:
        self._names = names

    def all(self):
        return [SimpleNamespace(name=name, description=name) for name in self._names]


def test_slash_provider_filters_by_prefix():
    registry = _Registry(
        [
            _Command("model", "Select model"),
            _Command("new", "Start new session"),
        ]
    )
    provider = create_slash_command_provider(registry)

    items = provider("/")
    assert {item["value"] for item in items} == {"/model ", "/new "}
    assert {item["label"] for item in items} == {"Select model", "Start new session"}

    assert [item["value"] for item in provider("/mo")] == ["/model "]
    assert provider("hello") == []
    assert provider("/model x") == []  # 已有参数时不再补全命令名


def test_slash_provider_includes_templates():
    registry = _Registry([_Command("model")])
    provider = create_slash_command_provider(registry, _Templates(["review"]))
    assert {item["value"] for item in provider("/")} == {"/model ", "/review "}
