"""slash 命令自动补全 provider 测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from pi_coding_agent.modes.interactive.autocomplete import (
    create_interactive_autocomplete_provider,
    create_slash_command_provider,
)


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


@pytest.mark.asyncio
async def test_unified_provider_combines_sources():
    registry = _Registry(
        [
            _Command("model", "Select model"),
            _Command("new", "Start new session"),
        ]
    )
    templates = _Templates(["review"])
    extension_runner = SimpleNamespace(
        get_registered_commands=lambda: [
            SimpleNamespace(
                name="ext-cmd",
                description="Extension command",
                argument_hint="<env>",
                get_argument_completions=lambda prefix: [
                    {"value": f"env-{prefix}", "label": f"env-{prefix}"}
                ],
            )
        ],
        get_autocomplete=lambda: [lambda text: [{"value": "ext:inline", "label": "Ext Inline"}]],
    )
    skill_loader = SimpleNamespace(
        all=lambda: [SimpleNamespace(name="brave-search", description="Search the web")]
    )
    settings_manager = SimpleNamespace(get_enable_skill_commands=lambda: True)
    provider = create_interactive_autocomplete_provider(
        slash_registry=registry,
        template_loader=templates,
        extension_runner=extension_runner,
        skill_loader=skill_loader,
        settings_manager=settings_manager,
        base_path="/tmp",
    )
    suggestions = await provider.get_suggestions("/")
    assert suggestions is not None
    names = {item.value for item in suggestions.items}
    assert {"model", "new", "review", "ext-cmd", "skill:brave-search"} <= names
    inline = await provider.get_suggestions("hello")
    assert inline is not None
    assert any(item.value == "ext:inline" for item in inline.items)
    ext_args = await provider.get_suggestions("/ext-cmd dev", force=True)
    assert ext_args is not None
    assert ext_args.items[0].value == "env-dev"


@pytest.mark.asyncio
async def test_unified_provider_model_and_login_completion():
    registry = _Registry(
        [
            _Command("model", "Select model"),
            _Command("login", "Configure provider authentication"),
        ]
    )

    class _Runtime:
        def get_available_snapshot(self):
            return [
                SimpleNamespace(
                    provider="faux",
                    id="faux-1",
                    name="Faux One",
                    aliases=["alias-1"],
                )
            ]

        def get_providers(self):
            return [SimpleNamespace(id="faux", auth=SimpleNamespace(env_vars=["X"]))]

        async def check_auth(self, provider_id):
            return None

    session = SimpleNamespace(scoped_models=[])
    provider = create_interactive_autocomplete_provider(
        slash_registry=registry,
        model_runtime=_Runtime(),
        session=session,
        base_path="/tmp",
    )
    model_suggestions = await provider.get_suggestions("/model faux", force=True)
    assert model_suggestions is not None
    assert model_suggestions.items[0].value == "faux/faux-1"
    alias_suggestions = await provider.get_suggestions("/model alias-1", force=True)
    assert alias_suggestions is not None
    assert alias_suggestions.items[0].value == "faux/faux-1"

    login_suggestions = await provider.get_suggestions("/login faux", force=True)
    assert login_suggestions is not None
    assert login_suggestions.items[0].value == "faux"


@pytest.mark.asyncio
async def test_model_completion_auto_derived_aliases():
    registry = _Registry([_Command("model", "Select model")])

    class _Runtime:
        def get_available_snapshot(self):
            return [
                SimpleNamespace(
                    provider="faux",
                    id="faux-1-20250101",
                    name="Faux One",
                    aliases=[],
                )
            ]

    session = SimpleNamespace(scoped_models=[])
    provider = create_interactive_autocomplete_provider(
        slash_registry=registry,
        model_runtime=_Runtime(),
        session=session,
        base_path="/tmp",
    )
    by_name = await provider.get_suggestions("/model faux one", force=True)
    assert by_name is not None
    assert by_name.items[0].value == "faux/faux-1-20250101"
    by_versionless = await provider.get_suggestions("/model faux-1", force=True)
    assert by_versionless is not None
    assert by_versionless.items[0].value == "faux/faux-1-20250101"


@pytest.mark.asyncio
async def test_unified_provider_skips_skills_when_disabled():
    registry = _Registry([_Command("new")])
    skill_loader = SimpleNamespace(
        all=lambda: [SimpleNamespace(name="hidden-skill", description="")]
    )
    settings_manager = SimpleNamespace(get_enable_skill_commands=lambda: False)
    provider = create_interactive_autocomplete_provider(
        slash_registry=registry,
        skill_loader=skill_loader,
        settings_manager=settings_manager,
        base_path="/tmp",
    )
    suggestions = await provider.get_suggestions("/")
    assert suggestions is not None
    assert all(item.value != "skill:hidden-skill" for item in suggestions.items)
