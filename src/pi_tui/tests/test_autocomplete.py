"""自动补全 provider 栈测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from pi_tui.autocomplete import CombinedAutocompleteProvider


@pytest.mark.asyncio
async def test_merge_and_dedupe_by_value() -> None:
    provider_a = lambda text: [  # noqa: E731
        {"value": "a", "label": "A"},
        {"value": "b"},
    ]
    provider_b = lambda text: [  # noqa: E731
        {"value": "b", "label": "B"},
        {"value": "c"},
    ]
    items = await CombinedAutocompleteProvider([provider_a, provider_b]).collect("")
    assert [item["value"] for item in items] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_async_provider() -> None:
    async def provider(text: str):
        await asyncio.sleep(0)
        return [{"value": "x", "label": "X"}]

    items = await CombinedAutocompleteProvider([provider]).collect("")
    assert items == [{"value": "x", "label": "X"}]


@pytest.mark.asyncio
async def test_exception_skipped() -> None:
    def bad_provider(text: str):
        raise RuntimeError("boom")

    good_provider = lambda text: [{"value": "ok"}]  # noqa: E731
    items = await CombinedAutocompleteProvider([bad_provider, good_provider]).collect("")
    assert items == [{"value": "ok"}]


@pytest.mark.asyncio
async def test_concurrent_collect_keeps_provider_order() -> None:
    order: list[str] = []

    async def provider_a(text: str):
        await asyncio.sleep(0.05)
        order.append("a")
        return [{"value": "a"}]

    async def provider_b(text: str):
        order.append("b")
        return [{"value": "b"}]

    items = await CombinedAutocompleteProvider([provider_a, provider_b]).collect("")
    assert order == ["b", "a"]
    assert [item["value"] for item in items] == ["a", "b"]


@pytest.mark.asyncio
async def test_empty_and_invalid_results() -> None:
    items = await CombinedAutocompleteProvider(
        [lambda text: None, lambda text: "not-a-list"]  # noqa: E731
    ).collect("")
    assert items == []


@pytest.mark.asyncio
async def test_empty_providers() -> None:
    assert await CombinedAutocompleteProvider().collect("x") == []


@pytest.mark.asyncio
async def test_command_suggestions_fuzzy() -> None:
    provider = CombinedAutocompleteProvider(
        commands=[
            SimpleNamespace(
                name="model",
                description="Select model",
                argument_hint="<provider/model>",
            ),
            SimpleNamespace(
                name="new",
                description="Start a new session",
                argument_hint=None,
            ),
        ],
        base_path="/tmp",
    )
    suggestions = await provider.get_suggestions("/mo")
    assert suggestions is not None
    assert suggestions.kind == "command"
    assert [item.value for item in suggestions.items] == ["model"]
    assert suggestions.items[0].description == "<provider/model> — Select model"


@pytest.mark.asyncio
async def test_argument_suggestions() -> None:
    provider = CombinedAutocompleteProvider(
        commands=[
            SimpleNamespace(
                name="model",
                description="Select model",
                argument_hint="<provider/model>",
                get_argument_completions=lambda prefix: [
                    {"value": "faux/faux-1", "label": "faux-1", "description": "faux"}
                ],
            )
        ],
        base_path="/tmp",
    )
    suggestions = await provider.get_suggestions("/model faux", force=True)
    assert suggestions is not None
    assert suggestions.kind == "argument"
    assert suggestions.items[0].value == "faux/faux-1"


@pytest.mark.asyncio
async def test_path_completion_directories_first(tmp_path) -> None:
    (tmp_path / "alpha.txt").write_text("a", encoding="utf-8")
    (tmp_path / "beta").mkdir()
    provider = CombinedAutocompleteProvider(base_path=str(tmp_path))
    suggestions = await provider.get_suggestions("", force=True)
    assert suggestions is not None
    assert suggestions.kind == "path"
    assert suggestions.items[0].value.endswith("/")
    assert {item.label for item in suggestions.items} == {"beta/", "alpha.txt"}


@pytest.mark.asyncio
async def test_path_completion_quotes_spaces(tmp_path) -> None:
    (tmp_path / "my docs").mkdir()
    provider = CombinedAutocompleteProvider(base_path=str(tmp_path))
    suggestions = await provider.get_suggestions("my", force=True)
    assert suggestions is not None
    assert suggestions.items[0].value == '"my docs/"'


@pytest.mark.asyncio
async def test_apply_completion() -> None:
    provider = CombinedAutocompleteProvider(
        commands=[SimpleNamespace(name="model", description="", argument_hint=None)],
        base_path="/tmp",
    )
    suggestions = await provider.get_suggestions("/mo")
    assert suggestions is not None
    new_text, cursor = provider.apply_completion(
        "/mo",
        suggestions.items[0],
        suggestions.prefix,
    )
    assert new_text == "/model "
    assert cursor == len("/model ")


@pytest.mark.asyncio
async def test_extension_provider_fallback(tmp_path) -> None:
    provider = CombinedAutocompleteProvider(
        [lambda text: [{"value": "ext:item", "label": "Ext"}]],
        base_path=str(tmp_path),
    )
    suggestions = await provider.get_suggestions("hello")
    assert suggestions is not None
    assert suggestions.items[0].value == "ext:item"
