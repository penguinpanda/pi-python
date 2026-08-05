"""自动补全 provider 栈测试。"""

from __future__ import annotations

import asyncio

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
