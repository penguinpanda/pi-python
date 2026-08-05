"""自动补全 provider 栈（对齐 TS CombinedAutocompleteProvider 的 Python 子集）。

provider 签名：`(text: str) -> list[dict] | Awaitable[list[dict]]`，
item 支持 `value` / `label`（与扩展 API 一致）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

AutocompleteProvider = Callable[[str], list[dict[str, Any]] | Awaitable[list[dict[str, Any]]]]


class CombinedAutocompleteProvider:
    """按注册顺序合并多个 provider；支持同步/异步；按 value 去重。"""

    def __init__(self, providers: list[AutocompleteProvider] | None = None) -> None:
        self._providers = list(providers or [])

    def add(self, provider: AutocompleteProvider) -> None:
        self._providers.append(provider)

    @property
    def providers(self) -> list[AutocompleteProvider]:
        return list(self._providers)

    async def collect(self, text: str) -> list[dict[str, Any]]:
        """并发收集所有 provider 结果，按 value 去重后保持注册顺序。"""
        if not self._providers:
            return []

        async def _run(provider: AutocompleteProvider) -> list[dict[str, Any]]:
            try:
                result = provider(text)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception:
                return []
            if not isinstance(result, list):
                return []
            return [item for item in result if isinstance(item, dict)]

        results = await asyncio.gather(*(_run(provider) for provider in self._providers))
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for items in results:
            for item in items:
                value = str(item.get("value", item.get("label", "")))
                if not value or value in seen:
                    continue
                seen.add(value)
                merged.append(item)
        return merged
