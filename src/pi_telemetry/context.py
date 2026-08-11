"""Vendor-neutral telemetry interfaces (对齐 TS packages/telemetry)。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol, TypeVar, runtime_checkable

AttributeValue = str | int | bool | tuple[str, ...] | tuple[int, ...] | tuple[bool, ...]
SpanAttributes = dict[str, AttributeValue | None]


@dataclass(slots=True)
class SpanOptions:
    name: str
    attributes: SpanAttributes | None = None


@dataclass(slots=True)
class SpanStatus:
    status: str
    error: dict[str, str] | None = None


T = TypeVar("T")


@runtime_checkable
class TelemetrySpan(Protocol):
    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None: ...

    def set_attributes(self, attributes: SpanAttributes) -> None: ...

    def set_status(self, status: SpanStatus) -> None: ...

    def start_span(
        self,
        options: SpanOptions,
        callback: Callable[["TelemetrySpan"], Awaitable[T]],
    ) -> Awaitable[T]: ...


@runtime_checkable
class TelemetryContext(Protocol):
    def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[TelemetrySpan], Awaitable[T]],
    ) -> Awaitable[T]: ...


__all__ = [
    "AttributeValue",
    "SpanAttributes",
    "SpanOptions",
    "SpanStatus",
    "TelemetrySpan",
    "TelemetryContext",
]
