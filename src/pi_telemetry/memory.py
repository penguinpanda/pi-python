"""In-memory telemetry context for tests and lightweight observability。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from .context import SpanAttributes, SpanOptions, SpanStatus, TelemetrySpan

T = TypeVar("T")


@dataclass(slots=True)
class RecordedSpan:
    name: str
    attributes: SpanAttributes = field(default_factory=dict)
    events: list[tuple[str, SpanAttributes]] = field(default_factory=list)
    status: SpanStatus | None = None


class InMemoryTelemetrySpan:
    def __init__(self, record: RecordedSpan) -> None:
        self._record = record

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        self._record.events.append((name, attributes or {}))

    def set_attributes(self, attributes: SpanAttributes) -> None:
        self._record.attributes.update(attributes)

    def set_status(self, status: SpanStatus) -> None:
        self._record.status = status

    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[TelemetrySpan], Awaitable[T]],
    ) -> T:
        child = RecordedSpan(name=options.name, attributes=dict(options.attributes or {}))
        self._record.events.append(("span", {"name": options.name}))
        return await callback(InMemoryTelemetrySpan(child))


class InMemoryTelemetryContext:
    def __init__(self) -> None:
        self.spans: list[RecordedSpan] = []

    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[TelemetrySpan], Awaitable[T]],
    ) -> T:
        record = RecordedSpan(name=options.name, attributes=dict(options.attributes or {}))
        self.spans.append(record)
        return await callback(InMemoryTelemetrySpan(record))


__all__ = ["RecordedSpan", "InMemoryTelemetrySpan", "InMemoryTelemetryContext"]
