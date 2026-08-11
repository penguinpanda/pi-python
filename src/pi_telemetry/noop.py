"""No-op telemetry implementation。"""

from __future__ import annotations

from typing import Awaitable, Callable, TypeVar

from .context import SpanAttributes, SpanOptions, SpanStatus, TelemetryContext, TelemetrySpan

T = TypeVar("T")


class NoopTelemetrySpan:
    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None:
        return None

    def set_attributes(self, attributes: SpanAttributes) -> None:
        return None

    def set_status(self, status: SpanStatus) -> None:
        return None

    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[TelemetrySpan], Awaitable[T]],
    ) -> T:
        return await callback(self)


class NoopTelemetryContext:
    async def start_span(
        self,
        options: SpanOptions,
        callback: Callable[[TelemetrySpan], Awaitable[T]],
    ) -> T:
        return await callback(NoopTelemetrySpan())


NOOP_TELEMETRY_CONTEXT: TelemetryContext = NoopTelemetryContext()


__all__ = ["NoopTelemetrySpan", "NoopTelemetryContext", "NOOP_TELEMETRY_CONTEXT"]
