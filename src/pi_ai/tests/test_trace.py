"""pi_ai.trace 运行时实现测试（类型定义在 pi_ai.types.trace）。"""

import pytest

from pi_ai.trace import (
    Trace,
    TraceSpanHandle,
    TraceTracer,
    current_trace,
    reset_current_trace,
    run_with_trace,
    set_current_trace,
    trace_duration_ms,
    trace_from_dict,
    trace_to_dict,
)


class TestTraceTracer:
    def test_auto_init_trace(self):
        tracer = TraceTracer(name="agent.run")
        assert tracer.trace is not None
        assert tracer.trace.trace_id
        assert tracer.trace.name == "agent.run"
        assert tracer.trace.start_time > 0
        assert tracer.trace.end_time is None

    def test_start_span_records_span(self):
        tracer = TraceTracer()
        with tracer.start_span("llm.call", metadata={"model": "gpt-4o"}) as span:
            assert isinstance(span, TraceSpanHandle)
            assert span.start_time > 0

        assert tracer.trace is not None
        assert len(tracer.trace.spans) == 1
        recorded = tracer.trace.spans[0]
        assert recorded["name"] == "llm.call"
        assert recorded["status"] == "ok"
        assert recorded["metadata"] == {"model": "gpt-4o"}
        assert recorded["end_time"] is not None
        assert recorded["end_time"] >= recorded["start_time"]

    def test_span_exception_sets_error_status(self):
        tracer = TraceTracer()
        with pytest.raises(RuntimeError):
            with tracer.start_span("tool.execute"):
                raise RuntimeError("boom")

        assert tracer.trace is not None
        assert tracer.trace.spans[0]["status"] == "error"

    def test_span_explicit_end_is_idempotent(self):
        tracer = TraceTracer()
        span = tracer.start_span("a")
        first = span.end(status="ok")
        second = span.end(status="error")
        assert first is second
        assert tracer.trace is not None
        assert len(tracer.trace.spans) == 1

    def test_tracer_finish_sets_end_time(self):
        tracer = TraceTracer()
        trace = tracer.finish()
        assert trace.end_time is not None
        assert trace_duration_ms(trace) is not None


class TestTraceContext:
    def test_tracer_context_sets_current_trace(self):
        with TraceTracer(name="ctx") as tracer:
            assert current_trace() is tracer.trace
        assert current_trace() is None

    def test_run_with_trace(self):
        trace = Trace(trace_id="t-1", name="outer")
        seen = []

        def _inner():
            seen.append(current_trace())

        run_with_trace(trace, _inner)
        assert seen == [trace]
        assert current_trace() is None

    def test_nested_traces_restore_previous(self):
        outer = Trace(trace_id="outer")
        inner = Trace(trace_id="inner")

        def _inner_fn():
            assert current_trace() is inner

        def _outer_fn():
            assert current_trace() is outer
            run_with_trace(inner, _inner_fn)
            assert current_trace() is outer

        run_with_trace(outer, _outer_fn)
        assert current_trace() is None

    def test_set_current_trace_token(self):
        trace = Trace(trace_id="t-2")
        token = set_current_trace(trace)
        try:
            assert current_trace() is trace
        finally:
            reset_current_trace(token)
        assert current_trace() is None


class TestTraceSerialization:
    def test_to_dict_roundtrip(self):
        tracer = TraceTracer(name="roundtrip")
        with tracer.start_span("span-1"):
            pass
        trace = tracer.finish()

        data = trace_to_dict(trace)
        restored = trace_from_dict(data)
        assert restored.trace_id == trace.trace_id
        assert restored.name == trace.name
        assert restored.start_time == trace.start_time
        assert restored.end_time == trace.end_time
        assert restored.spans == trace.spans

    def test_from_dict_missing_optional_fields(self):
        restored = trace_from_dict({"trace_id": "t-3"})
        assert restored.trace_id == "t-3"
        assert restored.spans == []
        assert restored.end_time is None
