"""cache_fingerprint 单元测试。"""

from __future__ import annotations

from pi_coding_agent.cache_fingerprint import (
    classify_context_change,
    compute_context_fingerprint,
)


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


def test_same_inputs_same_fingerprint():
    messages = [{"role": "user", "content": "hi"}]
    tools = [_Tool("read")]
    first = compute_context_fingerprint("sys", messages, tools)
    second = compute_context_fingerprint("sys", messages, tools)
    assert first == second


def test_system_change_only():
    messages = [{"role": "user", "content": "hi"}]
    tools = [_Tool("read")]
    first = compute_context_fingerprint("sys-a", messages, tools)
    second = compute_context_fingerprint("sys-b", messages, tools)
    assert first["messages"] == second["messages"]
    assert first["tools"] == second["tools"]
    assert first["system"] != second["system"]


def test_messages_change_only():
    tools = [_Tool("read")]
    first = compute_context_fingerprint("sys", [{"role": "user", "content": "hi"}], tools)
    second = compute_context_fingerprint("sys", [{"role": "user", "content": "hi2"}], tools)
    assert first["system"] == second["system"]
    assert first["tools"] == second["tools"]
    assert first["messages"] != second["messages"]


def test_tools_change_only():
    messages = [{"role": "user", "content": "hi"}]
    first = compute_context_fingerprint("sys", messages, [_Tool("read")])
    second = compute_context_fingerprint("sys", messages, [_Tool("bash")])
    assert first["system"] == second["system"]
    assert first["messages"] == second["messages"]
    assert first["tools"] != second["tools"]


def test_classify_no_change():
    fp = compute_context_fingerprint("sys", [], [])
    assert classify_context_change(fp, fp, []) == []


def test_classify_append_only():
    prev = compute_context_fingerprint("sys", [{"role": "user", "content": "hi"}], [])
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]
    cur = compute_context_fingerprint("sys", messages, [])
    assert classify_context_change(prev, cur, messages) == ["append"]


def test_classify_compaction():
    prev = compute_context_fingerprint("sys", [{"role": "user", "content": "hi"}], [])
    messages = [
        {"role": "compactionSummary", "summary": "s"},
        {"role": "user", "content": "hi"},
    ]
    cur = compute_context_fingerprint("sys", messages, [])
    assert classify_context_change(prev, cur, messages) == ["compaction"]


def test_classify_system_change():
    prev = compute_context_fingerprint("sys-a", [], [])
    cur = compute_context_fingerprint("sys-b", [], [])
    assert classify_context_change(prev, cur, []) == ["system"]


def test_classify_tools_change():
    prev = compute_context_fingerprint("sys", [], [_Tool("read")])
    cur = compute_context_fingerprint("sys", [], [_Tool("bash")])
    assert classify_context_change(prev, cur, []) == ["tools"]
