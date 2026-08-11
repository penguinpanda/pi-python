"""泛型 UndoStack 测试。"""

from __future__ import annotations

from pi_tui.engine import UndoStack


def test_push_clones_state() -> None:
    state = {"lines": ["a"]}
    stack = UndoStack[dict[str, list[str]]]()
    stack.push(state)
    state["lines"].append("b")
    restored = stack.pop()
    assert restored is not None
    assert restored["lines"] == ["a"]


def test_pop_empty_returns_none() -> None:
    stack = UndoStack[int]()
    assert stack.pop() is None


def test_max_length_drops_oldest() -> None:
    stack = UndoStack[int](max_length=2)
    stack.push(1)
    stack.push(2)
    stack.push(3)
    assert len(stack) == 2
    assert stack.pop() == 3
    assert stack.pop() == 2
    assert stack.pop() is None


def test_clear_and_bool() -> None:
    stack = UndoStack[int]()
    assert not stack
    stack.push(1)
    assert stack
    stack.clear()
    assert not stack
