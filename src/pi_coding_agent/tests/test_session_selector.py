"""Session Selector 排序 / 过滤 / scope / 删除 / 重命名测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_coding_agent._session_manager import SessionInfo
from pi_coding_agent.modes.interactive.session_selector import (
    NameFilter,
    SessionPickerModel,
    SessionScope,
    SessionSortMode,
    build_session_tree,
    delete_session_file,
    filter_and_sort_sessions,
    has_session_name,
    match_session,
    parse_search_query,
    rename_session_file,
)


def _session(
    session_id: str,
    *,
    modified: float = 0,
    name: str | None = None,
    parent_session_id: str | None = None,
    first_message: str = "",
    cwd: str = "/tmp",
) -> SessionInfo:
    return SessionInfo(
        path=f"/tmp/{session_id}.jsonl",
        session_id=session_id,
        cwd=cwd,
        modified=modified,
        name=name,
        parent_session_id=parent_session_id,
        first_message=first_message,
        message_count=1,
        search_text=f"{session_id} {name or ''} {first_message} {cwd}",
    )


def test_sort_recent_keeps_input_order() -> None:
    sessions = [
        _session("older", modified=100, first_message="alpha"),
        _session("newer", modified=200, first_message="beta"),
    ]
    result = filter_and_sort_sessions(sessions, "", SessionSortMode.RECENT)
    assert [session.session_id for session in result] == ["older", "newer"]


def test_sort_relevance_fuzzy() -> None:
    sessions = [
        _session("a", first_message="review auth flow"),
        _session("b", first_message="fix flaky tests"),
    ]
    result = filter_and_sort_sessions(sessions, "auth", SessionSortMode.RELEVANCE)
    assert [session.session_id for session in result] == ["a"]


def test_search_phrase_and_regex() -> None:
    sessions = [
        _session("a", first_message="node cve review"),
        _session("b", first_message="other"),
    ]
    assert match_session(sessions[0], parse_search_query('"node cve"')).matches
    assert not match_session(sessions[1], parse_search_query('"node cve"')).matches
    assert match_session(sessions[0], parse_search_query("re:node")).matches
    assert not match_session(sessions[1], parse_search_query("re:node")).matches


def test_named_filter() -> None:
    sessions = [_session("named", name="task"), _session("unnamed")]
    assert has_session_name(sessions[0])
    result = filter_and_sort_sessions(sessions, "", SessionSortMode.RECENT, NameFilter.NAMED)
    assert [session.session_id for session in result] == ["named"]


def test_threaded_tree_sorts_by_latest_activity() -> None:
    parent = _session("parent", modified=100)
    child = _session(
        "child",
        modified=300,
        parent_session_id="parent",
    )
    rows = build_session_tree([parent, child])
    flat = []

    def walk(node) -> None:
        flat.append(node.session.session_id)
        for child_node in node.children:
            walk(child_node)

    for root in rows:
        walk(root)
    assert flat == ["parent", "child"]


def test_picker_scope_and_path_toggle() -> None:
    current = [_session("current")]
    all_sessions = [_session("other"), _session("current")]
    model = SessionPickerModel(
        current_sessions=current,
        all_sessions=all_sessions,
        current_session_path="/tmp/current.jsonl",
    )
    assert [row.session.session_id for row in model.rows] == ["current"]
    model.toggle_scope()
    assert model.scope == SessionScope.ALL
    assert {row.session.session_id for row in model.rows} == {"other", "current"}
    assert model.show_path is False
    model.toggle_path()
    assert model.show_path is True


def test_picker_cannot_delete_current_session() -> None:
    model = SessionPickerModel(
        current_sessions=[_session("current")],
        current_session_path="/tmp/current.jsonl",
    )
    assert model.is_selected_current is True


@pytest.mark.asyncio
async def test_delete_session_file_falls_back_to_unlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "pi_coding_agent.modes.interactive.session_selector.shutil.which",
        lambda _name: None,
    )
    path = tmp_path / "session.jsonl"
    path.write_text("{}", encoding="utf-8")
    result = await delete_session_file(str(path))
    assert result.ok is True
    assert result.method == "unlink"
    assert not path.exists()


@pytest.mark.asyncio
async def test_rename_session_file(tmp_path: Path) -> None:
    from pi_ai.types import UserMessage

    from pi_coding_agent._session_manager_v4 import create_session_manager, open_session_manager

    manager = await create_session_manager(
        str(tmp_path),
        sessions_dir=str(tmp_path / "sessions"),
        session_id="rename-me",
    )
    await manager.append_message(UserMessage(role="user", content="hello"))
    path = manager.session_path
    assert path is not None
    close = getattr(manager, "close", None)
    if close is not None:
        await close()

    result = await rename_session_file(str(path), "task")
    assert result.ok is True
    reopened = await open_session_manager(path)
    try:
        assert reopened.session_name == "task"
    finally:
        close = getattr(reopened, "close", None)
        if close is not None:
            await close()


@pytest.mark.asyncio
async def test_list_sessions_reuses_search_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from pi_ai.types import UserMessage

    from pi_coding_agent._session_manager_v4 import (
        create_session_manager,
        list_sessions,
    )

    sessions_dir = tmp_path / "sessions"
    manager = await create_session_manager(
        str(tmp_path),
        sessions_dir=str(sessions_dir),
        session_id="index-me",
    )
    await manager.append_message(UserMessage(role="user", content="search me"))
    await manager.append_session_info("task")
    close = getattr(manager, "close", None)
    if close is not None:
        await close()

    await list_sessions(sessions_dir, cache_search_index=True)

    def fail_open(_self, _metadata):
        raise AssertionError("list_sessions should use the search index")

    monkeypatch.setattr(
        "pi_coding_agent._session_manager_v4.JsonlSessionRepo.open",
        fail_open,
    )
    infos = await list_sessions(sessions_dir, cache_search_index=True)
    assert len(infos) == 1
    assert infos[0].name == "task"
    assert infos[0].first_message == "search me"
    assert "search me" in infos[0].search_text
