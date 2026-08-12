"""SessionPicker 键盘状态与渲染测试（依赖 coding-agent 的 SessionPickerModel）。"""

from __future__ import annotations

from pi_coding_agent.modes.interactive.session_selector import SessionPickerModel
from pi_tui.engine.keys import Key
from pi_tui.keybindings import KeybindingsManager
from pi_tui.selectors import SessionPicker


def _picker() -> SessionPicker:
    sessions = [
        {
            "path": "/tmp/current.jsonl",
            "session_id": "current",
            "cwd": "/tmp",
            "modified": 200,
            "name": "current task",
        },
        {
            "path": "/tmp/other.jsonl",
            "session_id": "other",
            "cwd": "/tmp",
            "modified": 100,
            "name": None,
            "first_message": "other session",
        },
    ]
    model = SessionPickerModel(
        current_sessions=sessions,
        all_sessions=sessions,
        current_cwd="/tmp",
        current_session_path="/tmp/current.jsonl",
    )
    return SessionPicker(model)


def test_picker_scope_sort_and_path_toggles() -> None:
    picker = _picker()
    assert picker._model.scope.value == "current"
    picker.handle_key(Key(name="tab"))
    assert picker._model.scope.value == "all"
    picker.handle_key(Key(name="ctrl+s"))
    assert picker._model.sort_mode.value == "recent"
    picker.handle_key(Key(name="ctrl+p"))
    assert picker._model.show_path is True


def test_picker_named_filter() -> None:
    picker = _picker()
    picker.handle_key(Key(name="ctrl+n"))
    assert picker._model.name_filter.value == "named"
    assert [row.session.session_id for row in picker._model.rows] == ["current"]


def test_picker_current_session_delete_blocked() -> None:
    picker = _picker()
    picker.handle_key(Key(name="ctrl+d"))
    assert picker._confirming_path is None
    assert picker._status == ("error", "Cannot delete the currently active session")


def test_picker_rename_mode_and_render() -> None:
    picker = _picker()
    picker.handle_key(Key(name="ctrl+r"))
    assert picker._rename_mode is True
    assert picker._rename_input.value == "current task"
    lines = picker.render(80, 20)
    assert lines and lines[0].text().startswith("Rename Session")


def test_picker_renders_rows() -> None:
    picker = _picker()
    lines = picker.render(80, 20)
    text = "\n".join(line.text() for line in lines)
    assert "current task" in text
    assert "other session" in text


def test_picker_uses_keybindings_manager_override() -> None:
    manager = KeybindingsManager()
    manager.load_from_settings({"keybindings": {"app.session.toggleSort": "ctrl+k"}})
    model = SessionPickerModel(
        current_sessions=[
            {
                "path": "/tmp/a.jsonl",
                "session_id": "a",
                "cwd": "/tmp",
                "modified": 1,
            }
        ],
    )
    picker = SessionPicker(
        model,
        keybindings_manager=manager,
    )
    assert picker._session_key("app.session.toggleSort") == "ctrl+k"
