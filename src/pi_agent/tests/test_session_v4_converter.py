"""pi_agent.session.v4.converter 补充测试。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from pi_agent.session.v4 import converter
from pi_agent.session.v4.types import SessionError


def _iso_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def test_iso_to_ms_variants() -> None:
    assert converter._iso_to_ms(123) == 123
    assert converter._iso_to_ms(True) == 0
    assert converter._iso_to_ms("2026-08-10T00:00:00Z") == _iso_ms("2026-08-10T00:00:00Z")
    assert converter._iso_to_ms("not-a-date") == 0
    assert converter._iso_to_ms(None) == 0


def test_parent_session_id_from_path() -> None:
    assert (
        converter._parent_session_id_from_path("/tmp/2026-01-01T00-00-00_parent.jsonl") == "parent"
    )
    assert converter._parent_session_id_from_path("/tmp/plain.jsonl") is None
    assert converter._parent_session_id_from_path("/tmp/bad--id.jsonl") is None


@pytest.mark.parametrize(
    "header",
    [
        "not-json",
        json.dumps({"type": "wrong"}),
        json.dumps({"type": "session", "version": 2}),
    ],
)
def test_v3_header_metadata_invalid(header: str) -> None:
    with pytest.raises(SessionError):
        converter.v3_header_metadata(header, "/tmp/x.jsonl", 0)


def test_v3_header_metadata_parent_and_metadata() -> None:
    header = json.dumps(
        {
            "type": "session",
            "version": 3,
            "id": "s1",
            "timestamp": "2026-08-10T00:00:00Z",
            "cwd": "/tmp",
            "parentSession": "/tmp/2026-01-01T00-00-00_parent.jsonl",
            "metadata": {"key": "value"},
        }
    )
    metadata = converter.v3_header_metadata(header, "/tmp/x.jsonl", 7)
    assert metadata["parentSessionId"] == "parent"
    assert metadata["metadata"] == {"key": "value"}
    assert metadata["modifiedAt"] == 7

    header = json.dumps(
        {
            "type": "session",
            "version": 3,
            "id": "s2",
            "timestamp": "2026-08-10T00:00:00Z",
            "cwd": "/tmp",
            "parentSession": "legacy-path",
        }
    )
    metadata = converter.v3_header_metadata(header, "/tmp/y.jsonl", 0)
    assert metadata["legacyParentSessionPath"] == "legacy-path"


def test_entry_mutation_variants() -> None:
    base = {
        "id": "e1",
        "parentId": None,
        "timestamp": "2026-08-10T00:00:00Z",
    }
    message = converter._entry_mutation(
        {**base, "type": "message", "message": {"role": "user", "content": "q"}},
        1,
    )
    assert message["entry"]["type"] == "message"

    custom = converter._entry_mutation(
        {**base, "type": "custom", "customType": "note", "data": {"x": 1}},
        2,
    )
    assert custom["entry"]["customType"] == "note"
    assert custom["entry"]["data"] == {"x": 1}

    branch = converter._entry_mutation(
        {
            **base,
            "type": "branch_summary",
            "fromId": "old",
            "summary": "s",
            "details": {"readFiles": ["a.py"]},
            "usage": {"input": 1},
        },
        3,
    )
    assert branch["entry"]["details"] == {"readFiles": ["a.py"]}
    assert branch["entry"]["usage"] == {"input": 1}

    custom_message = converter._entry_mutation(
        {
            **base,
            "type": "custom_message",
            "customType": "note",
            "content": [{"type": "text", "text": "x"}],
            "display": True,
        },
        4,
    )
    assert custom_message["entry"]["message"]["role"] == "custom"

    with pytest.raises(SessionError, match="Unsupported"):
        converter._entry_mutation({**base, "type": "bogus"}, 5)


def test_compaction_mutation_retained_tail() -> None:
    entry = {
        "type": "compaction",
        "id": "c1",
        "summary": "s",
        "tokensBefore": 10,
        "parentId": None,
        "timestamp": "2026-08-10T00:00:00Z",
        "retainedTail": [{"role": "user", "content": "x"}],
        "details": {"readFiles": ["a.py"]},
        "usage": {"input": 1},
    }
    mutation = converter._compaction_mutation(entry, 1, [])
    assert mutation["entry"]["retainedTail"] == [{"role": "user", "content": "x"}]
    assert mutation["entry"]["details"] == {"readFiles": ["a.py"]}
    assert mutation["entry"]["usage"] == {"input": 1}

    derived = converter._compaction_mutation(
        {
            "type": "compaction",
            "id": "c2",
            "summary": "s",
            "firstKeptEntryId": "tail",
            "parentId": None,
            "timestamp": 1,
        },
        2,
        [
            {"id": "tail", "type": "message", "message": {"role": "user", "content": "tail"}},
            {"id": "after", "type": "message", "message": {"role": "assistant", "content": "a"}},
        ],
    )
    assert len(derived["entry"]["retainedTail"]) == 2


def test_v4_header_parent_variants() -> None:
    header = converter._v4_header(
        {"id": "s", "timestamp": "2026-08-10T00:00:00Z", "metadata": {"k": 1}},
        "/tmp",
        "/tmp/2026-01-01T00-00-00_parent.jsonl",
    )
    assert header["parentSessionId"] == "parent"
    assert header["metadata"] == {"k": 1}

    header = converter._v4_header({}, "/tmp", "legacy")
    assert header["legacyParentSessionPath"] == "legacy"


def _write_v3(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": "s1",
                "timestamp": "2026-08-10T00:00:00Z",
                "cwd": "/tmp",
            }
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_convert_rollback_on_load_failure(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "v3.jsonl"
    _write_v3(path)
    original = path.read_bytes()

    async def fail_load(*_args, **_kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(converter.JsonlSessionStorage, "load", fail_load)
    with pytest.raises(RuntimeError, match="load failed"):
        await converter.convert_v3_file_to_v4(str(path))
    assert path.read_bytes() == original
    assert not path.with_suffix(".jsonl.v4tmp").exists()


@pytest.mark.asyncio
async def test_convert_restores_backup_when_replace_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "v3.jsonl"
    _write_v3(path)
    original = path.read_bytes()
    real_replace = os.replace
    state = {"calls": 0}

    def fake_replace(src: str, dst: str) -> None:
        if str(dst) == str(path) and state["calls"] == 0:
            state["calls"] += 1
            raise OSError("replace failed")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fake_replace)
    with pytest.raises(OSError, match="replace failed"):
        await converter.convert_v3_file_to_v4(str(path))
    assert path.read_bytes() == original
