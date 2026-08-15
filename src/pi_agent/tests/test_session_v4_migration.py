"""v3 → v4 惰性迁移测试（M2 验收）。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from pi_agent.session.v4.repo import JsonlSessionRepo
from pi_agent.session.v4.types import SessionError


def _iso_ms(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def _v3_lines(cwd: str, custom_message: bool = False) -> list[str]:
    lines = [
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": "v3-session",
                "timestamp": "2026-08-10T00:00:00+00:00",
                "cwd": cwd,
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "type": "message",
                "id": "root",
                "parentId": None,
                "timestamp": "2026-08-10T00:00:01+00:00",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "root"}],
                    "timestamp": 1,
                },
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "type": "message",
                "id": "shared",
                "parentId": "root",
                "timestamp": "2026-08-10T00:00:02+00:00",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "shared"}],
                    "api": "deepseek",
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                    "stopReason": "stop",
                    "timestamp": 2,
                },
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "type": "thinking_level_change",
                "id": "thinking",
                "parentId": "shared",
                "timestamp": "2026-08-10T00:00:03+00:00",
                "thinkingLevel": "high",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "type": "model_change",
                "id": "model",
                "parentId": "thinking",
                "timestamp": "2026-08-10T00:00:04+00:00",
                "provider": "deepseek",
                "modelId": "deepseek-v4-pro",
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "type": "compaction",
                "id": "compact",
                "parentId": "model",
                "timestamp": "2026-08-10T00:00:05+00:00",
                "summary": "checkpoint",
                "firstKeptEntryId": "tail",
                "tokensBefore": 100,
            },
            ensure_ascii=False,
        ),
        json.dumps(
            {
                "type": "message",
                "id": "tail",
                "parentId": "compact",
                "timestamp": "2026-08-10T00:00:06+00:00",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "tail"}],
                    "timestamp": 6,
                },
            },
            ensure_ascii=False,
        ),
    ]
    if custom_message:
        lines.append(
            json.dumps(
                {
                    "type": "custom_message",
                    "id": "custom-msg",
                    "parentId": "tail",
                    "timestamp": "2026-08-10T00:00:07+00:00",
                    "customType": "note",
                    "content": [{"type": "text", "text": "note"}],
                    "display": True,
                },
                ensure_ascii=False,
            )
        )
    lines.extend(
        [
            json.dumps(
                {
                    "type": "label",
                    "id": "label",
                    "parentId": None,
                    "timestamp": "2026-08-10T00:00:08+00:00",
                    "targetId": "root",
                    "label": "checkpoint",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "type": "session_info",
                    "id": "info",
                    "parentId": None,
                    "timestamp": "2026-08-10T00:00:09+00:00",
                    "name": "Migrated",
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "type": "leaf",
                    "id": "leaf",
                    "parentId": None,
                    "timestamp": "2026-08-10T00:00:10+00:00",
                    "targetId": "tail",
                },
                ensure_ascii=False,
            ),
        ]
    )
    return [line + "\n" for line in lines]


def _write_v3(tmp_path: Path, path: str) -> None:
    Path(path).write_text("".join(_v3_lines(str(tmp_path))), encoding="utf-8")


def _v3_metadata(path: str, cwd: str) -> dict:
    return {
        "id": "v3-session",
        "createdAt": _iso_ms("2026-08-10T00:00:00+00:00"),
        "cwd": cwd,
        "path": path,
        "modifiedAt": 0,
        "sourceFormat": 3,
    }


class TestLazyConversion:
    @pytest.mark.asyncio
    async def test_open_converts_v3_to_v4(self, tmp_path):
        repo = JsonlSessionRepo(str(tmp_path / "sessions"))
        path = str(tmp_path / "v3.jsonl")
        _write_v3(tmp_path, path)
        metadata = _v3_metadata(path, str(tmp_path))

        session = await repo.open(metadata)

        assert await session.get_name() == "Migrated"
        assert await session.get_label("root") == "checkpoint"
        assert await session.get_leaf_id() == "tail"
        assert (await session.get_stats())["messageCount"] == 3
        entries = await session.find_entries({"order": "oldestFirst"})
        assert [entry["id"] for entry in entries] == [
            "root",
            "shared",
            "thinking",
            "model",
            "compact",
            "tail",
        ]
        compaction = entries[4]
        assert compaction["type"] == "compaction"
        assert [message["content"] for message in compaction["retainedTail"]] == [
            [{"type": "text", "text": "tail"}]
        ]
        assert [item["seq"] for item in await session.get_log()] == list(range(1, 10))

        # 原文件已替换为 v4，备份保留
        assert Path(path).read_text(encoding="utf-8").startswith('{"kind": "header", "version": 4')
        assert Path(f"{path}.bak").exists()

    @pytest.mark.asyncio
    async def test_reopen_after_conversion_is_idempotent(self, tmp_path):
        repo = JsonlSessionRepo(str(tmp_path / "sessions"))
        path = str(tmp_path / "v3.jsonl")
        _write_v3(tmp_path, path)
        metadata = _v3_metadata(path, str(tmp_path))

        await repo.open(metadata)
        before = Path(f"{path}.bak").read_bytes()
        session = await repo.open(metadata)

        assert await session.get_leaf_id() == "tail"
        assert Path(f"{path}.bak").read_bytes() == before

    @pytest.mark.asyncio
    async def test_list_reports_v3_then_open_converts(self, tmp_path):
        repo = JsonlSessionRepo(str(tmp_path / "sessions"))
        cwd = str(tmp_path / "project")
        # 与仓库的目录编码保持一致（Windows 含盘符/反斜杠，手拼会产出非法字符）。
        from pi_agent.session.v4.repo import _session_directory_name

        cwd_dir = tmp_path / "sessions" / _session_directory_name(cwd)
        cwd_dir.mkdir(parents=True)
        path = str(cwd_dir / "2026-08-10T00-00-00_v3-session.jsonl")
        Path(path).write_text("".join(_v3_lines(cwd)), encoding="utf-8")

        listed = await repo.list({"cwd": cwd})
        assert len(listed) == 1
        assert listed[0]["sourceFormat"] == 3
        assert listed[0]["id"] == "v3-session"
        assert listed[0]["createdAt"] == _iso_ms("2026-08-10T00:00:00+00:00")

        session = await repo.open(listed[0])
        assert await session.get_name() == "Migrated"

    @pytest.mark.asyncio
    async def test_v3_file_always_migrates(self, tmp_path):
        repo = JsonlSessionRepo(str(tmp_path / "sessions"))
        path = str(tmp_path / "v3.jsonl")
        _write_v3(tmp_path, path)

        session = await repo.open(_v3_metadata(path, str(tmp_path)))
        assert await session.get_name() == "Migrated"
        assert Path(f"{path}.bak").exists()

    @pytest.mark.asyncio
    async def test_invalid_v3_file_keeps_original(self, tmp_path):
        repo = JsonlSessionRepo(str(tmp_path / "sessions"))
        path = str(tmp_path / "v3.jsonl")
        _write_v3(tmp_path, path)
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write('{"type":"message","id":"broken"\n')
        original = Path(path).read_bytes()

        with pytest.raises(SessionError) as excinfo:
            await repo.open(_v3_metadata(path, str(tmp_path)))
        assert excinfo.value.code == "invalid_entry"
        assert Path(path).read_bytes() == original
        assert not Path(f"{path}.bak").exists()

    @pytest.mark.asyncio
    async def test_custom_message_converts_to_custom_role_message(self, tmp_path):
        repo = JsonlSessionRepo(str(tmp_path / "sessions"))
        path = str(tmp_path / "v3.jsonl")
        Path(path).write_text(
            "".join(_v3_lines(str(tmp_path), custom_message=True)),
            encoding="utf-8",
        )

        session = await repo.open(_v3_metadata(path, str(tmp_path)))
        custom = await session.find_entry({"customType": "note"})
        assert custom is None
        messages = await session.find_entries({"type": "message", "order": "oldestFirst"})
        assert messages[-1]["message"]["role"] == "custom"
        assert messages[-1]["message"]["customType"] == "note"

    @pytest.mark.asyncio
    async def test_fork_from_v3_source(self, tmp_path):
        repo = JsonlSessionRepo(str(tmp_path / "sessions"))
        path = str(tmp_path / "v3.jsonl")
        _write_v3(tmp_path, path)
        metadata = _v3_metadata(path, str(tmp_path))

        fork = await repo.fork(
            metadata,
            {
                "scope": "branch",
                "entryId": "tail",
                "position": "at",
                "id": "v4-fork",
                "cwd": str(tmp_path),
            },
        )
        assert await fork.get_name() == "Migrated"
        assert await fork.get_label("root") == "checkpoint"
        assert [entry["id"] for entry in await fork.find_entries({"order": "oldestFirst"})] == [
            "root",
            "shared",
            "thinking",
            "model",
            "compact",
            "tail",
        ]
