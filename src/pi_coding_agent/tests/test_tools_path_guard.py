"""coding-agent write/edit 工具 cwd 越界防护测试。"""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid

import pytest

from pi_agent.env import FileError
from pi_coding_agent.tools import create_edit_tool, create_write_tool


@pytest.fixture
def workdir() -> str:
    path = os.path.join(os.getcwd(), f"pi-test-wguard-{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_write_tool_rejects_outside_cwd(workdir: str) -> None:
    tool = create_write_tool(workdir)
    with pytest.raises(FileError) as excinfo:
        asyncio.run(tool.execute("tc1", {"path": "../escape.txt", "content": "x"}))
    assert excinfo.value.code == "permission_denied"
    assert not os.path.exists(os.path.join(workdir, "..", "escape.txt"))


def test_edit_tool_rejects_outside_cwd(workdir: str) -> None:
    tool = create_edit_tool(workdir)
    with pytest.raises(FileError) as excinfo:
        asyncio.run(
            tool.execute(
                "tc1",
                {"path": "../escape.py", "edits": [{"oldText": "a", "newText": "b"}]},
            )
        )
    assert excinfo.value.code == "permission_denied"


def test_write_tool_allows_within_cwd(workdir: str) -> None:
    tool = create_write_tool(workdir)
    result = asyncio.run(tool.execute("tc1", {"path": "ok.txt", "content": "ok"}))
    assert "Successfully wrote" in result.content[0]["text"]
    assert os.path.exists(os.path.join(workdir, "ok.txt"))
