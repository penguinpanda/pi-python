"""_ensure_tool 归档提取 zip-slip 防护测试。"""

from __future__ import annotations

import os
import shutil
import uuid
import zipfile

import pytest

from pi_coding_agent.tools._ensure_tool import _extract_archive


@pytest.fixture
def workdir() -> str:
    path = os.path.join(os.getcwd(), f"pi-test-archive-{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_zip_slip_member_rejected(workdir: str) -> None:
    dest = os.path.join(workdir, "dest")
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(workdir, "evil.zip")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "pwned")
        zf.writestr("ok.txt", "fine")
    with pytest.raises(RuntimeError, match="escapes"):
        _extract_archive(__import__("pathlib").Path(archive), __import__("pathlib").Path(dest))
    assert not os.path.exists(os.path.join(workdir, "evil.txt"))


def test_zip_absolute_member_rejected(workdir: str) -> None:
    dest = os.path.join(workdir, "dest")
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(workdir, "abs.zip")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("/tmp/pi-abs-escape.txt", "pwned")
    with pytest.raises(RuntimeError, match="escapes"):
        _extract_archive(__import__("pathlib").Path(archive), __import__("pathlib").Path(dest))


def test_zip_normal_extract_succeeds(workdir: str) -> None:
    dest = os.path.join(workdir, "dest")
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(workdir, "good.zip")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("tool.exe", "binary")
    _extract_archive(__import__("pathlib").Path(archive), __import__("pathlib").Path(dest))
    assert os.path.exists(os.path.join(dest, "tool.exe"))
