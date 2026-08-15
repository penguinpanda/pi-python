"""@path 注入大小/忽略/二进制限制测试。"""

from __future__ import annotations

import os
import shutil
import uuid

import pytest

from pi_coding_agent.file_processor import process_at_files


@pytest.fixture
def workdir() -> str:
    path = os.path.join(os.getcwd(), f"pi-test-atpath-{uuid.uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


async def test_ignores_git_and_node_modules(workdir: str) -> None:
    (os.path.join(workdir, ".git")).replace("", "")
    os.makedirs(os.path.join(workdir, ".git"), exist_ok=True)
    with open(os.path.join(workdir, ".git", "config"), "w", encoding="utf-8") as f:
        f.write("secret")
    os.makedirs(os.path.join(workdir, "node_modules"), exist_ok=True)
    with open(os.path.join(workdir, "node_modules", "index.js"), "w", encoding="utf-8") as f:
        f.write("module")
    with open(os.path.join(workdir, "keep.md"), "w", encoding="utf-8") as f:
        f.write("kept")
    texts, _images = await process_at_files(["@" + workdir], workdir)
    assert len(texts) == 1
    assert "kept" in texts[0]


async def test_skips_oversized_text(workdir: str) -> None:
    big = os.path.join(workdir, "big.txt")
    with open(big, "wb") as f:
        f.write(b"a" * 1_000_001)
    texts, _images = await process_at_files(["@big.txt"], workdir)
    assert texts == []


async def test_skips_binary_content(workdir: str) -> None:
    path = os.path.join(workdir, "blob.bin")
    with open(path, "wb") as f:
        f.write(b"abc\x00def")
    texts, _images = await process_at_files(["@blob.bin"], workdir)
    assert texts == []


async def test_normal_text_still_injected(workdir: str) -> None:
    with open(os.path.join(workdir, "ok.txt"), "w", encoding="utf-8") as f:
        f.write("hello")
    texts, _images = await process_at_files(["@ok.txt"], workdir)
    assert len(texts) == 1
    assert "hello" in texts[0]
