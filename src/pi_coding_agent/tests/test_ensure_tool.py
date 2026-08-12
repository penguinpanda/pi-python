"""托管工具下载/缓存机制测试。"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from pi_coding_agent.tools import _ensure_tool


def test_get_tool_path_prefers_cached_binary(tmp_path: Path) -> None:
    binary = tmp_path / "fd"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    assert _ensure_tool.get_tool_path("fd", bin_dir=tmp_path) == str(binary)


def test_get_tool_path_falls_back_to_system(monkeypatch) -> None:
    monkeypatch.setattr(_ensure_tool.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert _ensure_tool.get_tool_path("fd", bin_dir=Path("/nonexistent")) == "/usr/bin/fd"


@pytest.mark.asyncio
async def test_with_retry_succeeds_after_failures() -> None:
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("boom")
        return "ok"

    assert await _ensure_tool._with_retry(flaky) == "ok"
    assert calls == 3


@pytest.mark.asyncio
async def test_ensure_tool_offline_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(_ensure_tool, "is_offline_mode_enabled", lambda: True)
    monkeypatch.setattr(_ensure_tool.shutil, "which", lambda _name: None)
    assert await _ensure_tool.ensure_tool("fd", silent=True, bin_dir=Path("/nonexistent")) is None


@pytest.mark.asyncio
async def test_ensure_tool_returns_cached_path(tmp_path: Path) -> None:
    binary = tmp_path / "fd"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    result = await _ensure_tool.ensure_tool("fd", bin_dir=tmp_path)
    assert result == str(binary)


@pytest.mark.asyncio
async def test_download_tool_extracts_and_caches(tmp_path: Path, monkeypatch) -> None:
    async def fake_download(_url: str, dest: Path) -> None:
        with tarfile.open(dest, "w:gz") as archive:
            payload = b"#!/bin/sh\n"
            info = tarfile.TarInfo("fd-v1.0.0-x86_64-unknown-linux-gnu/fd")
            info.size = len(payload)
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(payload))

    async def fake_latest(_repo: str) -> str:
        return "1.0.0"

    monkeypatch.setattr(_ensure_tool, "_latest_version", fake_latest)
    monkeypatch.setattr(_ensure_tool, "_download_file", fake_download)
    config = _ensure_tool._ToolConfig(
        name="fd",
        repo="sharkdp/fd",
        binary_name="fd",
        system_binary_names=("fd",),
        tag_prefix="v",
        get_asset_name=lambda _version, _plat, _arch: "fd-test.tar.gz",
    )
    result = await _ensure_tool._download_tool(config, tmp_path)
    assert result == str(tmp_path / "fd")
    assert (tmp_path / "fd").is_file()
    assert not (tmp_path / "fd-test.tar.gz").exists()
