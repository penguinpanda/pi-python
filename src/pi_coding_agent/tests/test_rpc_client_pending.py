"""RpcClient 子进程退出 / 管道断裂时 pending 请求失败的回归测试。"""

from __future__ import annotations

import asyncio

import pytest

from pi_coding_agent.rpc.rpc_client import RpcClient


class _FakeStream:
    def __init__(self, mode: str) -> None:
        self._mode = mode  # "eof" | "boom"

    async def readline(self) -> bytes:
        if self._mode == "boom":
            raise OSError("pipe broken")
        return b""

    async def read(self, size: int = -1) -> bytes:
        if self._mode == "boom":
            raise OSError("pipe broken")
        return b""


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data.decode("utf-8"))

    async def drain(self) -> None:
        pass


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout_mode: str = "eof",
        exit_code: int | None = None,
        wait_forever: bool = False,
    ) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream(stdout_mode)
        self.stderr = _FakeStream("eof")
        self._exit_code = exit_code
        self._wait_forever = wait_forever
        self.returncode: int | None = None

    async def wait(self) -> int:
        if self._wait_forever:
            while self.returncode is None:
                await asyncio.sleep(0.01)
        if self.returncode is None:
            self.returncode = self._exit_code if self._exit_code is not None else 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 1

    def kill(self) -> None:
        self.returncode = 9


def _patch_exec(monkeypatch, process: _FakeProcess) -> None:
    async def fake_exec(*args, **kwargs):
        return process

    monkeypatch.setattr("pi_coding_agent.rpc.rpc_client.asyncio.create_subprocess_exec", fake_exec)


@pytest.mark.asyncio
async def test_pending_fails_fast_when_process_exits(monkeypatch) -> None:
    """子进程中途退出时,等待中的请求立即失败而不是悬挂到超时。"""
    process = _FakeProcess(exit_code=7)
    _patch_exec(monkeypatch, process)
    client = RpcClient({"command": ["fake"]})
    await client.start()
    try:
        with pytest.raises(RuntimeError, match="exited"):
            await client.send("get_state", timeout=5)
        assert client._pending == {}
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_pending_fails_fast_when_stdout_breaks(monkeypatch) -> None:
    """stdout 管道断裂时,等待中的请求立即失败。"""
    process = _FakeProcess(stdout_mode="boom", wait_forever=True)
    _patch_exec(monkeypatch, process)
    client = RpcClient({"command": ["fake"]})
    await client.start()
    try:
        with pytest.raises(RuntimeError, match="stdout closed"):
            await client.send("get_state", timeout=5)
        assert client._pending == {}
    finally:
        await client.stop()


@pytest.mark.asyncio
async def test_pending_fails_on_stop(monkeypatch) -> None:
    """stop() 后等待中的请求立即失败,而不是悬挂。"""
    process = _FakeProcess(wait_forever=True)
    _patch_exec(monkeypatch, process)
    client = RpcClient({"command": ["fake"]})
    await client.start()
    task = asyncio.create_task(client.send("get_state", timeout=5))
    await asyncio.sleep(0.05)
    await client.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await task
