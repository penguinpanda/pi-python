"""设备代码轮询测试。"""

import asyncio

import pytest

from pi_ai.auth.oauth.device_code import poll_oauth_device_code_flow


@pytest.mark.asyncio
async def test_complete_after_pending():
    calls = {"n": 0}

    async def poll():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "pending"}
        return {"status": "complete", "value": {"access": "tok"}}

    result = await poll_oauth_device_code_flow(
        poll,
        interval_seconds=0.001,
        expires_in_seconds=10,
    )
    assert result == {"access": "tok"}


@pytest.mark.asyncio
async def test_slow_down_uses_server_interval():
    calls = {"n": 0}

    async def poll():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"status": "slow_down", "interval_seconds": 0.001}
        return {"status": "complete", "value": 42}

    assert await poll_oauth_device_code_flow(poll, interval_seconds=5, expires_in_seconds=10) == 42


@pytest.mark.asyncio
async def test_failed_raises():
    async def poll():
        return {"status": "failed", "message": "access denied"}

    with pytest.raises(RuntimeError, match="access denied"):
        await poll_oauth_device_code_flow(poll, expires_in_seconds=10)


@pytest.mark.asyncio
async def test_timeout_raises():
    async def poll():
        return {"status": "pending"}

    with pytest.raises(RuntimeError, match="timed out"):
        await poll_oauth_device_code_flow(
            poll,
            interval_seconds=0.001,
            expires_in_seconds=0,
        )


@pytest.mark.asyncio
async def test_cancel_via_signal():
    signal = asyncio.Event()

    async def poll():
        signal.set()
        return {"status": "pending"}

    with pytest.raises(RuntimeError, match="Login cancelled"):
        await poll_oauth_device_code_flow(
            poll,
            interval_seconds=10,
            expires_in_seconds=10,
            signal=signal,
        )
