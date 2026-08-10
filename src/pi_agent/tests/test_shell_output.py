"""shell 输出捕获（shell_output.py）单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_agent import shell_output as so
from pi_agent.env import PythonExecutionEnv


async def _shell_available(env: PythonExecutionEnv) -> bool:
    return (await env._resolve_shell())[0]


@pytest.mark.asyncio
async def test_capture_callback_error_returned(tmp_path):
    env = PythonExecutionEnv(str(tmp_path))
    if not await _shell_available(env):
        pytest.skip("No bash shell available")

    def bad_callback(_chunk: str, _progress) -> None:
        raise RuntimeError("output callback exploded")

    ok, result = await so.execute_shell_with_capture(env, "printf 'x'", {"onChunk": bad_callback})
    assert ok is False
    assert "output callback exploded" in str(result)


@pytest.mark.asyncio
async def test_full_output_file_written_on_truncation(tmp_path, monkeypatch):
    env = PythonExecutionEnv(str(tmp_path))
    if not await _shell_available(env):
        pytest.skip("No bash shell available")

    monkeypatch.setattr(so, "DEFAULT_MAX_BYTES", 8)
    monkeypatch.setattr(so, "DEFAULT_MAX_LINES", 2)

    ok, result = await so.execute_shell_with_capture(env, "printf 'a\\nb\\nc\\nd\\ne\\n'")
    assert ok is True
    assert result.truncation.truncated is True
    assert result.full_output_path is not None
    written = Path(result.full_output_path).read_text(encoding="utf-8")
    assert "a" in written
