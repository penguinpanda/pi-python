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
async def test_full_output_file_includes_chunks_after_threshold(tmp_path, monkeypatch):
    """跨 chunk 输出必须在触发阈值后继续写入 full-output 文件。"""

    class _FakeEnv:
        def __init__(self) -> None:
            self.full_path = str(tmp_path / "full.log")

        async def exec(self, command, options):
            options.on_stdout("a" * 20)
            options.on_stdout("b" * 20)
            return True, type("_Shell", (), {"exit_code": 0})()

        async def create_temp_file(self, options):
            return True, self.full_path

        async def append_file(self, path, content):
            existing = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
            Path(path).write_text(existing + content, encoding="utf-8")
            return True, None

    monkeypatch.setattr(so, "DEFAULT_MAX_BYTES", 8)
    monkeypatch.setattr(so, "DEFAULT_MAX_LINES", 2)
    ok, result = await so.execute_shell_with_capture(_FakeEnv(), "ignored")
    assert ok is True
    assert result.full_output_path.endswith("full.log")
    assert Path(result.full_output_path).read_text(encoding="utf-8") == ("a" * 20) + ("b" * 20)


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
