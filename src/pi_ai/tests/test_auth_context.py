"""AuthContext 默认实现测试。"""

import pytest

from pi_ai.auth.context import default_auth_context


@pytest.mark.asyncio
async def test_env_trim_semantics(monkeypatch):
    ctx = default_auth_context()
    monkeypatch.setenv("PI_AUTH_TEST", "  value  ")
    assert await ctx.env("PI_AUTH_TEST") == "  value  "
    monkeypatch.setenv("PI_AUTH_TEST_BLANK", "   ")
    assert await ctx.env("PI_AUTH_TEST_BLANK") is None
    monkeypatch.delenv("PI_AUTH_TEST_MISSING", raising=False)
    assert await ctx.env("PI_AUTH_TEST_MISSING") is None


@pytest.mark.asyncio
async def test_file_exists(tmp_path):
    ctx = default_auth_context()
    target = tmp_path / "creds.json"
    assert await ctx.file_exists(str(target)) is False
    target.write_text("{}")
    assert await ctx.file_exists(str(target)) is True


@pytest.mark.asyncio
async def test_file_exists_expands_tilde(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    ctx = default_auth_context()
    assert await ctx.file_exists("~/pi-test-credentials.json") is False
    (tmp_path / "pi-test-credentials.json").write_text("{}")
    assert await ctx.file_exists("~/pi-test-credentials.json") is True
