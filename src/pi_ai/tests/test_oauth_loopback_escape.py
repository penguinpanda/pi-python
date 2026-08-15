"""OAuth loopback 回调页 HTML 转义回归测试。"""

from __future__ import annotations

import pytest


class _FakeWriter:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.chunks.append(data)

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_radius_error_page_escapes_html() -> None:
    from pi_ai.auth.oauth.radius import _respond

    writer = _FakeWriter()
    await _respond(writer, 400, "<script>alert(1)</script>")
    body = b"".join(writer.chunks)
    assert b"<script>" not in body
    assert b"&lt;script&gt;" in body


@pytest.mark.asyncio
async def test_openrouter_error_page_escapes_html() -> None:
    from pi_ai.auth.oauth.openrouter import _respond_openrouter

    writer = _FakeWriter()
    await _respond_openrouter(writer, 400, "<img src=x onerror=alert(1)>")
    body = b"".join(writer.chunks)
    assert b"<img" not in body
    assert b"&lt;img" in body


@pytest.mark.asyncio
async def test_codex_error_page_escapes_html() -> None:
    import asyncio

    from pi_ai.auth.oauth.openai_codex import _respond_html

    writer = _FakeWriter()
    _respond_html(writer, 400, "<b>boom</b>")
    await asyncio.sleep(0)  # 让 _finish 后台任务执行
    body = b"".join(writer.chunks)
    assert b"<b>" not in body
    assert b"&lt;b&gt;" in body
