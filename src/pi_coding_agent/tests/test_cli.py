"""CLI 入口错误处理回归测试。"""

from __future__ import annotations

from pi_coding_agent import _cli


class _FakeRuntime:
    """仅提供 set_agent_stream_fn 所需的最小 stream 属性。"""

    stream = staticmethod(lambda *args, **kwargs: None)


async def _fake_runtime() -> _FakeRuntime:
    return _FakeRuntime()


async def test_unknown_provider_returns_friendly_error(monkeypatch, capsys):
    """回归：--provider bogus 应输出友好错误并返回 1，而不是抛 traceback。"""

    async def fake_resolve(*_args, **_kwargs):
        raise ValueError(
            'Unknown provider "bogus". Use --list-models to see available providers/models.'
        )

    monkeypatch.setattr(_cli, "_create_runtime", _fake_runtime)
    monkeypatch.setattr(_cli, "_resolve_initial_model", fake_resolve)

    code = await _cli._async_main(
        ["--provider", "bogus", "--model", "x", "-p", "hi"]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "Unknown provider" in captured.err
    assert "Traceback" not in captured.err
