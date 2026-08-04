"""CLI 入口错误处理回归测试。"""

from __future__ import annotations

from types import SimpleNamespace

from pi_coding_agent import _cli
from pi_coding_agent._session_manager import SessionManager
from pi_ai import Model


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


async def test_cli_loads_skills_and_templates_on_startup(tmp_path, monkeypatch):
    """回归：CLI 启动时应调用 SkillLoader/PromptTemplateLoader 的 load()，
    否则 /skill: 与 /模板名 永远找不到资源。"""

    skill_load_calls: list = []
    template_load_calls: list = []

    original_skill_load = _cli.SkillLoader.load
    original_template_load = _cli.PromptTemplateLoader.load

    def skill_load(self, *args, **kwargs):
        skill_load_calls.append(self)
        return original_skill_load(self, *args, **kwargs)

    def template_load(self, *args, **kwargs):
        template_load_calls.append(self)
        return original_template_load(self, *args, **kwargs)

    monkeypatch.setattr(_cli.SkillLoader, "load", skill_load)
    monkeypatch.setattr(_cli.PromptTemplateLoader, "load", template_load)
    monkeypatch.setattr(_cli, "load_settings", lambda cwd: {})
    monkeypatch.setattr(_cli, "_create_runtime", _fake_runtime)

    async def fake_resolve(*_args, **_kwargs):
        return (
            Model(id="faux-1", provider="faux", api="openai-completions"),
            [],
        )

    async def fake_print_mode(*_args, **_kwargs):
        return 0

    def fake_session_create(cwd, **kwargs):
        return SessionManager.in_memory(cwd=cwd)

    monkeypatch.setattr(_cli, "_resolve_initial_model", fake_resolve)
    monkeypatch.setattr(_cli, "run_print_mode", fake_print_mode)
    monkeypatch.setattr(
        _cli.SessionManager, "create", staticmethod(fake_session_create)
    )

    monkeypatch.chdir(tmp_path)
    code = await _cli._async_main(["-p", "hi"])

    assert code == 0
    assert skill_load_calls
    assert template_load_calls
    # 项目目录应为 cwd 的 .pi/skills / .pi/prompts
    assert skill_load_calls[0]._project_dir == tmp_path / ".pi" / "skills"
    assert template_load_calls[0]._project_dir == tmp_path / ".pi" / "prompts"


async def test_cli_loads_extensions_and_warns_on_syntax_error(
    tmp_path, monkeypatch, capsys
):
    """回归（E-04/P17）：CLI 启动加载项目扩展；语法错误扩展输出 stderr
    Warning 且不崩溃；好扩展在 print 模式仍注册（跨模式一致性）。"""
    monkeypatch.setattr(_cli, "load_settings", lambda cwd: {})
    monkeypatch.setattr(_cli, "_create_runtime", _fake_runtime)

    async def fake_resolve(*_args, **_kwargs):
        return (
            Model(id="faux-1", provider="faux", api="openai-completions"),
            [],
        )

    captured_sessions = []

    async def fake_print_mode(session, *_args, **_kwargs):
        captured_sessions.append(session)
        return 0

    def fake_session_create(cwd, **kwargs):
        return SessionManager.in_memory(cwd=cwd)

    monkeypatch.setattr(_cli, "_resolve_initial_model", fake_resolve)
    monkeypatch.setattr(_cli, "run_print_mode", fake_print_mode)
    monkeypatch.setattr(
        _cli.SessionManager, "create", staticmethod(fake_session_create)
    )

    extensions_dir = tmp_path / ".pi" / "extensions"
    extensions_dir.mkdir(parents=True)
    (extensions_dir / "good.py").write_text(
        "def create_extension(api):\n"
        '    api.register_command("hello", '
        '{"handler": lambda ctx, args: "hi", "description": "Say hello"})\n',
        encoding="utf-8",
    )
    (extensions_dir / "bad.py").write_text(
        "def create_extension(api\n", encoding="utf-8"
    )

    monkeypatch.chdir(tmp_path)
    code = await _cli._async_main(["-p", "hi"])

    captured = capsys.readouterr()
    assert code == 0
    assert "Warning: Failed to load extension" in captured.err
    assert "bad.py" in captured.err
    assert captured_sessions
    runner = captured_sessions[0].extension_runner
    assert runner is not None
    assert len(runner.extensions) == 1
    names = [command.name for command in runner.get_registered_commands()]
    assert "hello" in names


def test_read_stdin_tty_returns_none(monkeypatch):
    """回归（C-14）：TTY 下不读 stdin。"""

    class TtyStdin:
        def isatty(self) -> bool:
            return True

        def read(self):
            raise AssertionError("TTY 下不应读取 stdin")

    monkeypatch.setattr(_cli.sys, "stdin", TtyStdin())
    assert _cli._read_stdin() is None


def test_read_stdin_pipe(monkeypatch):
    """回归（C-14）：管道输入被读取并去除首尾空白。"""
    import io

    class PipeStdin(io.StringIO):
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(_cli.sys, "stdin", PipeStdin("read README.md and summarize\n"))
    assert _cli._read_stdin() == "read README.md and summarize"


def test_read_stdin_blank_returns_empty(monkeypatch):
    import io

    class PipeStdin(io.StringIO):
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(_cli.sys, "stdin", PipeStdin("   \n"))
    assert _cli._read_stdin() == ""
