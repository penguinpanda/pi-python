"""ExtensionLoader 单元测试。"""

from __future__ import annotations

from pathlib import Path

from pi_coding_agent.extensions.loader import ExtensionLoader
from pi_coding_agent.extensions.types import ExtensionRuntime


def _write_extension(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


_SYNC_EXTENSION = """
def create_extension(pi):
    pi.on("message_end", lambda event, ctx: None)
    pi.register_tool({
        "name": "ext-tool",
        "description": "Extension tool",
        "parameters": {"type": "object"},
    })
    pi.register_command("extcmd", {
        "description": "Extension command",
        "getArgumentCompletions": lambda prefix: [{"value": "x"}],
    })
    pi.register_flag("ext-flag", {"type": "boolean", "default": True})
    pi.register_provider("acme", {"api_key": "sk-acme", "models": []})
"""

_ASYNC_EXTENSION = """
async def create_extension(pi):
    pi.on("agent_start", lambda event, ctx: None)
"""


class TestLoadExtension:
    async def test_loads_sync_extension(self, tmp_path):
        path = _write_extension(tmp_path / "myext.py", _SYNC_EXTENSION)
        loader = ExtensionLoader(global_dir=tmp_path / "empty", cwd=str(tmp_path))
        runtime = ExtensionRuntime()
        extension, error = await loader.load_extension(path, runtime)
        assert error is None
        assert extension is not None
        assert extension.path == str(path)
        assert "message_end" in extension.handlers
        assert "ext-tool" in extension.tools
        assert "extcmd" in extension.commands
        assert extension.commands["extcmd"].get_argument_completions is not None
        assert "ext-flag" in extension.flags
        assert extension.providers == [("acme", {"api_key": "sk-acme", "models": []})]

    async def test_loads_async_extension(self, tmp_path):
        path = _write_extension(tmp_path / "asyncext.py", _ASYNC_EXTENSION)
        loader = ExtensionLoader(global_dir=tmp_path / "empty", cwd=str(tmp_path))
        extension, error = await loader.load_extension(path, ExtensionRuntime())
        assert error is None
        assert "agent_start" in extension.handlers

    async def test_imports_sibling_package(self, tmp_path):
        ext_dir = tmp_path / "ext"
        (ext_dir / "ext_lib").mkdir(parents=True)
        (ext_dir / "ext_lib" / "__init__.py").write_text("", encoding="utf-8")
        (ext_dir / "ext_lib" / "helper.py").write_text(
            "def value():\n    return 42\n",
            encoding="utf-8",
        )
        path = _write_extension(
            ext_dir / "main.py",
            "from ext_lib.helper import value\n"
            "def create_extension(pi):\n"
            '    pi.register_tool({"name": "with-deps", "description": f"value={value()}", "parameters": {"type": "object"}})\n',
        )
        loader = ExtensionLoader(global_dir=tmp_path / "empty", cwd=str(tmp_path))
        extension, error = await loader.load_extension(path, ExtensionRuntime())
        assert error is None
        assert extension is not None
        assert extension.tools["with-deps"].description == "value=42"

    async def test_missing_factory_error(self, tmp_path):
        path = _write_extension(tmp_path / "bad.py", "VALUE = 42\n")
        loader = ExtensionLoader(global_dir=tmp_path / "empty", cwd=str(tmp_path))
        extension, error = await loader.load_extension(path, ExtensionRuntime())
        assert extension is None
        assert error is not None
        assert "factory function" in error.error

    async def test_module_error(self, tmp_path):
        path = _write_extension(tmp_path / "boom.py", "raise RuntimeError('boom')\n")
        loader = ExtensionLoader(global_dir=tmp_path / "empty", cwd=str(tmp_path))
        extension, error = await loader.load_extension(path, ExtensionRuntime())
        assert extension is None
        assert "Failed to load extension" in error.error


class TestDiscovery:
    def test_discovers_files_and_subdirs(self, tmp_path):
        _write_extension(tmp_path / "global" / "one.py", _SYNC_EXTENSION)
        _write_extension(tmp_path / "global" / "pkg" / "index.py", _SYNC_EXTENSION)
        _write_extension(tmp_path / "proj" / ".pi" / "extensions" / "two.py", _SYNC_EXTENSION)

        loader = ExtensionLoader(
            global_dir=tmp_path / "global",
            project_dir=tmp_path / "proj" / ".pi" / "extensions",
        )
        paths = loader.discover_all()
        names = {path.name for path in paths}
        assert names == {"one.py", "index.py", "two.py"}

    def test_explicit_path(self, tmp_path):
        _write_extension(tmp_path / "direct" / "ext.py", _SYNC_EXTENSION)
        loader = ExtensionLoader(global_dir=tmp_path / "empty")
        paths = loader.discover_all([str(tmp_path / "direct")])
        assert [path.name for path in paths] == ["ext.py"]


class TestLoadAll:
    async def test_load_collects_errors(self, tmp_path):
        _write_extension(tmp_path / "exts" / "good.py", _SYNC_EXTENSION)
        _write_extension(tmp_path / "exts" / "bad.py", "VALUE = 1\n")
        loader = ExtensionLoader(global_dir=tmp_path / "exts", cwd=str(tmp_path))
        result = await loader.load()
        assert len(result.extensions) == 1
        assert len(result.errors) == 1
        assert result.runtime is not None
