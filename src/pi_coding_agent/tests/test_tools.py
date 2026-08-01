"""工具单元测试。"""

import tempfile
from pathlib import Path

import pytest

from pi_coding_agent.tools._read import create_read_tool
from pi_coding_agent.tools._write import create_write_tool
from pi_coding_agent.tools._edit import create_edit_tool
from pi_coding_agent.tools._bash import create_bash_tool
from pi_coding_agent.tools._grep import create_grep_tool
from pi_coding_agent.tools._find import create_find_tool
from pi_coding_agent.tools._ls import create_ls_tool


class TestReadTool:
    """测试 read 工具。"""

    def test_read_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("line1\nline2\nline3\n")

            tool = create_read_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"path": "test.txt"}))
            assert result.content[0]["text"].endswith("line1\nline2\nline3\n")

    def test_read_with_offset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.txt"
            file_path.write_text("line1\nline2\nline3\nline4\n")

            tool = create_read_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"path": "test.txt", "offset": 2}))
            text = result.content[0]["text"]
            assert "line2" in text
            assert "line1" not in text or "[Lines 2" in text

    def test_read_with_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lines = [f"line{i}" for i in range(100)]
            file_path = Path(tmpdir) / "big.txt"
            file_path.write_text("\n".join(lines))

            tool = create_read_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"path": "big.txt", "limit": 5}))
            text = result.content[0]["text"]
            # 应只返回 5 行
            assert text.count("\n") <= 6  # 5 lines + header

    def test_read_nonexistent_file(self):
        tool = create_read_tool("/tmp")
        import asyncio
        result = asyncio.run(tool.execute("tc1", {"path": "nonexistent.txt"}))
        assert "Error" in result.content[0]["text"]


class TestWriteTool:
    """测试 write 工具。"""

    def test_write_new_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = create_write_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute(
                "tc1", {"path": "output.txt", "content": "hello world"}
            ))
            assert "File written" in result.content[0]["text"]
            assert (Path(tmpdir) / "output.txt").read_text() == "hello world"

    def test_write_overwrite_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "existing.txt"
            file_path.write_text("old content")

            tool = create_write_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute(
                "tc1", {"path": "existing.txt", "content": "new content"}
            ))
            assert "File written" in result.content[0]["text"]
            assert file_path.read_text() == "new content"


class TestEditTool:
    """测试 edit 工具。"""

    def test_edit_simple_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "code.py"
            file_path.write_text("def hello():\n    return 'old'\n")

            diff = (
                "--- a/code.py\n"
                "+++ b/code.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def hello():\n"
                "-    return 'old'\n"
                "+    return 'new'\n"
            )

            tool = create_edit_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"path": "code.py", "diff": diff}))
            assert "File edited" in result.content[0]["text"]
            assert "return 'new'" in file_path.read_text()

    def test_edit_nonexistent_file(self):
        tool = create_edit_tool("/tmp")
        import asyncio
        result = asyncio.run(tool.execute(
            "tc1", {"path": "nonexistent.py", "diff": "dummy"}
        ))
        assert "Error" in result.content[0]["text"]


class TestBashTool:
    """测试 bash 工具。"""

    def test_bash_echo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = create_bash_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"command": "echo hello"}))
            assert "hello" in result.content[0]["text"]

    def test_bash_exit_code(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import sys
            if sys.platform == "win32":
                command = "cmd /c exit 1"
            else:
                command = "exit 1"

            tool = create_bash_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"command": command}))
            rc = result.details.get("exit_code", 0)
            assert rc != 0 or "exit code" in result.content[0]["text"].lower()


class TestGrepTool:
    """测试 grep 工具。"""

    def test_grep_find_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

            tool = create_grep_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"pattern": "def ", "path": "."}))
            assert "def foo" in result.content[0]["text"]
            assert "def bar" in result.content[0]["text"]

    def test_grep_no_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text("hello world\n")

            tool = create_grep_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"pattern": "xyz123", "path": "."}))
            assert "No matches found" in result.content[0]["text"]


class TestFindTool:
    """测试 find 工具。"""

    def test_find_py_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "a.py").write_text("")
            (Path(tmpdir) / "b.txt").write_text("")
            (Path(tmpdir) / "c.py").write_text("")

            tool = create_find_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"pattern": "*.py", "path": "."}))
            assert "a.py" in result.content[0]["text"]
            assert "c.py" in result.content[0]["text"]
            assert "b.txt" not in result.content[0]["text"]


class TestLsTool:
    """测试 ls 工具。"""

    def test_ls_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "file1.txt").write_text("")
            (Path(tmpdir) / "file2.txt").write_text("")
            (Path(tmpdir) / "subdir").mkdir()

            tool = create_ls_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"path": "."}))
            assert "file1.txt" in result.content[0]["text"]
            assert "file2.txt" in result.content[0]["text"]
            # 目录应该有 /
            assert any("subdir" in line and "/" in line
                       for line in result.content[0]["text"].split("\n"))

    def test_ls_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tool = create_ls_tool(tmpdir)
            import asyncio
            result = asyncio.run(tool.execute("tc1", {"path": "."}))
            assert "empty" in result.content[0]["text"].lower()
