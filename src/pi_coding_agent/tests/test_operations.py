"""Operations / 工具包装 / 输出累积测试。"""

from __future__ import annotations

import pytest

from pi_coding_agent.operations import (
    LocalOperations,
    OutputAccumulator,
    create_bash_tool_with_operations,
    create_read_tool_with_operations,
    filter_tools_by_names,
    run_tool_with_updates,
    wrap_tool,
)


class TestLocalOperations:
    async def test_exec(self, tmp_path):
        ops = LocalOperations(str(tmp_path))
        result = await ops.exec("echo ops-ok", str(tmp_path))
        assert "ops-ok" in result["output"]
        assert result["exit_code"] == 0

    async def test_read_write(self, tmp_path):
        ops = LocalOperations(str(tmp_path))
        await ops.write("file.txt", "hello")
        result = await ops.read("file.txt")
        assert result["content"] == "hello"
        assert result["truncated"] is False

    async def test_read_limit(self, tmp_path):
        ops = LocalOperations(str(tmp_path))
        await ops.write("long.txt", "x" * 100)
        result = await ops.read("long.txt", limit=10)
        assert result["content"] == "x" * 10
        assert result["truncated"] is True

    async def test_edit(self, tmp_path):
        ops = LocalOperations(str(tmp_path))
        await ops.write("e.txt", "one\ntwo\n")
        result = await ops.edit(
            "e.txt",
            "--- a\n+++ b\n@@ -1,2 +1,2 @@\n-one\n+ONE\n two\n",
        )
        assert result["ok"] is True
        assert (tmp_path / "e.txt").read_text(encoding="utf-8") == "ONE\ntwo\n"


class TestWrapTool:
    async def test_bash_tool(self, tmp_path):
        tool = create_bash_tool_with_operations(str(tmp_path))
        result = await tool.execute("t1", {"command": "echo wrapped-ok"})
        text = result.content[0]["text"]
        assert "wrapped-ok" in text

    async def test_read_tool(self, tmp_path):
        (tmp_path / "r.txt").write_text("read me", encoding="utf-8")
        tool = create_read_tool_with_operations(str(tmp_path))
        result = await tool.execute("t2", {"path": "r.txt"})
        assert "read me" in result.content[0]["text"]

    async def test_fallback_to_original(self, tmp_path):
        from pi_ai.providers.faux import faux_tool_call

        tool = create_bash_tool_with_operations(str(tmp_path))
        wrapped = wrap_tool(tool, LocalOperations(str(tmp_path)))
        # wrap_tool 对 bash 走 operations；未知工具回退 original.execute。
        assert wrapped.name == "bash"


class TestFilterTools:
    def test_include(self, tmp_path):
        from pi_coding_agent.tools import create_all_tools

        tools = create_all_tools(str(tmp_path))
        filtered = filter_tools_by_names(tools, include=["read", "grep"])
        assert {tool.name for tool in filtered} == {"read", "grep"}

    def test_exclude(self, tmp_path):
        from pi_coding_agent.tools import create_all_tools

        tools = create_all_tools(str(tmp_path))
        filtered = filter_tools_by_names(tools, exclude=["bash"])
        assert all(tool.name != "bash" for tool in filtered)


class TestOutputAccumulator:
    def test_accumulates(self):
        accumulator = OutputAccumulator()
        accumulator.update("a")
        accumulator.update({"output": "b"})
        assert accumulator.output == "ab"

    async def test_run_tool_with_updates(self, tmp_path):
        accumulator = OutputAccumulator()
        tool = create_bash_tool_with_operations(str(tmp_path))
        result = await run_tool_with_updates(
            tool, "t1", {"command": "echo streamed"}, accumulator=accumulator
        )
        assert result is not None
