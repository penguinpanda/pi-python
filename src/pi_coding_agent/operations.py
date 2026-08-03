"""Operations 接口 + 工具包装器（对齐 TS tools/tool-definition-wrapper.ts）。

工具的默认执行后端为本地文件系统 + 本地 shell；可替换为 SSH / 容器等远程后端。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict

from pi_agent import AgentTool, AgentToolResult
from pi_ai import TextContent


class BashResult(TypedDict):
    output: str
    exit_code: int | None
    canceled: bool


class ReadResult(TypedDict):
    content: str
    truncated: bool


class EditResult(TypedDict):
    ok: bool
    error: str | None


class BashOperations(Protocol):
    async def exec(
        self, command: str, cwd: str, *, timeout: int = 120
    ) -> BashResult: ...


class ReadOperations(Protocol):
    async def read(self, path: str, *, limit: int | None = None) -> ReadResult: ...


class WriteOperations(Protocol):
    async def write(self, path: str, content: str) -> None: ...


class EditOperations(Protocol):
    async def edit(self, path: str, diff: str) -> EditResult: ...


class LocalOperations:
    """默认本地后端：本地 shell + 文件系统。"""

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = Path(self.cwd) / candidate
        return candidate.resolve()

    async def exec(
        self, command: str, cwd: str | None = None, *, timeout: int = 120
    ) -> BashResult:
        from .tools._bash import _run_command

        result = await _run_command(command, cwd or self.cwd, timeout)
        return {
            "output": result["output"],
            "exit_code": result["exit_code"],
            "canceled": result.get("canceled", False),
        }

    async def read(self, path: str, *, limit: int | None = None) -> ReadResult:
        content = self._resolve(path).read_text(encoding="utf-8", errors="replace")
        truncated = False
        if limit is not None and len(content) > limit:
            content = content[:limit]
            truncated = True
        return {"content": content, "truncated": truncated}

    async def write(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def edit(self, path: str, diff: str) -> EditResult:
        from .tools._edit import _apply_diff

        target = self._resolve(path)
        try:
            original = target.read_text(encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        patched = _apply_diff(original, diff)
        if patched is None:
            return {"ok": False, "error": "Failed to apply diff"}
        target.write_text(patched, encoding="utf-8")
        return {"ok": True, "error": None}


def wrap_tool(tool: AgentTool, operations: LocalOperations) -> AgentTool:
    """用可插拔 operations 后端包装工具。"""

    async def execute(tool_call_id: str, params: dict, signal=None, on_update=None):
        name = tool.name
        if name == "bash":
            result = await operations.exec(
                params["command"],
                timeout=int(params.get("timeout", 120)),
            )
            status = "completed"
            if result["canceled"]:
                status = "canceled"
            elif result["exit_code"] != 0:
                status = f"exit code {result['exit_code']}"
            text = result["output"] if result["output"].strip() else "(no output)"
            return AgentToolResult(
                content=[TextContent(type="text", text=text)],
                details={
                    "exit_code": result["exit_code"],
                    "canceled": result["canceled"],
                    "truncated": False,
                },
            )
        if name == "read":
            result = await operations.read(params["path"])
            return AgentToolResult(
                content=[TextContent(type="text", text=result["content"])],
                details={"truncated": result["truncated"]},
            )
        if name == "write":
            await operations.write(params["path"], params["content"])
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Wrote {params['path']}")],
                details={},
            )
        if name == "edit":
            result = await operations.edit(params["path"], params["diff"])
            if not result["ok"]:
                return AgentToolResult(
                    content=[TextContent(type="text", text=f"Error: {result['error']}")],
                    details={"error": "diff_apply_failed"},
                )
            return AgentToolResult(
                content=[TextContent(type="text", text=f"Edited {params['path']}")],
                details={},
            )
        return await tool.execute(tool_call_id, params, signal, on_update)

    return AgentTool(
        name=tool.name,
        description=tool.description,
        input_schema=tool.input_schema,
        label=tool.label,
        execute=execute,
    )


def create_bash_tool_with_operations(cwd: str, operations: LocalOperations | None = None) -> AgentTool:
    from .tools._bash import create_bash_tool

    return wrap_tool(create_bash_tool(cwd), operations or LocalOperations(cwd))


def create_read_tool_with_operations(cwd: str, operations: LocalOperations | None = None) -> AgentTool:
    from .tools._read import create_read_tool

    return wrap_tool(create_read_tool(cwd), operations or LocalOperations(cwd))


def filter_tools_by_names(
    tools: list[AgentTool],
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[AgentTool]:
    """工具白名单 / 黑名单过滤。"""
    if include is not None:
        include_set = set(include)
        tools = [tool for tool in tools if tool.name in include_set]
    if exclude:
        exclude_set = set(exclude)
        tools = [tool for tool in tools if tool.name not in exclude_set]
    return tools


class OutputAccumulator:
    """累积流式工具输出（on_update 回调）。"""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def update(self, data: Any) -> None:
        if isinstance(data, str):
            self.parts.append(data)
        elif isinstance(data, dict) and data.get("output"):
            self.parts.append(str(data["output"]))

    @property
    def output(self) -> str:
        return "".join(self.parts)


async def run_tool_with_updates(
    tool: AgentTool,
    tool_call_id: str,
    params: dict,
    *,
    signal=None,
    accumulator: OutputAccumulator | None = None,
) -> AgentToolResult:
    """执行工具并透传 on_update（流式输出）。"""

    def _on_update(data: Any) -> None:
        if accumulator is not None:
            accumulator.update(data)

    return await tool.execute(tool_call_id, params, signal, _on_update)


__all__ = [
    "BashResult",
    "ReadResult",
    "EditResult",
    "BashOperations",
    "ReadOperations",
    "WriteOperations",
    "EditOperations",
    "LocalOperations",
    "wrap_tool",
    "create_bash_tool_with_operations",
    "create_read_tool_with_operations",
    "filter_tools_by_names",
    "OutputAccumulator",
    "run_tool_with_updates",
]
