"""Kimi deferred-tool loading demo - search and activate tools at runtime.

Python port of kimi-deferred-tools.ts。
"""

from pi_coding_agent import ExtensionAPI, ToolDefinition


def _calculate(expr: str) -> str:
    return "42"


def create_extension(pi: ExtensionAPI):
    def calculator_execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        return {
            "content": [{"type": "text", "text": _calculate(params["expr"])}],
            "details": {},
        }

    def tool_search_execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        if "calc" not in params.get("query", "").lower():
            return {
                "content": [{"type": "text", "text": "The relevant tools do not exist."}],
                "details": {"matches": [], "added": []},
            }
        active = list(pi.get_active_tools())
        added = [] if "Calculator" in active else ["Calculator"]
        if added:
            pi.set_active_tools([*active, *added])
        return {
            "content": [{"type": "text", "text": "Success. Found 1 matching tool(s)"}],
            "details": {"matches": ["Calculator"], "added": added},
        }

    pi.register_tool(
        ToolDefinition(
            name="Calculator",
            label="Calculator",
            description="Evaluate a simple arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "An expression such as 100 + 500"}
                },
                "required": ["expr"],
            },
            execute=calculator_execute,
        )
    )
    pi.register_tool(
        ToolDefinition(
            name="tool_search",
            label="Tool Search",
            description="Find and activate tools for a capability.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Capability to search for"}
                },
                "required": ["query"],
            },
            execute=tool_search_execute,
        )
    )

    def on_session_start(event, ctx):
        pi.set_active_tools(["tool_search"])

    pi.on("session_start", on_session_start)
