"""Hello Tool - Minimal custom tool example (Python port of hello.ts)."""

from pi_coding_agent import ExtensionAPI, ToolDefinition


def create_extension(pi: ExtensionAPI):
    def execute(tool_call_id, params, signal, on_update, ctx):
        return {
            "content": [{"type": "text", "text": f"Hello, {params['name']}!"}],
            "details": {"greeted": params["name"]},
        }

    pi.register_tool(
        ToolDefinition(
            name="hello",
            label="Hello",
            description="A simple greeting tool",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Name to greet"}},
                "required": ["name"],
            },
            execute=execute,
        )
    )
