"""Structured Output Tool - terminate: true ends the turn on the tool call.

Python port of structured-output.ts。
"""

from pi_coding_agent import ExtensionAPI, ToolDefinition


def create_extension(pi: ExtensionAPI):
    def execute(tool_call_id, params, signal=None, on_update=None, ctx=None):
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Saved structured output: {params['headline']}",
                }
            ],
            "details": {
                "headline": params["headline"],
                "summary": params.get("summary", ""),
                "actionItems": params.get("actionItems", []),
            },
            "terminate": True,
        }

    pi.register_tool(
        ToolDefinition(
            name="structured_output",
            label="Structured Output",
            description=(
                "Return a final structured answer. Use this as your last action when "
                "the user asks for structured output or a machine-readable summary."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "headline": {"type": "string", "description": "Short title for the result"},
                    "summary": {"type": "string", "description": "One-paragraph summary"},
                    "actionItems": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete next steps or key bullets",
                    },
                },
                "required": ["headline", "summary", "actionItems"],
            },
            execute=execute,
        )
    )
